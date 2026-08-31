"""
agent.py
--------
LiveKit Agents entrypoint for the Hospital Appointment Voice AI.

Pipeline: Silero VAD -> Deepgram Nova-3 STT (bilingual) -> Claude 3.5 Sonnet
(function-calling) -> Cartesia Sonic 3.5 TTS.

Each LiveKit job (one call session) runs this entrypoint in its own worker
process, giving us the "1 isolated OS process per call session" isolation
required for HIPAA-adjacent zero cross-session data leakage.

Tool design: every FSM transition that requires reading back structured
data to the LLM (slots, appointment details, verification results) is
exposed as a `function_tool`. The LLM is never allowed to call the
HIS-mutating tools (`book_appointment`, `cancel_appointment_tool`,
`confirm_reschedule_tool`) without the FSM having first recorded a verbal
confirmation flag — enforced by checking `ctx.state` before executing.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from typing import Any, Optional
from dotenv import load_dotenv


# load dotenv here
load_dotenv()

from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RunContext,
    WorkerOptions,
    cli,
    function_tool,
)
from livekit.plugins import deepgram, cartesia, silero, openai as lk_openai

from config import (
    CallIntent,
    CallState,
    DoctorSelectionMode,
    HOSPITAL_NAME,
    Language,
    LIMITS,
    LLM_SETTINGS,
    STT_KEYTERMS_AR,
    STT_KEYTERMS_EN,
    SYSTEM_PROMPTS,
    TTS_VOICES,
    VAD_SETTINGS,
    load_notification_config,
    LANGUAGE_ALIASES
)
from escalation import EscalationEngine
from his_client import HISClient
from state_machine import CallContext, HospitalFSM, new_call_context

logger = logging.getLogger("agent")

MOBILE_RE = re.compile(r"^(?:\+?966|0)?5\d{8}$")  # Saudi mobile format


# --------------------------------------------------------------------------- #
# Language selection
# --------------------------------------------------------------------------- #

def handle_language_selection(user_text: str) -> Language | None:
    normalized = user_text.strip().lower()

    for phrase, language in LANGUAGE_ALIASES.items():
        if phrase.lower() in normalized:
            return language

    return None


# --------------------------------------------------------------------------- #
# Notification dispatch (SMS / WhatsApp)
# --------------------------------------------------------------------------- #

async def dispatch_notification(mobile: str, message: str) -> bool:
    """Best-effort SMS/WhatsApp dispatch. Failure here must never fail the
    underlying booking/cancel/reschedule operation that already succeeded
    in the HIS — we log and move on, informing the caller verbally instead."""
    import httpx

    try:
        cfg = load_notification_config()
    except RuntimeError:
        logger.error("Notification config missing; skipping SMS dispatch (booking still valid).")
        return False

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(
                f"{cfg.api_base_url}/messages",
                headers={"Authorization": f"Bearer {cfg.api_key}"},
                json={"to": mobile, "from": cfg.sender_id, "body": message},
            )
            resp.raise_for_status()
        return True
    except Exception:
        logger.exception("Notification dispatch failed for a completed HIS transaction.")
        return False


# --------------------------------------------------------------------------- #
# Session wrapper: holds the immutable CallContext + FSM + collaborators for
# one call, and exposes tool methods bound to `self` for the Agent class.
# --------------------------------------------------------------------------- #

class HospitalCallSession:
    def __init__(self, his_client: HISClient):
        self.his_client = his_client
        self.escalation_engine = EscalationEngine()
        self.fsm = HospitalFSM(his_client, self.escalation_engine)
        self.ctx: CallContext = new_call_context()

    def _set(self, new_ctx: CallContext) -> None:
        self.ctx = new_ctx
        if new_ctx.is_escalated:
            logger.warning(
                "Session %s escalated: %s", new_ctx.session_id, new_ctx.escalation_reason
            )

    async def guard_emergency(self, transcript: str) -> bool:
        """Call at the top of every tool/turn. Returns True if an emergency
        was detected and the session is now escalated."""
        escalated = await self.fsm.check_emergency_and_escalate(self.ctx, transcript)
        if escalated is not None:
            self._set(escalated)
            return True
        return False


# --------------------------------------------------------------------------- #
# Agent definition: tools exposed to the LLM
# --------------------------------------------------------------------------- #

class HospitalAgent(Agent):
    def __init__(self, session_state: HospitalCallSession, language: Language):
        self.state = session_state
        instructions = SYSTEM_PROMPTS[language].format(
            hospital_name=HOSPITAL_NAME,
            patient_status="verified" if self.state.ctx.is_verified else "not yet verified",
        )
        super().__init__(instructions=instructions)



          # -------------------------------------------------------------------
    # Language selection
    # -------------------------------------------------------------------

    @function_tool()
    async def select_language_tool(
        self,
        run_ctx: RunContext,
        language: str,
    ) -> str:
        """
        Select the caller's preferred language.

        Args:
            language: The requested language.
                Supported values are English, Arabic, Malayalam, or Tamil.
        """

        selected_language = handle_language_selection(language)

        if selected_language is None:
            return (
                "Unsupported language. Ask the caller to choose "
                "English, Arabic, Malayalam, or Tamil."
            )

        # Update FSM
        self.state._set(
            self.state.fsm.set_language(
                self.state.ctx,
                selected_language,
            )
        )

        # Change TTS voice/language
        voice_config = TTS_VOICES[selected_language]

        if self.session.tts is not None:
            self.session.tts.update_options(
                language=voice_config.language,
                voice=voice_config.voice_id,
            )

        # Change agent/system prompt
        await self.update_instructions(
            SYSTEM_PROMPTS[selected_language].format(
                hospital_name=HOSPITAL_NAME,
                patient_status=(
                    "verified"
                    if self.state.ctx.is_verified
                    else "not yet verified"
                ),
            )
        )

        logger.info(
            "Session %s language changed to %s",
            self.state.ctx.session_id,
            selected_language.value,
        )

        return (
            f"Language successfully changed to "
            f"{selected_language.name}. "
            "Continue speaking in that language and ask "
            "what the caller would like help with."
        )

    # ------------------------------------------------------------------- #
    # Identity & verification
    # ------------------------------------------------------------------- #

    @function_tool()
    async def set_call_intent(self, run_ctx: RunContext, intent: str) -> str:
        """Record what the caller wants to do.

        Args:
            intent: One of 'book_new_appointment', 'check_appointment',
                'reschedule_appointment', 'cancel_appointment',
                'new_patient_registration'.
        """
        try:
            parsed = CallIntent(intent)
        except ValueError:
            return f"Invalid intent '{intent}'. Ask the caller to clarify what they need."
        self.state._set(self.state.fsm.set_intent(self.state.ctx, parsed))
        return f"Intent set to {parsed.value}. Proceed to the next required step."

    @function_tool()
    async def verify_patient_identity(
        self, run_ctx: RunContext, mobile_number: str, national_id: Optional[str] = None
    ) -> str:
        """Verify a registered patient by mobile number against the HIS.

        Args:
            mobile_number: Caller's mobile number, digits only or with +966/0 prefix.
            national_id: Optional national ID / Iqama, used to disambiguate
                if multiple patient records share a mobile number.
        """
        if await self.state.guard_emergency(mobile_number):
            return "EMERGENCY DETECTED. Escalating to a human receptionist immediately."

        cleaned = re.sub(r"[\s\-()]", "", mobile_number)
        if not MOBILE_RE.match(cleaned):
            return (
                "That doesn't look like a valid Saudi mobile number. "
                "Ask the caller to repeat their 10-digit mobile number."
            )
        logger.info("Starting patient verification. state=%s",self.state.ctx.state.value,)

        new_ctx = await self.state.fsm.verify_patient(self.state.ctx, cleaned, national_id)
        self.state._set(new_ctx)

        if new_ctx.is_escalated:
            return "Verification failed twice. Escalating to a human receptionist now."
        if not new_ctx.is_verified:
            return (
                f"No matching patient record found (attempt "
                f"{new_ctx.verification_attempts}/{LIMITS.max_verification_attempts}). "
                "Ask the caller to double-check their mobile number, or offer new patient registration."
            )
        return f"Patient verified: {new_ctx.patient_name}. Proceed with the requested flow."

    # ------------------------------------------------------------------- #
    # Booking flow
    # ------------------------------------------------------------------- #

    @function_tool()
    async def list_departments_tool(self, run_ctx: RunContext) -> str:
        """List hospital departments available for booking."""
        result = await self.state.his_client.list_departments()
        if not result.ok:
            return "Could not retrieve the department list right now. Offer to transfer to a receptionist."
        depts = (result.data or {}).get("departments", [])
        return "; ".join(f"{d['department_id']}:{d['name']}" for d in depts) or "No departments returned."

    @function_tool()
    async def select_department_tool(
        self, run_ctx: RunContext, department_id: str, department_name: str
    ) -> str:
        """Record the caller's chosen department.

        Args:
            department_id: The department's internal ID from list_departments_tool.
            department_name: Human-readable department name to read back to the caller.
        """
        new_ctx = await self.state.fsm.select_department(
            self.state.ctx, department_id, department_name
        )
        self.state._set(new_ctx)
        return f"Department set to {department_name}. Now ask how they'd like to pick a doctor."

    @function_tool()
    async def select_doctor_tool(
        self,
        run_ctx: RunContext,
        selection_mode: str,
        specific_doctor_id: Optional[str] = None,
        specific_doctor_name: Optional[str] = None,
    ) -> str:
        """Choose a doctor for the appointment.

        Args:
            selection_mode: One of 'earliest_available', 'previous_doctor', 'specific_doctor'.
            specific_doctor_id: Required if selection_mode is 'specific_doctor'.
            specific_doctor_name: Required if selection_mode is 'specific_doctor'.
        """
        try:
            mode = DoctorSelectionMode(selection_mode)
        except ValueError:
            return f"Invalid selection_mode '{selection_mode}'."

        new_ctx = await self.state.fsm.select_doctor(
            self.state.ctx,
            mode,
            specific_doctor_id=specific_doctor_id,
            specific_doctor_name=specific_doctor_name,
        )
        self.state._set(new_ctx)

        if new_ctx.is_escalated:
            return "HIS API unavailable. Escalating to a human receptionist now."
        if mode == DoctorSelectionMode.PREVIOUS_DOCTOR and not new_ctx.doctor_id:
            return "No previous doctor on record for this department. Ask if they'd like the earliest available doctor instead."
        return f"Doctor set to {new_ctx.doctor_name or '(earliest available, TBD at search)'}. Now search for slots."

    @function_tool()
    async def search_slots_tool(self, run_ctx: RunContext) -> str:
        """Search for the next available appointment slots with the selected doctor."""
        new_ctx = await self.state.fsm.search_slots(self.state.ctx)
        self.state._set(new_ctx)

        if new_ctx.is_escalated:
            return "HIS API unavailable. Escalating to a human receptionist now."
        if not new_ctx.offered_slots:
            return "No open slots found. Tell the caller and offer to try a different doctor or department."

        lines = [
            f"{i+1}) slot_id={s.slot_id} | Dr. {s.doctor_name} | {s.start_time} | {s.location}"
            for i, s in enumerate(new_ctx.offered_slots)
        ]
        return "Offer these slots (max 3), spoken naturally as day/date/time:\n" + "\n".join(lines)

    @function_tool()
    async def hold_slot_tool(self, run_ctx: RunContext, slot_id: str) -> str:
        """Place a temporary hold on the caller's chosen slot while they confirm verbally.

        Args:
            slot_id: The slot_id the caller selected from search_slots_tool results.
        """
        new_ctx = await self.state.fsm.hold_selected_slot(self.state.ctx, slot_id)
        self.state._set(new_ctx)

        if new_ctx.is_escalated:
            return "HIS API unavailable. Escalating to a human receptionist now."
        if new_ctx.state == CallState.SEARCH_SLOTS:
            return "That slot was just taken. Apologize and search again."

        s = new_ctx.selected_slot
        return (
            f"Slot held for {LIMITS.slot_hold_duration_seconds} seconds. "
            f"Read back to the caller: Dr. {s.doctor_name}, {s.department}, {s.start_time}, "
            f"{s.location}. Ask for explicit confirmation before calling confirm_booking_tool."
        )

    @function_tool()
    async def confirm_booking_tool(self, run_ctx: RunContext, caller_confirmed: bool) -> str:
        """Finalize or cancel the held booking based on the caller's explicit verbal confirmation.

        Args:
            caller_confirmed: True only if the caller explicitly said yes/confirm/correct.
        """
        new_ctx = await self.state.fsm.confirm_booking(self.state.ctx, caller_confirmed)
        self.state._set(new_ctx)

        if new_ctx.is_escalated:
            return "HIS API unavailable while booking. Escalating to a human receptionist now."
        if not caller_confirmed:
            return "Booking not confirmed. Return to slot search."
        if new_ctx.state != CallState.BOOKING_COMPLETE:
            return "Booking failed (slot no longer available). Search for new slots and inform the caller."

        message = (
            f"Your appointment with Dr. {new_ctx.selected_slot.doctor_name} on "
            f"{new_ctx.selected_slot.start_time} is confirmed. Confirmation ID: "
            f"{new_ctx.confirmed_appointment_id}."
        )
        sms_sent = await dispatch_notification(new_ctx.mobile, message)
        suffix = "" if sms_sent else " (SMS dispatch failed — mention this to the caller and read the details aloud instead.)"
        return f"Booking confirmed: appointment_id={new_ctx.confirmed_appointment_id}.{suffix} Tell the caller it's booked."

    # ------------------------------------------------------------------- #
    # Check appointment
    # ------------------------------------------------------------------- #

    @function_tool()
    async def retrieve_appointments_tool(self, run_ctx: RunContext) -> str:
        """Fetch the verified patient's upcoming appointments."""
        new_ctx = await self.state.fsm.retrieve_appointments(self.state.ctx)
        self.state._set(new_ctx)

        if new_ctx.is_escalated:
            return "HIS API unavailable. Escalating to a human receptionist now."
        if not new_ctx.upcoming_appointments:
            return "No upcoming appointments found. Inform the caller."

        lines = [
            f"{a.get('appointment_id')}: Dr. {a.get('doctor_name')} ({a.get('department')}) "
            f"on {a.get('start_time')} at {a.get('location')}"
            for a in new_ctx.upcoming_appointments
        ]
        return "Upcoming appointments (nearest first):\n" + "\n".join(lines)

    # ------------------------------------------------------------------- #
    # Reschedule / cancel (Release 2)
    # ------------------------------------------------------------------- #

    @function_tool()
    async def select_appointment_to_modify_tool(
        self, run_ctx: RunContext, appointment_id: str
    ) -> str:
        """Select which existing appointment the caller wants to reschedule or cancel.

        Args:
            appointment_id: appointment_id from retrieve_appointments_tool results.
        """
        new_ctx = await self.state.fsm.select_appointment_to_modify(
            self.state.ctx, appointment_id
        )
        self.state._set(new_ctx)
        if new_ctx.intent == CallIntent.RESCHEDULE_APPOINTMENT:
            return "Now search for new slots for this appointment's doctor."
        return "Now read back the appointment details and ask for explicit cancellation confirmation."

    @function_tool()
    async def confirm_reschedule_tool(
        self, run_ctx: RunContext, new_slot_id: str, caller_confirmed: bool
    ) -> str:
        """Finalize or abandon a reschedule based on explicit caller confirmation.

        Args:
            new_slot_id: The new slot_id the caller selected.
            caller_confirmed: True only if the caller explicitly confirmed.
        """
        new_ctx = await self.state.fsm.confirm_reschedule(
            self.state.ctx, new_slot_id, caller_confirmed
        )
        self.state._set(new_ctx)

        if new_ctx.is_escalated:
            return "HIS API unavailable while rescheduling. Escalating to a human receptionist now."
        if not caller_confirmed:
            return "Reschedule not confirmed. Search for a different slot."
        if new_ctx.state != CallState.BOOKING_COMPLETE:
            return "Reschedule failed. Inform the caller and search again."

        message = f"Your appointment has been rescheduled. New confirmation ID: {new_ctx.confirmed_appointment_id}."
        sms_sent = await dispatch_notification(new_ctx.mobile, message)
        suffix = "" if sms_sent else " (SMS dispatch failed — read the new details aloud instead.)"
        return f"Reschedule confirmed: appointment_id={new_ctx.confirmed_appointment_id}.{suffix}"

    @function_tool()
    async def confirm_cancellation_tool(
        self, run_ctx: RunContext, caller_confirmed: bool
    ) -> str:
        """Finalize or abandon a cancellation based on explicit caller confirmation.

        Args:
            caller_confirmed: True only if the caller explicitly confirmed.
        """
        new_ctx = await self.state.fsm.confirm_cancellation(self.state.ctx, caller_confirmed)
        self.state._set(new_ctx)

        if new_ctx.is_escalated:
            return "HIS API unavailable while cancelling. Escalating to a human receptionist now."
        if not caller_confirmed:
            return "Cancellation not confirmed. Ask if there's anything else."
        if new_ctx.state != CallState.CALL_ENDED:
            return "Cancellation failed. Offer to transfer to a receptionist."

        message = "Your appointment has been cancelled as requested."
        await dispatch_notification(new_ctx.mobile, message)
        return "Appointment cancelled. Confirm with the caller and ask if there's anything else, then end the call."

    # ------------------------------------------------------------------- #
    # New patient registration (Release 3)
    # ------------------------------------------------------------------- #

    @function_tool()
    async def collect_registration_details_tool(
        self,
        run_ctx: RunContext,
        full_name: Optional[str] = None,
        mobile: Optional[str] = None,
        national_id: Optional[str] = None,
        date_of_birth: Optional[str] = None,
        gender: Optional[str] = None,
        nationality: Optional[str] = None,
    ) -> str:
        """Incrementally collect new-patient registration details. Call this
        repeatedly as the caller provides each piece of information.

        Args:
            full_name: Patient's full legal name.
            mobile: Saudi mobile number.
            national_id: National ID or Iqama number.
            date_of_birth: ISO 8601 date (YYYY-MM-DD).
            gender: 'male' or 'female'.
            nationality: Patient's nationality.
        """
        payload = {
            k: v
            for k, v in {
                "full_name": full_name,
                "mobile": mobile,
                "national_id": national_id,
                "date_of_birth": date_of_birth,
                "gender": gender,
                "nationality": nationality,
            }.items()
            if v is not None
        }
        new_ctx = await self.state.fsm.collect_registration_details(self.state.ctx, payload)
        self.state._set(new_ctx)

        if new_ctx.state == CallState.SEND_OTP:
            return "All required registration details collected. Now send the OTP."
        missing = {"full_name", "mobile", "date_of_birth", "gender", "nationality"} - set(
            new_ctx.registration_payload.keys()
        )
        return f"Still missing: {', '.join(sorted(missing))}. Ask the caller for these next."

    @function_tool()
    async def send_registration_otp_tool(self, run_ctx: RunContext) -> str:
        """Send a one-time verification code via SMS to the mobile number on file."""
        new_ctx = await self.state.fsm.send_registration_otp(self.state.ctx)
        self.state._set(new_ctx)
        if new_ctx.is_escalated:
            return "Could not send OTP. Escalating to a human receptionist now."
        return "OTP sent via SMS. Ask the caller to read back the 6-digit code."

    @function_tool()
    async def verify_registration_otp_tool(self, run_ctx: RunContext, otp_code: str) -> str:
        """Verify the OTP code spoken back by the caller.

        Args:
            otp_code: The digits the caller read out from the SMS.
        """
        digits = re.sub(r"\D", "", otp_code)
        new_ctx = await self.state.fsm.verify_registration_otp(self.state.ctx, digits)
        self.state._set(new_ctx)

        if new_ctx.is_escalated:
            return "OTP verification failed too many times. Escalating to a human receptionist now."
        if new_ctx.state != CallState.CREATE_PATIENT_RECORD:
            return f"Incorrect OTP (attempt {new_ctx.otp_attempts}/{LIMITS.otp_max_attempts}). Ask the caller to try again."
        return "OTP verified. Now create the patient record."

    @function_tool()
    async def create_patient_record_tool(self, run_ctx: RunContext) -> str:
        """Create the new patient record in the HIS after successful OTP verification."""
        new_ctx = await self.state.fsm.create_patient_record(self.state.ctx)
        self.state._set(new_ctx)

        if new_ctx.is_escalated:
            return "Registration failed. Escalating to a human receptionist now."
        if new_ctx.new_mrn:
            return (
                f"New patient record created, MRN {new_ctx.new_mrn}. Welcome the caller "
                "and proceed directly into department selection for their first appointment."
            )
        return (
            "A matching existing record was found (avoided creating a duplicate). "
            "Proceed directly into department selection using the existing record."
        )

    # ------------------------------------------------------------------- #
    # Escalation
    # ------------------------------------------------------------------- #

    @function_tool()
    async def request_human_transfer(self, run_ctx: RunContext, reason: str) -> str:
        """Transfer the caller to a human receptionist.

        Args:
            reason: Brief reason for the transfer (e.g., 'caller asked for a human',
                'repeated misunderstanding', 'complex request outside scope').
        """
        new_ctx = await self.state.fsm.handle_human_request(self.state.ctx)
        self.state._set(new_ctx)
        return f"Transferring now ({reason}). Tell the caller they're being connected to a receptionist."


# --------------------------------------------------------------------------- #
# LiveKit entrypoint
# --------------------------------------------------------------------------- #

async def entrypoint(job_ctx: JobContext) -> None:
    await job_ctx.connect()

    his_client = HISClient()
    session_state = HospitalCallSession(his_client)

    # Cleanup must happen when the LiveKit JOB ends,
    # not when entrypoint() returns.
    async def cleanup() -> None:
        await his_client.aclose()
        logger.info(
            "Session %s ended and resources released.",
            session_state.ctx.session_id,
        )

    job_ctx.add_shutdown_callback(cleanup)

    # Default to English at greeting; the caller's first utterance / an
    # explicit language prompt can switch this via `set_call_intent`-adjacent
    # flow in a fuller implementation. Kept simple here per the FSM's
    # LANGUAGE_CONFIRM state, which agent.py drives via STT language hint.
    initial_language = Language.ENGLISH
    # session_state.ctx = session_state.fsm.set_language(session_state.ctx, initial_language)

    stt_engine = deepgram.STT(
        model="nova-3-medical",
        language="en",
        keyterm=STT_KEYTERMS_EN,
        interim_results=True,
    )

    llm_engine = lk_openai.LLM(
        model=LLM_SETTINGS.model,
        temperature=LLM_SETTINGS.temperature,
    )

    tts_engine = cartesia.TTS(
        voice=TTS_VOICES[initial_language].voice_id,
        language=TTS_VOICES[initial_language].language,
    )

    vad_engine = silero.VAD.load(
        min_speech_duration=VAD_SETTINGS.min_speech_duration,
        min_silence_duration=VAD_SETTINGS.min_silence_duration,
        activation_threshold=VAD_SETTINGS.activation_threshold,
    )

    agent = HospitalAgent(session_state, initial_language)

    session = AgentSession(
        stt=stt_engine,
        llm=llm_engine,
        tts=tts_engine,
        vad=vad_engine,
    )

    async def _guard_turn(state: HospitalCallSession, transcript: str) -> None:
            try:
                await state.guard_emergency(transcript)
            except Exception:
                logger.exception("Emergency guardrail check failed for session %s", state.ctx.session_id)

    @session.on("user_speech_committed")
    def _on_user_speech(msg: Any) -> None:
        # Run emergency-language guardrail on every committed user turn,
        # independent of whatever tool the LLM decides to call. This is the
        # deterministic backstop described in escalation.py.
        transcript = getattr(msg, "text", "") or ""
        asyncio.create_task(_guard_turn(session_state, transcript))

    

    try:
        await session.start(agent=agent, room=job_ctx.room)
        await session.generate_reply(
            instructions=(
                f"Welcome the caller to {HOSPITAL_NAME}. "
                "Do not ask about appointments yet. "
                "Ask the caller to choose their preferred language: "
                "English, Arabic, Malayalam, or Tamil."
                "When the caller chooses a language, you MUST call "
                "select_language_tool before continuing the conversation."
            )
        )

        

    finally:
        # HIPAA zero-retention: ensure no session state or HIS client
        # resources leak past this call. `session_state.ctx` lives only in
        # process memory and is discarded when this worker process exits;
        # nothing here writes transcripts or PII to persistent disk.
        await his_client.aclose()
        logger.info("Session %s ended and resources released.", session_state.ctx.session_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        agent_name="hospital-voice-agent"
        ))
