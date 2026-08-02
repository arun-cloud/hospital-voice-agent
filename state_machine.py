"""
state_machine.py
-----------------
Async Finite State Machine driving a single call session.

Design notes:
  - `CallContext` is the single source of truth for a session. It is
    intentionally an *immutable-by-convention* dataclass: state transitions
    are performed via `CallContext.evolve(**changes)`, which returns a new
    context rather than mutating in place. This keeps the FSM's history
    auditable (each turn's context can be logged/replayed) and avoids a
    whole class of bugs where a tool call partially mutates shared state
    before failing.
  - `HospitalFSM` holds the *behavior* (transition functions); `CallContext`
    holds the *data*. One `HospitalFSM` instance is created per LiveKit job
    (per call), matching the "1 isolated OS process per call session" model.
  - Every transition method is defensive: HIS calls are wrapped, and any
    `HISUnavailableError` is translated into an escalation trigger rather
    than propagating and crashing the session.
  - This module does not know about LiveKit, STT, or TTS — it is pure
    business logic so it can be unit tested without a live voice pipeline.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Optional

from config import CallIntent, CallState, DoctorSelectionMode, Language, LIMITS
from escalation import EscalationEngine, EscalationSignal
from his_client import HISClient, HISUnavailableError

logger = logging.getLogger("state_machine")


# --------------------------------------------------------------------------- #
# Call context (immutable snapshot of session state)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SlotOption:
    slot_id: str
    doctor_id: str
    doctor_name: str
    department: str
    start_time: str  # ISO 8601
    location: str = ""


@dataclass(frozen=True)
class CallContext:
    session_id: str
    language: Language = Language.ENGLISH
    state: CallState = CallState.GREETING
    intent: CallIntent = CallIntent.UNKNOWN

    # Identity / verification
    mobile: Optional[str] = None
    national_id: Optional[str] = None
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    is_verified: bool = False
    verification_attempts: int = 0

    # Booking flow
    department_id: Optional[str] = None
    department_name: Optional[str] = None
    doctor_selection_mode: Optional[DoctorSelectionMode] = None
    doctor_id: Optional[str] = None
    doctor_name: Optional[str] = None
    offered_slots: tuple[SlotOption, ...] = field(default_factory=tuple)
    selected_slot: Optional[SlotOption] = None
    active_hold_id: Optional[str] = None
    confirmed_appointment_id: Optional[str] = None

    # Existing appointment (check / reschedule / cancel)
    upcoming_appointments: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    target_appointment_id: Optional[str] = None

    # Registration flow
    registration_payload: dict[str, Any] = field(default_factory=dict)
    otp_sent: bool = False
    otp_attempts: int = 0
    new_mrn: Optional[str] = None

    # Reliability tracking
    speech_uninterpretable_count: int = 0
    last_confidence: float = 1.0

    # Escalation
    is_escalated: bool = False
    escalation_reason: Optional[str] = None

    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def evolve(self, **changes: Any) -> "CallContext":
        return replace(self, **changes)


def new_call_context(session_id: Optional[str] = None) -> CallContext:
    return CallContext(session_id=session_id or str(uuid.uuid4()))


# --------------------------------------------------------------------------- #
# FSM
# --------------------------------------------------------------------------- #

class TransitionError(Exception):
    """Raised for programmer errors — calling a transition from an invalid
    state. This should never surface to the caller; it indicates a bug in
    agent.py's tool-routing logic."""


class HospitalFSM:
    def __init__(self, his_client: HISClient, escalation_engine: EscalationEngine):
        self._his = his_client
        self._escalation = escalation_engine

    # ------------------------------------------------------------------- #
    # Language / intent
    # ------------------------------------------------------------------- #

    def set_language(self, ctx: CallContext, language: Language) -> CallContext:
        return ctx.evolve(language=language, state=CallState.INTENT_DETECTION)

    def set_intent(self, ctx: CallContext, intent: CallIntent) -> CallContext:
        next_state = {
            CallIntent.BOOK_NEW_APPOINTMENT: CallState.COLLECT_MOBILE,
            CallIntent.CHECK_APPOINTMENT: CallState.COLLECT_MOBILE,
            CallIntent.RESCHEDULE_APPOINTMENT: CallState.COLLECT_MOBILE,
            CallIntent.CANCEL_APPOINTMENT: CallState.COLLECT_MOBILE,
            CallIntent.NEW_PATIENT_REGISTRATION: CallState.COLLECT_REGISTRATION_DETAILS,
        }.get(intent, CallState.INTENT_DETECTION)
        return ctx.evolve(intent=intent, state=next_state)

    # ------------------------------------------------------------------- #
    # Verification
    # ------------------------------------------------------------------- #

    async def verify_patient(
        self, ctx: CallContext, mobile: str, national_id: Optional[str] = None
    ) -> CallContext:
        try:
            result = await self._his.verify_patient(mobile, national_id)
        except HISUnavailableError as exc:
            return await self._escalate(ctx, EscalationSignal.his_api_unavailable(str(exc)))

        if not result.ok:
            attempts = ctx.verification_attempts + 1
            if attempts >= LIMITS.max_verification_attempts:
                return await self._escalate(
                    ctx.evolve(verification_attempts=attempts, mobile=mobile),
                    EscalationSignal.verification_failed(attempts),
                )
            return ctx.evolve(
                verification_attempts=attempts,
                mobile=mobile,
                state=CallState.COLLECT_MOBILE,
            )

        data = result.data or {}
        target_state = self._post_verification_state(ctx.intent)
        return ctx.evolve(
            mobile=mobile,
            national_id=national_id,
            patient_id=data.get("patient_id"),
            patient_name=data.get("full_name"),
            is_verified=True,
            state=target_state,
        )

    def _post_verification_state(self, intent: CallIntent) -> CallState:
        return {
            CallIntent.BOOK_NEW_APPOINTMENT: CallState.SELECT_DEPARTMENT,
            CallIntent.CHECK_APPOINTMENT: CallState.RETRIEVE_APPOINTMENT,
            CallIntent.RESCHEDULE_APPOINTMENT: CallState.SELECT_APPOINTMENT_TO_MODIFY,
            CallIntent.CANCEL_APPOINTMENT: CallState.SELECT_APPOINTMENT_TO_MODIFY,
        }.get(intent, CallState.INTENT_DETECTION)

    # ------------------------------------------------------------------- #
    # Booking flow (Release 1)
    # ------------------------------------------------------------------- #

    async def select_department(
        self, ctx: CallContext, department_id: str, department_name: str
    ) -> CallContext:
        return ctx.evolve(
            department_id=department_id,
            department_name=department_name,
            state=CallState.SELECT_DOCTOR,
        )

    async def select_doctor(
        self,
        ctx: CallContext,
        mode: DoctorSelectionMode,
        *,
        specific_doctor_id: Optional[str] = None,
        specific_doctor_name: Optional[str] = None,
    ) -> CallContext:
        if not ctx.department_id:
            raise TransitionError("select_doctor called before a department was chosen")

        doctor_id = specific_doctor_id
        doctor_name = specific_doctor_name

        try:
            if mode == DoctorSelectionMode.PREVIOUS_DOCTOR:
                if not ctx.patient_id:
                    raise TransitionError("previous-doctor lookup requires a verified patient")
                result = await self._his.get_previous_doctor(ctx.patient_id, ctx.department_id)
                if not result.ok or not result.data:
                    # Fall back gracefully: no prior doctor on record for this
                    # department, so ask the agent layer to offer earliest-available.
                    return ctx.evolve(
                        doctor_selection_mode=DoctorSelectionMode.EARLIEST_AVAILABLE,
                        state=CallState.SELECT_DOCTOR,
                    )
                doctor_id = result.data.get("doctor_id")
                doctor_name = result.data.get("doctor_name")
        except HISUnavailableError as exc:
            return await self._escalate(ctx, EscalationSignal.his_api_unavailable(str(exc)))

        return ctx.evolve(
            doctor_selection_mode=mode,
            doctor_id=doctor_id,
            doctor_name=doctor_name,
            state=CallState.SEARCH_SLOTS,
        )

    async def search_slots(
        self, ctx: CallContext, *, earliest: bool = True
    ) -> CallContext:
        target_doctor_id = ctx.doctor_id

        try:
            if ctx.doctor_selection_mode == DoctorSelectionMode.EARLIEST_AVAILABLE and not target_doctor_id:
                doctors_result = await self._his.list_doctors(ctx.department_id)
                if not doctors_result.ok or not doctors_result.data:
                    return await self._escalate(
                        ctx, EscalationSignal.his_api_unavailable("No doctors returned for department")
                    )
                doctors = doctors_result.data.get("doctors", [])
                if not doctors:
                    return ctx.evolve(offered_slots=tuple(), state=CallState.SELECT_DEPARTMENT)
                target_doctor_id = doctors[0]["doctor_id"]

            result = await self._his.search_slots(
                target_doctor_id, earliest=earliest, max_results=LIMITS.max_slot_offers
            )
        except HISUnavailableError as exc:
            return await self._escalate(ctx, EscalationSignal.his_api_unavailable(str(exc)))

        if not result.ok:
            return ctx.evolve(offered_slots=tuple(), state=CallState.SELECT_DOCTOR)

        raw_slots = (result.data or {}).get("slots", [])[: LIMITS.max_slot_offers]
        slots = tuple(
            SlotOption(
                slot_id=s["slot_id"],
                doctor_id=s.get("doctor_id", target_doctor_id),
                doctor_name=s.get("doctor_name", ctx.doctor_name or ""),
                department=ctx.department_name or "",
                start_time=s["start_time"],
                location=s.get("location", ""),
            )
            for s in raw_slots
        )

        return ctx.evolve(
            doctor_id=target_doctor_id,
            offered_slots=slots,
            state=CallState.OFFER_SLOTS if slots else CallState.SELECT_DOCTOR,
        )

    async def hold_selected_slot(self, ctx: CallContext, slot_id: str) -> CallContext:
        selected = next((s for s in ctx.offered_slots if s.slot_id == slot_id), None)
        if selected is None:
            raise TransitionError(f"slot_id {slot_id} was not among offered slots")

        try:
            result = await self._his.hold_slot(slot_id, ctx.patient_id)
        except HISUnavailableError as exc:
            return await self._escalate(ctx, EscalationSignal.his_api_unavailable(str(exc)))

        if not result.ok:
            # Slot likely taken between offer and selection — go back to search.
            return ctx.evolve(
                offered_slots=tuple(s for s in ctx.offered_slots if s.slot_id != slot_id),
                state=CallState.SEARCH_SLOTS,
            )

        hold_id = (result.data or {}).get("hold_id")
        return ctx.evolve(
            selected_slot=selected,
            active_hold_id=hold_id,
            state=CallState.CONFIRM_BOOKING,
        )

    async def confirm_booking(self, ctx: CallContext, caller_confirmed: bool) -> CallContext:
        if not caller_confirmed:
            return ctx.evolve(
                selected_slot=None,
                active_hold_id=None,
                state=CallState.SEARCH_SLOTS,
            )

        if not ctx.active_hold_id:
            raise TransitionError("confirm_booking called with no active hold")

        idempotency_key = f"book-{ctx.session_id}-{ctx.active_hold_id}"
        try:
            result = await self._his.book_appointment(
                ctx.active_hold_id, ctx.patient_id, idempotency_key=idempotency_key
            )
        except HISUnavailableError as exc:
            return await self._escalate(ctx, EscalationSignal.his_api_unavailable(str(exc)))

        if not result.ok:
            return ctx.evolve(
                selected_slot=None, active_hold_id=None, state=CallState.SEARCH_SLOTS
            )

        appointment_id = (result.data or {}).get("appointment_id")
        return ctx.evolve(
            confirmed_appointment_id=appointment_id,
            state=CallState.BOOKING_COMPLETE,
        )

    # ------------------------------------------------------------------- #
    # Check existing appointment
    # ------------------------------------------------------------------- #

    async def retrieve_appointments(self, ctx: CallContext) -> CallContext:
        try:
            result = await self._his.get_upcoming_appointments(ctx.patient_id)
        except HISUnavailableError as exc:
            return await self._escalate(ctx, EscalationSignal.his_api_unavailable(str(exc)))

        if not result.ok:
            return ctx.evolve(upcoming_appointments=tuple(), state=CallState.READ_APPOINTMENT)

        appts = tuple((result.data or {}).get("appointments", []))
        return ctx.evolve(upcoming_appointments=appts, state=CallState.READ_APPOINTMENT)

    # ------------------------------------------------------------------- #
    # Reschedule (Release 2)
    # ------------------------------------------------------------------- #

    async def select_appointment_to_modify(
        self, ctx: CallContext, appointment_id: str
    ) -> CallContext:
        next_state = (
            CallState.SEARCH_RESCHEDULE_SLOTS
            if ctx.intent == CallIntent.RESCHEDULE_APPOINTMENT
            else CallState.CONFIRM_CANCELLATION
        )
        return ctx.evolve(target_appointment_id=appointment_id, state=next_state)

    async def confirm_reschedule(
        self, ctx: CallContext, new_slot_id: str, caller_confirmed: bool
    ) -> CallContext:
        if not caller_confirmed:
            return ctx.evolve(state=CallState.SEARCH_RESCHEDULE_SLOTS)

        idempotency_key = f"reschedule-{ctx.session_id}-{ctx.target_appointment_id}"
        try:
            result = await self._his.reschedule_appointment(
                ctx.target_appointment_id, new_slot_id, ctx.patient_id,
                idempotency_key=idempotency_key,
            )
        except HISUnavailableError as exc:
            return await self._escalate(ctx, EscalationSignal.his_api_unavailable(str(exc)))

        if not result.ok:
            return ctx.evolve(state=CallState.SEARCH_RESCHEDULE_SLOTS)

        new_appointment_id = (result.data or {}).get("appointment_id")
        return ctx.evolve(
            confirmed_appointment_id=new_appointment_id, state=CallState.BOOKING_COMPLETE
        )

    # ------------------------------------------------------------------- #
    # Cancel (Release 2)
    # ------------------------------------------------------------------- #

    async def confirm_cancellation(
        self, ctx: CallContext, caller_confirmed: bool
    ) -> CallContext:
        if not caller_confirmed:
            return ctx.evolve(state=CallState.READ_APPOINTMENT)

        idempotency_key = f"cancel-{ctx.session_id}-{ctx.target_appointment_id}"
        try:
            result = await self._his.cancel_appointment(
                ctx.target_appointment_id,
                reason="patient_requested",
                idempotency_key=idempotency_key,
            )
        except HISUnavailableError as exc:
            return await self._escalate(ctx, EscalationSignal.his_api_unavailable(str(exc)))

        if not result.ok:
            return await self._escalate(
                ctx, EscalationSignal.his_api_unavailable(result.error_message or "cancel failed")
            )

        return ctx.evolve(state=CallState.CALL_ENDED)

    # ------------------------------------------------------------------- #
    # New patient registration (Release 3)
    # ------------------------------------------------------------------- #

    async def collect_registration_details(
        self, ctx: CallContext, payload: dict[str, Any]
    ) -> CallContext:
        merged = {**ctx.registration_payload, **payload}
        required = {"full_name", "mobile", "date_of_birth", "gender", "nationality"}
        if not required.issubset(merged.keys()):
            return ctx.evolve(
                registration_payload=merged, state=CallState.COLLECT_REGISTRATION_DETAILS
            )
        return ctx.evolve(registration_payload=merged, state=CallState.SEND_OTP)

    async def send_registration_otp(self, ctx: CallContext) -> CallContext:
        mobile = ctx.registration_payload.get("mobile")
        if not mobile:
            raise TransitionError("send_registration_otp called with no mobile on file")

        try:
            result = await self._his.send_otp(mobile)
        except HISUnavailableError as exc:
            return await self._escalate(ctx, EscalationSignal.his_api_unavailable(str(exc)))

        if not result.ok:
            return await self._escalate(
                ctx, EscalationSignal.his_api_unavailable(result.error_message or "OTP send failed")
            )

        return ctx.evolve(otp_sent=True, state=CallState.VERIFY_OTP)

    async def verify_registration_otp(self, ctx: CallContext, otp_code: str) -> CallContext:
        mobile = ctx.registration_payload.get("mobile")
        try:
            result = await self._his.verify_otp(mobile, otp_code)
        except HISUnavailableError as exc:
            return await self._escalate(ctx, EscalationSignal.his_api_unavailable(str(exc)))

        if not result.ok:
            attempts = ctx.otp_attempts + 1
            if attempts >= LIMITS.otp_max_attempts:
                return await self._escalate(
                    ctx.evolve(otp_attempts=attempts),
                    EscalationSignal.verification_failed(attempts),
                )
            return ctx.evolve(otp_attempts=attempts, state=CallState.VERIFY_OTP)

        return ctx.evolve(state=CallState.CREATE_PATIENT_RECORD)

    async def create_patient_record(self, ctx: CallContext) -> CallContext:
        try:
            dup = await self._his.check_duplicate_patient(
                ctx.registration_payload.get("mobile"),
                ctx.registration_payload.get("national_id"),
            )
        except HISUnavailableError as exc:
            return await self._escalate(ctx, EscalationSignal.his_api_unavailable(str(exc)))

        if dup.ok and dup.data and dup.data.get("duplicate_found"):
            # Route into the existing-patient verification path instead of
            # creating a duplicate MRN.
            existing_patient_id = dup.data.get("patient_id")
            return ctx.evolve(
                patient_id=existing_patient_id,
                is_verified=True,
                state=CallState.SELECT_DEPARTMENT,
            )

        try:
            result = await self._his.create_patient(ctx.registration_payload)
        except HISUnavailableError as exc:
            return await self._escalate(ctx, EscalationSignal.his_api_unavailable(str(exc)))

        if not result.ok:
            return await self._escalate(
                ctx, EscalationSignal.his_api_unavailable(result.error_message or "registration failed")
            )

        data = result.data or {}
        return ctx.evolve(
            new_mrn=data.get("mrn"),
            patient_id=data.get("patient_id"),
            patient_name=ctx.registration_payload.get("full_name"),
            is_verified=True,
            state=CallState.SELECT_DEPARTMENT,  # Route directly into Release 1 booking flow
        )

    # ------------------------------------------------------------------- #
    # Speech reliability tracking
    # ------------------------------------------------------------------- #

    async def register_uninterpretable_speech(self, ctx: CallContext) -> CallContext:
        count = ctx.speech_uninterpretable_count + 1
        if count >= LIMITS.max_speech_uninterpretable_attempts:
            return await self._escalate(
                ctx.evolve(speech_uninterpretable_count=count),
                EscalationSignal.speech_uninterpretable(count),
            )
        return ctx.evolve(speech_uninterpretable_count=count)

    async def register_low_confidence(self, ctx: CallContext, confidence: float) -> CallContext:
        ctx = ctx.evolve(last_confidence=confidence)
        if confidence < LIMITS.low_confidence_threshold:
            return await self._escalate(ctx, EscalationSignal.low_confidence(confidence))
        return ctx

    # ------------------------------------------------------------------- #
    # Escalation bridge
    # ------------------------------------------------------------------- #

    async def _escalate(self, ctx: CallContext, signal: EscalationSignal) -> CallContext:
        await self._escalation.log_event(ctx.session_id, signal)
        return ctx.evolve(
            is_escalated=True,
            escalation_reason=signal.reason.value,
            state=CallState.ESCALATED,
        )

    async def check_emergency_and_escalate(
        self, ctx: CallContext, transcript: str
    ) -> Optional[CallContext]:
        """Should be called on EVERY user turn regardless of current state.
        Returns an updated (escalated) context if an emergency phrase was
        detected, else None so the caller can proceed with normal routing."""
        if self._escalation.contains_emergency_language(transcript, ctx.language):
            return await self._escalate(ctx, EscalationSignal.emergency_detected(transcript))
        return None

    async def handle_human_request(self, ctx: CallContext) -> CallContext:
        return await self._escalate(ctx, EscalationSignal.explicit_human_request())
