"""
escalation.py
--------------
Deterministic escalation / guardrail engine.

This module is intentionally simple and rule-based rather than LLM-based:
escalation triggers are safety-critical, and a keyword/threshold check is
auditable and testable in a way that an LLM judgment call is not. The LLM
(in agent.py) is still free to *request* a transfer via tool call, but the
hard triggers below (emergency language, repeated verification/speech
failure, HIS outage) are enforced here independent of what the LLM decides,
so a model hallucination or missed tool call cannot suppress a safety
escalation.

Telephony transfer itself (SIP REFER) is out of scope for this phase per
the spec — `SIPTransferStub` records the intent to transfer and exposes a
single extension point (`execute`) that the telephony integration phase
will replace with a real SIP REFER action.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Protocol

from config import EMERGENCY_KEYWORDS, EscalationReason, Language

logger = logging.getLogger("escalation")


# --------------------------------------------------------------------------- #
# Escalation signal
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EscalationSignal:
    reason: EscalationReason
    detail: str
    severity: str  # "critical" | "high" | "medium"

    @staticmethod
    def emergency_detected(transcript_snippet: str) -> "EscalationSignal":
        return EscalationSignal(
            reason=EscalationReason.EMERGENCY_DETECTED,
            detail=f"Emergency language detected in caller speech.",
            severity="critical",
        )

    @staticmethod
    def explicit_human_request() -> "EscalationSignal":
        return EscalationSignal(
            reason=EscalationReason.EXPLICIT_HUMAN_REQUEST,
            detail="Caller explicitly requested a human agent.",
            severity="medium",
        )

    @staticmethod
    def verification_failed(attempts: int) -> "EscalationSignal":
        return EscalationSignal(
            reason=EscalationReason.VERIFICATION_FAILED_TWICE,
            detail=f"Patient verification failed {attempts} time(s).",
            severity="high",
        )

    @staticmethod
    def speech_uninterpretable(attempts: int) -> "EscalationSignal":
        return EscalationSignal(
            reason=EscalationReason.SPEECH_UNINTELLIGIBLE_THRICE,
            detail=f"Speech uninterpretable after {attempts} attempt(s).",
            severity="medium",
        )

    @staticmethod
    def his_api_unavailable(detail: str) -> "EscalationSignal":
        return EscalationSignal(
            reason=EscalationReason.HIS_API_UNAVAILABLE,
            detail=f"HIS API unavailable: {detail}",
            severity="high",
        )

    @staticmethod
    def low_confidence(confidence: float) -> "EscalationSignal":
        return EscalationSignal(
            reason=EscalationReason.LOW_CONFIDENCE,
            detail=f"ASR/NLU confidence {confidence:.2f} below baseline.",
            severity="medium",
        )

    @staticmethod
    def unhandled_exception(detail: str) -> "EscalationSignal":
        return EscalationSignal(
            reason=EscalationReason.UNHANDLED_EXCEPTION,
            detail=f"Unhandled exception: {detail}",
            severity="critical",
        )


# --------------------------------------------------------------------------- #
# SIP transfer stub
# --------------------------------------------------------------------------- #

class TransferSink(Protocol):
    """Interface the telephony phase will implement with a real SIP REFER
    call. Kept as a Protocol so `SIPTransferStub` and a future
    `LiveKitSIPTransfer` are interchangeable from the FSM's perspective."""

    async def execute(self, session_id: str, reason: EscalationReason, detail: str) -> bool:
        ...


class SIPTransferStub:
    """Stub transfer sink. Logs the transfer intent and returns True
    (transfer "succeeded" from the FSM's perspective) so downstream flow
    (e.g., playing a "transferring you now" prompt) can be developed and
    tested end-to-end before the real telephony integration lands.

    DO NOT ship this to production without replacing `execute` with a real
    SIP REFER call — this stub does not actually move the caller anywhere.
    """

    async def execute(self, session_id: str, reason: EscalationReason, detail: str) -> bool:
        logger.critical(
            "[SIP TRANSFER STUB] session=%s reason=%s detail=%s "
            "-- Telephony integration not yet wired; no real transfer performed.",
            session_id, reason.value, detail,
        )
        return True


# --------------------------------------------------------------------------- #
# Escalation engine
# --------------------------------------------------------------------------- #

class EscalationEngine:
    def __init__(self, transfer_sink: Optional[TransferSink] = None):
        self._transfer_sink = transfer_sink or SIPTransferStub()

    def contains_emergency_language(self, transcript: str, language: Language) -> bool:
        if not transcript:
            return False
        lowered = transcript.lower()
        keywords = EMERGENCY_KEYWORDS.get(language, [])
        # Also check the English list as a fallback in case a bilingual
        # caller code-switches mid-sentence.
        combined = keywords if language == Language.ENGLISH else keywords + EMERGENCY_KEYWORDS[Language.ENGLISH]
        return any(kw.lower() in lowered for kw in combined)

    async def log_event(self, session_id: str, signal: EscalationSignal) -> None:
        """Persist the escalation event and trigger the transfer stub.
        This method never raises — escalation logging must not itself
        become a new failure mode during an already-degraded call."""
        try:
            logger.warning(
                "ESCALATION session=%s reason=%s severity=%s detail=%s ts=%d",
                session_id, signal.reason.value, signal.severity, signal.detail,
                int(time.time()),
            )
            await self._transfer_sink.execute(session_id, signal.reason, signal.detail)
        except Exception:
            logger.exception(
                "Failed to record/execute escalation for session=%s reason=%s "
                "(continuing — caller must still be handed off by the agent layer)",
                session_id, signal.reason.value,
            )
