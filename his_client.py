"""
his_client.py
-------------
Async client for the hospital's existing C# HIS (Hospital Information System)
REST API. All calls go through `httpx.AsyncClient`.

Design notes:
  - The client owns its own OAuth2 client-credentials token lifecycle and
    refreshes proactively before expiry.
  - Every public method returns a `HISResult` (success/failure envelope)
    instead of raising on expected failure modes (timeout, 4xx, 5xx), so
    calling code (the FSM / escalation engine) can react deterministically
    without wrapping every call in its own try/except for control flow.
  - Unexpected exceptions (bugs, DNS failures, etc.) still raise
    `HISUnavailableError` after retries are exhausted, which the escalation
    engine treats as an immediate transfer trigger.
  - Atomic reschedule is implemented as: hold new slot -> confirm new
    booking -> only then release the old slot. If the new booking fails,
    the old slot is left untouched. If the release fails after a
    successful new booking, we log a reconciliation event rather than
    fail the caller-facing operation (the caller now legitimately has a
    valid new appointment).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from config import HISConfig, load_his_config, PII_LOG_REDACT_FIELDS

logger = logging.getLogger("his_client")


# --------------------------------------------------------------------------- #
# Errors & result envelope
# --------------------------------------------------------------------------- #

class HISUnavailableError(Exception):
    """Raised when the HIS API is unreachable or erroring after all retries.
    Callers (the FSM) should catch this and route to the escalation engine —
    do not attempt to recover from this inside business logic."""


@dataclass
class HISResult:
    ok: bool
    data: Any = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    @staticmethod
    def success(data: Any) -> "HISResult":
        return HISResult(ok=True, data=data)

    @staticmethod
    def failure(error_code: str, error_message: str) -> "HISResult":
        return HISResult(ok=False, error_code=error_code, error_message=error_message)


def _redact(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of `payload` with PII fields masked, for safe logging."""
    redacted = dict(payload)
    for field_name in PII_LOG_REDACT_FIELDS:
        if field_name in redacted and redacted[field_name] is not None:
            redacted[field_name] = "***REDACTED***"
    return redacted


# --------------------------------------------------------------------------- #
# HIS Client
# --------------------------------------------------------------------------- #

class HISClient:
    """Async wrapper around the hospital's C# HIS REST API."""

    def __init__(self, config: Optional[HISConfig] = None):
        self._config = config or load_his_config()
        self._http = httpx.AsyncClient(
            base_url=self._config.base_url,
            timeout=self._config.timeout_seconds,
        )
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0
        self._token_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "HISClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.aclose()

    # ------------------------------------------------------------------- #
    # Auth
    # ------------------------------------------------------------------- #

    async def _ensure_token(self) -> str:
        async with self._token_lock:
            now = time.monotonic()
            if self._access_token and now < (
                self._token_expires_at - self._config.token_refresh_margin_seconds
            ):
                return self._access_token

            try:
                resp = await self._http.post(
                    "/oauth/token",
                    data={
                        "grant_type": "client_credentials",
                        "client_id": self._config.client_id,
                        "client_secret": self._config.client_secret,
                    },
                )
                resp.raise_for_status()
                payload = resp.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise HISUnavailableError(f"HIS auth failed: {exc}") from exc

            token = payload.get("access_token")
            expires_in = payload.get("expires_in", 3600)
            if not token:
                raise HISUnavailableError("HIS auth response missing access_token")

            self._access_token = token
            self._token_expires_at = time.monotonic() + float(expires_in)
            return token

    async def _auth_headers(self) -> dict[str, str]:
        if not self._config.auth_enabled:
            return {}

        token = await self._ensure_token()
        return {"Authorization": f"Bearer {token}"}

    # ------------------------------------------------------------------- #
    # Core request wrapper: retries transient failures, never retries on
    # 4xx (those are expected business outcomes, not transient errors).
    # ------------------------------------------------------------------- #

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> HISResult:
        last_exc: Optional[Exception] = None

        for attempt in range(self._config.max_retries + 1):
            try:
                headers = await self._auth_headers()
                if idempotency_key:
                    headers["Idempotency-Key"] = idempotency_key

                resp = await self._http.request(
                    method,
                    path,
                    json=json_body,
                    params=params,
                    headers=headers,
                )

                if resp.status_code == 401 and attempt == 0:
                    # Token may have been invalidated server-side; force refresh once.
                    self._access_token = None
                    continue

                if 400 <= resp.status_code < 500:
                    try:
                        body = resp.json()
                    except ValueError:
                        body = {}
                    logger.warning(
                        "HIS API rejected request %s %s [%s]: %s",
                        method, path, resp.status_code, _redact(body),
                    )
                    return HISResult.failure(
                        error_code=str(body.get("code", resp.status_code)),
                        error_message=body.get("message", "Request rejected by HIS API"),
                    )

                resp.raise_for_status()
                return HISResult.success(resp.json())

            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                logger.warning(
                    "HIS API transient error on attempt %d/%d for %s %s: %s",
                    attempt + 1, self._config.max_retries + 1, method, path, exc,
                )
                if attempt < self._config.max_retries:
                    await asyncio.sleep(self._config.retry_backoff_seconds * (attempt + 1))
                    continue

        raise HISUnavailableError(
            f"HIS API unavailable after {self._config.max_retries + 1} attempts "
            f"for {method} {path}: {last_exc}"
        )

    # ------------------------------------------------------------------- #
    # Patient verification
    # ------------------------------------------------------------------- #

    async def verify_patient(self, mobile: str, national_id: Optional[str] = None) -> HISResult:
        """Look up a registered patient by mobile number, optionally
        disambiguated by national ID / Iqama when multiple records share a
        mobile number (family-shared phones are common)."""
        logger.info("Calling HIS patient verification")
        
        return await self._request(
            "POST",
            "/api/v1/patients/verify",
            json_body={"mobile": mobile, "national_id": national_id},
        )

    # ------------------------------------------------------------------- #
    # Departments / doctors / slots
    # ------------------------------------------------------------------- #

    async def list_departments(self) -> HISResult:
        return await self._request("GET", "/api/v1/departments")

    async def list_doctors(self, department_id: str) -> HISResult:
        return await self._request(
            "GET", "/api/v1/doctors", params={"department_id": department_id}
        )

    async def get_previous_doctor(self, patient_id: str, department_id: str) -> HISResult:
        return await self._request(
            "GET",
            "/api/v1/patients/last-doctor",
            params={"patient_id": patient_id, "department_id": department_id},
        )

    async def search_slots(
        self,
        doctor_id: str,
        *,
        earliest: bool = True,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        max_results: int = 3,
    ) -> HISResult:
        return await self._request(
            "GET",
            "/api/v1/slots/search",
            params={
                "doctor_id": doctor_id,
                "earliest": earliest,
                "date_from": date_from,
                "date_to": date_to,
                "limit": max_results,
            },
        )

    async def hold_slot(self, slot_id: str, patient_id: str) -> HISResult:
        """Places a short-lived hold on a slot so it can't be double-booked
        while the caller is confirming verbally."""
        return await self._request(
            "POST",
            "/api/v1/slots/hold",
            json_body={"slotId": slot_id, "patientId": patient_id},
        )

    async def release_slot_hold(self, hold_id: str) -> HISResult:
        return await self._request(
            "POST", "/api/v1/slots/release-hold", json_body={"holdId": hold_id}
        )

    # ------------------------------------------------------------------- #
    # Booking / appointments
    # ------------------------------------------------------------------- #

    async def book_appointment(
        self, hold_id: str, patient_id: str, *, idempotency_key: str
    ) -> HISResult:
        return await self._request(
            "POST",
            "/api/v1/appointments/book",
            json_body={"holdId": hold_id, "patientId": patient_id},
            idempotency_key=idempotency_key,
        )

    async def get_upcoming_appointments(self, patient_id: str, limit: int = 5) -> HISResult:
        return await self._request(
            "GET",
            "/api/v1/appointments/upcoming",
            params={"patient_id": patient_id, "limit": limit},
        )

    async def cancel_appointment(
        self, appointment_id: str, *, reason: str = "patient_requested", idempotency_key: str
    ) -> HISResult:
        return await self._request(
            "POST",
            "/api/v1/appointments/cancel",
            json_body={"appointmentId": appointment_id, "reason": reason},
            idempotency_key=idempotency_key,
        )

    async def reschedule_appointment(
        self,
        old_appointment_id: str,
        new_slot_id: str,
        patient_id: str,
        *,
        idempotency_key: str,
    ) -> HISResult:
        """Atomic-swap semantics: hold the new slot, book it, and only
        release/cancel the old appointment after the new booking succeeds.
        If any step fails, previously-held resources are unwound so the
        patient is never left worse off than before the call."""
        hold_result = await self.hold_slot(new_slot_id, patient_id)
        if not hold_result.ok:
            return hold_result

        hold_id = hold_result.data.get("hold_id") if hold_result.data else None
        if not hold_id:
            return HISResult.failure("MISSING_HOLD_ID", "HIS API did not return a hold_id")

        book_result = await self.book_appointment(
            hold_id, patient_id, idempotency_key=idempotency_key
        )
        if not book_result.ok:
            # Unwind: release the hold we just took on the new slot.
            await self.release_slot_hold(hold_id)
            return book_result

        new_appointment_id = book_result.data.get("appointment_id") if book_result.data else None

        cancel_result = await self.cancel_appointment(
            old_appointment_id,
            reason="rescheduled",
            idempotency_key=f"{idempotency_key}-release-old",
        )
        if not cancel_result.ok:
            # The caller now has TWO active appointments (new one succeeded,
            # old one failed to release). This is a reconciliation issue, not
            # a caller-facing failure — surface it as a warning, not an error,
            # and flag it for back-office follow-up rather than telling the
            # caller their reschedule failed.
            logger.error(
                "Reschedule reconciliation needed: new appointment %s booked but "
                "old appointment %s failed to release: %s",
                new_appointment_id, old_appointment_id, cancel_result.error_message,
            )
            return HISResult.success({
                **(book_result.data or {}),
                "reconciliation_needed": True,
                "old_appointment_id": old_appointment_id,
            })

        return book_result

    # ------------------------------------------------------------------- #
    # New patient registration & OTP
    # ------------------------------------------------------------------- #

    async def send_otp(self, mobile: str) -> HISResult:
        return await self._request(
            "POST", "/api/v1/auth/send-otp", json_body={"mobile": mobile}
        )

    async def verify_otp(self, mobile: str, otp_code: str) -> HISResult:
        return await self._request(
            "POST",
            "/api/v1/auth/verify-otp",
            json_body={"mobile": mobile, "otpCode": otp_code},
        )

    async def check_duplicate_patient(
        self, mobile: str, national_id: Optional[str] = None
    ) -> HISResult:
        return await self._request(
            "POST",
            "/api/v1/patients/check-duplicate",
            json_body={"mobile": mobile, "nationalId": national_id},
        )

    async def create_patient(self, registration_payload: dict[str, Any]) -> HISResult:
        """`registration_payload` should contain: full_name, mobile, national_id
        (or iqama), date_of_birth, gender, nationality. Returns the new MRN on
        success."""
        return await self._request(
            "POST", "/api/v1/patients/register", json_body=registration_payload
        )
