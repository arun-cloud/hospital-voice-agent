"""
config.py
---------
Central configuration for the Hospital Voice AI Agent.

Contains:
  - Language / intent / escalation enums
  - Bilingual system prompts (EN / AR-SA)
  - STT keyterm lists for Deepgram Nova-3 (healthcare + Arabic)
  - Silero VAD configuration
  - C# HIS Gateway environment configuration
  - Global timing / retry constants used across the pipeline

This module has NO side effects at import time beyond reading environment
variables via `os.environ`. It intentionally does not construct any network
clients so it can be safely imported by tests without live credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum


# --------------------------------------------------------------------------- #
# Environment helpers
# --------------------------------------------------------------------------- #

def _require_env(key: str) -> str:
    """Fetch a required environment variable or raise a clear startup error."""
    value = os.environ.get(key)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{key}'. "
            f"Set it in your deployment environment or .env file before starting the agent."
        )
    return value


def _optional_env(key: str, default: str) -> str:
    return os.environ.get(key, default)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

class Language(str, Enum):
    ENGLISH = "en"
    ARABIC = "ar-SA"


class CallIntent(str, Enum):
    BOOK_NEW_APPOINTMENT = "book_new_appointment"
    CHECK_APPOINTMENT = "check_appointment"
    RESCHEDULE_APPOINTMENT = "reschedule_appointment"
    CANCEL_APPOINTMENT = "cancel_appointment"
    NEW_PATIENT_REGISTRATION = "new_patient_registration"
    UNKNOWN = "unknown"


class CallState(str, Enum):
    """Top-level FSM states shared across all flows."""
    GREETING = "greeting"
    LANGUAGE_CONFIRM = "language_confirm"
    INTENT_DETECTION = "intent_detection"
    COLLECT_MOBILE = "collect_mobile"
    VERIFY_PATIENT = "verify_patient"
    SELECT_DEPARTMENT = "select_department"
    SELECT_DOCTOR = "select_doctor"
    SEARCH_SLOTS = "search_slots"
    OFFER_SLOTS = "offer_slots"
    HOLD_SLOT = "hold_slot"
    CONFIRM_BOOKING = "confirm_booking"
    BOOKING_COMPLETE = "booking_complete"
    RETRIEVE_APPOINTMENT = "retrieve_appointment"
    READ_APPOINTMENT = "read_appointment"
    SELECT_APPOINTMENT_TO_MODIFY = "select_appointment_to_modify"
    SEARCH_RESCHEDULE_SLOTS = "search_reschedule_slots"
    CONFIRM_RESCHEDULE = "confirm_reschedule"
    CONFIRM_CANCELLATION = "confirm_cancellation"
    COLLECT_REGISTRATION_DETAILS = "collect_registration_details"
    SEND_OTP = "send_otp"
    VERIFY_OTP = "verify_otp"
    CREATE_PATIENT_RECORD = "create_patient_record"
    ESCALATED = "escalated"
    CALL_ENDED = "call_ended"


class EscalationReason(str, Enum):
    EXPLICIT_HUMAN_REQUEST = "explicit_human_request"
    EMERGENCY_DETECTED = "emergency_detected"
    VERIFICATION_FAILED_TWICE = "verification_failed_twice"
    SPEECH_UNINTELLIGIBLE_THRICE = "speech_unintelligible_thrice"
    HIS_API_UNAVAILABLE = "his_api_unavailable"
    LOW_CONFIDENCE = "low_confidence_threshold"
    UNHANDLED_EXCEPTION = "unhandled_exception"


class DoctorSelectionMode(str, Enum):
    EARLIEST_AVAILABLE = "earliest_available"
    PREVIOUS_DOCTOR = "previous_doctor"
    SPECIFIC_DOCTOR = "specific_doctor"


# --------------------------------------------------------------------------- #
# System prompts (bilingual)
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT_EN = """You are Sara, the automated appointment assistant for {hospital_name}.
You speak clearly, warmly, and concisely — this is a live phone call, not a chat window.

STRICT RULES:
1. Never invent appointment slots, doctor names, or patient data. Only state facts returned by
   the `his_*` tools. If a tool fails or returns no data, say so plainly and offer to transfer
   the caller to a human receptionist.
2. Always verify the caller's identity (mobile number + patient lookup) before discussing or
   modifying any appointment.
3. When offering appointment slots, offer at most 3 options at a time, spoken as day, date, and
   time (e.g., "Tuesday, August 4th at 10:30 AM").
4. Before finalizing any booking, reschedule, or cancellation, read back the full details
   (doctor, department, date, time) and wait for the caller's explicit verbal confirmation
   ("yes", "confirm", "that's correct") before calling any booking/cancellation tool.
5. If the caller mentions a medical emergency (e.g., chest pain, difficulty breathing, severe
   bleeding, loss of consciousness), immediately stop the booking flow and trigger escalation —
   do not attempt to schedule anything for an active emergency.
6. If the caller explicitly asks for a human, or you fail to understand them after repeated
   attempts, trigger escalation immediately rather than continuing to guess.
7. Keep turns short. Ask one question at a time. Do not read out internal system identifiers
   (patient IDs, MRNs) unless the caller asks for them.
8. Never provide medical advice, diagnosis, or clinical guidance of any kind. Your sole function
   is appointment administration.

Current call context:
- Hospital: {hospital_name}
- Language: English
- Verified patient: {patient_status}
"""

SYSTEM_PROMPT_AR = """أنتِ سارة، المساعدة الآلية لحجز المواعيد في {hospital_name}.
تحدثي بوضوح ودفء وإيجاز — هذه مكالمة هاتفية مباشرة وليست محادثة نصية.

قواعد صارمة:
1. لا تخترعي أبدًا مواعيد أو أسماء أطباء أو بيانات مرضى. اذكري فقط الحقائق التي تُرجعها أدوات
   `his_*`. إذا فشلت إحدى الأدوات أو لم تُرجع بيانات، أخبري المتصل بوضوح واعرضي تحويله إلى موظف استقبال بشري.
2. تحققي دائمًا من هوية المتصل (رقم الجوال + البحث عن المريض) قبل مناقشة أو تعديل أي موعد.
3. عند عرض المواعيد المتاحة، اعرضي 3 خيارات كحد أقصى في كل مرة.
4. قبل إتمام أي حجز أو إعادة جدولة أو إلغاء، أعيدي قراءة التفاصيل الكاملة (الطبيب، القسم،
   التاريخ، الوقت) وانتظري تأكيدًا صريحًا من المتصل قبل استدعاء أي أداة حجز أو إلغاء.
5. إذا ذكر المتصل حالة طوارئ طبية (مثل ألم في الصدر، صعوبة في التنفس، نزيف حاد، فقدان الوعي)،
   أوقفي فورًا عملية الحجز وابدئي إجراء التحويل — لا تحاولي جدولة أي شيء أثناء حالة طوارئ نشطة.
6. إذا طلب المتصل التحدث إلى إنسان صراحةً، أو تعذّر عليكِ فهمه بعد محاولات متكررة، ابدئي التحويل
   فورًا بدلاً من الاستمرار بالتخمين.
7. اجعلي الأدوار قصيرة. اطرحي سؤالاً واحدًا في كل مرة. لا تذكري المعرّفات الداخلية للنظام إلا إذا طلبها المتصل.
8. لا تقدمي أبدًا أي نصيحة طبية أو تشخيص أو إرشاد سريري من أي نوع. مهمتك الوحيدة هي إدارة المواعيد.

سياق المكالمة الحالي:
- المستشفى: {hospital_name}
- اللغة: العربية
- حالة التحقق من المريض: {patient_status}
"""

SYSTEM_PROMPTS: dict[Language, str] = {
    Language.ENGLISH: SYSTEM_PROMPT_EN,
    Language.ARABIC: SYSTEM_PROMPT_AR,
}

EMERGENCY_KEYWORDS: dict[Language, list[str]] = {
    Language.ENGLISH: [
        "chest pain", "can't breathe", "cannot breathe", "difficulty breathing",
        "severe bleeding", "bleeding heavily", "unconscious", "passed out",
        "stroke", "heart attack", "not breathing", "severe pain", "overdose",
        "seizure", "choking", "suicidal", "want to die", "allergic reaction",
        "anaphylaxis",
    ],
    Language.ARABIC: [
        "ألم في الصدر", "لا أستطيع التنفس", "صعوبة في التنفس", "نزيف شديد",
        "فقدان الوعي", "إغماء", "سكتة دماغية", "نوبة قلبية", "توقف التنفس",
        "ألم شديد", "جرعة زائدة", "تشنج", "اختناق", "انتحار", "حساسية شديدة",
    ],
}


# --------------------------------------------------------------------------- #
# Deepgram STT keyterm injection (healthcare vocabulary)
# --------------------------------------------------------------------------- #

# Nova-3 supports keyterm prompting to bias recognition toward domain vocabulary.
# English list uses Nova-3 Medical's clinical vocabulary; Arabic list is a curated
# set of transliterated department/procedure/doctor-title terms since Nova-3
# General (ar-SA) does not ship a medical variant.

STT_KEYTERMS_EN: list[str] = [
    "cardiology", "endocrinology", "gastroenterology", "nephrology",
    "neurology", "obstetrics", "gynecology", "oncology", "ophthalmology",
    "orthopedics", "otolaryngology", "pediatrics", "psychiatry",
    "pulmonology", "radiology", "rheumatology", "urology", "dermatology",
    "dr", "doctor", "consultant", "specialist", "appointment", "reschedule",
    "cancellation", "medical record number", "MRN", "outpatient",
    "follow-up", "referral", "insurance", "co-pay", "national ID",
    "Iqama",
]

STT_KEYTERMS_AR: list[str] = [
    "قسم القلب", "الغدد الصماء", "الجهاز الهضمي", "الكلى", "الأعصاب",
    "النساء والولادة", "الأورام", "العيون", "العظام", "الأنف والأذن والحنجرة",
    "الأطفال", "الطب النفسي", "الجهاز التنفسي", "الأشعة", "الروماتيزم",
    "المسالك البولية", "الجلدية", "دكتور", "دكتورة", "استشاري", "أخصائي",
    "موعد", "إعادة جدولة", "إلغاء", "رقم الملف الطبي", "عيادات خارجية",
    "متابعة", "تحويلة", "تأمين", "الهوية الوطنية", "الإقامة",
]


# --------------------------------------------------------------------------- #
# Silero VAD configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class VADSettings:
    min_speech_duration: float = 0.05
    min_silence_duration: float = 0.55
    activation_threshold: float = 0.5
    sample_rate: int = 16000


VAD_SETTINGS = VADSettings()


# --------------------------------------------------------------------------- #
# LLM configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LLMSettings:
    model: str = "gemini-2.0-flash-001"
    temperature: float = 0.2
    max_tokens: int = 1024
    # Extended thinking is off by default for this model tier; kept explicit
    # here so the intent is auditable rather than implicit.
    thinking_enabled: bool = False


LLM_SETTINGS = LLMSettings()


# --------------------------------------------------------------------------- #
# TTS voice configuration (Cartesia Sonic 3.5)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TTSVoiceConfig:
    voice_id: str
    language: str
    speed: str = "normal"


TTS_VOICES: dict[Language, TTSVoiceConfig] = {
    Language.ENGLISH: TTSVoiceConfig(
        voice_id=_optional_env("CARTESIA_VOICE_ID_EN", "a0e99841-438c-4a64-b679-ae501e7d6091"),
        language="en",
    ),
    Language.ARABIC: TTSVoiceConfig(
        voice_id=_optional_env("CARTESIA_VOICE_ID_AR_SA", "REPLACE_WITH_SAUDI_ARABIC_VOICE_ID"),
        language="ar",
    ),
}


# --------------------------------------------------------------------------- #
# C# HIS Gateway configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class HISConfig:
    base_url: str
    client_id: str
    client_secret: str
    timeout_seconds: float = 6.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.75
    token_refresh_margin_seconds: int = 60


def load_his_config() -> HISConfig:
    """Build HIS gateway config from environment. Called lazily, not at import time,
    so unit tests can import `config` without live credentials present."""
    return HISConfig(
        base_url=_require_env("HIS_API_BASE_URL").rstrip("/"),
        client_id=_require_env("HIS_API_CLIENT_ID"),
        client_secret=_require_env("HIS_API_CLIENT_SECRET"),
        timeout_seconds=float(_optional_env("HIS_API_TIMEOUT_SECONDS", "6.0")),
        max_retries=int(_optional_env("HIS_API_MAX_RETRIES", "2")),
        retry_backoff_seconds=float(_optional_env("HIS_API_RETRY_BACKOFF_SECONDS", "0.75")),
    )


# --------------------------------------------------------------------------- #
# SMS / WhatsApp dispatch configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class NotificationConfig:
    provider: str
    api_base_url: str
    api_key: str
    sender_id: str


def load_notification_config() -> NotificationConfig:
    return NotificationConfig(
        provider=_optional_env("NOTIFICATION_PROVIDER", "unifonic"),
        api_base_url=_require_env("NOTIFICATION_API_BASE_URL"),
        api_key=_require_env("NOTIFICATION_API_KEY"),
        sender_id=_optional_env("NOTIFICATION_SENDER_ID", "HOSPITAL"),
    )


# --------------------------------------------------------------------------- #
# Global operational constants
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OperationalLimits:
    max_verification_attempts: int = 2
    max_speech_uninterpretable_attempts: int = 3
    max_slot_offers: int = 3
    otp_length: int = 6
    otp_expiry_seconds: int = 300
    otp_max_attempts: int = 3
    slot_hold_duration_seconds: int = 180
    pipeline_latency_budget_ms: int = 800
    low_confidence_threshold: float = 0.55


LIMITS = OperationalLimits()

HOSPITAL_NAME = _optional_env("HOSPITAL_NAME", "Al Jedaani Hospitals")

# PII fields that must never be written to persistent logs in plaintext.
PII_LOG_REDACT_FIELDS: tuple[str, ...] = (
    "mobile", "national_id", "iqama", "mrn", "full_name", "otp", "dob",
)
