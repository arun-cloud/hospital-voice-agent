# Hospital Appointment Voice AI Agent

Production-grade, bilingual (EN/AR-SA) LiveKit voice agent for automated hospital appointment workflows.

## Architecture
```
Caller Audio -> Silero VAD -> Deepgram Nova-3 STT -> Claude 3.5 Sonnet (tool-calling) -> Cartesia Sonic 3.5 TTS -> Caller Audio
                                                            |
                                                      state_machine.py (FSM)
                                                            |
                                                      his_client.py (async C# HIS REST client)
                                                            |
                                                      escalation.py (deterministic safety guardrails)
```

## Files
| File | Responsibility |
|---|---|
| `config.py` | Enums, bilingual system prompts, STT keyterms, VAD/LLM/TTS settings, HIS env config |
| `his_client.py` | Async C# HIS REST client: auth, patient verify, slots, atomic booking/reschedule, OTP |
| `state_machine.py` | Immutable async FSM driving all 7 call flows (Releases 1–3) |
| `escalation.py` | Deterministic emergency/failure detection + SIP transfer stub |
| `agent.py` | LiveKit entrypoint: pipeline wiring + LLM function tools |

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env   # fill in real credentials
python agent.py dev    # or `start` for production worker mode
```

## Design decisions worth knowing about
- **CallContext is immutable.** Every FSM transition returns a *new* context via `.evolve()`.
  This makes every turn replayable/auditable and prevents partial-mutation bugs when a tool
  call fails halfway through.
- **Escalation is rule-based, not LLM-judged.** Emergency phrases, repeated verification
  failures, and HIS outages are caught deterministically in `escalation.py` so a model
  hallucination can't suppress a safety-critical transfer. This runs on *every* user turn,
  independent of what tool (if any) the LLM decides to call.
- **Reschedule is atomic by construction**: hold new slot -> book new slot -> only then
  release the old one. If the old-slot release fails after a successful new booking, we do
  not fail the call — the caller legitimately has a valid new appointment; we log a
  reconciliation event for back-office follow-up instead.
- **Idempotency keys** are attached to every mutating HIS call (book/cancel/reschedule) so a
  retried request (e.g., after a transient timeout) can't double-book or double-cancel.
- **SIP transfer is a stub** (`SIPTransferStub` in `escalation.py`), exactly as scoped — it
  logs the transfer intent so the FSM and prompts around it can be fully built and tested
  before telephony integration lands. Swap in a real `TransferSink` implementation later.
- **Zero persistent PII**: `CallContext` lives only in the per-call worker process's memory
  (LiveKit's "1 process per job" model) and is discarded on process exit. `PII_LOG_REDACT_FIELDS`
  in `config.py` ensures any HIS request/response logging masks PII before it ever reaches disk.

## Verified (this build)
- All 5 modules compile and import cleanly.
- Emergency-language detection tested for both English and Arabic.
- Full booking flow (verify -> department -> doctor -> slots -> hold -> confirm) tested
  end-to-end against a mocked HIS client.
- HIS outage mid-flow correctly triggers escalation rather than crashing the session.

## Not yet wired (by design, per spec scope)
- Real SIP REFER telephony transfer (stubbed).
- Live credentials for Deepgram / Cartesia / Anthropic / the real HIS Gateway.
- Automated test suite beyond the smoke tests above (recommend `pytest` + `respx` for HTTP mocking next).
