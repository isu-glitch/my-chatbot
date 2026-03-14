# Mental Health Support Chatbot — Implementation Guide

---

## Quick Start

```bash
pip install anthropic fastapi uvicorn python-dotenv
cp .env.example .env          # fill in your ANTHROPIC_API_KEY
uvicorn server:app --reload
```

---

## File Structure

```
mental_health_chatbot/
├── chatbot.py          # Core logic: system prompt, crisis detection, API call
├── server.py           # FastAPI endpoints
├── .env.example        # Template — copy to .env, never commit .env
└── README.md           # This file
```

---

## How Each Part Works (Plain English)

### chatbot.py

| Part | What it does |
|---|---|
| `SYSTEM_PROMPT` | Tells the model who it is, what it must never do, and how to write a crisis flag |
| `CRISIS_PATTERNS` | Regex that catches obvious danger words *before* the model even sees the message — a fast safety net |
| `moderate_input()` | Runs the regex check + a basic prompt-injection guard |
| `trim_history()` | Keeps only the last 20 turns so the API call stays affordable |
| `parse_model_response()` | Looks for `CRISIS_FLAG` that the model inserts when it detects danger |
| `chat()` | Puts it all together: moderate → call API → check output → update history → return |

### server.py

| Part | What it does |
|---|---|
| `/disclaimer` | Returns the consent text; frontend must show this before enabling chat |
| `/chat` | Accepts message + history, calls `chat()`, returns reply + crisis flag |
| `ChatRequest` | Validates input length and role values before any processing |
| CORS middleware | Restricts which domains can call your API |

---

## Example Frontend Request Payload

```json
POST /chat
Content-Type: application/json

{
  "session_id": "a3f1c2d4-...",
  "message": "I've been feeling really overwhelmed lately and can't sleep.",
  "history": [
    {
      "role": "assistant",
      "content": "Just so you know, I'm a supportive AI companion — not a licensed therapist..."
    }
  ]
}
```

## Example Response

```json
{
  "session_id": "a3f1c2d4-...",
  "reply": "That sounds really exhausting. Feeling overwhelmed and losing sleep can feed each other...",
  "crisis": false,
  "error": null
}
```

## Crisis Response Example

When `crisis: true` the reply will end with a `---` separator followed by the fallback message and hotlines. Your frontend should:
- Change the chat background to a calm, attention-drawing colour
- Show the crisis resources prominently
- Disable the "send" button for a few seconds to prevent rapid re-sends

---

## Recommended System Prompt (for reference)

The full prompt is in `chatbot.py`. Key design decisions:
1. **Explicit identity boundary** — "You are NOT a licensed…" stated clearly
2. **Hard rules numbered** — LLMs respect numbered lists as constraints
3. **CRISIS_FLAG token** — a parseable signal the model can reliably emit
4. **Tone instruction** — "calm, warm, non-judgmental, unhurried"
5. **First-turn disclaimer** — legal protection + user trust

---

## Short User-Facing Disclaimer

> **Before you begin:** This is an AI-powered emotional support companion, not a licensed therapist or medical professional. Nothing here is a diagnosis or medical advice. In an emergency, call your local emergency services or a crisis line immediately. By chatting, you agree that this service is for general emotional support only.

Show this as a modal before the chat opens. Require an "I understand" click to proceed.

---

## Storing Chat History Safely

### Option A — Stateless (simplest, shown in this code)
The frontend sends the full `history` array with every request. Nothing is stored server-side. History is lost when the user closes the tab.

**Pros:** No database, minimal compliance burden.
**Cons:** Long conversations increase token cost; history lost on refresh.

### Option B — Server-side encrypted store (recommended for production)
1. Generate a `session_id` UUID on the backend at session start.
2. Store history in a database (e.g. PostgreSQL) with the session_id as the key.
3. **Encrypt the content column at rest** using AES-256 or a KMS-managed key.
4. Set a TTL — delete conversations after 30/60/90 days automatically.
5. Never store the user's name, email, or IP in the same table as message content.

```python
# Pseudocode for encrypted storage
from cryptography.fernet import Fernet

key = os.environ["HISTORY_ENCRYPTION_KEY"]  # store in secrets manager
f = Fernet(key)

encrypted = f.encrypt(json.dumps(history).encode())
db.save(session_id=session_id, data=encrypted)

history = json.loads(f.decrypt(db.load(session_id)))
```

---

## Privacy, Consent & Legal Precautions

### Before Launch — Non-Negotiable

- [ ] **Privacy Policy** — explain what data you collect, how long you keep it, who can access it. GDPR/CCPA may apply.
- [ ] **Informed Consent screen** — users must actively agree before chatting.
- [ ] **Disclaimer visible at all times** — "Not a therapist. Not medical advice."
- [ ] **Crisis escalation mandatory** — never remove the crisis fallback; this is a liability floor.
- [ ] **Data residency** — ensure API calls and stored data comply with your users' country regulations.

### Security Checklist

- [ ] API key in environment variable, never in code or client-side JS.
- [ ] HTTPS only — no plain HTTP in production.
- [ ] Rate limiting per session_id or IP (e.g., 30 requests/minute).
- [ ] Input max_length enforced server-side (already in Pydantic model).
- [ ] Log session IDs and crisis flags — never log raw message content in production.
- [ ] Regular dependency updates (`pip audit`, Dependabot).

### Legal Precautions

1. **Consult a healthcare attorney** before launch, especially if operating in the US (HIPAA, FTC), EU (GDPR, MDR), or UK (CQC).
2. **Do NOT market as a clinical tool** — this is a *support companion*, not a *therapy tool*.
3. **Add a "Report a problem" button** so users can flag bad responses.
4. **Consider professional review** — have a licensed psychologist review the system prompt and sample conversations before launch.
5. **Carry product liability insurance** — consumer mental health apps are subject to tort claims.
6. **Data breach plan** — know exactly what you'd do if conversation data was exposed.

### Ongoing Safety

- Monitor `crisis=true` log events weekly.
- Sample random conversations monthly for quality review.
- Keep crisis hotline numbers up to date.
- Version-control all system prompt changes with a change log.

---

## Extending This Code

| Feature | How |
|---|---|
| Streaming replies | Use `client.messages.stream()` + SSE on FastAPI |
| User auth | Add JWT middleware; tie session_id to user_id |
| Multi-language | Add language detection; adjust system prompt language instruction |
| Voice input | Transcribe with Whisper before calling `chat()` |
| Therapist handoff | When `crisis=True`, trigger a webhook to a human review queue |
