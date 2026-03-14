"""
Mental Health Support Chatbot — Backend Core
Uses the Anthropic Python SDK (anthropic>=0.25.0) with the Messages API.

Install:
    pip install anthropic python-dotenv

Usage:
    Import `chat()` into your FastAPI app (see server.py).
"""

import os
import re
import logging
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
import anthropic

# ─────────────────────────────────────────────
# 1. ENVIRONMENT & LOGGING
# ─────────────────────────────────────────────
# Loads ANTHROPIC_API_KEY from a .env file so the key never lives in code.
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 2. CONSTANTS
# ─────────────────────────────────────────────
MODEL = "claude-sonnet-4-6"          # Current capable model; swap to opus-4 for higher quality
MAX_TOKENS = 512                      # Keep replies concise; raise if needed
MAX_HISTORY_TURNS = 20               # Sliding window — avoids runaway context costs

# ─────────────────────────────────────────────
# 3. SYSTEM PROMPT
#    This is the most important safety layer.
#    It tells the model exactly who it is, what
#    it must never do, and how to handle crises.
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
You are Sage, a warm and empathetic emotional-support companion.
You are NOT a licensed psychologist, psychiatrist, therapist, or medical professional.
You do NOT diagnose, treat, or prescribe anything.

YOUR ROLE
- Listen actively and reflect feelings back with compassion.
- Help users feel heard and less alone.
- Gently encourage healthy coping strategies (rest, movement, social connection, professional help).
- Maintain clear, warm boundaries — you are a supportive conversation partner, not a therapist.

HARD RULES (never break these)
1. Never diagnose any mental illness or medical condition.
2. Never claim to be a licensed professional.
3. Never roleplay as a romantic partner or encourage emotional dependency.
4. Never reinforce delusional thinking (e.g., if someone says they are being controlled by aliens,
   do not validate the delusion — gently acknowledge distress and suggest professional support).
5. Never provide methods, details, or encouragement for self-harm, suicide, or harming others.
6. Never give specific medication advice.
7. Always recommend professional help when distress is significant or persistent.

CRISIS ESCALATION (highest priority)
If the user expresses ANY of the following, immediately output the CRISIS_FLAG token
on its own line BEFORE your compassionate response:
  - Suicidal thoughts, intent, or plans
  - Self-harm (current or planned)
  - Harm to others
  - Signs of acute psychosis (e.g., hearing commanding voices, losing touch with reality)
  - Signs of mania (e.g., not sleeping for days, grandiose dangerous plans)
  - Abuse (ongoing domestic violence, child abuse, elder abuse)
  - Immediate physical danger

Format for crisis response:
CRISIS_FLAG
<your warm, grounding response that prioritises their safety and directs them to emergency help>

TONE
- Calm, warm, non-judgmental, unhurried.
- Use plain language — no jargon.
- Short paragraphs; do not lecture.
- Ask one gentle follow-up question at a time (never interrogate).

DISCLAIMER REMINDER
At the start of every new conversation (first turn only), include a one-sentence reminder:
"Just so you know, I'm a supportive AI companion — not a licensed therapist — and nothing I say
is a substitute for professional mental health care."
"""

# ─────────────────────────────────────────────
# 4. CRISIS KEYWORDS (pre-LLM safety net)
#    A fast regex pass runs BEFORE sending to the
#    model so obviously high-risk messages are
#    caught even if the model misses them.
# ─────────────────────────────────────────────
CRISIS_PATTERNS = re.compile(
    r"\b(kill myself|suicide|suicidal|end my life|want to die|self.harm|"
    r"cut myself|overdose|hurt myself|no reason to live|can't go on|"
    r"harming others|kill (him|her|them)|voices telling me to|"
    r"being controlled|they.re watching me)\b",
    re.IGNORECASE,
)

CRISIS_FALLBACK = (
    "I'm really sorry you're going through this. "
    "I may not be the right kind of help for an emergency. "
    "Please contact local emergency services or a licensed mental health professional right away, "
    "and if possible reach out to someone you trust who can stay with you now.\n\n"
    "Crisis lines available 24/7:\n"
    "• International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/\n"
    "• US: 988 Suicide & Crisis Lifeline — call or text 988\n"
    "• UK: Samaritans — 116 123\n"
    "• EU: https://findahelpline.com"
)


# ─────────────────────────────────────────────
# 5. CONVERSATION HISTORY HELPERS
#    History is a simple list of dicts:
#    [{"role": "user"|"assistant", "content": str}]
#    In production, persist this per session in
#    an encrypted store — see advice in server.py.
# ─────────────────────────────────────────────

def trim_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> list[dict]:
    """Keep only the most recent `max_turns` full turns (each turn = 1 user + 1 assistant)."""
    # Each turn = 2 messages; we keep last max_turns * 2 messages
    limit = max_turns * 2
    return history[-limit:] if len(history) > limit else history


def is_first_turn(history: list[dict]) -> bool:
    """True when the user is sending their very first message."""
    return len(history) == 0


# ─────────────────────────────────────────────
# 6. INPUT MODERATION
#    A lightweight check before calling the API.
# ─────────────────────────────────────────────

def moderate_input(user_message: str) -> dict:
    """
    Returns a dict with:
      - "crisis": bool  — True if obvious crisis language detected
      - "blocked": bool — True if message should be fully blocked (e.g., prompt injection attempt)
    """
    crisis_detected = bool(CRISIS_PATTERNS.search(user_message))

    # Very basic prompt-injection guard: block attempts to override system prompt
    injection_patterns = re.compile(
        r"(ignore (all |previous |your )?(instructions|rules|system prompt)|"
        r"you are now|pretend you are|new system prompt|disregard)",
        re.IGNORECASE,
    )
    blocked = bool(injection_patterns.search(user_message))

    return {"crisis": crisis_detected, "blocked": blocked}


# ─────────────────────────────────────────────
# 7. OUTPUT MODERATION
#    Check if the model flagged a crisis itself.
# ─────────────────────────────────────────────

def parse_model_response(raw_text: str) -> tuple[bool, str]:
    """
    Checks whether the model inserted CRISIS_FLAG.
    Returns (crisis_flagged: bool, clean_text: str).
    """
    crisis_flagged = "CRISIS_FLAG" in raw_text
    clean_text = raw_text.replace("CRISIS_FLAG", "").strip()
    return crisis_flagged, clean_text


# ─────────────────────────────────────────────
# 8. MAIN CHAT FUNCTION
#    This is what your FastAPI endpoint calls.
# ─────────────────────────────────────────────

def chat(
    user_message: str,
    history: list[dict],
    session_id: Optional[str] = None,   # for logging / audit trail
) -> dict:
    """
    Send a user message and return the assistant reply.

    Parameters
    ----------
    user_message : str
        The latest message from the user.
    history : list[dict]
        Conversation so far: [{"role": ..., "content": ...}, ...]
        Pass an empty list for a new conversation.
    session_id : str, optional
        Identifier used in logs (never log PII here).

    Returns
    -------
    dict with keys:
        "reply"         : str  — text to show the user
        "crisis"        : bool — True if escalation triggered
        "updated_history": list[dict] — history including this turn
        "error"         : str | None
    """
    # ── 8a. Input moderation ──
    mod = moderate_input(user_message)

    if mod["blocked"]:
        logger.warning("session=%s blocked_message=prompt_injection_attempt", session_id)
        return {
            "reply": "I'm not able to process that message. How are you feeling today?",
            "crisis": False,
            "updated_history": history,
            "error": None,
        }

    # ── 8b. Fast-path crisis check (pre-LLM) ──
    if mod["crisis"]:
        logger.critical("session=%s pre_llm_crisis_detected", session_id)
        # Still call the model so the user gets a warm, human-feeling response,
        # but we will merge in the crisis resources regardless.
        # (If you prefer a hard block, return CRISIS_FALLBACK here instead.)

    # ── 8c. Build message list for the API ──
    trimmed_history = trim_history(history)
    messages = trimmed_history + [{"role": "user", "content": user_message}]

    # ── 8d. Call the Anthropic Messages API ──
    try:
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        raw_reply = response.content[0].text

    except anthropic.AuthenticationError:
        logger.error("session=%s api_key_invalid", session_id)
        return {
            "reply": "I'm having trouble connecting right now. Please try again shortly.",
            "crisis": False,
            "updated_history": history,
            "error": "auth_error",
        }
    except anthropic.RateLimitError:
        logger.warning("session=%s rate_limit_hit", session_id)
        return {
            "reply": "I'm a little overwhelmed right now — please try again in a moment.",
            "crisis": False,
            "updated_history": history,
            "error": "rate_limit",
        }
    except Exception as exc:
        logger.exception("session=%s unexpected_error=%s", session_id, exc)
        return {
            "reply": "Something went wrong on my end. Please try again.",
            "crisis": False,
            "updated_history": history,
            "error": "unexpected",
        }

    # ── 8e. Output moderation ──
    model_crisis, clean_reply = parse_model_response(raw_reply)
    crisis = mod["crisis"] or model_crisis

    if crisis:
        logger.critical("session=%s crisis_escalation_triggered", session_id)
        # Append crisis resources to the model's compassionate reply
        final_reply = clean_reply + "\n\n---\n" + CRISIS_FALLBACK
    else:
        final_reply = clean_reply

    # ── 8f. Update history ──
    updated_history = trimmed_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": final_reply},
    ]

    # Audit log — timestamp + session only, never log message content in production
    logger.info(
        "session=%s turn=%d crisis=%s ts=%s",
        session_id,
        len(updated_history) // 2,
        crisis,
        datetime.utcnow().isoformat(),
    )

    return {
        "reply": final_reply,
        "crisis": crisis,
        "updated_history": updated_history,
        "error": None,
    }
