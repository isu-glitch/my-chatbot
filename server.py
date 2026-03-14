"""
Mental Health Support Chatbot — FastAPI Server
Run with:  uvicorn server:app --reload

Install:
    pip install fastapi uvicorn python-dotenv anthropic
"""

import os
import uuid
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from chatbot import chat

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Mental Health Support Chatbot API",
    description="Empathetic AI support — not a replacement for professional care.",
    version="1.0.0",
    docs_url=None,   # Disable Swagger in production for privacy
    redoc_url=None,
)

# CORS: lock this down to your actual frontend domain in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    """
    Frontend sends this JSON body.
    session_id should be a UUID generated client-side and stored
    in sessionStorage (not localStorage — cleared when tab closes).
    """
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[Message] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Message cannot be blank.")
        return v.strip()


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    crisis: bool               # Frontend can use this to show a prominent crisis banner
    error: str | None = None


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Simple liveness probe — no sensitive data returned."""
    return {"status": "ok"}


@app.get("/disclaimer")
def get_disclaimer():
    """
    Frontend should fetch and display this before allowing any chat.
    Require the user to click 'I understand' before enabling the chat input.
    """
    return {
        "disclaimer": (
            "This tool is an AI-powered emotional support companion. "
            "It is NOT a licensed therapist, psychologist, or medical professional. "
            "Nothing shared here constitutes medical advice or diagnosis. "
            "In an emergency or crisis, please call your local emergency services immediately.\n\n"
            "By continuing, you acknowledge that:\n"
            "• This service is for general emotional support only.\n"
            "• Your conversations may be stored securely for safety and quality purposes.\n"
            "• You will seek professional help for serious mental health concerns."
        )
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, req: Request):
    """
    Main chat endpoint.

    The frontend sends the full history on every request so the server
    stays stateless. For longer sessions, consider storing history
    server-side in an encrypted database keyed by session_id.
    """
    # Convert Pydantic models → plain dicts for chatbot.py
    history_dicts = [m.model_dump() for m in request.history]

    result = chat(
        user_message=request.message,
        history=history_dicts,
        session_id=request.session_id,
    )

    if result["error"] in ("auth_error",):
        # Don't expose internal errors to the client
        raise HTTPException(status_code=503, detail="Service temporarily unavailable.")

    return ChatResponse(
        session_id=request.session_id,
        reply=result["reply"],
        crisis=result["crisis"],
        error=result["error"],
    )
