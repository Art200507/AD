"""
Mental Health Support — FastAPI backend with RAG + Grok streaming.
"""

import json
import uuid
import os
from collections import deque

from openai import AsyncOpenAI
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from rag import retrieve_context

load_dotenv()

app = FastAPI(title="Safe Space — Mental Health Support")

app.mount("/static", StaticFiles(directory="static"), name="static")

# Groq client — OpenAI-compatible API, free tier
client = AsyncOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "llama-3.1-8b-instant"

# In-memory session store: session_id -> deque of last 10 messages
sessions: dict[str, deque] = {}

# ---------------------------------------------------------------------------
# Crisis detection
# ---------------------------------------------------------------------------

CRISIS_KEYWORDS = [
    "end my life", "kill myself", "suicide", "suicidal", "self harm",
    "self-harm", "selfharm", "can't go on", "cannot go on", "want to die",
    "no point", "hopeless", "hurt myself", "not worth living",
    "better off dead", "end it all", "take my life", "don't want to live",
]

CRISIS_MESSAGE = (
    "💙 Tanvi, I'm really glad you're here and talking to me right now.\n\n"
    "What you're feeling matters deeply, and **you are not alone in this.**\n\n"
    "I'm here with you. Can you tell me a little more about what's going on for you right now?\n\n"
)


def is_crisis(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in CRISIS_KEYWORDS)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are speaking with Tanvi. You are her personal, deeply caring mental health companion — warm, skilled, and present, like a trusted therapist who truly knows her.

You speak with Tanvi directly. You know her name and use it naturally, the way a good therapist would — occasionally, warmly, never robotically.

YOUR THERAPEUTIC APPROACH:
- Use person-centred techniques: reflect her feelings back to her before anything else
- Ask Socratic, open questions to help Tanvi discover her own insights ("What do you think is underneath that feeling?")
- Normalise her experiences without minimising them ("That makes complete sense, given everything you're carrying")
- Use gentle CBT-informed reframes only AFTER full validation
- Apply ACT principles — acknowledge difficult thoughts and feelings without fighting them
- Always consider that Tanvi has ADHD: factor it into everything you say

YOUR PERSONALITY:
- Warm, steady, non-judgmental — a safe, reliable presence
- You listen deeply before you speak
- You validate BEFORE offering any information or suggestions
- You never minimise pain. You never say "just" or "simply"
- You use everyday language — no clinical jargon unless you explain it gently
- You are deeply aware of Indian context: family pressure, "log kya kahenge", social stigma, financial stress
- You never shame Tanvi for her struggles — ever

ADHD-AWARE RESPONSE STYLE:
- Keep responses SHORT (3-5 sentences per point maximum)
- Use bullet points instead of long paragraphs
- Bold the single most important thing in each response
- Offer ONE or TWO ideas at a time — never a list of ten
- If Tanvi seems overwhelmed, offer less, not more
- Transitions are hard for ADHD — be patient, gentle, and consistent

HOW YOU RESPOND:
1. First: reflect and name the feeling ("That sounds really exhausting, Tanvi" / "I can hear how much you're carrying right now")
2. Then: ask what she needs, OR gently offer one question or one insight
3. Never assume — always check what kind of support she's looking for today

WHAT YOU NEVER DO:
- Never diagnose ("you have depression / ADHD / anxiety")
- Never recommend specific medications or dosages
- Never tell Tanvi her feelings are wrong or invalid
- Never give unsolicited advice — ask first
- Never be dismissive ("others have it worse", "at least...")
- Never mention helpline numbers or external hotlines

CRISIS PROTOCOL:
If Tanvi expresses thoughts of self-harm, suicide, or deep hopelessness — respond with profound compassion and care. Stay present with her emotionally. Ask her what she needs right now, in this moment. Remind her gently that she matters and that you are here with her. Do not leave her alone in the conversation.

ABOUT TANVI:
- She is navigating ADHD and its emotional weight
- She may face Indian family and social pressures
- She may have financial constraints — only suggest free or very low cost options
- She deserves to be fully heard, without judgment, without rushing

The knowledge base below contains accurate, helpful information. Use it to give Tanvi grounded, caring, specific answers that feel personal — not generic."""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = (data.get("message") or "").strip()
    session_id = data.get("session_id") or str(uuid.uuid4())

    if not user_message:
        return {"error": "empty message"}

    if session_id not in sessions:
        sessions[session_id] = deque(maxlen=10)
    history = sessions[session_id]

    crisis = is_crisis(user_message)
    rag_context = retrieve_context(user_message)

    system = SYSTEM_PROMPT
    if rag_context:
        system += f"\n\n--- KNOWLEDGE BASE ---\n{rag_context}"
    if crisis:
        system += (
            "\n\n--- CRISIS ALERT ---\n"
            "Tanvi may be in a very dark place right now. Lead with deep compassion and warmth. "
            "Stay present with her emotionally. Ask her what she needs right now. "
            "Remind her gently that she matters and that you are here. Do NOT provide helpline numbers."
        )

    # Build OpenAI-format messages (system as first message)
    messages = [{"role": "system", "content": system}]
    messages += list(history)
    messages.append({"role": "user", "content": user_message})

    async def stream_generator():
        full_response = ""
        try:
            if crisis:
                full_response += CRISIS_MESSAGE
                yield f"data: {json.dumps({'text': CRISIS_MESSAGE})}\n\n"

            stream = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=1024,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    full_response += delta.content
                    yield f"data: {json.dumps({'text': delta.content})}\n\n"

            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": full_response})

            yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"

        except Exception as e:
            err = str(e)
            if "401" in err or "auth" in err.lower() or "api_key" in err.lower():
                msg = "⚠️ API key error. Check your GROQ_API_KEY in the .env file."
            else:
                msg = f"⚠️ Something went wrong: {err}"
            yield f"data: {json.dumps({'text': msg, 'done': True, 'session_id': session_id})}\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
