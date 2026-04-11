"""
api/routes/questions.py — Profile questions HTTP boundary.

Single responsibility: parse request → call service → return JSONResponse.
No business logic here. All LLM calls and Q&A persistence live in services/questions.py.

Endpoints:
  GET  /api/questions/active         — active (one) strategic question
  POST /api/questions/active/answer  — submit answer, advance to next question
  GET  /api/pending_message          — proactive message from Phi
  DELETE /api/pending_message        — dismiss proactive message
  GET  /api/questions                — (legacy) all profile questions batch
  POST /api/questions/answer         — (legacy) answer a profile question
  GET  /api/profile/qa               — all saved Q&A pairs
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.context import BASE_DIR
from services.questions import load_profile_qa, get_active_question, answer_active

router = APIRouter()

_PENDING_MSG_FILE = BASE_DIR / "data" / "pending_message.json"
_PROFILE_QA_FILE  = BASE_DIR / "data" / "profile_qa.jsonl"


@router.get("/api/pending_message")
async def get_pending_message():
    """Retorna el mensaje pendiente de Phi (generado por proactive_loop). Lo marca como leído."""
    if not _PENDING_MSG_FILE.exists():
        return JSONResponse({"message": None})
    try:
        payload = json.loads(_PENDING_MSG_FILE.read_text(encoding="utf-8"))
        if payload.get("read"):
            return JSONResponse({"message": None})
        payload["read"] = True
        _PENDING_MSG_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        return JSONResponse({"message": payload.get("text", ""), "created_at": payload.get("created_at")})
    except Exception:
        return JSONResponse({"message": None})


@router.delete("/api/pending_message")
async def delete_pending_message():
    """Borra el mensaje pendiente (dismiss)."""
    try:
        _PENDING_MSG_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    return JSONResponse({"ok": True})


@router.get("/api/questions")
async def get_questions():
    """Legacy endpoint — returns empty list. Use /api/questions/active instead."""
    return JSONResponse({"questions": [], "cached": False, "age_minutes": 0})


@router.post("/api/questions/answer")
async def answer_question(req: Request):
    """Persist David's answer to a profile question so Phi remembers it."""
    body = await req.json()
    question = (body.get("question") or "").strip()
    answer   = (body.get("answer") or "").strip()
    if not question or not answer:
        return JSONResponse({"error": "question and answer required"}, status_code=422)

    entry = {
        "question":    question,
        "answer":      answer,
        "answered_at": datetime.now(timezone.utc).isoformat()[:19] + "Z",
    }
    _PROFILE_QA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_PROFILE_QA_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return JSONResponse({"ok": True})


@router.get("/api/profile/qa")
async def get_profile_qa():
    """Return all saved Q&A pairs (David's answers to Phi's profile questions)."""
    entries = await asyncio.to_thread(load_profile_qa, 100)
    return JSONResponse({"entries": entries, "count": len(entries)})


# ── Active question system (one question at a time) ────────────────────────

@router.get("/api/questions/active")
async def get_active_question_endpoint():
    """
    Return the current active strategic question.
    Generates a new one if missing or >72h old.
    """
    question = await asyncio.to_thread(get_active_question)
    return JSONResponse(question)


@router.post("/api/questions/active/answer")
async def answer_active_question(req: Request):
    """
    Submit answer to the active question.
    Applies context changes (directives / job filters) and advances to next question.
    """
    body = await req.json()
    question_id = (body.get("question_id") or "").strip()
    answer      = (body.get("answer") or "").strip()
    if not question_id or not answer:
        return JSONResponse({"error": "question_id and answer required"}, status_code=422)

    result = await asyncio.to_thread(answer_active, question_id, answer)
    if result.get("error"):
        return JSONResponse({"error": result["error"]}, status_code=400)
    return JSONResponse(result)
