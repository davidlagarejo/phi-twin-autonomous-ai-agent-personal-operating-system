"""
api/routes/chats.py — Persistent chat history + smart routing.

Smart router:
  @web  / /buscar → SearXNG + Phi summarizes (Perplexity-style)
  @claude          → Claude API directly (writing, code, analysis)
  auto             → detect intent → route to best engine

Endpoints:
  GET    /api/chats                   — list all chats
  POST   /api/chats                   — create new chat
  GET    /api/chats/{id}              — full chat with messages
  DELETE /api/chats/{id}              — delete chat
  PATCH  /api/chats/{id}              — update title / project_id
  POST   /api/chats/{id}/message      — send message (SSE streaming)
  GET    /api/chat_projects           — list projects
  POST   /api/chat_projects           — create project
  DELETE /api/chat_projects/{pid}     — delete project
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from tools.chat_store import (
    create_chat, get_chat, list_chats, delete_chat, update_chat_meta, append_message,
    load_projects, create_project, delete_project,
)

router = APIRouter()
_log   = logging.getLogger("phi.routes.chats")

# ── System prompts ────────────────────────────────────────────────────────────
from pathlib import Path
_BASE        = Path(__file__).parent.parent.parent
_CHAT_SYSTEM = (_BASE / "prompts" / "chat_system.md").read_text(encoding="utf-8")

# Translator (lazy-loaded; gracefully falls back to Phi if argostranslate unavailable)
from services.translator import translate_to_spanish, is_english

# ── Request models ────────────────────────────────────────────────────────────

class NewChatReq(BaseModel):
    title:      str = ""
    project_id: Optional[str] = None

class PatchChatReq(BaseModel):
    title:      Optional[str] = None
    project_id: Optional[str] = None

class MessageReq(BaseModel):
    content:    str
    mode:       Optional[str] = None   # 'web' | 'claude' | 'phi' | None=auto

class NewProjectReq(BaseModel):
    name:  str
    color: str = "#7c5cbf"


# ── Smart router ──────────────────────────────────────────────────────────────

_WEB_SIGNALS = re.compile(
    r'\b(noticias?|latest|recent|current price|precio de|cuánto cuesta|how much is|'
    r'what is the current|what happened|who is|quién es|dónde está|cuándo fue|'
    r'stock price|weather|clima|busca en internet|search for|qué hay de nuevo|'
    r'find me|look up|hay (trabajos?|empleos?)|job openings?|tendencias?|trends?|'
    r'bloomberg|reuters|nytimes|this week|esta semana)\b',
    re.I,
)
_CLAUDE_SIGNALS = re.compile(
    r'\b(escribe un|redacta|crea un|write a|create a|draft a|código|code|'
    r'función que|function that|script|programa que|explícame en detalle|'
    r'explain in detail|análisis completo|help me write|ayúdame a (escribir|crear)|'
    r'generate a|genera un|diseña un|build me|make me a)\b',
    re.I,
)

def _detect_mode(text: str) -> tuple[str, str]:
    """
    Returns (mode, cleaned_text).
    mode: 'web' | 'claude' | 'phi'
    """
    t = text.strip()

    # Explicit prefixes
    m = re.match(r'^(@web|/buscar)\s+', t, re.I)
    if m:
        return "web", t[m.end():]

    m = re.match(r'^(@claude|/claude)\s+', t, re.I)
    if m:
        return "claude", t[m.end():]

    m = re.match(r'^(@phi|/phi)\s+', t, re.I)
    if m:
        return "phi", t[m.end():]

    # Auto-detect
    if _WEB_SIGNALS.search(t):
        return "web", t

    if _CLAUDE_SIGNALS.search(t) and len(t) > 40:
        return "claude", t

    return "phi", t


# ── Engine calls ──────────────────────────────────────────────────────────────

async def _run_web(query: str, history: list[dict]):
    """SearXNG search → Phi-4 summarizes results."""
    from tools.search import run_queries, format_for_prompt
    from llm.client import call_phi

    yield json.dumps({"status": "searching", "query": query[:60]}) + "\n"

    results = run_queries([query], max_per_query=6)
    if not results:
        # Fallback: simplify query
        simple = " ".join(w for w in query.split() if len(w) > 3)[:60]
        results = run_queries([simple], max_per_query=5)

    if not results:
        yield json.dumps({"content": "No encontré resultados para esa búsqueda."}) + "\n"
        yield json.dumps({"done": True}) + "\n"
        return

    yield json.dumps({"status": "found", "count": len(results)}) + "\n"

    search_block = format_for_prompt(results)
    system = (
        "Eres Phi, asistente de David Lagarejo. Responde siempre en español.\n"
        "Analiza los resultados de búsqueda y responde de forma concisa y directa.\n"
        "Cita las fuentes relevantes con [fuente: título] al final de cada afirmación importante."
    )
    msgs = [{"role": "system", "content": system}]
    for h in history[-6:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({
        "role": "user",
        "content": f"{query}\n\n[RESULTADOS DE BÚSQUEDA]:\n{search_block}"
    })

    raw = await call_phi(msgs, num_ctx=5000, temperature=0.1)
    chunk_size = 10
    for i in range(0, len(raw), chunk_size):
        yield json.dumps({"content": raw[i:i + chunk_size]}) + "\n"
    yield json.dumps({"done": True, "mode": "web", "sources": len(results)}) + "\n"


async def _run_claude(query: str, history: list[dict]):
    """Claude API — streaming via httpx."""
    import os, httpx

    api_key = os.environ.get("CLAUDE_API_KEY", "").strip()
    if not api_key:
        yield json.dumps({"error": "CLAUDE_API_KEY no configurada. Agrega la clave en .env"}) + "\n"
        yield json.dumps({"done": True}) + "\n"
        return

    yield json.dumps({"status": "claude", "model": "claude-sonnet-4-6"}) + "\n"

    system = (
        "Eres el asistente de David Lagarejo, un ingeniero físico especialista en "
        "eficiencia energética, IIoT y gestión de proyectos en NYC. "
        "Responde siempre en español. Sé directo y concreto. Sin relleno."
    )

    msgs = []
    for h in history[-10:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": query})

    body = {
        "model":      "claude-sonnet-4-6",
        "max_tokens": 4096,
        "system":     system,
        "messages":   msgs,
        "stream":     True,
    }

    full_text = ""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":         api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                },
                json=body,
            ) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    yield json.dumps({"error": f"Claude API error {resp.status_code}: {err[:200].decode()}"}) + "\n"
                    yield json.dumps({"done": True}) + "\n"
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        ev = json.loads(raw)
                        delta = (ev.get("delta") or {}).get("text", "")
                        if delta:
                            full_text += delta
                            yield json.dumps({"content": delta}) + "\n"
                    except Exception:
                        pass
    except Exception as e:
        yield json.dumps({"error": f"Error conectando con Claude: {str(e)[:120]}"}) + "\n"

    yield json.dumps({"done": True, "mode": "claude"}) + "\n"


async def _run_phi(query: str, history: list[dict]):
    """Phi-4 local (Ollama) — existing chat pipeline with EN→ES translation."""
    from llm.client import call_phi

    msgs = [{"role": "system", "content": _CHAT_SYSTEM}]
    for h in history[-10:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": query})

    raw = await call_phi(msgs, num_ctx=4096, temperature=0.1)

    # Translate English response to Spanish (context-preserving)
    if raw and is_english(raw):
        try:
            translated = await asyncio.to_thread(translate_to_spanish, raw)
            if translated and len(translated) > 5:
                raw = translated
        except Exception as te:
            import logging
            logging.getLogger("phi.chats").warning("translation failed: %s", te)

    chunk_size = 10
    for i in range(0, len(raw), chunk_size):
        yield json.dumps({"content": raw[i:i + chunk_size]}) + "\n"
    yield json.dumps({"done": True, "mode": "phi"}) + "\n"


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/api/chats")
async def chats_list():
    return JSONResponse({"chats": list_chats(60)})


@router.post("/api/chats")
async def chats_create(req: NewChatReq):
    chat = create_chat(req.title, req.project_id)
    return JSONResponse(chat)


@router.get("/api/chats/{chat_id}")
async def chats_get(chat_id: str):
    chat = get_chat(chat_id)
    if not chat:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(chat)


@router.delete("/api/chats/{chat_id}")
async def chats_delete(chat_id: str):
    ok = delete_chat(chat_id)
    if not ok:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"ok": True})


@router.patch("/api/chats/{chat_id}")
async def chats_patch(chat_id: str, req: PatchChatReq):
    chat = update_chat_meta(chat_id, req.title, req.project_id)
    if not chat:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(chat)


@router.post("/api/chats/{chat_id}/message")
async def chats_message(chat_id: str, req: MessageReq):
    """Send a message. Returns SSE stream with content chunks."""
    chat = get_chat(chat_id)
    if not chat:
        return JSONResponse({"error": "chat not found"}, status_code=404)

    # Detect mode
    forced_mode = req.mode  # explicit from client
    if forced_mode and forced_mode in ("web", "claude", "phi"):
        mode     = forced_mode
        content  = req.content.strip()
    else:
        mode, content = _detect_mode(req.content)

    # Save user message
    append_message(chat_id, "user", req.content, mode)
    history = chat.get("messages", [])

    async def stream():
        full_response = ""
        async_gen = None

        if mode == "web":
            async_gen = _run_web(content, history)
        elif mode == "claude":
            async_gen = _run_claude(content, history)
        else:
            async_gen = _run_phi(content, history)

        async for chunk in async_gen:
            yield f"data: {chunk}\n\n"
            try:
                data = json.loads(chunk)
                if "content" in data:
                    full_response += data["content"]
            except Exception:
                pass

        # Persist assistant response
        if full_response:
            append_message(chat_id, "assistant", full_response, mode)

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── Projects ──────────────────────────────────────────────────────────────────

@router.get("/api/chat_projects")
async def projects_list():
    return JSONResponse({"projects": load_projects()})


@router.post("/api/chat_projects")
async def projects_create(req: NewProjectReq):
    proj = create_project(req.name, req.color)
    return JSONResponse(proj)


@router.delete("/api/chat_projects/{project_id}")
async def projects_delete(project_id: str):
    ok = delete_project(project_id)
    if not ok:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"ok": True})
