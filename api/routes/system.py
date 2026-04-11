"""
api/routes/system.py — System-level utility endpoints.

Endpoints:
  GET  /api/memory/stats    — long-term memory statistics
  POST /api/like            — toggle like for a dossier card
  POST /api/library_fetch   — fetch/locate a scientific document
  GET  /api/dev_queue       — list Phi → Claude Code dev requests
  POST /api/dev_queue/add   — add a dev request (called by UI)
  POST /api/dev_queue/done/{req_id} — mark request completed

Single responsibility: HTTP boundary only. No business logic here.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from api.context import BASE_DIR, request_executor, workspace
from tools.memory import stats as memory_stats
from tools.library_fetch import fetch_library_document
from services.timeline import invalidate_cache as _invalidate_timeline
from services.directives import load_directives as _load_directives, get_exclusions, get_focus_areas

_log = logging.getLogger("phi.routes.system")
router = APIRouter()

_LIKES_FILE     = BASE_DIR / "data" / "likes.json"
_DEV_QUEUE_FILE = BASE_DIR / "data" / "dev_queue.json"


# ── Memory stats ───────────────────────────────────────────────────────────────

@router.get("/api/memory/stats")
async def memory_stats_endpoint():
    return JSONResponse(memory_stats())


# ── Likes ──────────────────────────────────────────────────────────────────────

@router.post("/api/like")
async def toggle_like(body: dict):
    """Toggle like/unlike for a dossier card. Likes influence research priority."""
    _invalidate_timeline()
    entity_id = str(body.get("entity_id", ""))
    liked     = bool(body.get("liked", True))
    try:
        likes: dict = json.loads(_LIKES_FILE.read_text(encoding="utf-8")) if _LIKES_FILE.exists() else {}
        current = likes.get(entity_id, 0)
        likes[entity_id] = max(0, current + (1 if liked else -1))
        _LIKES_FILE.write_text(json.dumps(likes, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        _log.warning("like_write_failed: %s", exc)
    return JSONResponse({"ok": True})


# ── Library fetch ──────────────────────────────────────────────────────────────

class LibraryFetchRequest(BaseModel):
    doi_or_url: str
    title: Optional[str] = None
    credibility_score: float = 0.0
    relevance_score: float = 0.0
    open_pdf_url: Optional[str] = None


@router.post("/api/library_fetch")
async def library_fetch(req: LibraryFetchRequest):
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        request_executor,
        fetch_library_document,
        req.doi_or_url,
        req.title,
        req.credibility_score,
        req.relevance_score,
        req.open_pdf_url,
        workspace,
    )
    return JSONResponse({
        "status":            result.status,
        "doc_id":            result.doc_id,
        "local_path":        result.local_path,
        "title":             result.title,
        "message":           result.message,
        "unlocking_question": result.unlocking_question,
    })


# ── Dev request queue (Phi → Claude Code) ─────────────────────────────────────

def _load_dev_queue() -> list:
    try:
        return json.loads(_DEV_QUEUE_FILE.read_text(encoding="utf-8")) if _DEV_QUEUE_FILE.exists() else []
    except Exception:
        return []


def _save_dev_queue(items: list) -> None:
    _DEV_QUEUE_FILE.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")


@router.get("/api/dev_queue")
async def get_dev_queue():
    items   = _load_dev_queue()
    pending = [r for r in items if r.get("status") == "pending"]
    return JSONResponse({"requests": items, "pending_count": len(pending)})


@router.post("/api/dev_queue/add")
async def add_dev_request(req: Request):
    body   = await req.json()
    items  = _load_dev_queue()
    req_id = "req_" + str(uuid.uuid4())[:8]
    item   = {
        "id":          req_id,
        "created_at":  datetime.now(timezone.utc).isoformat()[:19] + "Z",
        "status":      "pending",
        "type":        body.get("type", "unknown"),
        "description": body.get("description", ""),
        "file":        body.get("file", ""),
        "detail":      body.get("detail", ""),
    }
    items.append(item)
    _save_dev_queue(items)
    _log.info("dev_request_added id=%s type=%s desc=%r", req_id, item["type"], item["description"][:60])
    return JSONResponse({"ok": True, "id": req_id})


@router.get("/api/directives")
async def get_directives():
    """Return current strategic directives (focus, exclusions, priorities)."""
    return JSONResponse({
        "content":    _load_directives(),
        "focus":      get_focus_areas(),
        "exclusions": get_exclusions(),
    })


@router.post("/api/directives/update")
async def update_directives(req: Request):
    """Direct update of directives.md content (from UI editor)."""
    body = await req.json()
    content = body.get("content", "").strip()
    if not content:
        return JSONResponse({"error": "content required"}, status_code=400)
    try:
        _directives_path = BASE_DIR / "data" / "directives.md"
        _directives_path.write_text(content, encoding="utf-8")
        _log.info("directives_updated_via_api")
        return JSONResponse({"ok": True, "focus": get_focus_areas(), "exclusions": get_exclusions()})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@router.post("/api/dev_queue/done/{req_id}")
async def mark_dev_request_done(req_id: str):
    items = _load_dev_queue()
    for item in items:
        if item.get("id") == req_id:
            item["status"]  = "done"
            item["done_at"] = datetime.now(timezone.utc).isoformat()[:19] + "Z"
    _save_dev_queue(items)
    return JSONResponse({"ok": True})


# ── Data version (for frontend auto-refresh) ──────────────────────────────────

_VERSION_FILE = BASE_DIR / "data" / "last_modified.ts"

_WATCHED_FILES = [
    "data/tasks.json",
    "data/directives.md",
    "data/job_filters.json",
    "data/last_modified.ts",
]


@router.get("/api/data/version")
async def data_version():
    """
    Returns the latest modification timestamp across key data files.
    Frontend polls this after each chat exchange and every 30s.
    If ts is newer than last known → reload timeline / jobs panel.
    """
    ts = 0.0
    for rel in _WATCHED_FILES:
        p = BASE_DIR / rel
        try:
            mt = p.stat().st_mtime
            if mt > ts:
                ts = mt
        except FileNotFoundError:
            pass
    # Also check explicit touch file
    if _VERSION_FILE.exists():
        try:
            stored = float(_VERSION_FILE.read_text().strip())
            if stored > ts:
                ts = stored
        except Exception:
            pass
    return JSONResponse({"ts": ts})
