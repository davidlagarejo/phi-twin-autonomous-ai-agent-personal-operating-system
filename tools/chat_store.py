"""
tools/chat_store.py — Persistent chat history and projects.

Storage:
  data/chats/{chat_id}.json   — one file per conversation
  data/chat_projects.json     — project list
"""
from __future__ import annotations

import json
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_BASE       = Path(__file__).parent.parent
_CHATS_DIR  = _BASE / "data" / "chats"
_PROJ_FILE  = _BASE / "data" / "chat_projects.json"
_CHATS_DIR.mkdir(parents=True, exist_ok=True)


# ── Projects ──────────────────────────────────────────────────────────────────

def load_projects() -> list[dict]:
    try:
        return json.loads(_PROJ_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_projects(projects: list[dict]):
    _PROJ_FILE.write_text(json.dumps(projects, ensure_ascii=False, indent=2), encoding="utf-8")


def create_project(name: str, color: str = "#7c5cbf") -> dict:
    projects = load_projects()
    proj = {
        "id":         uuid.uuid4().hex[:12],
        "name":       name.strip()[:60],
        "color":      color,
        "created_at": _now(),
    }
    projects.append(proj)
    save_projects(projects)
    return proj


def delete_project(project_id: str) -> bool:
    projects = load_projects()
    new = [p for p in projects if p["id"] != project_id]
    if len(new) == len(projects):
        return False
    save_projects(new)
    # Remove project_id from chats
    for path in _CHATS_DIR.glob("*.json"):
        try:
            chat = json.loads(path.read_text(encoding="utf-8"))
            if chat.get("project_id") == project_id:
                chat["project_id"] = None
                path.write_text(json.dumps(chat, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return True


# ── Chats ─────────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()[:19] + "Z"


def _chat_path(chat_id: str) -> Path:
    return _CHATS_DIR / f"{chat_id}.json"


def create_chat(title: str = "", project_id: Optional[str] = None) -> dict:
    chat_id = uuid.uuid4().hex[:16]
    chat = {
        "id":         chat_id,
        "title":      title.strip()[:80] or "Nueva conversación",
        "project_id": project_id,
        "created_at": _now(),
        "updated_at": _now(),
        "messages":   [],
    }
    _chat_path(chat_id).write_text(
        json.dumps(chat, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return chat


def get_chat(chat_id: str) -> Optional[dict]:
    p = _chat_path(chat_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_chat(chat: dict):
    chat["updated_at"] = _now()
    _chat_path(chat["id"]).write_text(
        json.dumps(chat, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def append_message(chat_id: str, role: str, content: str, mode: str = "phi") -> Optional[dict]:
    chat = get_chat(chat_id)
    if chat is None:
        return None
    msg = {
        "id":      uuid.uuid4().hex[:8],
        "role":    role,
        "content": content,
        "mode":    mode,
        "ts":      _now(),
    }
    chat["messages"].append(msg)
    # Auto-title from first user message
    if role == "user" and chat["title"] in ("Nueva conversación", ""):
        chat["title"] = _auto_title(content)
    save_chat(chat)
    return msg


def _auto_title(text: str) -> str:
    """Generate short title from first message."""
    clean = re.sub(r'^(@web|@claude|@phi|/buscar|/claude|/phi)\s+', '', text.strip(), flags=re.I)
    words = clean.split()[:8]
    title = " ".join(words)
    return (title[:55] + "…") if len(title) > 55 else title


def list_chats(limit: int = 60) -> list[dict]:
    """Return all chats sorted by updated_at desc, summary only."""
    chats = []
    paths = sorted(_CHATS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in paths[:limit]:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            last_msg = ""
            for m in reversed(d.get("messages", [])):
                if m.get("role") == "assistant":
                    last_msg = (m.get("content") or "")[:80]
                    break
            chats.append({
                "id":         d["id"],
                "title":      d.get("title", "Nueva conversación"),
                "project_id": d.get("project_id"),
                "updated_at": d.get("updated_at", ""),
                "created_at": d.get("created_at", ""),
                "msg_count":  len(d.get("messages", [])),
                "last_msg":   last_msg,
            })
        except Exception:
            pass
    return chats


def delete_chat(chat_id: str) -> bool:
    p = _chat_path(chat_id)
    if p.exists():
        p.unlink()
        return True
    return False


def update_chat_meta(chat_id: str, title: Optional[str] = None, project_id: Optional[str] = None) -> Optional[dict]:
    chat = get_chat(chat_id)
    if not chat:
        return None
    if title is not None:
        chat["title"] = title.strip()[:80]
    if project_id is not None:
        chat["project_id"] = project_id or None
    save_chat(chat)
    return chat
