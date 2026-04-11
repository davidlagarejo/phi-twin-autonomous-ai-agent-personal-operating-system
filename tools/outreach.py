"""
tools/outreach.py — Outreach task store.

Single responsibility: thread-safe read/write of outreach_tasks.json.
No business logic, no LLM calls.

Consumers: api/routes/outreach.py, services/outreach.py.
"""
from __future__ import annotations

import json
from pathlib import Path

_BASE = Path(__file__).parent.parent
_TASKS_FILE = _BASE / "data" / "outreach_tasks.json"

# Statuses considered "active" for the outreach pipeline
ACTIVE_STATUSES: frozenset[str] = frozenset({
    "pending", "draft_ready", "sent", "followup_due", "replied", "reply_ready",
})

# Display sort order (lower = shown first)
STATUS_ORDER: dict[str, int] = {
    "replied":      0,
    "reply_ready":  1,
    "followup_due": 2,
    "sent":         3,
    "draft_ready":  4,
    "pending":      5,
}


def load_tasks() -> list:
    try:
        return json.loads(_TASKS_FILE.read_text(encoding="utf-8")) if _TASKS_FILE.exists() else []
    except Exception:
        return []


def save_tasks(tasks: list) -> None:
    _TASKS_FILE.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
