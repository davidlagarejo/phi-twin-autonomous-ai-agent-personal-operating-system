"""
api/routes/timeline.py — Timeline, workspace state, and research job endpoints.

Endpoints:
  GET    /api/timeline
  POST   /api/timeline/add
  DELETE /api/timeline/{card_id}
  GET    /api/state
  GET    /api/workspace
  GET    /api/activity
  POST   /api/enqueue
  GET    /api/diagnostics
  POST   /api/execute
  GET    /api/audit

Single responsibility: HTTP boundary. Business logic (timeline assembly) lives
in services/timeline.py. Job state lives in core/job_state.py.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from api.context import BASE_DIR, request_executor, workspace
from core.job_state import job_state, ResearchJob, JOB_INTERACTIVE, JOB_BACKGROUND
from services.timeline import build_timeline_response, invalidate_cache as _invalidate_timeline
from tools.research_engine import execute_research_cycle, RunBudget

_log = logging.getLogger("phi.routes.timeline")
router = APIRouter()

# Config loaded by server.py — mirror constants here (same source: settings.json)
_settings_path = BASE_DIR / "config" / "settings.json"
try:
    with open(_settings_path, encoding="utf-8") as _f:
        _cfg = json.load(_f)
except Exception:
    _cfg = {}

_OLLAMA_MODEL          = _cfg.get("ollama", {}).get("model", "phi4:14b")
_SEARXNG_URL           = _cfg.get("searxng", {}).get("url", "")
_SCHEDULER_INTERVAL    = _cfg.get("scheduler", {}).get("interval_sec", 1800)
_AUDIT_LOG             = Path(_cfg.get("data", {}).get(
    "audit_log", str(BASE_DIR / "data" / "audit_logs" / "audit.jsonl")))
_ACTIVITY_FILE         = BASE_DIR / "data" / "activity.jsonl"
_CARDS_FILE            = BASE_DIR / "data" / "timeline_cards.json"

BUDGET_INTERACTIVE = RunBudget(max_seconds=120.0, max_web_queries=8,  max_sources=20, max_tasks=1)
BUDGET_BACKGROUND  = RunBudget(max_seconds=90.0,  max_web_queries=5,  max_sources=10, max_tasks=1)


# ── Timeline feed ──────────────────────────────────────────────────────────────

@router.get("/api/timeline")
async def timeline_data():
    payload = await build_timeline_response(_AUDIT_LOG)
    return JSONResponse(payload)


class TimelineCardRequest(BaseModel):
    id: str
    project: str
    priority: int = 2
    tag: str = ""
    title: str = ""
    desc: str = ""
    meta: str = ""
    metrics: list = []
    actions: list = []
    viz: Optional[dict] = None


@router.post("/api/timeline/add")
async def timeline_add(card: TimelineCardRequest):
    _invalidate_timeline()
    _CARDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    cards: list = []
    if _CARDS_FILE.exists():
        try:
            cards = json.loads(_CARDS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    card_dict = card.model_dump()
    if not card_dict.get("meta"):
        card_dict["meta"] = f"Phi-4 · {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"

    cards = [c for c in cards if c.get("id") != card_dict["id"]]
    cards.append(card_dict)
    cards.sort(key=lambda c: c.get("priority", 99))
    _CARDS_FILE.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")

    return JSONResponse({"status": "added", "id": card_dict["id"], "total_cards": len(cards)})


@router.delete("/api/timeline/{card_id}")
async def timeline_remove(card_id: str):
    _invalidate_timeline()
    if not _CARDS_FILE.exists():
        return JSONResponse({"status": "not_found"}, status_code=404)
    cards  = json.loads(_CARDS_FILE.read_text(encoding="utf-8"))
    before = len(cards)
    cards  = [c for c in cards if c.get("id") != card_id]
    _CARDS_FILE.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")
    return JSONResponse({"status": "removed", "removed": before - len(cards)})


# ── Workspace state ────────────────────────────────────────────────────────────

@router.get("/api/state")
async def get_workspace_state():
    return JSONResponse(workspace.get_state_summary())


@router.get("/api/workspace")
async def get_workspace_detail():
    dossiers = workspace.list_dossiers()
    queue    = workspace.read_queue()

    dossier_rows = [
        {
            "entity_id":    d.get("entity_id"),
            "name":         d.get("name") or d.get("entity_id"),
            "type":         d.get("type", "ORG"),
            "status":       d.get("status", "DRAFT"),
            "fit_score":    d.get("fit_score", 0),
            "last_updated": (d.get("last_updated") or "")[:19],
        }
        for d in sorted(dossiers, key=lambda x: x.get("last_updated") or "", reverse=True)
    ]

    queue_rows = [
        {
            "task_id":    t.get("task_id"),
            "strategy":   t.get("strategy"),
            "status":     t.get("status"),
            "entity_id":  t.get("entity_id"),
            "enqueued_at": (t.get("enqueued_at") or "")[:19],
            "attempts":   t.get("attempts", 0),
        }
        for t in sorted(queue, key=lambda x: x.get("enqueued_at") or "", reverse=True)[:30]
    ]

    return JSONResponse({
        "dossiers": dossier_rows,
        "queue":    queue_rows,
        "job_runner": {
            "job_running": job_state.running,
            "queue_depth": job_state.queue.qsize() if job_state.queue else 0,
            "last_job":    job_state.last_result,
        },
        "summary": workspace.get_state_summary(),
    })


# ── Activity log ───────────────────────────────────────────────────────────────

@router.get("/api/activity")
async def get_activity(limit: int = 60):
    entries: list = []
    if _ACTIVITY_FILE.exists():
        try:
            lines = _ACTIVITY_FILE.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines[-limit:]):
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
        except Exception:
            pass
    return JSONResponse({
        "entries":      entries,
        "job_running":  job_state.running,
        "current_goal": job_state.current_goal,
    })


# ── Enqueue research job ───────────────────────────────────────────────────────

class EnqueueRequest(BaseModel):
    goal: str
    priority: int = JOB_INTERACTIVE


@router.post("/api/enqueue")
async def enqueue_research_job(req: EnqueueRequest):
    if job_state.queue is None:
        return JSONResponse({"error": "job_runner_not_ready"}, status_code=503)

    budget = BUDGET_INTERACTIVE if req.priority == JOB_INTERACTIVE else BUDGET_BACKGROUND
    jid    = str(uuid.uuid4())
    job    = ResearchJob(
        priority=req.priority,
        created_at=time.time(),
        job_id=jid,
        goal=req.goal[:200],
        budget=budget,
        triggered_by="user_enqueue",
    )
    await job_state.queue.put(job)
    queue_depth = job_state.queue.qsize()
    _log.info("enqueue_request job_id=%s goal=%r queue_depth=%d", jid, req.goal[:60], queue_depth)
    return JSONResponse({
        "status":      "queued",
        "job_id":      jid,
        "queue_depth": queue_depth,
        "job_running": job_state.running,
        "message":     f"Investigando en segundo plano (trabajo {jid[:8]}). Los resultados aparecerán en el timeline.",
    })


# ── Diagnostics ────────────────────────────────────────────────────────────────

@router.get("/api/diagnostics")
async def diagnostics():
    audit_entries: list = []
    if _AUDIT_LOG.exists():
        for line in _AUDIT_LOG.read_text(encoding="utf-8").splitlines()[-20:]:
            try:
                audit_entries.append(json.loads(line))
            except Exception:
                pass

    timeline_count = 0
    timeline_last  = None
    if _CARDS_FILE.exists():
        try:
            cards = json.loads(_CARDS_FILE.read_text(encoding="utf-8"))
            timeline_count = len(cards)
            if cards:
                timeline_last = cards[-1].get("meta")
        except Exception:
            pass

    queue_depth   = job_state.queue.qsize() if job_state.queue else -1
    searxng_local = bool(_SEARXNG_URL and ("127.0.0.1" in _SEARXNG_URL or "localhost" in _SEARXNG_URL))

    return JSONResponse({
        "workspace":  workspace.get_state_summary(),
        "job_runner": {
            "job_running": job_state.running,
            "queue_depth": queue_depth,
            "last_job":    job_state.last_result,
            "last_job_at": job_state.last_at,
        },
        "audit": {
            "total_entries": len(_AUDIT_LOG.read_text(encoding="utf-8").splitlines()) if _AUDIT_LOG.exists() else 0,
            "last_20":       audit_entries,
        },
        "timeline": {
            "card_count":    timeline_count,
            "last_card_meta": timeline_last,
        },
        "config": {
            "ollama_model":         _OLLAMA_MODEL,
            "searxng_url":          _SEARXNG_URL or "(not set)",
            "searxng_local":        searxng_local,
            "scheduler_interval_sec": _SCHEDULER_INTERVAL,
            "budget_interactive": {
                "max_seconds":    BUDGET_INTERACTIVE.max_seconds,
                "max_web_queries": BUDGET_INTERACTIVE.max_web_queries,
            },
            "budget_background": {
                "max_seconds":    BUDGET_BACKGROUND.max_seconds,
                "max_web_queries": BUDGET_BACKGROUND.max_web_queries,
            },
        },
    })


# ── Execute research cycle ─────────────────────────────────────────────────────

class ExecuteResearchRequest(BaseModel):
    max_seconds: float = 480.0
    max_web_queries: int = 16
    max_sources: int = 40
    resume_checkpoint_id: Optional[str] = None


@router.post("/api/execute")
async def execute_research(req: ExecuteResearchRequest):
    budget = RunBudget(
        max_seconds=min(req.max_seconds, 600.0),
        max_web_queries=min(req.max_web_queries, 32),
        max_sources=min(req.max_sources, 80),
        max_tasks=2,
    )
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            request_executor,
            execute_research_cycle,
            workspace,
            budget,
            req.resume_checkpoint_id,
        )
        return JSONResponse(result.to_dict())
    except Exception as exc:
        _log.error("execute_research failed: %s", exc)
        return JSONResponse(
            {"status": "FAILED", "result_summary": str(exc), "artifacts": [],
             "tasks_run": 0, "queries_used": 0, "elapsed_seconds": 0.0,
             "next_task_suggestions": [], "gate_results": []},
            status_code=500,
        )


# ── Audit log ──────────────────────────────────────────────────────────────────

@router.get("/api/audit")
async def audit_log(limit: int = 50):
    if not _AUDIT_LOG.exists():
        return JSONResponse({"entries": []})
    entries = []
    with open(_AUDIT_LOG, encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    return JSONResponse({"entries": entries[-limit:]})
