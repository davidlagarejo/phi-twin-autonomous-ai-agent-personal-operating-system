"""
api/routes/outreach.py — Outreach task management and email drafting endpoints.

Endpoints:
  GET  /api/tasks
  POST /api/tasks/save
  POST /api/tasks/update
  POST /api/tasks/mark_sent
  POST /api/tasks/mark_replied
  POST /api/open_mail
  POST /api/draft_email
  POST /api/draft_reply
  POST /api/check_mail_replies
  POST /api/timeline/cleanup

Single responsibility: HTTP boundary only.
Business logic (LLM drafts) lives in services/outreach.py.
Persistence lives in tools/outreach.py.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from api.context import BASE_DIR, request_executor
from tools.outreach import load_tasks, save_tasks, ACTIVE_STATUSES, STATUS_ORDER
from services.outreach import build_email_draft, build_reply_draft
from services.timeline import invalidate_cache as _invalidate_timeline

_log = logging.getLogger("phi.routes.outreach")
router = APIRouter()

_CARDS_FILE = BASE_DIR / "data" / "timeline_cards.json"


# ── Task CRUD ──────────────────────────────────────────────────────────────────

@router.get("/api/tasks")
async def get_tasks(status: str = ""):
    tasks = load_tasks()
    if status == "active":
        tasks = [t for t in tasks if t.get("status") in ACTIVE_STATUSES]
    elif status:
        tasks = [t for t in tasks if t.get("status") == status]
    tasks.sort(key=lambda t: (STATUS_ORDER.get(t.get("status", ""), 9), t.get("created_at", "")))
    return JSONResponse({"tasks": tasks})


@router.post("/api/tasks/save")
async def save_task(req: Request):
    body    = await req.json()
    tasks   = load_tasks()
    task_id = body.get("task_id") or str(uuid.uuid4())[:8]
    body["task_id"] = task_id
    body.setdefault("status", "pending")
    body.setdefault("created_at", datetime.now(timezone.utc).isoformat()[:19] + "Z")
    existing = next((i for i, t in enumerate(tasks) if t.get("task_id") == task_id), None)
    if existing is not None:
        tasks[existing] = {**tasks[existing], **body}
    else:
        tasks.append(body)
    save_tasks(tasks)
    return JSONResponse({"ok": True, "task_id": task_id})


@router.post("/api/tasks/update")
async def update_task(req: Request):
    body    = await req.json()
    task_id = body.get("task_id")
    tasks   = load_tasks()
    for t in tasks:
        if t.get("task_id") == task_id:
            t.update(body)
            t["updated_at"] = datetime.now(timezone.utc).isoformat()[:19] + "Z"
    save_tasks(tasks)
    return JSONResponse({"ok": True})


# ── Mail actions ───────────────────────────────────────────────────────────────

def _open_mail_compose(to: str, subject: str, body: str) -> None:
    """Fire-and-forget: open Apple Mail compose window via osascript."""
    to_a  = to.replace('"', '\\"')
    subj_a = subject.replace('"', '\\"').replace("\n", " ")
    body_a = body.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    script = f'''tell application "Mail"
    set newMsg to make new outgoing message with properties {{subject:"{subj_a}", content:"{body_a}", visible:true}}
    tell newMsg
        make new to recipient at end of to recipients with properties {{address:"{to_a}"}}
    end tell
    activate
end tell'''
    subprocess.Popen(["osascript", "-e", script])


@router.post("/api/open_mail")
async def open_mail(req: Request):
    body = await req.json()
    to   = (body.get("to") or "").replace('"', '\\"')
    subj = (body.get("subject") or "").replace('"', '\\"').replace("\n", " ")
    cont = (body.get("body") or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    try:
        _open_mail_compose(to, subj, cont)
        return JSONResponse({"ok": True, "message": "Apple Mail compose window opened"})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ── Email draft ────────────────────────────────────────────────────────────────

class DraftEmailRequest(BaseModel):
    entity_id: str
    contact_name: Optional[str] = None
    contact_role: Optional[str] = None
    action: str
    from_email: str = "davidlagarejo@gmail.com"
    opportunity_title: Optional[str] = None
    deadline_label: Optional[str] = None
    why_yes: Optional[list] = None
    why_contact: Optional[str] = None


@router.post("/api/draft_email")
async def draft_email(req: DraftEmailRequest):
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(request_executor, build_email_draft, req.dict())
    return JSONResponse(result)


@router.post("/api/draft_reply")
async def draft_reply(req: Request):
    body   = await req.json()
    loop   = asyncio.get_event_loop()
    result = await loop.run_in_executor(request_executor, build_reply_draft, body)
    return JSONResponse(result)


# ── Mark sent ──────────────────────────────────────────────────────────────────

@router.post("/api/tasks/mark_sent")
async def mark_task_sent(req: Request):
    body       = await req.json()
    task_id    = body.get("task_id")
    subject    = body.get("subject", "")
    from_email = body.get("from_email", "")
    to_email   = body.get("to_email", "")
    email_body = body.get("body", "")

    tasks = load_tasks()
    for t in tasks:
        if t.get("task_id") == task_id:
            t["status"]       = "sent"
            t["sent_at"]      = datetime.now(timezone.utc).isoformat()[:19] + "Z"
            t["sent_subject"] = subject
            t["sent_to"]      = to_email
            t["from_email"]   = from_email
            t["updated_at"]   = datetime.now(timezone.utc).isoformat()[:19] + "Z"
            break
    save_tasks(tasks)

    if to_email and subject and email_body:
        try:
            _open_mail_compose(to_email, subject, email_body)
        except Exception:
            pass

    return JSONResponse({"ok": True, "task_id": task_id})


# ── Mark replied ───────────────────────────────────────────────────────────────

@router.post("/api/tasks/mark_replied")
async def mark_task_replied(req: Request):
    body          = await req.json()
    task_id       = body.get("task_id")
    reply_from    = body.get("reply_from", "")
    reply_snippet = body.get("reply_snippet", "")

    tasks = load_tasks()
    for t in tasks:
        if t.get("task_id") == task_id:
            t["status"]        = "replied"
            t["replied_at"]    = datetime.now(timezone.utc).isoformat()[:19] + "Z"
            t["reply_from"]    = reply_from
            t["reply_snippet"] = reply_snippet[:500]
            t["updated_at"]    = datetime.now(timezone.utc).isoformat()[:19] + "Z"
            break
    save_tasks(tasks)
    return JSONResponse({"ok": True, "task_id": task_id})


# ── Check mail replies ─────────────────────────────────────────────────────────

@router.post("/api/check_mail_replies")
async def check_mail_replies():
    """
    Scan Apple Mail for replies to sent outreach tasks.

    Strategy (script only, no LLM):
    1. For tasks with a known sent_to email → search all mailboxes for messages
       FROM that address received after sent_at.
    2. For tasks without email → fall back to subject matching (Re: <sent_subject>)
       in inbox only.
    Both paths update task status to 'replied' when a match is found.
    """
    tasks      = load_tasks()
    sent_tasks = [t for t in tasks if t.get("status") == "sent"]
    if not sent_tasks:
        return JSONResponse({"replies": [], "checked": 0})

    # ── Check if Mail is running ───────────────────────────────────────────────
    mail_check = subprocess.run(
        ["osascript", "-e", 'tell application "System Events" to (name of processes) contains "Mail"'],
        capture_output=True, text=True, timeout=5,
    )
    if mail_check.stdout.strip().lower() != "true":
        return JSONResponse({"replies": [], "checked": 0, "error": "Mail not running"})

    found_replies: list = []
    tasks_by_id = {t["task_id"]: t for t in tasks}

    def _run_as(script: str, timeout: int = 20) -> str:
        try:
            r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
            return r.stdout.strip()
        except Exception:
            return ""

    def _mark_replied(task_id: str, reply_from: str, snippet: str):
        t = tasks_by_id.get(task_id)
        if t and t.get("status") == "sent":
            t["status"]        = "replied"
            t["replied_at"]    = datetime.now(timezone.utc).isoformat()[:19] + "Z"
            t["reply_from"]    = reply_from
            t["reply_snippet"] = snippet[:300]
            t["updated_at"]    = datetime.now(timezone.utc).isoformat()[:19] + "Z"
            return True
        return False

    for task in sent_tasks:
        task_id   = task["task_id"]
        sent_to   = (task.get("sent_to") or "").strip()
        sent_subj = (task.get("sent_subject") or "").strip()
        sent_at   = (task.get("sent_at") or "2024-01-01").strip()

        if sent_to and "@" in sent_to:
            # ── Strategy 1: search by sender email across all mailboxes ──────
            safe_email = sent_to.replace('"', "")
            script = f"""
tell application "Mail"
    set hits to {{}}
    repeat with acct in every account
        repeat with mb in every mailbox of acct
            try
                set msgs to (messages of mb whose sender contains "{safe_email}")
                repeat with m in msgs
                    set msgDate to date received of m
                    try
                        set snip to text 1 thru 300 of content of m
                    on error
                        set snip to ""
                    end try
                    set end of hits to ((subject of m) & "|||" & (sender of m) & "|||" & snip)
                end repeat
            end try
        end repeat
    end repeat
    return hits
end tell
"""
            raw = _run_as(script)
            if raw:
                for line in raw.split(",\n"):
                    line = line.strip()
                    if "|||" not in line:
                        continue
                    parts = line.split("|||")
                    subj    = parts[0].strip()
                    sender  = parts[1].strip() if len(parts) > 1 else ""
                    snippet = parts[2].strip() if len(parts) > 2 else ""
                    if _mark_replied(task_id, sender, snippet):
                        found_replies.append({
                            "task_id":       task_id,
                            "entity_name":   task.get("entity_name", ""),
                            "reply_from":    sender,
                            "reply_snippet": snippet[:200],
                            "match":         "email",
                        })
                    break  # first match is enough

        elif sent_subj:
            # ── Strategy 2: subject match in inbox (fallback) ────────────────
            safe_subj = sent_subj.replace('"', "").replace("\\", "")
            script = f"""
tell application "Mail"
    set hits to {{}}
    try
        set msgs to (messages of inbox whose subject contains "{safe_subj}")
        repeat with m in msgs
            try
                set snip to text 1 thru 300 of content of m
            on error
                set snip to ""
            end try
            set end of hits to ((subject of m) & "|||" & (sender of m) & "|||" & snip)
        end repeat
    end try
    return hits
end tell
"""
            raw = _run_as(script)
            if raw:
                for line in raw.split(",\n"):
                    line = line.strip()
                    if "|||" not in line:
                        continue
                    parts   = line.split("|||")
                    sender  = parts[1].strip() if len(parts) > 1 else ""
                    snippet = parts[2].strip() if len(parts) > 2 else ""
                    if _mark_replied(task_id, sender, snippet):
                        found_replies.append({
                            "task_id":       task_id,
                            "entity_name":   task.get("entity_name", ""),
                            "reply_from":    sender,
                            "reply_snippet": snippet[:200],
                            "match":         "subject",
                        })
                    break

    if found_replies:
        save_tasks(tasks)

    return JSONResponse({"replies": found_replies, "checked": len(sent_tasks)})


# ── Deadline scanner ──────────────────────────────────────────────────────────

@router.post("/api/scan_deadlines")
async def scan_deadlines(req: Request):
    """
    Scan dossiers for deadlines using web fetch + regex (no LLM).
    Body: {} to scan all, or {"entity_ids": ["ent_abc..."]} for subset.
    Invalidates dossier index and timeline cache when done.
    """
    body       = await req.json() if req.headers.get("content-length", "0") != "0" else {}
    entity_ids = body.get("entity_ids") or None
    loop       = asyncio.get_event_loop()

    def _run():
        from scripts.deadline_lookup import lookup_deadlines
        from tools.dossier_index import invalidate as invalidate_index
        results = lookup_deadlines(entity_ids)
        invalidate_index()
        return results

    results = await loop.run_in_executor(request_executor, _run)
    _invalidate_timeline()

    found     = [r for r in results if r.get("status") == "found"]
    not_found = [r for r in results if r.get("status") == "not_found"]
    return JSONResponse({
        "found":     len(found),
        "not_found": len(not_found),
        "results":   results,
    })


# ── Inbox scan ────────────────────────────────────────────────────────────────

@router.get("/api/mail/scan")
async def mail_scan(days: int = 7):
    """
    Scan Apple Mail inbox for emails related to timeline opportunities.
    Returns categorized matches (INVESTMENT, GRANT, JOB, REPLY, ENTITY).
    """
    from services.mail_scanner import scan_inbox
    result = await asyncio.to_thread(scan_inbox, days)
    return JSONResponse(result)


@router.post("/api/mail/draft_reply")
async def mail_draft_reply(req: Request):
    """
    Draft a reply to an email found during inbox scan.
    Body: {subject, sender, snippet, entity_id?, lang: "en"|"es"}
    Returns: {subject, body, lang}
    """
    body = await req.json()
    lang = body.get("lang", "en")  # default: English for outreach

    result = await asyncio.to_thread(build_reply_draft, {
        **body,
        "lang": lang,
    })

    # If lang is "es" and draft came out in English, translate
    if lang == "es" and result.get("body"):
        from services.translator import translate_to_spanish, is_english
        draft_body = result["body"]
        if is_english(draft_body):
            try:
                translated = await asyncio.to_thread(translate_to_spanish, draft_body)
                if translated:
                    result["body"] = translated
            except Exception:
                pass

    result["lang"] = lang
    return JSONResponse(result)


# ── Timeline cleanup ───────────────────────────────────────────────────────────

@router.post("/api/timeline/cleanup")
async def timeline_cleanup():
    _invalidate_timeline()
    """
    Daily cleanup:
    - Remove timeline cards older than 10 days (except priority=1 seguimiento cards).
    - Archive outreach tasks with status done/cancelled/archived older than 3 days.
    """
    now            = datetime.now(timezone.utc)
    removed_cards  = 0
    archived_tasks = 0

    if _CARDS_FILE.exists():
        try:
            cards = json.loads(_CARDS_FILE.read_text(encoding="utf-8"))
            kept  = []
            for c in cards:
                meta       = c.get("meta", "")
                date_match = None
                for part in meta.split():
                    if re.match(r"\d{4}-\d{2}-\d{2}", part):
                        try:
                            date_match = datetime.fromisoformat(part + "T00:00:00+00:00")
                        except Exception:
                            pass
                        break
                if date_match:
                    age_days = (now - date_match).days
                    if age_days > 10 and c.get("priority", 2) > 1:
                        removed_cards += 1
                        continue
                kept.append(c)
            _CARDS_FILE.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            _log.warning("timeline_cleanup cards failed: %s", exc)

    tasks  = load_tasks()
    active = []
    for t in tasks:
        status = t.get("status", "")
        if status in ("done", "cancelled", "archived"):
            updated = t.get("updated_at", t.get("created_at", ""))[:10]
            try:
                age = (now.date() - datetime.fromisoformat(updated).date()).days if updated else 999
            except Exception:
                age = 999
            if age > 3:
                archived_tasks += 1
                continue
        active.append(t)
    save_tasks(active)

    return JSONResponse({"removed_cards": removed_cards, "archived_tasks": archived_tasks})
