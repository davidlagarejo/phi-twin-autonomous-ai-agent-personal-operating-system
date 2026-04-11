"""
services/session.py — Session briefing for chat context injection.

Single responsibility: build a compact state summary to inject at the start of
each chat session so Phi knows the current project state without the user
having to repeat it.

Called once per conversation (when messages length == 1 — first user turn).
Kept synchronous and fast (<5ms): reads only already-cached or small files.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from core.job_state import job_state
from tools.dossier_index import load_dossiers
from tools.outreach import load_tasks, ACTIVE_STATUSES

_BASE = Path(__file__).parent.parent
_AUDIT_LOG = _BASE / "data" / "audit_logs" / "audit.jsonl"


def _last_audit_event() -> Optional[str]:
    """Return the last audit log timestamp (ISO, truncated to minute)."""
    try:
        if not _AUDIT_LOG.exists():
            return None
        lines = _AUDIT_LOG.read_text(encoding="utf-8").splitlines()
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            import json
            entry = json.loads(line)
            ts = entry.get("timestamp", "")[:16]  # "2026-04-04T12:34"
            flow = entry.get("flow", "")
            return f"{ts} ({flow})" if flow else ts
    except Exception:
        return None


def build_session_brief() -> str:
    """
    Return a compact Markdown block with current project state.
    Injected as a system-level context message at the start of each chat session.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ── Top dossiers ──────────────────────────────────────────────────────────
    dossiers = load_dossiers()
    top_lines: list[str] = []
    for d in dossiers[:5]:
        name = d.get("name", "?")
        fit  = (d.get("fit_assessment") or {}).get("fit_score", 0) or 0
        etype = d.get("type", "")
        na   = (d.get("next_actions") or [])
        next_action = ""
        if na:
            first = na[0]
            next_action = (first if isinstance(first, str) else first.get("action", ""))[:60]
        line = f"- **{name}** ({etype}, fit {fit}%)"
        if next_action:
            line += f" → {next_action}"
        top_lines.append(line)

    # ── Pending tasks ─────────────────────────────────────────────────────────
    tasks = load_tasks()
    pending  = [t for t in tasks if t.get("status") == "pending"]
    followup = [t for t in tasks if t.get("status") == "followup_due"]
    replied  = [t for t in tasks if t.get("status") == "replied"]

    task_parts: list[str] = []
    if pending:
        task_parts.append(f"{len(pending)} pendiente(s)")
    if followup:
        task_parts.append(f"{len(followup)} followup")
    if replied:
        task_parts.append(f"{len(replied)} con respuesta")
    task_summary = ", ".join(task_parts) if task_parts else "ninguna"

    # ── Background job ────────────────────────────────────────────────────────
    if job_state.running and job_state.current_goal:
        job_line = f"🔄 Investigando ahora: *{job_state.current_goal[:80]}*"
    elif job_state.current_goal and job_state.last_at:
        from datetime import datetime as _dt
        import time as _time
        elapsed_min = int((_time.monotonic() - job_state.last_at) / 60)
        job_line = f"Último research: *{job_state.current_goal[:80]}* (hace ~{elapsed_min}min)"
    else:
        job_line = "Sin research activo"

    # ── Last audit event ──────────────────────────────────────────────────────
    last_audit = _last_audit_event()
    audit_line = f"Último ciclo LLM: {last_audit}" if last_audit else ""

    # ── Assemble ──────────────────────────────────────────────────────────────
    sections: list[str] = [
        f"[ESTADO DEL PROYECTO — {now}]",
        "",
        f"**Oportunidades top** ({len(dossiers)} total):",
    ]
    sections.extend(top_lines or ["- (sin dossiers aún)"])
    sections += [
        "",
        f"**Outreach:** {task_summary}",
        f"**Research:** {job_line}",
    ]
    if audit_line:
        sections.append(audit_line)

    return "\n".join(sections)
