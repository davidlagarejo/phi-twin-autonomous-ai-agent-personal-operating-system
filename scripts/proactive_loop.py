#!/usr/bin/env python3
"""
proactive_loop.py — Phi proactive agent daemon
================================================
Reemplaza los workflows de n8n. Corre como daemon en background.

Comportamiento:
  - Cada 2.5h: dispara ciclo de research → genera preguntas → notifica macOS
  - Daily 7AM:  brief matutino (estado del workspace)
  - Al encontrar hallazgos urgentes: guarda pending_message para el UI

Inicio:
  python3 scripts/proactive_loop.py &

  O desde start.sh (ya incluido).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_URL       = os.environ.get("PHI_SERVER", "http://127.0.0.1:8080")
RESEARCH_INTERVAL_SEC = int(os.environ.get("RESEARCH_INTERVAL", 9000))  # 2.5h
BRIEF_HOUR           = int(os.environ.get("BRIEF_HOUR", 7))             # 7AM local
POLL_AFTER_ENQUEUE   = int(os.environ.get("POLL_AFTER_ENQUEUE", 180))   # max wait 3min
LOG_LEVEL            = os.environ.get("LOG_LEVEL", "info").upper()

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
STATE_FILE = DATA_DIR / "proactive_state.json"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [proactive] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log = logging.getLogger("proactive_loop")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _api(method: str, path: str, body: dict | None = None, timeout: int = 30) -> dict:
    url = SERVER_URL + path
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        _log.warning("api_error %s %s → %d", method, path, e.code)
        return {}
    except Exception as e:
        _log.warning("api_error %s %s → %s", method, path, e)
        return {}


def _notify(title: str, message: str):
    """Envía notificación macOS vía osascript."""
    # Sanitize for AppleScript
    title_safe   = title.replace('"', "'").replace("\\", "")[:80]
    message_safe = message.replace('"', "'").replace("\\", "")[:200]
    script = (
        f'display notification "{message_safe}" '
        f'with title "{title_safe}" '
        f'sound name "Glass"'
    )
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
        _log.info("notify_sent title=%r", title_safe)
    except Exception as e:
        _log.warning("notify_failed: %s", e)


def _save_pending_message(text: str):
    """Guarda mensaje pendiente para que el UI lo muestre al abrir."""
    payload = {
        "text": text,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "read": False,
    }
    pending_path = DATA_DIR / "pending_message.json"
    try:
        pending_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        _log.info("pending_message_saved len=%d", len(text))
    except Exception as e:
        _log.warning("pending_message_save_failed: %s", e)


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    except Exception:
        return {}


def _save_state(state: dict):
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    except Exception as e:
        _log.warning("state_save_failed: %s", e)


# ── Core actions ──────────────────────────────────────────────────────────────

def run_research_cycle() -> dict:
    """Dispara un ciclo de research y espera hasta POLL_AFTER_ENQUEUE segundos."""
    _log.info("enqueuing_research_cycle")
    result = _api("POST", "/api/enqueue", {"goal": "proactive_loop_cycle", "priority": 1})
    if not result.get("job_id"):
        _log.warning("enqueue_failed: %s", result)
        return {}

    job_id = result["job_id"]
    _log.info("enqueued job_id=%s — polling up to %ds", job_id, POLL_AFTER_ENQUEUE)

    deadline = time.time() + POLL_AFTER_ENQUEUE
    while time.time() < deadline:
        time.sleep(10)
        diag = _api("GET", "/api/diagnostics")
        last = diag.get("job_runner", {}).get("last_job", {})
        if last.get("job_id") == job_id and not diag.get("job_runner", {}).get("job_running"):
            _log.info("cycle_done tasks=%d queries=%d artifacts=%d",
                      last.get("tasks_run", 0), last.get("queries_used", 0),
                      len(last.get("artifacts", [])))
            return last
        if not diag.get("job_runner", {}).get("job_running"):
            # Cycle completed quickly (or was skipped)
            break

    return {}


def generate_questions() -> list[dict]:
    """Llama a /api/questions con refresh para obtener preguntas frescas."""
    _log.info("generating_questions")
    result = _api("GET", "/api/questions?refresh=true", timeout=120)
    questions = result.get("questions", [])
    _log.info("questions_generated count=%d", len(questions))
    return questions


def get_workspace_state() -> dict:
    result = _api("GET", "/api/state")
    return result


def run_proactive_cycle():
    """Ciclo completo: research → preguntas → notificación → pending_message."""
    _log.info("=== PROACTIVE CYCLE START ===")

    cycle_result = run_research_cycle()
    questions    = generate_questions()
    ws           = get_workspace_state()

    tasks_run    = cycle_result.get("tasks_run", 0)
    queries_used = cycle_result.get("queries_used", 0)
    artifacts    = cycle_result.get("artifacts", [])
    n_dossiers   = (ws.get("dossier_summary") or {}).get("DRAFT", 0)
    n_pending    = (ws.get("pending_approvals") or [])
    n_pending    = len(n_pending) if isinstance(n_pending, list) else 0

    # ── Notificación macOS ────────────────────────────────────────────────────
    high_q = [q for q in questions if q.get("urgency") == "high"]
    medium_q = [q for q in questions if q.get("urgency") == "medium"]

    if high_q or artifacts:
        notify_title = "Phi · Nuevos hallazgos"
        lines = []
        if artifacts:
            lines.append(f"🔍 {len(artifacts)} artefacto(s) nuevos")
        if n_pending:
            lines.append(f"📋 {n_pending} contacto(s) para revisar")
        if high_q:
            lines.append(f"❓ {high_q[0]['question'][:120]}")
        _notify(notify_title, " · ".join(lines) if lines else "Ciclo completado.")
    elif tasks_run > 0:
        _notify("Phi · Ciclo completado", f"✅ {tasks_run} tarea(s) · {queries_used} búsquedas")

    # ── Pending message para el UI ────────────────────────────────────────────
    if questions or artifacts:
        msg_lines = []
        if artifacts:
            msg_lines.append(f"Encontré **{len(artifacts)} hallazgo(s)** en el ciclo de investigación.")
        if n_pending:
            msg_lines.append(f"Hay **{n_pending} contacto(s)** esperando tu revisión en el timeline.")
        if high_q:
            msg_lines.append("")
            msg_lines.append("**Pregunto esto porque necesito tu input para avanzar:**")
            for q in high_q[:2]:
                msg_lines.append(f"- {q['question']}")
                if q.get("context"):
                    msg_lines.append(f"  _{q['context']}_")
        elif medium_q:
            msg_lines.append("")
            msg_lines.append(f"**Pregunta pendiente:** {medium_q[0]['question']}")

        if msg_lines:
            _save_pending_message("\n".join(msg_lines))

    _log.info("=== PROACTIVE CYCLE END tasks=%d questions=%d ===", tasks_run, len(questions))


def check_followups():
    """Notifica sobre contactos CRM con seguimiento vencido."""
    try:
        sys.path.insert(0, str(BASE_DIR))
        from tools.crm import get_followup_due, update_status
        due = get_followup_due()
        if not due:
            return
        _log.info("followup_due count=%d", len(due))
        for c in due[:3]:
            company = c.get("company", "?")
            name    = c.get("name") or "contacto"
            _notify(
                f"Phi · Seguimiento pendiente",
                f"{name} @ {company} — es momento de hacer follow-up",
            )
            update_status(c["id"], "followup_due")
        # Save pending message if multiple
        if len(due) > 1:
            lines = [f"**{len(due)} seguimientos pendientes:**"]
            for c in due[:5]:
                lines.append(f"- {c.get('name') or 'Contacto'} @ {c.get('company','?')}")
            _save_pending_message("\n".join(lines))
    except Exception as e:
        _log.warning("check_followups failed: %s", e)


def _ensure_research_queue_not_empty():
    """
    If the research queue has fewer than MIN_QUEUE entities, re-enqueue
    all dossiers (oldest-updated first) so Phi is always investigating.
    """
    MIN_QUEUE = 3
    ws    = _api("GET", "/api/state")
    queue = ws.get("queue_summary") or {}
    total_queued = sum(queue.values()) if isinstance(queue, dict) else 0

    if total_queued >= MIN_QUEUE:
        _log.info("queue_ok total_queued=%d", total_queued)
        return

    _log.info("queue_low total=%d — refilling from dossiers", total_queued)

    # Load all dossiers, sort by last_updated ascending (oldest first)
    from pathlib import Path as _Path
    import json as _json
    dossier_dir = BASE_DIR / "workspace" / "dossiers"
    dossiers = []
    for p in dossier_dir.glob("*.json"):
        try:
            d = _json.loads(p.read_text(encoding="utf-8"))
            dossiers.append((d.get("last_updated", ""), d.get("entity_id", ""), d.get("name", "")))
        except Exception:
            continue

    dossiers.sort(key=lambda x: x[0])  # oldest first

    enqueued = 0
    for _, entity_id, name in dossiers[:5]:
        if not entity_id:
            continue
        result = _api("POST", "/api/enqueue", {
            "goal":      f"due_diligence {name}",
            "entity_id": entity_id,
            "priority":  2,
        })
        if result.get("job_id"):
            enqueued += 1
            _log.info("requeued entity=%s job_id=%s", name, result["job_id"])

    _log.info("queue_refill done enqueued=%d", enqueued)


def run_morning_brief():
    """Brief matutino diario."""
    _log.info("=== MORNING BRIEF ===")

    ws         = get_workspace_state()
    questions  = generate_questions()

    dossiers   = ws.get("dossier_summary") or {}
    queue      = ws.get("queue_summary") or {}
    n_draft    = dossiers.get("DRAFT", 0)
    n_complete = dossiers.get("COMPLETE", 0)
    n_pending  = len(ws.get("pending_approvals") or [])
    n_runs     = ws.get("run_counter", 0)
    high_q     = [q for q in questions if q.get("urgency") == "high"]

    brief_lines = [
        f"📊 {n_draft} dossiers · {n_pending} contactos para revisar · {n_runs} ciclos acumulados"
    ]
    if high_q:
        brief_lines.append(f"❓ {high_q[0]['question'][:140]}")

    brief_text = "\n".join(brief_lines)
    _notify("Phi · Brief matutino", brief_text)

    # Pending message con el brief completo
    msg_lines = ["**Buenos días. Resumen de lo que encontré:**", ""]
    msg_lines.append(f"- **{n_draft}** dossiers en borrador · **{n_complete}** completos")
    msg_lines.append(f"- **{n_pending}** contactos/organizaciones esperando tu aprobación")
    msg_lines.append(f"- **{n_runs}** ciclos de investigación acumulados")
    if high_q:
        msg_lines.append("")
        msg_lines.append("**Lo más urgente hoy:**")
        for q in high_q[:3]:
            msg_lines.append(f"- {q['question']}")

    _save_pending_message("\n".join(msg_lines))
    _log.info("=== MORNING BRIEF DONE ===")


# ── Scheduler loop ────────────────────────────────────────────────────────────

def _should_run_brief(state: dict) -> bool:
    last_brief = state.get("last_brief_date", "")
    today = datetime.now().strftime("%Y-%m-%d")
    if last_brief == today:
        return False
    # Solo si estamos en la hora del brief (±10 min)
    now = datetime.now()
    return now.hour == BRIEF_HOUR and now.minute <= 10


def main():
    _log.info("proactive_loop started server=%s interval=%ds brief_hour=%d",
              SERVER_URL, RESEARCH_INTERVAL_SEC, BRIEF_HOUR)

    state = _load_state()
    last_cycle_at = state.get("last_cycle_at", 0)

    # Esperar a que el servidor esté listo
    for attempt in range(30):
        health = _api("GET", "/health")
        if health.get("status") == "ok":
            _log.info("server_ready")
            break
        _log.info("waiting_for_server attempt=%d", attempt + 1)
        time.sleep(10)
    else:
        _log.error("server_not_ready after 300s — exiting")
        sys.exit(1)

    # Espera adicional al arrancar (dar tiempo al server para inicializarse)
    time.sleep(90)

    while True:
        state = _load_state()
        now   = time.time()

        # Brief matutino
        if _should_run_brief(state):
            try:
                run_morning_brief()
            except Exception as e:
                _log.error("morning_brief_failed: %s", e)
            state["last_brief_date"] = datetime.now().strftime("%Y-%m-%d")
            _save_state(state)

        # Ciclo de research cada 2.5h
        elapsed_since_cycle = now - float(state.get("last_cycle_at", 0))
        if elapsed_since_cycle >= RESEARCH_INTERVAL_SEC:
            try:
                run_proactive_cycle()
            except Exception as e:
                _log.error("proactive_cycle_failed: %s", e)

            # Enforce: always populate missing websites after each research cycle
            try:
                from scripts.website_filler import fill_missing_websites
                fill_missing_websites()
            except Exception as e:
                _log.warning("website_filler_failed: %s", e)

            # Enforce: seed investor stubs if fewer than 10 investor dossiers exist
            try:
                from scripts.seed_investors import seed_investors as _seed_investors
                from tools.dossier_index import load_dossiers as _load_dossiers
                _investor_count = sum(1 for d in _load_dossiers() if d.get("type") == "INVESTOR")
                if _investor_count < 10:
                    _log.info("investor_count=%d — seeding investor stubs", _investor_count)
                    _seed_investors(overwrite=False)
            except Exception as e:
                _log.warning("seed_investors_failed: %s", e)

            # Enforce: re-enqueue all dossiers if research queue is empty
            try:
                _ensure_research_queue_not_empty()
            except Exception as e:
                _log.warning("queue_refill_failed: %s", e)

            state["last_cycle_at"] = now
            _save_state(state)

        # Revisar seguimientos CRM en cada ciclo
        try:
            check_followups()
        except Exception as e:
            _log.warning("followup_check_failed: %s", e)

        # Revisar replies en Apple Mail (solo si Mail está abierto)
        try:
            from scripts.mail_tracker import check_mail_replies
            check_mail_replies()
        except Exception as e:
            _log.warning("mail_tracker_failed: %s", e)

        # ── Job tracker: escaneo cada 4h ──────────────────────────────────────
        JOB_SCAN_INTERVAL = 14400  # 4 horas
        elapsed_job_scan = now - float(state.get("last_job_scan_at", 0))
        if elapsed_job_scan >= JOB_SCAN_INTERVAL:
            _log.info("job_scan_start")
            try:
                import importlib.util, sys as _sys
                _spec = importlib.util.spec_from_file_location(
                    "job_tracker", BASE_DIR / "scripts" / "job_tracker.py"
                )
                _jt = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_jt)
                summary = _jt.run_scan_cycle()
                _log.info("job_scan_done %s", summary)
            except Exception as e:
                _log.warning("job_scan_failed: %s", e)
            state["last_job_scan_at"] = now
            _save_state(state)

        # Sleep 5 minutos entre checks
        time.sleep(300)


if __name__ == "__main__":
    main()
