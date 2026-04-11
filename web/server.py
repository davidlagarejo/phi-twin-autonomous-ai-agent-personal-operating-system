"""
Phi Twin — FastAPI server
Port 8080, localhost only.

Responsibilities of this file:
  - Application factory (app, middleware, lifespan)
  - Background job runner (_job_consumer) and periodic scheduler (_periodic_scheduler)
  - Startup health checks (Ollama, SEARXNG config validation)
  - Static assets + / (HTML) + /health routes
  - Router registration (all domain routes live in api/routes/)

What does NOT live here:
  - Route handlers (see api/routes/)
  - LLM retry/audit logic (see llm/pipeline.py)
  - Business logic (see services/)
  - Persistence (see tools/)
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from api.context import workspace as _workspace_state, request_executor as _research_executor  # noqa: E402
from core.job_state import job_state, ResearchJob, JOB_INTERACTIVE, JOB_BACKGROUND             # noqa: E402
from tools.research_engine import execute_research_cycle, RunBudget                            # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
with open(BASE_DIR / "config" / "settings.json") as f:
    SETTINGS = json.load(f)

OLLAMA_URL             = SETTINGS["ollama"]["base_url"]
OLLAMA_MODEL           = SETTINGS["ollama"]["model"]
TEMPERATURE            = SETTINGS["ollama"]["temperature"]
SCHEDULER_INTERVAL_SEC = 1800  # 30 minutes

# Budget profiles (shared constants — route handlers import their own from timeline.py)
BUDGET_INTERACTIVE = RunBudget(max_seconds=480.0, max_web_queries=30, max_sources=60, max_tasks=1)
BUDGET_BACKGROUND  = RunBudget(max_seconds=360.0, max_web_queries=25, max_sources=50, max_tasks=1)
BUDGET_SCHEDULED   = RunBudget(max_seconds=480.0, max_web_queries=30, max_sources=60, max_tasks=1)

_ACTIVITY_FILE = BASE_DIR / "data" / "activity.jsonl"
_ACTIVITY_FILE.parent.mkdir(parents=True, exist_ok=True)

_log = logging.getLogger("phi.server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

# ── Activity log writer ───────────────────────────────────────────────────────

def _log_activity(project: str, action: str, detail: str, status: str = "info"):
    label = (project or "background")[:40].rstrip()
    entry = {
        "ts":      datetime.now(timezone.utc).isoformat()[:19] + "Z",
        "project": label,
        "action":  action,
        "detail":  detail[:180],
        "status":  status,
    }
    try:
        with open(_ACTIVITY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


class _ResearchActivityHandler(logging.Handler):
    """Taps phi.research_engine logger; converts ACT: prefixed messages to activity entries."""
    _ICONS = {
        "search_start":        "🔍",
        "search_done":         "📥",
        "entity_found":        "🏢",
        "due_diligence_start": "🔬",
        "dossier_saved":       "📋",
    }
    _LABELS = {
        "search_start":        "Buscando",
        "search_done":         "Resultados",
        "entity_found":        "Entidad encontrada",
        "due_diligence_start": "Analizando",
        "dossier_saved":       "Dossier creado",
    }

    def emit(self, record):
        msg = record.getMessage()
        if not msg.startswith("ACT:"):
            return
        rest  = msg[4:]
        code  = rest.split(" ")[0]
        icon  = self._ICONS.get(code, "·")
        label = self._LABELS.get(code, code)
        pairs = re.findall(r'\w+=\'([^\']+)\'|\w+="([^"]+)"|\w+=(\S+)', rest)
        first_val = next((a or b or c for a, b, c in pairs), "")
        detail = f"{icon} {label}: {first_val}" if first_val else f"{icon} {label}"
        _log_activity(job_state.current_goal, code, detail, "working")


# ── Background job consumer ───────────────────────────────────────────────────

async def _job_consumer():
    """Single consumer — max 1 heavy research job running at a time."""
    while True:
        try:
            job: ResearchJob = await job_state.queue.get()
        except asyncio.CancelledError:
            return

        job_state.running      = True
        job_state.current_goal = job.goal
        _log.info("research_job_start job_id=%s goal=%r triggered_by=%s",
                  job.job_id, job.goal[:80], job.triggered_by)
        _log_activity(job.goal, "cycle_start",
                      f"▶ Iniciando ciclo · {job.budget.max_web_queries} búsquedas máx · {int(job.budget.max_seconds)}s budget",
                      "working")
        t0 = time.monotonic()
        try:
            loop   = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _research_executor,
                execute_research_cycle,
                _workspace_state,
                job.budget,
                None,
            )
            elapsed              = time.monotonic() - t0
            job_state.last_result = {
                "job_id":          job.job_id,
                "goal":            job.goal,
                "triggered_by":    job.triggered_by,
                "status":          result.status,
                "tasks_run":       result.tasks_run,
                "queries_used":    result.queries_used,
                "elapsed_seconds": round(elapsed, 1),
                "artifacts":       result.artifacts,
                "completed_at":    datetime.now(timezone.utc).isoformat(),
            }
            job_state.last_at = time.time()
            _log.info(
                "research_job_done job_id=%s status=%s tasks=%d queries=%d elapsed=%.0fs",
                job.job_id, result.status, result.tasks_run, result.queries_used, elapsed,
            )
            artifact_names = [a.get("id", "?") for a in result.artifacts[:4]]
            _log_activity(
                job.goal, "cycle_done",
                f"✅ {result.status} · {result.tasks_run} tarea(s) · {result.queries_used} búsquedas · {round(elapsed)}s"
                + (f" · artefactos: {', '.join(artifact_names)}" if artifact_names else ""),
                "done",
            )

            # Auto-post timeline card if artifacts produced
            if result.artifacts:
                _CARDS_FILE = BASE_DIR / "data" / "timeline_cards.json"
                _CARDS_FILE.parent.mkdir(parents=True, exist_ok=True)
                try:
                    cards = json.loads(_CARDS_FILE.read_text(encoding="utf-8")) if _CARDS_FILE.exists() else []
                except Exception:
                    cards = []

                raw_goal = job.goal.strip()
                display_project = "Ciclo automático" if raw_goal.lower() in ("proactive_scheduled_cycle", "") else raw_goal[:40]
                summary = (result.result_summary or "").split(".")[0].strip()
                if summary:
                    m = re.match(r"(DISCOVER|DUE_DILIGENCE|CORRELATE|VALIDATE)\s+(?:for\s+)?(.+?):", summary, re.I)
                    strategy_label = {"DISCOVER": "Descubrimiento", "DUE_DILIGENCE": "Análisis",
                                      "CORRELATE": "Correlación", "VALIDATE": "Validación"}
                    if m:
                        slabel     = strategy_label.get(m.group(1).upper(), m.group(1))
                        card_title = f"{slabel} · {m.group(2)[:50]}"
                    else:
                        card_title = summary[:70]
                else:
                    card_title = display_project

                card = {
                    "id":       f"job_{job.job_id[:8]}",
                    "project":  display_project,
                    "priority": 2,
                    "tag":      "investigación",
                    "title":    card_title,
                    "meta":     f"Phi-4 · {datetime.now(timezone.utc).strftime('%Y-%m-%d')} · {result.queries_used} búsquedas · {round(elapsed)}s",
                    "desc":     (result.result_summary or "")[:180],
                    "metrics":  [{"label": "Artefactos", "value": str(len(result.artifacts))}],
                    "actions":  [],
                    "viz":      None,
                }
                cards = [c for c in cards if c.get("id") != card["id"]]
                cards.append(card)
                research_cards = [c for c in cards if c.get("tag") == "investigación"][-5:]
                other_cards    = [c for c in cards if c.get("tag") != "investigación"]
                cards = other_cards + research_cards
                cards.sort(key=lambda c: c.get("priority", 99))
                _CARDS_FILE.write_text(json.dumps(cards, ensure_ascii=False, indent=2), encoding="utf-8")

        except Exception as exc:
            elapsed = time.monotonic() - t0
            _log.error("research_job_failed job_id=%s error=%s elapsed=%.0fs", job.job_id, exc, elapsed)
            _log_activity(job.goal, "cycle_error", f"❌ Error: {str(exc)[:120]}", "error")
            job_state.last_result = {
                "job_id":          job.job_id,
                "goal":            job.goal,
                "triggered_by":    job.triggered_by,
                "status":          "FAILED",
                "error":           str(exc),
                "elapsed_seconds": round(elapsed, 1),
                "completed_at":    datetime.now(timezone.utc).isoformat(),
            }
        finally:
            job_state.running      = False
            job_state.current_goal = ""
            job_state.queue.task_done()


# ── Periodic scheduler ────────────────────────────────────────────────────────

async def _periodic_scheduler():
    """Enqueues a background research cycle every SCHEDULER_INTERVAL_SEC."""
    _log.info("scheduler_start interval=%ds", SCHEDULER_INTERVAL_SEC)
    await asyncio.sleep(60)  # grace period after startup
    while True:
        jid = str(uuid.uuid4())
        job = ResearchJob(
            priority=JOB_BACKGROUND,
            created_at=time.time(),
            job_id=jid,
            goal="proactive_scheduled_cycle",
            budget=BUDGET_SCHEDULED,
            triggered_by="scheduler",
        )
        await job_state.queue.put(job)
        _log.info("scheduler_enqueued job_id=%s", jid)
        await asyncio.sleep(SCHEDULER_INTERVAL_SEC)


# ── App factory ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app_instance):
    """Startup: validate config, start job runner and scheduler."""
    # SEARXNG validation
    searxng_url = SETTINGS.get("searxng", {}).get("url") or ""
    if not searxng_url:
        _log.warning("SEARXNG_URL not configured — web search will be blocked.")
    elif "127.0.0.1" not in searxng_url and "localhost" not in searxng_url:
        _log.warning("SEARXNG_URL '%s' is not local — web search will be blocked by privacy gate.", searxng_url)
    else:
        os.environ["SEARXNG_URL"] = searxng_url
        _log.info("SEARXNG_URL configured: %s", searxng_url)

    # Ollama check
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r      = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            if any(OLLAMA_MODEL in m for m in models):
                _log.info("ollama_ready model=%s", OLLAMA_MODEL)
            else:
                _log.warning("ollama model %s not found. Available: %s", OLLAMA_MODEL, models)
    except Exception as e:
        _log.warning("ollama_unreachable: %s", e)

    # Tap research engine logger for live activity log
    _re_logger = logging.getLogger("phi.research_engine")
    _activity_handler = _ResearchActivityHandler()
    _activity_handler.setLevel(logging.INFO)
    _re_logger.addHandler(_activity_handler)

    # Start background job runner
    job_state.queue = asyncio.PriorityQueue()
    consumer  = asyncio.create_task(_job_consumer(),          name="job_consumer")
    scheduler = asyncio.create_task(_periodic_scheduler(),    name="research_scheduler")
    _log.info("background_runner_started scheduler_interval=%ds", SCHEDULER_INTERVAL_SEC)

    # Auto-seed on first startup
    state_summary = _workspace_state.get_state_summary()
    if state_summary.get("run_counter", 0) == 0:
        _log.info("first_startup: seeding initial research goals")
        for seed_goal in [
            "Zircular IIoT seed funding investors cleantech",
            "ZION ING energy consulting clients grants NYC",
        ]:
            jid      = str(uuid.uuid4())
            seed_job = ResearchJob(
                priority=JOB_BACKGROUND,
                created_at=time.time(),
                job_id=jid,
                goal=seed_goal,
                budget=BUDGET_BACKGROUND,
                triggered_by="auto_seed",
            )
            async def _seed(j=seed_job):
                await asyncio.sleep(5)
                await job_state.queue.put(j)
            asyncio.create_task(_seed())

    # Warm up translator (downloads en→es model if not present; non-blocking)
    try:
        from services.translator import warm_up as _translator_warm_up
        _translator_warm_up()
    except Exception as _te:
        _log.warning("translator warm-up skipped: %s", _te)

    yield

    consumer.cancel()
    scheduler.cancel()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "null",
    ],
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)

# ── Domain routers ────────────────────────────────────────────────────────────
from api.routes.crm       import router as _crm_router        # noqa: E402
from api.routes.roadmap   import router as _roadmap_router    # noqa: E402
from api.routes.questions import router as _questions_router  # noqa: E402
from api.routes.system    import router as _system_router     # noqa: E402
from api.routes.outreach  import router as _outreach_router   # noqa: E402
from api.routes.timeline  import router as _timeline_router   # noqa: E402
from api.routes.chat      import router as _chat_router       # noqa: E402
from api.routes.jobs      import router as _jobs_router       # noqa: E402
from api.routes.chats     import router as _chats_router      # noqa: E402

app.include_router(_crm_router)
app.include_router(_roadmap_router)
app.include_router(_questions_router)
app.include_router(_system_router)
app.include_router(_outreach_router)
app.include_router(_timeline_router)
app.include_router(_chat_router)
app.include_router(_jobs_router)
app.include_router(_chats_router)

# ── Static assets ─────────────────────────────────────────────────────────────
_STATIC_DIR = BASE_DIR / "web" / "static"
if _STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ── Core routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    for name in ("davidsan.html", "index.html"):
        html = BASE_DIR / "web" / name
        if html.exists():
            return HTMLResponse(html.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Phi Twin — starting up</h1>")


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r      = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
            phi_ready = any(OLLAMA_MODEL in m for m in models)
    except Exception:
        phi_ready = False

    return {
        "status":      "ok",
        "model":       OLLAMA_MODEL,
        "phi_ready":   phi_ready,
        "temperature": TEMPERATURE,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host=SETTINGS["web"]["host"], port=SETTINGS["web"]["port"], reload=False)
