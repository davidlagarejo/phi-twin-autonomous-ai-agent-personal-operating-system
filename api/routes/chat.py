"""
api/routes/chat.py — Chat, triage, dossier, plan, and orchestrator endpoints.

Endpoints:
  POST /api/chat           — streaming chat (SSE)
  POST /api/triage         — classify a signal: INVESTIGATE | ASK | DISCARD
  POST /api/dossier        — generate an opportunity dossier
  POST /api/plan           — return a PLAN_JSON routing plan
  POST /api/execute/step   — execute a single plan step (SEARCH_WEB / RESPOND_DIRECT)
  POST /api/ask_claude     — forward a safe abstract spec to Claude API

Single responsibility: HTTP boundary only.
LLM retry logic lives in llm/pipeline.py. Raw LLM calls in llm/client.py.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from api.context import BASE_DIR, request_executor
from core.gates import GateResult
from core.orchestrator_policy import (
    decide_next_action,
    RESPOND_DIRECT,
    ASK_CLAUDE_FOR_INSTRUCTIONS,
)
from core.plan_validator import validate_plan_json, plan_is_json_only
from llm.client import call_phi, call_phi_with_tools
from llm.pipeline import call_phi_with_retry, write_audit
from llm.tool_executor import run_tool_loop
from tools.tool_registry import TOOLS
from tools.search import search_web, run_queries, format_for_prompt, results_to_sources
from tools.memory import (
    store as memory_store,
    retrieve as memory_retrieve,
    format_for_context,
)
from services.questions import load_profile_qa as _load_profile_qa
from services.session import build_session_brief as _build_session_brief
from services.context_compressor import compress_if_needed as _compress_context
from services.skills import parse_skill_invocation as _parse_skill
from services.sub_agent import run_sub_agent as _run_sub_agent
from services.directives import extract_and_save as _extract_directives, load_directives as _load_directives
from services.intent_classifier import classify as _classify_intent

_log = logging.getLogger("phi.routes.chat")
router = APIRouter()

_VERSION_FILE = BASE_DIR / "data" / "last_modified.ts"


def _touch_version() -> None:
    """Mark that data changed so frontend knows to reload timeline/jobs."""
    import time
    try:
        _VERSION_FILE.write_text(str(time.time()), encoding="utf-8")
    except Exception:
        pass


# Prompts loaded once at import
_SYSTEM_PROMPT      = (BASE_DIR / "prompts" / "system.md").read_text(encoding="utf-8")
_PLAN_PROMPT        = (BASE_DIR / "prompts" / "orchestrator_plan.md").read_text(encoding="utf-8")
_CHAT_SYSTEM_PROMPT = (BASE_DIR / "prompts" / "chat_system.md").read_text(encoding="utf-8")

_settings_path = BASE_DIR / "config" / "settings.json"
try:
    with open(_settings_path, encoding="utf-8") as _f:
        _cfg = json.load(_f)
except Exception:
    _cfg = {}
_TEMPERATURE = float(_cfg.get("ollama", {}).get("temperature", 0.1))


# ── Models ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    messages: list[dict]
    search_first: bool = True


class TriageRequest(BaseModel):
    signal: str
    source: str = "manual"
    context_json: Optional[dict] = None


class DossierRequest(BaseModel):
    title: str
    type: str = "OPPORTUNITY"
    context: str = ""


class PlanRequest(BaseModel):
    context: Optional[dict] = None
    state: Optional[dict] = None
    intent: str = ""
    raw_input: Optional[str] = None
    has_private: bool = False


class ExecuteStepRequest(BaseModel):
    action: str
    web_query: Optional[str] = None
    workflow_input: Optional[dict] = None


class AskClaudeRequest(BaseModel):
    safe_abstract_spec: dict
    max_tokens: int = 2048


# ── Search query simplifier (auto-correction fallback) ────────────────────────

_STOP_WORDS = frozenset({
    "busca", "encuentra", "hay", "existe", "investiga", "dime", "explica",
    "cuál", "cuáles", "qué", "cómo", "puede", "puedes", "sobre", "acerca",
    "de", "la", "el", "los", "las", "un", "una", "me", "te", "se", "si",
    "search", "find", "check", "look", "tell", "show", "what", "how", "is",
    "are", "the", "a", "an", "for", "in", "on", "at", "to", "of",
})


def _simplify_query(text: str) -> str:
    """Strip stop words and filler; return first 8 content words as a clean query."""
    words = text.split()
    content = [w.strip("¿?.,;:!") for w in words if w.lower().strip("¿?.,;:!") not in _STOP_WORDS and len(w) > 2]
    simplified = " ".join(content[:8])
    return simplified if len(simplified) >= 10 else ""


# ── Intent action helpers (called from background task after each chat turn) ──

def _enqueue_investigate(entity_name: str) -> None:
    """Enqueue a DUE_DILIGENCE task for a specific entity named in chat."""
    import hashlib
    from tools.state_manager import DuplicateError, fingerprint_opportunity
    from api.context import workspace
    entity_id = f"ent_{hashlib.sha256(entity_name.lower().encode()).hexdigest()[:8]}"
    # Create stub dossier if it doesn't exist
    existing = workspace.load_dossier(entity_id)
    if not existing:
        stub = {
            "entity_id": entity_id, "schema_version": "1.0",
            "status": "DRAFT", "type": "ORG", "name": entity_name,
            "aliases": [], "description": f"Requested by David via chat: '{entity_name}'",
            "profile": {}, "fit_assessment": {"fit_score": 0, "why_yes": [], "why_not": []},
            "evidence_ids": [], "open_loops": [], "next_actions": [], "approval_status": "NONE",
        }
        workspace.save_dossier(stub)
    # Enqueue with high priority
    try:
        workspace.enqueue_task({
            "strategy": "DUE_DILIGENCE",
            "priority": 0,  # highest priority — chat-initiated
            "payload": {"entity_id": entity_id, "query_hint": entity_name, "source": "chat_request"},
        })
        _log.info("chat_investigate_enqueued entity=%r id=%s", entity_name[:60], entity_id)
    except DuplicateError:
        # Already in queue — bump priority by resetting to PENDING
        all_q = workspace.read_queue()
        for t in all_q:
            if (t.get("payload") or {}).get("entity_id") == entity_id:
                workspace.mark_task_status(t["task_id"], "PENDING")
                _log.info("chat_investigate_reset_to_pending entity=%r", entity_name[:60])
                break


def _cancel_queue_topic(topic: str) -> None:
    """Mark PENDING/IN_PROGRESS tasks matching a topic as CANCELLED."""
    from api.context import workspace
    topic_lower = topic.lower()
    all_q = workspace.read_queue()
    cancelled = 0
    for t in all_q:
        if t.get("status") not in ("PENDING", "IN_PROGRESS"):
            continue
        payload = t.get("payload") or {}
        hint = (payload.get("query_hint") or "").lower()
        entity_id = payload.get("entity_id", "")
        # Also check dossier name
        dossier_name = ""
        try:
            d = workspace.load_dossier(entity_id)
            dossier_name = (d.get("name") or "").lower() if d else ""
        except Exception:
            pass
        if topic_lower in hint or topic_lower in dossier_name or hint in topic_lower:
            workspace.mark_task_status(t["task_id"], "CANCELLED")
            cancelled += 1
    _log.info("chat_cancel_queue topic=%r cancelled=%d tasks", topic[:60], cancelled)


# ── Job feedback (script-based, called from _store background task) ───────────

import re as _re

_JOB_EXCLUDE_RE = _re.compile(
    r"(?:no me interesa|no quiero|excluye?|elimina?|filtra?|descarta?|skip|quita)\s+"
    r"(?:(?:ese|este|el|la|los|las)\s+)?(?:empleo|trabajo|rol|puesto|oferta|job)s?"
    r"(?:\s+de\s+|\s+en\s+|\s+como\s+|\s+de\s+tipo\s+)?([^\.,\?!]{3,50})?",
    _re.I,
)
_JOB_PREFER_RE = _re.compile(
    r"(?:busca|encuentra|quiero|prioriza)\s+"
    r"(?:empleos?|trabajos?|roles?|puestos?|ofertas?)\s+"
    r"(?:de\s+|en\s+|como\s+|de\s+tipo\s+)?([^\.,\?!]{3,50})",
    _re.I,
)

_JOB_FILTERS_FILE = BASE_DIR / "data" / "job_filters.json"


def _apply_job_feedback(user_msg: str, phi_msg: str) -> bool:
    """
    Detect job preference signals in conversation and update job_filters.json.
    Returns True if any change was made.
    """
    combined = f"{user_msg}\n{phi_msg}"
    changed = False
    try:
        filters = json.loads(_JOB_FILTERS_FILE.read_text(encoding="utf-8")) if _JOB_FILTERS_FILE.exists() else {}
    except Exception:
        filters = {}
    filters.setdefault("excluded_roles", [])
    filters.setdefault("preferred_types", [])

    for m in _JOB_EXCLUDE_RE.finditer(combined):
        val = (m.group(1) or "").strip().rstrip(".,;")
        if val and val not in filters["excluded_roles"]:
            filters["excluded_roles"].append(val)
            changed = True
            _log.info("job_feedback_exclude: %r", val)

    for m in _JOB_PREFER_RE.finditer(combined):
        val = (m.group(1) or "").strip().rstrip(".,;")
        if val and val not in filters["preferred_types"]:
            filters["preferred_types"].append(val)
            changed = True
            _log.info("job_feedback_prefer: %r", val)

    if changed:
        try:
            _JOB_FILTERS_FILE.parent.mkdir(parents=True, exist_ok=True)
            _JOB_FILTERS_FILE.write_text(json.dumps(filters, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            _log.warning("job_filters_save failed: %s", e)
    return changed


# ── Chat (streaming SSE) ───────────────────────────────────────────────────────

@router.post("/api/chat")
async def chat(req: ChatRequest):
    """Streaming chat. Uses CHAT_SYSTEM_PROMPT (plain-text). Returns SSE."""

    async def run():
        messages = [{"role": "system", "content": _CHAT_SYSTEM_PROMPT}] + req.messages
        last_msg = req.messages[-1].get("content", "") if req.messages else ""

        # ── Skill invocation (/dossier, /research, /status, …) ───────────────
        expanded, is_skill = _parse_skill(last_msg)
        if is_skill:
            # Skills that trigger sub-agent (research, dossier with no existing data)
            _SUB_AGENT_SKILLS = {"research"}
            skill_name = last_msg.strip().lstrip("/").split()[0].lower() if last_msg.startswith("/") else ""
            if skill_name in _SUB_AGENT_SKILLS:
                yield f"data: {json.dumps({'status': 'sub_agent', 'goal': expanded[:80]})}\n\n"
                result = await _run_sub_agent(expanded)
            else:
                # Non-research skills: expand prompt → run through normal tool loop
                messages[-1]["content"] = expanded
                last_msg = expanded
                # Fall through to normal flow below
                is_skill = False  # let normal flow handle it

            if is_skill:  # sub-agent was used — stream result directly
                chunk_size = 8
                for i in range(0, len(result), chunk_size):
                    yield f"data: {json.dumps({'content': result[i:i + chunk_size]})}\n\n"
                yield "data: [DONE]\n\n"
                return

        # ── Mail check — intercept early, no LLM needed ─────────────────────
        mail_intent = _classify_intent(last_msg)
        if mail_intent.type == "MAIL_CHECK":
            from services.mail_scanner import scan_inbox, format_for_chat
            from services.translator import translate_to_spanish, is_english
            yield f"data: {json.dumps({'status': 'scanning', 'action': 'MAIL_CHECK'})}\n\n"
            try:
                scan_result = await asyncio.to_thread(scan_inbox, 7, 60)
                formatted   = format_for_chat(scan_result)
            except Exception as _me:
                _log.warning("mail_scan_failed: %s", _me)
                formatted = "No se pudo acceder a Apple Mail. Asegúrate de que la app esté abierta."
            if is_english(formatted):
                try:
                    formatted = await asyncio.to_thread(translate_to_spanish, formatted)
                except Exception:
                    pass
            chunk_size = 8
            for i in range(0, len(formatted), chunk_size):
                yield f"data: {json.dumps({'content': formatted[i:i + chunk_size]})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # ── Strategic directives (first turn only) ───────────────────────────
        if len(req.messages) == 1:
            try:
                directives_text = _load_directives()
                if directives_text.strip():
                    messages.insert(1, {"role": "user", "content": f"[TUS DIRECTIVAS ESTRATÉGICAS ACTUALES]\n{directives_text[:1200]}"})
                    messages.insert(2, {"role": "assistant", "content": "Tengo claras mis directivas actuales. Investigo según este foco."})
            except Exception as _de:
                _log.warning("directives_load failed: %s", _de)

        # ── Session briefing (first turn only) ───────────────────────────────
        if len(req.messages) == 1:
            try:
                brief = _build_session_brief()
                if brief:
                    messages.insert(1, {"role": "user",      "content": brief})
                    messages.insert(2, {"role": "assistant", "content": "Entendido, tengo el estado actual del proyecto en mente."})
            except Exception as _be:
                _log.warning("session_brief failed: %s", _be)

        memories = memory_retrieve(last_msg, n=3)
        if memories:
            block = format_for_context(memories)
            messages.insert(1, {"role": "user",      "content": block})
            messages.insert(2, {"role": "assistant", "content": "Entendido, tengo ese contexto previo en cuenta."})

        profile_qa = _load_profile_qa(15)
        if profile_qa:
            qa_lines = "\n".join(f"- P: {e['question']}\n  R: {e['answer']}" for e in profile_qa)
            profile_block = f"[PERFIL ACTUALIZADO — respuestas recientes de David]\n{qa_lines}"
            messages.insert(1, {"role": "user",      "content": profile_block})
            messages.insert(2, {"role": "assistant", "content": "Tengo en cuenta lo que me has contado recientemente."})

        # ── Plan → Execute (phi4 only — Claude plans via tool selection) ────────
        from llm.claude_client import is_available as _claude_available
        if req.search_first and last_msg and not _claude_available():
            try:
                p = await _resolve_plan(last_msg)
                action = p.get("action", RESPOND_DIRECT)

                if action == "SEARCH_WEB":
                    queries = [q["query"] for q in p.get("web_queries", []) if q.get("query")]
                    if not queries:
                        queries = [last_msg]
                    yield f"data: {json.dumps({'status': 'planning', 'action': action})}\n\n"
                    results = run_queries(queries[:2], max_per_query=4)

                    # ── Auto-correction: retry with simplified query if too few results ──
                    if len(results) < 2:
                        simplified = _simplify_query(last_msg)
                        if simplified and simplified != last_msg:
                            _log.info("search_fallback: retrying with simplified query %r", simplified)
                            results = results + run_queries([simplified], max_per_query=3)
                            if results:
                                yield f"data: {json.dumps({'status': 'searching', 'count': len(results), 'retry': True})}\n\n"

                    if results:
                        search_block = format_for_prompt(results)
                        messages[-1]["content"] += (
                            f"\n\n[SEARCH RESULTS — cite source_id for any claim]:\n{search_block}"
                        )
                        if not (len(results) < 2):
                            yield f"data: {json.dumps({'status': 'searching', 'count': len(results)})}\n\n"

            except Exception as _pe:
                _log.warning("plan_resolve failed: %s", _pe)

        # ── Context compression (keeps window under ~2800 tokens) ────────────
        messages = await _compress_context(messages, call_phi)

        # ── Tool calling loop (search_web / read_dossier / save_task) ─────────
        tool_events: list = []

        async def _collect_event(event_json: str) -> None:
            tool_events.append(event_json)

        raw = await run_tool_loop(
            messages,
            tools=TOOLS,
            call_phi_fn=call_phi_with_tools,
            num_ctx=4096,
            yield_fn=_collect_event,
        )

        for ev in tool_events:
            yield f"data: {ev}\n\n"

        null_gate = GateResult(valid=True, score=100, failures=[], gate_scores={})
        write_audit("chat", "CHAT_RESPONSE", {}, raw, null_gate)

        async def _store():
            changed = False
            phi_raw = raw[:600] if isinstance(raw, str) else ""

            try:
                if isinstance(raw, str) and raw.strip():
                    await asyncio.to_thread(memory_store, last_msg[:500], raw[:500])
            except Exception as mem_err:
                _log.warning("memory_store failed: %s", mem_err)

            # ── Intent classification on user message (deterministic) ─────────
            try:
                intent = _classify_intent(last_msg)
                if intent.type == "INVESTIGATE_NOW" and intent.entity:
                    await asyncio.to_thread(_enqueue_investigate, intent.entity)
                    changed = True
                elif intent.type == "CANCEL_QUEUE" and intent.topic:
                    await asyncio.to_thread(_cancel_queue_topic, intent.topic)
                    changed = True
                elif intent.type == "DIRECTIVE_CHANGE":
                    directive_update = await asyncio.to_thread(_extract_directives, last_msg, phi_raw)
                    if directive_update:
                        _log.info("strategic_directive_saved: %s", directive_update)
                        changed = True
            except Exception as intent_err:
                _log.warning("intent_handler_user failed: %s", intent_err)

            # ── Intent classification on Phi's response (catches implicit signals) ─
            # Example: Phi says "Voy a investigar a Congruent Ventures" →
            # "He actualizado tus directivas" → these mirror the user's intent
            # so we run the same classifier on phi's text to catch stragglers.
            if phi_raw and not changed:
                try:
                    phi_intent = _classify_intent(phi_raw)
                    if phi_intent.type == "INVESTIGATE_NOW" and phi_intent.entity:
                        await asyncio.to_thread(_enqueue_investigate, phi_intent.entity)
                        changed = True
                    elif phi_intent.type == "CANCEL_QUEUE" and phi_intent.topic:
                        await asyncio.to_thread(_cancel_queue_topic, phi_intent.topic)
                        changed = True
                except Exception as phi_intent_err:
                    _log.warning("intent_handler_phi failed: %s", phi_intent_err)

            # ── Job feedback detection (script, no LLM) ────────────────────────
            # Patterns: "no me interesa ese empleo / rol", "filtra trabajos de X"
            try:
                changed |= await asyncio.to_thread(_apply_job_feedback, last_msg, phi_raw)
            except Exception as jf_err:
                _log.warning("job_feedback failed: %s", jf_err)

            # ── Touch version file so frontend knows to reload ─────────────────
            if changed:
                _touch_version()

        asyncio.create_task(_store())

        # Translate English response to Spanish before streaming.
        # Skip translation when user explicitly requested English output
        # (email drafts, messages to keep in English).
        _KEEP_EN_RE = _re.compile(
            r"(?:en inglés|in english|keep.*english|escribe.*english|draft.*english"
            r"|email.*english|correo.*english|redacta.*inglés|escribe.*inglés"
            r"|en inglés por favor|mantén.*inglés|keep it in english)",
            _re.I,
        )
        skip_translation = bool(_KEEP_EN_RE.search(last_msg))

        display_raw = raw
        if display_raw and isinstance(display_raw, str) and not skip_translation:
            from services.translator import translate_to_spanish, is_english
            if is_english(display_raw):
                try:
                    translated = await asyncio.to_thread(translate_to_spanish, display_raw)
                    if translated and len(translated) > 5:
                        display_raw = translated
                except Exception as _te:
                    pass  # fallback: stream original

        chunk_size = 8
        for i in range(0, len(display_raw), chunk_size):
            yield f"data: {json.dumps({'content': display_raw[i:i + chunk_size]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(run(), media_type="text/event-stream")


# ── Triage ─────────────────────────────────────────────────────────────────────

@router.post("/api/triage")
async def triage(req: TriageRequest):
    source_id  = f"src_{uuid.uuid4().hex[:8]}"
    context_obj = req.context_json or {
        "signal":    req.signal,
        "source":    req.source,
        "source_id": source_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    context_str = json.dumps(context_obj, ensure_ascii=False)

    prompt = f"""Return ONLY valid JSON matching TRIAGE_DECISION schema v1.0.

CONTEXT_JSON:
{context_str}

Triage rules:
- INVESTIGATE: topic overlaps investor/client/grant/partner profile AND has specific entity
- ASK: ambiguous AND search cannot resolve (max 2 blocking questions)
- DISCARD: B2C, consumer, software-only, Series A+, requires revenue, off-profile

Search before asking. Return JSON only."""

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    raw, gate, retries = await call_phi_with_retry(messages, "TRIAGE_DECISION", "triage", context_json=context_obj)

    return JSONResponse({
        "output":  json.loads(raw) if gate.valid else {"raw": raw, "error": "gates_failed"},
        "gate":    {"valid": gate.valid, "score": gate.score, "failures": gate.failures},
        "retries": retries,
    })


# ── Dossier ────────────────────────────────────────────────────────────────────

@router.post("/api/dossier")
async def dossier(req: DossierRequest):
    queries = [
        f"{req.title} 2026",
        f"{req.title} application requirements",
        f"{req.title} portfolio industrial",
    ]
    results        = run_queries(queries, max_per_query=4)
    search_context = format_for_prompt(results)
    sources        = results_to_sources(results)

    prompt = f"""Return ONLY valid JSON matching OPPORTUNITY_DOSSIER schema v1.0.

SEARCH RESULTS (Carril A — use these as sources, cite source_id for every claim):
{search_context}

INPUT:
{{
  "title": "{req.title}",
  "type": "{req.type}",
  "context": "{req.context}",
  "available_sources": {json.dumps(sources)}
}}

Rules:
- Every factual claim must reference a source_id from the search results above
- If a person's name is not confirmed in sources: "not available"
- fit_score = weighted: profile_match 40% + timing 30% + effort_vs_reward 20% + (100-risk) 10%
- why_not must be honest
- All drafts must pass Style Gate (no filler, CTA present, lead with credential/metric)
- Return JSON only."""

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    # 6144: search ctx (~2500 tokens) + system + prompt fits comfortably; saves ~40% vs 8192
    raw, gate, retries = await call_phi_with_retry(messages, "OPPORTUNITY_DOSSIER", "dossier", engine="ollama", num_ctx=6144)

    if gate.valid:
        dossier_path = BASE_DIR / "data" / "dossiers" / f"{uuid.uuid4().hex[:8]}.json"
        dossier_path.write_text(raw, encoding="utf-8")

    return JSONResponse({
        "output":  json.loads(raw) if gate.valid else {"raw": raw, "error": "gates_failed"},
        "gate":    {"valid": gate.valid, "score": gate.score, "failures": gate.failures},
        "retries": retries,
    })


# ── Plan (shared helper + HTTP endpoint) ──────────────────────────────────────

async def _resolve_plan(intent: str, raw_input: str = "", has_private: bool = False) -> dict:
    """
    Core planning logic shared by /api/plan and the chat handler.
    Returns a PLAN_JSON dict (action, web_queries, goal, …).
    Never raises — falls back to heuristic on any LLM error.
    """
    context_json = {
        "task":        intent,
        "raw_input":   raw_input or intent,
        "has_private": has_private,
    }
    user_content = (
        f"CONTEXT_JSON:\n{json.dumps(context_json, ensure_ascii=False)}\n\n"
        "CURRENT_STATE:\n{}\n\n"
        "Now produce PLAN_JSON only."
    )
    lm_messages = [
        {"role": "system", "content": _PLAN_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    plan_result: Optional[dict] = None
    lm_errors: list = []

    try:
        raw = await call_phi(lm_messages, num_ctx=4096)
        for fence in ("```json", "```"):
            if fence in raw:
                raw = raw.split(fence)[1].split("```")[0].strip()
                break
        if plan_is_json_only(raw):
            candidate = json.loads(raw)
            is_valid, lm_errors = validate_plan_json(candidate)
            if is_valid:
                plan_result = candidate
    except Exception as exc:
        lm_errors = [f"lm_exception:{type(exc).__name__}"]

    if plan_result is None:
        heuristic = decide_next_action({
            "intent":      intent,
            "raw_input":   raw_input or intent,
            "has_private": has_private,
        })
        mapped_action = (
            ASK_CLAUDE_FOR_INSTRUCTIONS
            if heuristic["action"] == ASK_CLAUDE_FOR_INSTRUCTIONS
            else heuristic["action"]
        )
        raw_spec = heuristic.get("safe_abstract_spec") or {}
        claude_spec: dict = {}
        if mapped_action == ASK_CLAUDE_FOR_INSTRUCTIONS and raw_spec:
            goal_tag = str(raw_spec.get("goal", "")).split(":")[0].strip() or "workflow_spec"
            claude_spec = {
                "goal":           goal_tag,
                "language":       raw_spec.get("language", "en"),
                "inputs_schema":  raw_spec.get("inputs_schema", {}),
                "outputs_schema": raw_spec.get("outputs_schema", {}),
                "constraints":    raw_spec.get("constraints", []),
            }
        web_queries = (
            [{"query": heuristic.get("web_query", ""), "why": "heuristic_router"}]
            if mapped_action == "SEARCH_WEB" and heuristic.get("web_query")
            else []
        )
        plan_result = {
            "status":            "IN_PROGRESS",
            "action":            mapped_action,
            "goal":              heuristic.get("reason", intent),
            "why":               [heuristic.get("reason", "heuristic_fallback")],
            "state_update":      {"hypotheses_delta": [], "evidence_add": [], "open_loops_add": [], "queue_add": []},
            "questions_to_user": [],
            "web_queries":       web_queries,
            "claude_spec":       claude_spec,
            "next_step":         heuristic.get("direct_answer_outline") or heuristic.get("web_query") or "",
            "_source":           "heuristic_fallback",
            "_lm_errors":        lm_errors,
        }

    return plan_result


@router.post("/api/plan")
async def plan(req: PlanRequest):
    """Return a PLAN_JSON routing plan. Falls back to heuristic on LLM failure."""
    intent   = req.intent or (req.context or {}).get("task", "") or ""
    raw_in   = req.raw_input or (req.context or {}).get("raw_input", "") or intent
    plan_result = await _resolve_plan(intent, raw_in, req.has_private)
    return JSONResponse(plan_result)


# ── Execute step ───────────────────────────────────────────────────────────────

@router.post("/api/execute/step")
async def execute_step(req: ExecuteStepRequest):
    action = req.action.upper()
    if action == "SEARCH_WEB":
        if not req.web_query:
            return JSONResponse({"error": "web_query required for SEARCH_WEB"}, status_code=400)
        results = search_web(req.web_query)
        return JSONResponse({
            "status":  "ok",
            "count":   len(results),
            "results": [{"title": r.title, "url": r.url, "snippet": r.snippet[:200]} for r in results],
        })
    if action == "RESPOND_DIRECT":
        return JSONResponse({"status": "ok", "message": "Use /api/chat for direct responses."})
    return JSONResponse({"error": f"Unsupported action: {action}"}, status_code=400)


# ── Ask Claude ─────────────────────────────────────────────────────────────────

@router.post("/api/ask_claude")
async def ask_claude_endpoint(req: AskClaudeRequest):
    """
    Forward a SAFE_ABSTRACT_SPEC to Claude API.
    Disabled unless CLAUDE_API_KEY env var is set.
    Payload must contain ONLY abstract technical structure — no PII.
    """
    import os
    if not os.environ.get("CLAUDE_API_KEY", "").strip():
        return JSONResponse(
            {"error": "Claude integration disabled. Set CLAUDE_API_KEY to enable."},
            status_code=503,
        )
    from tools.ask_claude import ask_claude, ClaudeDisabledError, ClaudeBlockedError, ClaudeAPIError
    try:
        response = ask_claude(req.safe_abstract_spec, max_tokens=req.max_tokens)
        return JSONResponse({"status": "ok", "response": response})
    except ClaudeDisabledError as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)
    except ClaudeBlockedError as exc:
        return JSONResponse(
            {"error": "BLOCKED by privacy gate", "reason": exc.reason, "pii_detected": exc.pii_detected},
            status_code=422,
        )
    except ClaudeAPIError as exc:
        return JSONResponse({"error": str(exc)}, status_code=502)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
