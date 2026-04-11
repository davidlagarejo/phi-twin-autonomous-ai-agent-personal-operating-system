"""
llm/pipeline.py — LLM call pipeline: gate validation, audit writing, retry logic.

Single responsibility: wrap raw LLM calls with schema validation, structured
audit logging, and a repair-first retry policy.

Surfaces:
  write_audit()          — append one entry to the triage/dossier audit log.
  call_phi_mlx()         — async wrapper for synchronous MLX inference.
  call_phi_with_retry()  — main retry loop: call → validate → repair → re-ask.

Config loaded once at import time (no per-call disk reads).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Resolve phi-twin/ root and ensure web/ is importable for mlx_runner ────────
_BASE = Path(__file__).parent.parent
_WEB_DIR = _BASE / "web"
if str(_WEB_DIR) not in sys.path:
    sys.path.insert(0, str(_WEB_DIR))

from core.gates import validate, GateResult  # noqa: E402
from llm.client import call_phi              # noqa: E402

_log = logging.getLogger("phi.pipeline")

# ── Config loaded once ────────────────────────────────────────────────────────
_settings_path = _BASE / "config" / "settings.json"
try:
    with open(_settings_path, encoding="utf-8") as _f:
        _cfg = json.load(_f)
except Exception as _e:
    _log.error("pipeline: failed to load settings: %s", _e)
    _cfg = {}

_OLLAMA_MODEL = _cfg.get("ollama", {}).get("model", "phi4:14b")
_TEMPERATURE  = float(_cfg.get("ollama", {}).get("temperature", 0.1))
_AUDIT_LOG    = Path(_cfg.get("data", {}).get("audit_log",
                     str(_BASE / "data" / "audit_logs" / "audit.jsonl")))
_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

_SYSTEM_PROMPT_PATH = _BASE / "prompts" / "system.md"
try:
    _SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
except Exception:
    _SYSTEM_PROMPT = ""
    _log.warning("pipeline: could not load system.md from %s", _SYSTEM_PROMPT_PATH)

# ── MLX thread pool (single-threaded: MLX is not re-entrant) ──────────────────
_mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")

try:
    from mlx_runner import call_mlx_sync as _call_mlx_sync
    _MLX_AVAILABLE = True
except ImportError:
    _MLX_AVAILABLE = False

    def _call_mlx_sync(messages, max_tokens=512):  # type: ignore[misc]
        return ""


# ── Audit writer ──────────────────────────────────────────────────────────────

def write_audit(
    flow: str,
    schema: str,
    input_data: dict,
    output: str,
    gate: GateResult,
    retries: int = 0,
    model: str | None = None,
) -> None:
    entry = {
        "log_id":        str(uuid.uuid4()),
        "timestamp":     datetime.now(timezone.utc).isoformat(),
        "flow":          flow,
        "schema":        schema,
        "output_preview": output[:200],
        "gates_result":  gate.gate_scores,
        "gates_passed":  gate.valid,
        "score":         gate.score,
        "failures":      gate.failures,
        "retries":       retries,
        "model":         model or _OLLAMA_MODEL,
        "input_keys":    list(input_data.keys()),
    }
    try:
        with open(_AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        _log.warning("write_audit failed: %s", exc)


# ── Async MLX wrapper ─────────────────────────────────────────────────────────

async def call_phi_mlx(messages: list, max_tokens: int = 512) -> str:
    """Run synchronous MLX inference in the dedicated single-worker thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_mlx_executor, _call_mlx_sync, messages, max_tokens)


# ── Repair-first retry helpers ────────────────────────────────────────────────

REPAIR_PROMPT = (
    "REPAIR the JSON below to match the required schema exactly. "
    "Output ONLY the corrected JSON. No explanations. No markdown. Preserve all existing meaning. "
    "Apply these defaults for any missing required fields: "
    "draft_message.channel = 'email' (unless context implies linkedin_dm/whatsapp/sms); "
    "any required string that is unknown = empty string \"\"; "
    "any required array that is missing = []. "
    "Do NOT change decision, score, yes, no, or next_actions unless they violate the schema.\n\n"
    "BROKEN JSON:\n"
)

_MINIMAL_EXTRACT_PROMPT = (
    "Extract ONLY the minimum valid JSON object from the text below. "
    "Output a single JSON object with no prose, no markdown, no fences. "
    "If no JSON is present, output: {}\n\nTEXT:\n"
)

_PLACEHOLDER_RE = re.compile(
    r"(\[|\{|<)(FIRST_NAME|LAST_NAME|FULL_NAME|NAME)(\]|\}|>)",
    re.IGNORECASE,
)


def _needs_repair(gate: GateResult, raw: str) -> bool:
    """True when failures are structural/schema issues repairable without re-asking."""
    if not gate.failures:
        return False
    return (
        any(f.startswith(("INVALID_JSON", "missing_field", "bad_channel")) for f in gate.failures)
        or "channel" in json.dumps(gate.failures).lower()
        or not _is_parseable(raw)
    )


def _is_parseable(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def _strip_placeholders(text: str) -> str:
    cleaned = _PLACEHOLDER_RE.sub("", text)
    cleaned = re.sub(r"\b(Hi|Hello|Dear|Hola|Estimado)\s*[,.]", r"\1 there,", cleaned)
    return cleaned.strip()


def _normalize_draft_message(raw: str, context_json: dict | None = None) -> str:
    """Coerce flat-string draft_message → {channel,subject,body}, strip placeholders."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if not isinstance(obj, dict):
        return raw

    dm = obj.get("draft_message")
    if isinstance(dm, str):
        dm = {"channel": "email", "subject": "", "body": dm}

    if not isinstance(dm, dict):
        return raw

    if "body" in dm:
        dm["body"] = _strip_placeholders(dm["body"])
    if "subject" in dm:
        dm["subject"] = _strip_placeholders(dm["subject"])

    if dm.get("channel", "email") == "email" and not dm.get("subject"):
        ctx = context_json or {}
        title   = ctx.get("title") or ctx.get("opportunity") or ctx.get("signal", "")[:60]
        company = ctx.get("company") or ctx.get("org") or ctx.get("source", "")
        if title and company:
            dm["subject"] = f"{title} — {company}"
        elif title:
            dm["subject"] = str(title)
        elif company:
            dm["subject"] = f"Question re: {company}"
        else:
            dm["subject"] = "Quick question"

    obj["draft_message"] = dm
    return json.dumps(obj, ensure_ascii=False)


# ── Main retry loop ───────────────────────────────────────────────────────────

async def call_phi_with_retry(
    messages: list,
    schema: str,
    flow: str,
    max_retries: int = 3,
    engine: str = "mlx",
    context_json: dict | None = None,
    num_ctx: int | None = None,
) -> tuple[str, GateResult, int]:
    """Call the model with repair-first retry policy, then full re-ask.

    engine="mlx"    — 600-iter LoRA adapter (default for triage)
    engine="ollama" — Ollama phi4:14b (dossier, chat, repair passes)
    num_ctx         — override context window (default: llm/client.py default 8192)
                      Pass 4096 for structured JSON tasks, 6144 for dossier w/ search ctx.

    Returns (raw_output, gate_result, attempt_count).
    """
    temp = _TEMPERATURE
    raw  = ""
    gate = GateResult(valid=False, score=0, failures=["not_started"])
    model_label = "mlx" if engine == "mlx" else _OLLAMA_MODEL

    for attempt in range(max_retries):
        raw = await call_phi_mlx(messages) if engine == "mlx" else await call_phi(messages, temperature=temp, num_ctx=num_ctx)

        # Strip markdown fences
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        raw  = _normalize_draft_message(raw, context_json)
        gate = validate({"output": raw})

        if gate.valid:
            write_audit(flow, schema, {}, raw, gate, retries=attempt, model=model_label)
            return raw, gate, attempt

        # ── Repair pass (attempt 0 only, structural issues) ─────────────────
        if attempt == 0 and _needs_repair(gate, raw):
            repair_msgs = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": REPAIR_PROMPT + raw},
            ]
            repaired = await call_phi(repair_msgs, temperature=0.05, num_ctx=4096)
            for fence in ("```json", "```"):
                if fence in repaired:
                    repaired = repaired.split(fence)[1].split("```")[0].strip()
                    break
            repaired = _normalize_draft_message(repaired, context_json)
            repair_gate = validate({"output": repaired})
            if repair_gate.valid:
                write_audit(flow, schema, {}, repaired, repair_gate, retries=attempt + 0.5, model=model_label)
                return repaired, repair_gate, attempt
            raw  = repaired
            gate = repair_gate

        # ── Minimal extraction pass (last attempt only) ──────────────────────
        if attempt == max_retries - 2 and not _is_parseable(raw):
            extract_msgs = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _MINIMAL_EXTRACT_PROMPT + raw},
            ]
            extracted = await call_phi(extract_msgs, temperature=0.0, num_ctx=2048)
            for fence in ("```json", "```"):
                if fence in extracted:
                    extracted = extracted.split(fence)[1].split("```")[0].strip()
                    break
            extracted = _normalize_draft_message(extracted, context_json)
            extract_gate = validate({"output": extracted})
            if extract_gate.valid:
                write_audit(flow, schema, {}, extracted, extract_gate, retries=attempt + 0.7, model=model_label)
                return extracted, extract_gate, attempt
            raw  = extracted
            gate = extract_gate

        # ── Full re-ask with gate feedback ────────────────────────────────────
        temp = max(0.05, temp - 0.03)
        feedback = (
            f"\n\nPREVIOUS ATTEMPT FAILED QUALITY GATES: {gate.failures}"
            "\nFix these issues and return valid JSON only."
        )
        messages = messages[:-1] + [{"role": "user", "content": messages[-1]["content"] + feedback}]

    write_audit(flow, schema, {}, raw, gate, retries=max_retries, model=model_label)
    return raw, gate, max_retries
