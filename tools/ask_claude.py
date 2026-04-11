"""
tools/ask_claude.py — Claude API client for PrivateClaw.

Always-on mode: Claude is enabled when CLAUDE_API_KEY is present.
If CLAUDE_API_KEY is absent, ask_claude() degrades gracefully to a
RESPOND_DIRECT plan — no exception raised, no network call attempted.

Every outbound payload is screened by privacy_pre_hook("ASK_CLAUDE") first.
Audit entry written for every call (ALLOW / BLOCK); raw payload never logged.

Configuration (env vars):
    CLAUDE_API_KEY    — if absent, degrades to RESPOND_DIRECT (no error)
    CLAUDE_MODEL      — default: claude-3-7-sonnet-latest
    CLAUDE_BASE_URL   — default: https://api.anthropic.com/v1/messages
    CLAUDE_TIMEOUT    — seconds; default: 60

Goal field allowlist (spec["goal"] must start with one of):
    automation_spec   — request for Python automation script specification
    workflow_spec     — request for workflow architecture / instructions
    debug_fix         — request for debugging help / fix

Usage:
    from tools.ask_claude import ask_claude, ClaudeBlockedError
    reply = ask_claude(safe_abstract_spec)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.privacy_pre_hook import privacy_pre_hook
from core.audit_append import append_audit, _payload_hash


# ── Model selection ───────────────────────────────────────────────────────────
# Pinned to latest Claude generation. Update when a newer model ships.
_DEFAULT_MODEL = "claude-3-7-sonnet-latest"

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


# ── Exceptions ────────────────────────────────────────────────────────────────

class ClaudeDisabledError(RuntimeError):
    """CLAUDE_API_KEY is not set. Claude client is disabled."""


class ClaudeBlockedError(RuntimeError):
    """privacy_pre_hook returned BLOCK. Payload contains PII; not sent."""
    def __init__(self, reason: str, pii_detected: list):
        super().__init__(reason)
        self.reason = reason
        self.pii_detected = pii_detected


class ClaudeAPIError(RuntimeError):
    """Non-recoverable error from the Claude API."""


# ── Safe abstract spec validation ─────────────────────────────────────────────

_REQUIRED_SPEC_FIELDS = ("goal", "inputs_schema", "outputs_schema", "constraints")

# Only these goal prefixes are accepted — prevents free-form private text
# from being smuggled in via the goal field.
_GOAL_ALLOWLIST = ("automation_spec", "workflow_spec", "debug_fix")

# Maximum characters allowed in any single string value (doc-blob guard).
_MAX_STRING_VALUE_LEN = 800

# Non-Latin Unicode blocks that signal non-English content.
import unicodedata as _ud
_NON_LATIN_CATEGORIES = {"Lo", "So"}  # Letter-other (CJK/Arabic/…), Symbol-other


def _has_non_latin_content(text: str) -> bool:
    """Return True if >5 % of chars are non-Latin-script characters."""
    if not text:
        return False
    non_latin = sum(
        1 for ch in text if _ud.category(ch) in _NON_LATIN_CATEGORIES
    )
    return (non_latin / len(text)) > 0.05


def _validate_spec(spec: dict) -> None:
    """
    Raise ValueError if spec:
      - is missing a required key
      - has a goal that doesn't start with an allowlisted prefix
      - has a 'language' field that isn't 'en'
      - contains any string value longer than _MAX_STRING_VALUE_LEN (doc-blob guard)
      - contains non-Latin-script characters (non-English signal)
      - has a top-level key that looks like private data
    """
    for field in _REQUIRED_SPEC_FIELDS:
        if field not in spec:
            raise ValueError(f"safe_abstract_spec missing required field: '{field}'")

    # Goal must start with an allowed prefix
    goal: str = str(spec.get("goal", ""))
    goal_prefix = goal.split(":")[0].strip().lower()
    if goal_prefix not in _GOAL_ALLOWLIST:
        raise ValueError(
            f"safe_abstract_spec 'goal' must start with one of "
            f"{_GOAL_ALLOWLIST}. Got prefix: '{goal_prefix}'"
        )

    # Language field, if present, must be English
    lang = spec.get("language", "en")
    if str(lang).lower() not in ("en", "english"):
        raise ValueError(
            f"safe_abstract_spec must be in ENGLISH. Got language='{lang}'."
        )

    # Doc-blob guard: reject any string value exceeding the length cap
    import json as _json
    for value in _flatten_strings(spec):
        if len(value) > _MAX_STRING_VALUE_LEN:
            raise ValueError(
                f"safe_abstract_spec contains a string value longer than "
                f"{_MAX_STRING_VALUE_LEN} chars (doc-blob guard). "
                "Summarise or truncate the content."
            )
        if _has_non_latin_content(value):
            raise ValueError(
                "safe_abstract_spec contains non-Latin characters. "
                "Spec must be in ENGLISH only."
            )

    # Reject specs with private-looking top-level keys
    _FORBIDDEN_KEYS = {
        "email", "phone", "name", "address", "contract",
        "client", "password", "secret", "token", "api_key",
    }
    for key in spec:
        if key.lower() in _FORBIDDEN_KEYS:
            raise ValueError(
                f"safe_abstract_spec contains forbidden key '{key}'. "
                "Use abstract field names only (e.g. 'inputs_schema')."
            )


def _flatten_strings(obj, _depth: int = 0) -> list:
    """Recursively collect all string values from a nested dict/list."""
    if _depth > 6:
        return []
    results = []
    if isinstance(obj, str):
        results.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            results.extend(_flatten_strings(v, _depth + 1))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            results.extend(_flatten_strings(item, _depth + 1))
    return results


# ── Core client ───────────────────────────────────────────────────────────────

def ask_claude(
    safe_abstract_spec: dict,
    system_prompt: Optional[str] = None,
    max_tokens: int = 2048,
) -> dict:
    """
    Send a SAFE_ABSTRACT_SPEC to Claude and return a structured response.

    Pipeline:
      1. Verify CLAUDE_API_KEY is set (else raise ClaudeDisabledError).
      2. Validate spec structure (English-only, no private keys).
      3. Run privacy_pre_hook("ASK_CLAUDE") on the full payload.
         - BLOCK  → raise ClaudeBlockedError (no network call).
         - ALLOW / REDACT_AND_PROCEED → proceed with sanitised payload.
      4. POST to Claude API.
      5. Write audit entry (hash only; no raw payload).
      6. Return parsed response dict.

    Parameters
    ----------
    safe_abstract_spec : dict
        Must contain: goal, inputs_schema, outputs_schema, constraints.
        Must contain ONLY technical structure — no real names, emails, etc.
    system_prompt : str, optional
        Override the default system prompt. Must also be PII-free.
    max_tokens : int
        Maximum tokens for Claude's response. Default: 2048.

    Returns
    -------
    dict with keys:
        "content"  : str  — Claude's response text
        "model"    : str  — model used
        "decision" : str  — gate decision (ALLOW / REDACT_AND_PROCEED)
    """
    api_key = os.environ.get("CLAUDE_API_KEY", "").strip()
    if not api_key:
        # Graceful degrade: return a RESPOND_DIRECT plan so the caller can
        # answer locally without the caller needing to handle an exception.
        return {
            "content": (
                "Claude API is not configured (CLAUDE_API_KEY missing). "
                "Please answer this request using local reasoning and best practices. "
                "Goal: " + str(safe_abstract_spec.get("goal", ""))
            ),
            "model": "local-fallback",
            "decision": "DEGRADED",
            "action": "RESPOND_DIRECT",
        }

    model = os.environ.get("CLAUDE_MODEL", _DEFAULT_MODEL).strip()
    base_url = os.environ.get("CLAUDE_BASE_URL", _ANTHROPIC_API_URL).strip()
    timeout = float(os.environ.get("CLAUDE_TIMEOUT", "60"))

    # Step 2: validate spec structure
    _validate_spec(safe_abstract_spec)

    # Step 3: privacy gate
    payload_for_gate = {
        "message": json.dumps(safe_abstract_spec, ensure_ascii=False),
        "model": model,
        "action": "ASK_CLAUDE",
    }
    hook = privacy_pre_hook(
        action_type="ASK_CLAUDE",
        payload=payload_for_gate,
        policy={"on_pii_remote": "block"},   # strict: block on any PII
    )

    if hook.decision == "BLOCK":
        raise ClaudeBlockedError(
            reason=hook.reason,
            pii_detected=hook.pii_detected,
        )

    # Use redacted payload if pre-hook produced one
    if hook.decision == "REDACT_AND_PROCEED" and hook.redacted_payload:
        outbound_message = hook.redacted_payload.get("message", payload_for_gate["message"])
    else:
        outbound_message = payload_for_gate["message"]

    # Step 4: build API request
    _default_system = (
        "You are a technical architecture assistant. "
        "Respond ONLY in ENGLISH. "
        "Output structured plans, workflow JSON, or instructions as requested. "
        "Never invent private data; use placeholders (PERSON_A, COMPANY_X, etc.)."
    )
    system = system_prompt or _default_system

    request_body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [
            {"role": "user", "content": outbound_message},
        ],
    }

    ph = _payload_hash(request_body)

    try:
        resp = httpx.post(
            base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": _ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            json=request_body,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        _write_claude_audit(
            model=model,
            decision=hook.decision,
            payload_hash=ph,
            pii_detected=hook.pii_detected,
            outcome="API_ERROR",
            error=str(exc),
        )
        raise ClaudeAPIError(f"Claude API returned {exc.response.status_code}") from exc
    except Exception as exc:
        _write_claude_audit(
            model=model,
            decision=hook.decision,
            payload_hash=ph,
            pii_detected=hook.pii_detected,
            outcome="NETWORK_ERROR",
            error=type(exc).__name__,
        )
        raise

    content_blocks = data.get("content", [])
    response_text = " ".join(
        b.get("text", "") for b in content_blocks if b.get("type") == "text"
    ).strip()

    _write_claude_audit(
        model=data.get("model", model),
        decision=hook.decision,
        payload_hash=ph,
        pii_detected=hook.pii_detected,
        outcome="SUCCESS",
    )

    return {
        "content": response_text,
        "model": data.get("model", model),
        "decision": hook.decision,
    }


# ── Audit helper ──────────────────────────────────────────────────────────────

def _write_claude_audit(
    model: str,
    decision: str,
    payload_hash: str,
    pii_detected: list,
    outcome: str,
    error: str = "",
) -> None:
    """Write an audit entry for a Claude call. Never includes raw payload."""
    try:
        append_audit(
            action_type="ASK_CLAUDE",
            decision=decision,
            pii_detected=pii_detected,
            redaction_notes=[],
            payload_hash=payload_hash,
            reason=f"Claude call outcome={outcome}" + (f" error={error}" if error else ""),
            extra={"model": model, "outcome": outcome},
        )
    except Exception:
        pass  # audit must never crash the caller
