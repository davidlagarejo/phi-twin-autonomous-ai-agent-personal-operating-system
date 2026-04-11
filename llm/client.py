"""
llm/client.py — Unified LLM client. Single definition for the entire project.

Single responsibility: call the local Ollama instance and return the response string.

Design:
  - Config read once at module import from config/settings.json — no per-call disk reads.
  - Two surfaces:
      call_phi()       async — for FastAPI route handlers (non-blocking event loop).
      call_phi_sync()  sync  — for ThreadPoolExecutor workers (research, drafts, questions).
  - Both log duration and status. Errors are caught and return "" (non-fatal by design —
    callers check for empty string and handle gracefully).
  - One place to change: model, timeout, base_url, temperature.
  - No prompt building. No output parsing. No domain logic. Just the HTTP call.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import httpx

_log = logging.getLogger(__name__)

# ── Config loaded once at import time ─────────────────────────────────────────
# Path is relative to this file: phi-twin/llm/client.py → phi-twin/config/settings.json
_SETTINGS_PATH = Path(__file__).parent.parent / "config" / "settings.json"

try:
    with open(_SETTINGS_PATH, encoding="utf-8") as _f:
        _cfg = json.load(_f)["ollama"]
except Exception as _e:
    _log.error("llm/client.py: failed to load settings from %s: %s", _SETTINGS_PATH, _e)
    _cfg = {
        "base_url": "http://localhost:11434",
        "model": "phi4:14b",
        "temperature": 0.1,
        "context_window": 8192,
    }

_BASE_URL: str = _cfg["base_url"].rstrip("/")
_MODEL: str = _cfg["model"]
_TEMPERATURE: float = float(_cfg.get("temperature", 0.1))
_DEFAULT_CTX: int = int(_cfg.get("context_window", 8192))

_CHAT_ENDPOINT = f"{_BASE_URL}/api/chat"


# ── Shared request builder ────────────────────────────────────────────────────

def _build_payload(
    messages: list[dict],
    num_ctx: Optional[int],
    temperature: Optional[float],
    tools: Optional[list] = None,
) -> dict:
    payload: dict = {
        "model": _MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature if temperature is not None else _TEMPERATURE,
            "num_ctx": num_ctx if num_ctx is not None else _DEFAULT_CTX,
        },
    }
    if tools:
        payload["tools"] = tools
    return payload


def _extract_content(response_json: dict) -> str:
    return response_json.get("message", {}).get("content", "")


def _extract_tool_response(response_json: dict) -> dict:
    """
    Return {"type": "text", "content": str} or {"type": "tool_calls", "calls": list}.
    """
    msg = response_json.get("message", {})
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        return {"type": "tool_calls", "calls": tool_calls, "content": msg.get("content", "")}
    return {"type": "text", "content": msg.get("content", "")}


# ── Async surface (FastAPI route handlers) ────────────────────────────────────

async def call_phi(
    messages: list[dict],
    num_ctx: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout: float = 300.0,
) -> str:
    """
    Async LLM call. Use in FastAPI route handlers and async contexts.
    Returns "" on any error — callers must handle empty string.
    """
    payload = _build_payload(messages, num_ctx, temperature)
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(_CHAT_ENDPOINT, json=payload)
            r.raise_for_status()
            content = _extract_content(r.json())
            _log.debug("call_phi ok model=%s ctx=%s msgs=%d chars=%d elapsed=%.1fs",
                       _MODEL, payload["options"]["num_ctx"],
                       len(messages), len(content), time.monotonic() - t0)
            return content
    except Exception as exc:
        _log.error("call_phi failed model=%s elapsed=%.1fs err=%s",
                   _MODEL, time.monotonic() - t0, exc)
        return ""


# ── Tool-calling surface (returns structured response) ────────────────────────

async def call_phi_with_tools(
    messages: list[dict],
    tools: list,
    num_ctx: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout: float = 300.0,
) -> dict:
    """
    Async LLM call with tool schemas. Returns a structured dict:
      {"type": "text",       "content": str}          — plain text response
      {"type": "tool_calls", "calls": list, ...}       — tool call request

    On error returns {"type": "text", "content": ""}.
    """
    payload = _build_payload(messages, num_ctx, temperature, tools=tools)
    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(_CHAT_ENDPOINT, json=payload)
            r.raise_for_status()
            result = _extract_tool_response(r.json())
            _log.debug(
                "call_phi_with_tools ok type=%s model=%s elapsed=%.1fs",
                result["type"], _MODEL, time.monotonic() - t0,
            )
            return result
    except Exception as exc:
        _log.error("call_phi_with_tools failed model=%s elapsed=%.1fs err=%s",
                   _MODEL, time.monotonic() - t0, exc)
        return {"type": "text", "content": ""}


# ── Sync surface (ThreadPoolExecutor workers) ─────────────────────────────────

def call_phi_sync(
    messages: list[dict],
    num_ctx: Optional[int] = None,
    temperature: Optional[float] = None,
    timeout: float = 180.0,
) -> str:
    """
    Synchronous LLM call. Use inside thread pool executors only.
    Returns "" on any error — callers must handle empty string.
    """
    payload = _build_payload(messages, num_ctx, temperature)
    t0 = time.monotonic()
    try:
        r = httpx.post(_CHAT_ENDPOINT, json=payload, timeout=timeout)
        r.raise_for_status()
        content = _extract_content(r.json())
        _log.debug("call_phi_sync ok model=%s ctx=%s msgs=%d chars=%d elapsed=%.1fs",
                   _MODEL, payload["options"]["num_ctx"],
                   len(messages), len(content), time.monotonic() - t0)
        return content
    except Exception as exc:
        _log.error("call_phi_sync failed model=%s elapsed=%.1fs err=%s",
                   _MODEL, time.monotonic() - t0, exc)
        return ""
