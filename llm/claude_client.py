"""
llm/claude_client.py — Claude API client with native tool calling.

Hybrid architecture:
  phi4 (local) — private data: dossiers, tasks, memory, email drafts.
  Claude API   — reasoning: planning, tool selection, response synthesis.

Privacy contract (enforced here, not delegated to ask_claude.py):
  ALLOWED outbound: user messages, web search results, dossier company info
                    (public company names, fit scores, descriptions, next actions).
  BLOCKED outbound: email addresses, phone numbers, personal contact details,
                    long-term memory entries, profile Q&A, email drafts.

Tool schema conversion: Ollama format → Anthropic format (done automatically).

Degrades gracefully: if CLAUDE_API_KEY is absent, all functions return None
and callers fall back to phi4.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

import httpx

_log = logging.getLogger("phi.llm.claude_client")

_ANTHROPIC_URL     = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MODEL     = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
_TIMEOUT           = float(os.environ.get("CLAUDE_TIMEOUT", "60"))

# Simple PII patterns — blocks outbound if found in any string value
_PII_RE = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"  # email
    r"|\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",  # phone
    re.IGNORECASE,
)


def is_available() -> bool:
    """Return True if CLAUDE_API_KEY is configured."""
    return bool(os.environ.get("CLAUDE_API_KEY", "").strip())


# ── Schema conversion: Ollama → Anthropic ─────────────────────────────────────

def _to_anthropic_tools(ollama_tools: list) -> list:
    """Convert Ollama/OpenAI tool format to Anthropic tool format."""
    result = []
    for t in ollama_tools:
        fn = t.get("function", {})
        result.append({
            "name":         fn.get("name", ""),
            "description":  fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


# ── Privacy filter ────────────────────────────────────────────────────────────

def _contains_pii(text: str) -> bool:
    return bool(_PII_RE.search(text))


def _strip_injected_context(messages: list) -> list:
    """
    Remove injected context blocks (session briefing, memories, profile Q&A)
    that start with '[' — these contain private/internal data.
    Keep only genuine user/assistant conversation turns and tool results.
    """
    clean = []
    i = 0
    while i < len(messages):
        m = messages[i]
        role    = m.get("role", "")
        content = m.get("content") or ""

        # Skip system messages (handled separately as system prompt)
        if role == "system":
            i += 1
            continue

        # Skip injected context pairs: user msg starting with '[' + following assistant ack
        if role == "user" and isinstance(content, str) and content.startswith("["):
            # Skip this message and its assistant acknowledgment
            if i + 1 < len(messages) and messages[i + 1].get("role") == "assistant":
                i += 2
                continue
            i += 1
            continue

        # Skip if content contains PII
        if isinstance(content, str) and _contains_pii(content):
            _log.warning("claude_client: skipping message with PII (role=%s)", role)
            i += 1
            continue

        clean.append(m)
        i += 1
    return clean


# ── Core async client ─────────────────────────────────────────────────────────

async def call_claude_with_tools(
    messages: list,
    tools: list,
    system: str = "",
    num_ctx: int = 4096,   # ignored — kept for API compatibility with call_phi_with_tools
) -> dict:
    """
    Call Claude API with tool schemas. Returns same format as call_phi_with_tools:
      {"type": "text",       "content": str}
      {"type": "tool_calls", "calls": list, "call_ids": list}

    Returns {"type": "text", "content": ""} on any error or if API key missing.
    """
    api_key = os.environ.get("CLAUDE_API_KEY", "").strip()
    if not api_key:
        return {"type": "text", "content": ""}

    anthropic_tools = _to_anthropic_tools(tools)
    clean_messages  = _strip_injected_context(messages)

    if not clean_messages:
        return {"type": "text", "content": ""}

    # Convert Ollama tool_calls in messages to Anthropic tool_use/tool_result format
    anthropic_messages = _convert_messages_to_anthropic(clean_messages)

    request_body = {
        "model":      _DEFAULT_MODEL,
        "max_tokens": 1024,
        "system":     system or _DEFAULT_SYSTEM,
        "messages":   anthropic_messages,
        "tools":      anthropic_tools,
    }

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(
                _ANTHROPIC_URL,
                headers={
                    "x-api-key":          api_key,
                    "anthropic-version":  _ANTHROPIC_VERSION,
                    "content-type":       "application/json",
                },
                json=request_body,
            )
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        _log.error("claude_client: API call failed: %s", exc)
        return {"type": "text", "content": ""}

    stop_reason = data.get("stop_reason", "")
    content_blocks = data.get("content", [])

    if stop_reason == "tool_use":
        calls = []
        call_ids = []
        for block in content_blocks:
            if block.get("type") == "tool_use":
                calls.append({
                    "function": {
                        "name":      block["name"],
                        "arguments": block.get("input", {}),
                    }
                })
                call_ids.append(block.get("id", ""))
        return {"type": "tool_calls", "calls": calls, "call_ids": call_ids, "raw_blocks": content_blocks}

    text = " ".join(
        b.get("text", "") for b in content_blocks if b.get("type") == "text"
    ).strip()
    return {"type": "text", "content": text}


def call_claude_extract_sync(prompt: str, system: str = "Extract information. JSON only. No markdown.") -> str:
    """
    Synchronous Claude call for JSON extraction tasks (e.g. person discovery in research_engine).
    Uses httpx sync client — safe to call from synchronous code.
    Returns raw text response, or empty string on failure/unavailable.
    Only sends public web search results — no private data.
    """
    api_key = os.environ.get("CLAUDE_API_KEY", "").strip()
    if not api_key:
        return ""
    request_body = {
        "model":      _DEFAULT_MODEL,
        "max_tokens": 512,
        "system":     system,
        "messages":   [{"role": "user", "content": prompt}],
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            r = client.post(
                _ANTHROPIC_URL,
                headers={
                    "x-api-key":         api_key,
                    "anthropic-version": _ANTHROPIC_VERSION,
                    "content-type":      "application/json",
                },
                json=request_body,
            )
            r.raise_for_status()
            data = r.json()
        blocks = data.get("content", [])
        return " ".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
    except Exception as exc:
        _log.warning("call_claude_extract_sync failed: %s", exc)
        return ""


_DEFAULT_SYSTEM = (
    "You are Phi, a personal AI assistant for David Lagarejo — physicist-engineer, IIoT/cleantech CEO in NYC. "
    "David evaluates opportunities (grants, investors, partners, jobs) with his startup focused on "
    "ultrasonic sensor technology for industrial sustainability. "
    "Be direct and actionable. Answer in Spanish unless the user writes in English. "
    "Use tools proactively when you need current information. "
    "For decisions and analysis, draw on the search results you retrieve."
)


# ── Message format conversion ─────────────────────────────────────────────────

def _convert_messages_to_anthropic(messages: list) -> list:
    """
    Convert messages (Ollama format) to Anthropic format.
    Handles tool_calls in assistant messages and tool results.
    """
    result = []
    for m in messages:
        role    = m.get("role", "")
        content = m.get("content") or ""

        if role == "assistant":
            tool_calls = m.get("tool_calls")
            if tool_calls:
                # Assistant message with tool calls → content array with tool_use blocks
                blocks = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    blocks.append({
                        "type":  "tool_use",
                        "id":    tc.get("id", f"tool_{fn.get('name','')}"),
                        "name":  fn.get("name", ""),
                        "input": args,
                    })
                result.append({"role": "assistant", "content": blocks})
            else:
                result.append({"role": "assistant", "content": content})

        elif role == "tool":
            # Tool result — must follow as user message in Anthropic format
            # Find the corresponding tool_use id from previous assistant message
            tool_use_id = m.get("tool_use_id", "tool_result")
            # Wrap in user message with tool_result block
            if result and result[-1].get("role") == "user" and isinstance(result[-1]["content"], list):
                # Append to existing user content list
                result[-1]["content"].append({
                    "type":        "tool_result",
                    "tool_use_id": tool_use_id,
                    "content":     str(content),
                })
            else:
                result.append({
                    "role": "user",
                    "content": [{
                        "type":        "tool_result",
                        "tool_use_id": tool_use_id,
                        "content":     str(content),
                    }],
                })

        elif role == "user":
            result.append({"role": "user", "content": content})

    # Anthropic requires messages to alternate roles — merge consecutive same-role messages
    return _merge_consecutive_roles(result)


def _merge_consecutive_roles(messages: list) -> list:
    """Merge consecutive messages with the same role into one."""
    if not messages:
        return messages
    merged = [messages[0]]
    for m in messages[1:]:
        last = merged[-1]
        if m["role"] == last["role"]:
            # Merge content
            if isinstance(last["content"], list) and isinstance(m["content"], list):
                last["content"].extend(m["content"])
            elif isinstance(last["content"], str) and isinstance(m["content"], str):
                last["content"] += "\n" + m["content"]
            else:
                merged.append(m)
        else:
            merged.append(m)
    return merged
