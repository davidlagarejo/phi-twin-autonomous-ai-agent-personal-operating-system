"""
llm/tool_executor.py — Agentic tool call loop for chat.

Single responsibility: given a messages list and a set of tool schemas,
run the LLM → execute tool calls → inject results → repeat until the model
returns a plain text response or max_rounds is reached.

Auto-selects backend:
  Claude API  — when CLAUDE_API_KEY is set (reliable tool calling, ~95%)
  phi4 local  — fallback (best-effort tool calling, ~50-60%)

The two backends differ in how tool results are injected:
  Anthropic: tool results go in a user message as {type: tool_result, tool_use_id: id}
  Ollama:    tool results go as {role: tool, content: result}

This module handles both formats transparently.
"""
from __future__ import annotations

import json
import logging

from tools.tool_registry import execute_tool

_log = logging.getLogger("phi.llm.tool_executor")

_MAX_ROUNDS = 3


async def run_tool_loop(
    messages: list,
    tools: list,
    call_phi_fn,          # async callable — call_phi_with_tools (phi4 fallback)
    num_ctx: int = 4096,
    yield_fn=None,
) -> str:
    """
    Run the LLM with tool calling until a text response is produced.
    Uses Claude when CLAUDE_API_KEY is available, phi4 otherwise.

    Parameters
    ----------
    messages     : conversation history (NOT mutated — we work on a copy)
    tools        : list of Ollama-format tool schemas
    call_phi_fn  : async callable — call_phi_with_tools from llm.client (phi4 fallback)
    num_ctx      : context window size (phi4 only)
    yield_fn     : optional async callable(str) to emit SSE status events

    Returns the final text response string.
    """
    from llm.claude_client import is_available, call_claude_with_tools

    if is_available():
        return await _run_claude(messages, tools, yield_fn)
    return await _run_phi4(messages, tools, call_phi_fn, num_ctx, yield_fn)


# ── Claude backend ────────────────────────────────────────────────────────────

async def _run_claude(messages: list, tools: list, yield_fn) -> str:
    from llm.claude_client import call_claude_with_tools

    # Work on a copy — Claude needs its own message history with Anthropic tool format
    working: list = list(messages)

    for round_num in range(_MAX_ROUNDS):
        result = await call_claude_with_tools(working, tools=tools)

        if result["type"] == "text":
            return result["content"]

        tool_calls = result.get("calls", [])
        call_ids   = result.get("call_ids", [])
        raw_blocks = result.get("raw_blocks", [])

        if not tool_calls:
            return result.get("content", "")

        _log.info("tool_loop[claude] round=%d calls=%d", round_num + 1, len(tool_calls))

        # Append assistant message with tool_use blocks (Anthropic format)
        working.append({
            "role":       "assistant",
            "content":    "",
            "tool_calls": [
                {**tc, "id": cid}
                for tc, cid in zip(tool_calls, call_ids + [""] * len(tool_calls))
            ],
        })

        # Execute tools, inject results as tool_result blocks
        tool_result_blocks = []
        for i, call in enumerate(tool_calls):
            fn   = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            if yield_fn:
                await yield_fn(json.dumps({"status": "tool_call", "tool": name, "args": args}))

            tool_result = execute_tool(name, args)
            _log.debug("tool[claude]:%s result_len=%d", name, len(tool_result))

            tool_use_id = call_ids[i] if i < len(call_ids) else f"tool_{name}"
            tool_result_blocks.append({
                "type":        "tool_result",
                "tool_use_id": tool_use_id,
                "content":     tool_result,
            })
            # Also store on message for _convert_messages_to_anthropic
            working.append({
                "role":        "tool",
                "content":     tool_result,
                "tool_use_id": tool_use_id,
            })

    # Max rounds — force text response
    _log.warning("tool_loop[claude]: max_rounds=%d, forcing text", _MAX_ROUNDS)
    working.append({
        "role":    "user",
        "content": "Based on the tool results above, provide your final answer now.",
    })
    final = await call_claude_with_tools(working, tools=[])
    return final.get("content", "")


# ── phi4 backend ──────────────────────────────────────────────────────────────

async def _run_phi4(messages: list, tools: list, call_phi_fn, num_ctx: int, yield_fn) -> str:
    working = list(messages)

    for round_num in range(_MAX_ROUNDS):
        result = await call_phi_fn(working, tools=tools, num_ctx=num_ctx)

        if result["type"] == "text":
            return result["content"]

        tool_calls = result.get("calls", [])
        if not tool_calls:
            return result.get("content", "")

        _log.info("tool_loop[phi4] round=%d calls=%d", round_num + 1, len(tool_calls))

        working.append({
            "role":       "assistant",
            "content":    "",
            "tool_calls": tool_calls,
        })

        for call in tool_calls:
            fn   = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            if yield_fn:
                await yield_fn(json.dumps({"status": "tool_call", "tool": name, "args": args}))

            tool_result = execute_tool(name, args)
            _log.debug("tool[phi4]:%s result_len=%d", name, len(tool_result))

            working.append({"role": "tool", "content": tool_result})

    _log.warning("tool_loop[phi4]: max_rounds=%d, forcing text", _MAX_ROUNDS)
    working.append({
        "role":    "user",
        "content": "Based on the tool results above, provide your final answer now.",
    })
    from llm.client import call_phi
    return await call_phi(working, num_ctx=num_ctx)
