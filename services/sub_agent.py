"""
services/sub_agent.py — Sub-agent execution for deep, multi-step tasks.

A sub-agent is an LLM loop with its own isolated message history, goal,
and tool set. It runs inline (not background) and returns a summary.

Used by research-type skills (/research, /dossier) to do exhaustive
multi-tool investigations without polluting the main conversation context.

The sub-agent:
  1. Gets a goal and optional context string
  2. Runs up to max_rounds of: think → tool_call → result → think
  3. Produces a final summary when done or when max_rounds is reached
  4. Returns only the summary to the caller — internal tool calls stay private

Privacy: sub-agent uses the same tool registry (search_web, read_dossier,
save_task). The same privacy rules apply as in the main tool loop.
"""
from __future__ import annotations

import logging
from typing import Optional

_log = logging.getLogger("phi.services.sub_agent")

_SUB_AGENT_SYSTEM = """You are a specialized research sub-agent for Phi, David Lagarejo's personal AI.
David is a physicist-engineer, IIoT/cleantech CEO in NYC working on ultrasonic sensor technology.

Your job: execute a specific research goal exhaustively using the available tools.
- Use search_web multiple times with different queries to gather comprehensive information
- Use read_dossier to check existing knowledge before searching
- Be thorough: 3-5 tool calls is normal for a deep research task
- When done, synthesize everything into a clear, actionable summary in Spanish

Focus only on your assigned goal. Be specific and cite sources."""


async def run_sub_agent(
    goal: str,
    context: str = "",
    max_rounds: int = 5,
) -> str:
    """
    Run a sub-agent to accomplish a specific research goal.

    Parameters
    ----------
    goal      : what the sub-agent should accomplish
    context   : optional background context (injected as first message)
    max_rounds: maximum tool call rounds (default 5 — more than main loop)

    Returns a text summary of findings.
    """
    from tools.tool_registry import TOOLS, execute_tool
    import json

    _log.info("sub_agent start goal=%r max_rounds=%d", goal[:80], max_rounds)

    # Build initial messages
    messages: list = [{"role": "system", "content": _SUB_AGENT_SYSTEM}]

    user_content = f"**Goal:** {goal}"
    if context:
        user_content += f"\n\n**Context:**\n{context}"
    user_content += "\n\nProceed with your research. Use tools as needed."

    messages.append({"role": "user", "content": user_content})

    # Determine backend
    from llm.claude_client import is_available as _claude_available

    if _claude_available():
        result = await _run_sub_agent_claude(messages, TOOLS, goal, max_rounds)
    else:
        result = await _run_sub_agent_phi4(messages, TOOLS, goal, max_rounds)

    _log.info("sub_agent done goal=%r result_len=%d", goal[:80], len(result))
    return result


async def _run_sub_agent_claude(messages: list, tools: list, goal: str, max_rounds: int) -> str:
    from llm.claude_client import call_claude_with_tools
    from tools.tool_registry import execute_tool
    import json

    working = list(messages)

    for round_num in range(max_rounds):
        result = await call_claude_with_tools(working, tools=tools)

        if result["type"] == "text":
            return result["content"] or f"Sub-agente completó investigación sobre: {goal}"

        tool_calls = result.get("calls", [])
        call_ids   = result.get("call_ids", [])

        if not tool_calls:
            return result.get("content", "")

        _log.debug("sub_agent[claude] round=%d calls=%d", round_num + 1, len(tool_calls))

        working.append({
            "role":       "assistant",
            "content":    "",
            "tool_calls": [
                {**tc, "id": call_ids[i] if i < len(call_ids) else f"sa_{i}"}
                for i, tc in enumerate(tool_calls)
            ],
        })

        for i, call in enumerate(tool_calls):
            fn   = call.get("function", {})
            name = fn.get("name", "")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            tool_result = execute_tool(name, args)
            _log.debug("sub_agent tool:%s result_len=%d", name, len(tool_result))

            tool_use_id = call_ids[i] if i < len(call_ids) else f"sa_{i}"
            working.append({
                "role":        "tool",
                "content":     tool_result,
                "tool_use_id": tool_use_id,
            })

    # Max rounds — force final synthesis
    working.append({
        "role":    "user",
        "content": "Research complete. Provide your final comprehensive summary now in Spanish.",
    })
    final = await call_claude_with_tools(working, tools=[])
    return final.get("content", f"Investigación completada para: {goal}")


async def _run_sub_agent_phi4(messages: list, tools: list, goal: str, max_rounds: int) -> str:
    from llm.client import call_phi_with_tools, call_phi
    from tools.tool_registry import execute_tool
    import json

    working = list(messages)

    for round_num in range(max_rounds):
        result = await call_phi_with_tools(working, tools=tools, num_ctx=6144)

        if result["type"] == "text":
            return result["content"] or f"Sub-agente completó: {goal}"

        tool_calls = result.get("calls", [])
        if not tool_calls:
            return result.get("content", "")

        _log.debug("sub_agent[phi4] round=%d calls=%d", round_num + 1, len(tool_calls))

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

            tool_result = execute_tool(name, args)
            working.append({"role": "tool", "content": tool_result})

    # Force final summary
    working.append({
        "role":    "user",
        "content": "Summarize all findings in Spanish. Be specific and actionable.",
    })
    return await call_phi(working, num_ctx=6144)
