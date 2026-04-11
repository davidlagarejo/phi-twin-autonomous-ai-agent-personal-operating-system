"""
services/context_compressor.py — Context window management for long conversations.

Single responsibility: detect when the messages list is approaching the context
limit and summarize middle turns to keep the window under budget.

Strategy:
  - Estimate tokens as len(content) // 4 (rough but fast, no tokenizer needed).
  - Keep: system[0], session briefing pair (if present), last 3 turns.
  - Summarize: everything in between via a single LLM call.
  - The summary is injected as a user/assistant pair so the model treats it as
    prior context it already acknowledged.

Called synchronously from the async chat handler via await (the LLM call inside
is async). Safe to call on every turn — returns early when under budget.
"""
from __future__ import annotations

import logging
from typing import Optional

_log = logging.getLogger("phi.services.context_compressor")

# Compress when estimated tokens exceed this threshold.
# phi4:14b default context is 4096 (chat uses num_ctx=4096).
# We start compressing at ~2800 to leave room for the response.
_COMPRESS_THRESHOLD = 2800
_KEEP_TAIL_TURNS    = 3   # always keep the last N user/assistant pairs


def _estimate_tokens(messages: list) -> int:
    return sum(len(m.get("content") or "") for m in messages) // 4


def _is_system(m: dict) -> bool:
    return m.get("role") == "system"


def _split_messages(messages: list) -> tuple[list, list, list]:
    """
    Split messages into three segments:
      head  — system message(s) + any injected context pairs at the front
      body  — middle turns eligible for compression
      tail  — last _KEEP_TAIL_TURNS user/assistant pairs (always preserved)

    Returns (head, body, tail).
    """
    # Head: system messages at position 0 (and up to 4 injected context pairs after)
    head: list = []
    i = 0
    while i < len(messages) and _is_system(messages[i]):
        head.append(messages[i])
        i += 1
    # Keep injected context pairs (max 4 pairs = 8 messages) as part of head
    context_pairs = 0
    while i + 1 < len(messages) and context_pairs < 4:
        role_a = messages[i].get("role")
        role_b = messages[i + 1].get("role")
        if role_a == "user" and role_b == "assistant":
            content_a = messages[i].get("content", "")
            # Injected context messages start with "[" (briefing, memories, profile)
            if content_a.startswith("["):
                head.append(messages[i])
                head.append(messages[i + 1])
                i += 2
                context_pairs += 1
                continue
        break

    # Tail: last _KEEP_TAIL_TURNS user/assistant pairs
    tail_msgs: list = []
    j = len(messages) - 1
    pairs_kept = 0
    while j >= i and pairs_kept < _KEEP_TAIL_TURNS:
        if messages[j].get("role") == "assistant" and j - 1 >= i and messages[j - 1].get("role") == "user":
            tail_msgs = [messages[j - 1], messages[j]] + tail_msgs
            j -= 2
            pairs_kept += 1
        else:
            # Odd message (e.g. last user message without response yet)
            tail_msgs = [messages[j]] + tail_msgs
            j -= 1
            pairs_kept += 1

    body = messages[i:j + 1]
    return head, body, tail_msgs


async def compress_if_needed(messages: list, call_phi_fn) -> list:
    """
    Return a (possibly compressed) copy of messages.

    Parameters
    ----------
    messages    : current conversation message list
    call_phi_fn : async callable matching llm.client.call_phi signature
                  (messages, num_ctx=...) → str

    Returns the original list unchanged if under threshold.
    """
    if _estimate_tokens(messages) < _COMPRESS_THRESHOLD:
        return messages

    head, body, tail = _split_messages(messages)

    if not body:
        # Nothing to compress — all content is in head/tail
        return messages

    _log.info(
        "context_compress: total~%d tokens, compressing %d middle messages",
        _estimate_tokens(messages),
        len(body),
    )

    # Build a summary of the body turns
    body_text = "\n".join(
        f"{m['role'].upper()}: {(m.get('content') or '')[:400]}"
        for m in body
    )
    summary_prompt = [
        {
            "role": "system",
            "content": (
                "You are a conversation summarizer. "
                "Summarize the following conversation excerpt in under 200 words. "
                "Preserve all factual claims, decisions made, entities mentioned, and action items. "
                "Output only the summary, no preamble."
            ),
        },
        {"role": "user", "content": body_text},
    ]

    try:
        summary = await call_phi_fn(summary_prompt, num_ctx=2048)
        summary = summary.strip()
    except Exception as exc:
        _log.warning("context_compress: summary call failed (%s), skipping compression", exc)
        return messages

    if not summary:
        return messages

    compressed_pair = [
        {"role": "user",      "content": f"[RESUMEN DE CONVERSACIÓN ANTERIOR]\n{summary}"},
        {"role": "assistant", "content": "Entendido, tengo en cuenta lo que hemos hablado antes."},
    ]

    result = head + compressed_pair + tail
    _log.info(
        "context_compress: %d → %d tokens after compression",
        _estimate_tokens(messages),
        _estimate_tokens(result),
    )
    return result
