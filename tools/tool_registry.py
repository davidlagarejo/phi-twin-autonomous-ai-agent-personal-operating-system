"""
tools/tool_registry.py — Formal tool definitions for Ollama function calling.

Defines the 3 tools available to Phi during chat:
  search_web   — query the web and return formatted results
  read_dossier — retrieve a dossier by partial name match
  save_task    — create an outreach task for follow-up

Each tool is defined in Ollama/OpenAI function-calling format (JSON schema).
`execute_tool()` is the synchronous dispatcher — runs the real function and
returns a plain-text result suitable for injection as a `tool` message.

Only imports from tools/ layer to avoid circular imports.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any
from pathlib import Path

_log = logging.getLogger("phi.tools.tool_registry")

_BASE = Path(__file__).parent.parent

# ── Tool schemas (Ollama / OpenAI format) ─────────────────────────────────────

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for current information. Use when you need facts, "
                "recent news, grant details, company info, or anything not in the dossiers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Specific search query in English or Spanish",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_dossier",
            "description": (
                "Read the full dossier for an opportunity or entity. Use when the user asks "
                "about a specific company, grant, investor, or contact already in the system."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Partial or full name of the entity to look up",
                    }
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_task",
            "description": (
                "Save an outreach action item for follow-up. Use when the user confirms "
                "they want to contact someone, send a proposal, or schedule a meeting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {
                        "type": "string",
                        "description": "Name of the company, grant, or person",
                    },
                    "action": {
                        "type": "string",
                        "description": "Specific action to take (e.g. 'email CTO about pilot')",
                    },
                    "contact_name": {
                        "type": "string",
                        "description": "Name of the person to contact, if known",
                    },
                },
                "required": ["entity_name", "action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_skill",
            "description": (
                "Create a new reusable skill (slash command) from a task or workflow. "
                "Call this ONLY when the user explicitly confirms they want to save an action as a skill."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Short slug for the command (no spaces, e.g. 'contact_doe')",
                    },
                    "description": {
                        "type": "string",
                        "description": "One-line description of what this skill does",
                    },
                    "template": {
                        "type": "string",
                        "description": (
                            "The prompt template body. Use {0}, {1} for user-supplied arguments. "
                            "Should be a complete, actionable instruction in Spanish."
                        ),
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Argument names, e.g. ['entidad', 'contacto']. Empty if no args needed.",
                    },
                },
                "required": ["skill_name", "description", "template"],
            },
        },
    },
]


# ── Tool executor ─────────────────────────────────────────────────────────────

def execute_tool(name: str, arguments: dict) -> str:
    """
    Dispatch a tool call by name and return a plain-text result.
    Never raises — returns an error description on failure.
    """
    try:
        if name == "search_web":
            return _tool_search_web(arguments.get("query", ""))
        if name == "read_dossier":
            return _tool_read_dossier(arguments.get("name", ""))
        if name == "save_task":
            return _tool_save_task(
                arguments.get("entity_name", ""),
                arguments.get("action", ""),
                arguments.get("contact_name"),
            )
        if name == "create_skill":
            return _tool_create_skill(
                arguments.get("skill_name", ""),
                arguments.get("description", ""),
                arguments.get("template", ""),
                arguments.get("args", []),
            )
        return f"Unknown tool: {name}"
    except Exception as exc:
        _log.warning("execute_tool %s failed: %s", name, exc)
        return f"Tool error ({name}): {exc}"


# ── Individual tool implementations ──────────────────────────────────────────

def _tool_search_web(query: str) -> str:
    if not query.strip():
        return "No query provided."
    from tools.search import run_queries, format_for_prompt
    results = run_queries([query], max_per_query=5)
    if not results:
        return f"No results found for: {query}"
    return format_for_prompt(results)


def _tool_read_dossier(name: str) -> str:
    if not name.strip():
        return "No name provided."
    from tools.dossier_index import load_dossiers
    name_lower = name.lower()
    dossiers = load_dossiers()
    # Exact match first, then partial
    match = next(
        (d for d in dossiers if d.get("name", "").lower() == name_lower),
        None,
    )
    if match is None:
        match = next(
            (d for d in dossiers if name_lower in d.get("name", "").lower()),
            None,
        )
    if match is None:
        names = [d.get("name", "") for d in dossiers[:10]]
        return f"No dossier found for '{name}'. Available: {', '.join(names)}"

    fa  = match.get("fit_assessment") or {}
    pr  = match.get("profile") or {}
    out = {
        "name":         match.get("name"),
        "type":         match.get("type"),
        "fit_score":    fa.get("fit_score"),
        "description":  (match.get("description") or "")[:400],
        "why_yes":      (fa.get("why_yes") or [])[:3],
        "why_not":      (fa.get("why_not") or [])[:2],
        "next_actions": (match.get("next_actions") or [])[:3],
        "website":      pr.get("website"),
        "country":      pr.get("country"),
        "key_people":   (pr.get("key_people") or [])[:3],
    }
    return json.dumps(out, ensure_ascii=False, indent=2)


def _tool_save_task(entity_name: str, action: str, contact_name: str | None) -> str:
    if not entity_name.strip() or not action.strip():
        return "entity_name and action are required."
    from tools.outreach import load_tasks, save_tasks
    import uuid
    tasks = load_tasks()
    task_id = str(uuid.uuid4())[:8]
    tasks.append({
        "task_id":      task_id,
        "entity_name":  entity_name,
        "action":       action,
        "contact_name": contact_name or "",
        "status":       "pending",
        "created_at":   datetime.now(timezone.utc).isoformat()[:19] + "Z",
        "source":       "chat_tool_call",
    })
    save_tasks(tasks)
    _log.info("tool:save_task id=%s entity=%r action=%r", task_id, entity_name, action)
    return (
        f"Task saved (id={task_id}): {action} → {entity_name}\n"
        f"SKILL_SUGGESTION: ¿Quieres guardar esta acción como un skill reutilizable? "
        f"Si dices que sí, crearé el comando /outreach_{entity_name.split()[0].lower()} "
        f"para que puedas repetirla fácilmente."
    )


def _tool_create_skill(
    skill_name: str,
    description: str,
    template: str,
    args: list,
) -> str:
    """Write a new skill .md file to the skills/ directory."""
    skill_name = skill_name.strip().lower().replace(" ", "_")
    if not skill_name or not template.strip():
        return "skill_name and template are required."

    # Sanitize skill_name to safe slug
    import re as _re
    skill_name = _re.sub(r"[^a-z0-9_\-]", "", skill_name)[:40]
    if not skill_name:
        return "Invalid skill_name: use only letters, numbers, _ or -"

    skills_dir = _BASE / "skills"
    skills_dir.mkdir(exist_ok=True)
    skill_path = skills_dir / f"{skill_name}.md"

    args_yaml = f"[{', '.join(args)}]" if args else "[]"
    content = f"""---
name: {skill_name}
description: {description}
args: {args_yaml}
---

{template.strip()}
"""

    try:
        skill_path.write_text(content, encoding="utf-8")
        _log.info("tool:create_skill name=%r path=%s", skill_name, skill_path)
        # Invalidate skills cache so it's immediately available
        try:
            from services.skills import _cache_mtime
            if skill_path in _cache_mtime:
                del _cache_mtime[skill_path]
        except Exception:
            pass
        return f"Skill /{skill_name} creado. Ya puedes usarlo escribiendo `/{skill_name}` en el chat."
    except Exception as exc:
        _log.warning("tool:create_skill failed: %s", exc)
        return f"Error al crear skill: {exc}"
