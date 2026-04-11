"""
services/skills.py — Skills loader and expander for Phi chat.

Skills are Markdown files in phi-twin/skills/*.md with YAML frontmatter.
Users invoke them with /skill-name [arg1] [arg2] from the chat UI.

Format:
  ---
  name: dossier
  description: Genera un dossier completo para una oportunidad
  args: [nombre_entidad]
  ---
  Prompt template with {0}, {1}, ... placeholders for args.

Usage:
  from services.skills import expand_skill, list_skills, parse_skill_invocation
  msg, is_skill = parse_skill_invocation("/dossier NYCEDC")
  # msg → expanded prompt ready to inject as user message
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_SKILLS_DIR = Path(__file__).parent.parent / "skills"

# Cache: name → {"description": str, "args": list, "template": str}
_skill_cache: dict = {}
_cache_mtime: dict = {}


def _load_skill_file(path: Path) -> Optional[dict]:
    """Parse a skill .md file. Returns dict or None on error."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None

    # Parse YAML frontmatter
    fm: dict = {}
    template = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            raw_fm = parts[1].strip()
            template = parts[2].strip()
            for line in raw_fm.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    key = key.strip()
                    val = val.strip()
                    # Parse list values like [arg1, arg2]
                    if val.startswith("[") and val.endswith("]"):
                        items = [v.strip().strip("'\"") for v in val[1:-1].split(",") if v.strip()]
                        fm[key] = items
                    else:
                        fm[key] = val.strip("'\"")

    name = fm.get("name", path.stem)
    return {
        "name":        name,
        "description": fm.get("description", ""),
        "args":        fm.get("args", []),
        "template":    template,
    }


def _refresh_cache() -> None:
    """Reload skills that have changed on disk."""
    if not _SKILLS_DIR.exists():
        return
    for path in _SKILLS_DIR.glob("*.md"):
        mtime = path.stat().st_mtime
        if _cache_mtime.get(path) != mtime:
            skill = _load_skill_file(path)
            if skill:
                _skill_cache[skill["name"]] = skill
                _cache_mtime[path] = mtime


def list_skills() -> list[dict]:
    """Return list of {name, description, args} for all available skills."""
    _refresh_cache()
    return [
        {"name": s["name"], "description": s["description"], "args": s["args"]}
        for s in _skill_cache.values()
    ]


def expand_skill(name: str, args: list[str]) -> Optional[str]:
    """
    Expand a skill template with the given args.
    Returns the expanded prompt string, or None if skill not found.
    """
    _refresh_cache()
    skill = _skill_cache.get(name)
    if not skill:
        return None

    template = skill["template"]
    # Replace {0}, {1}, ... with args; leave unreplaced if arg missing
    for i, arg in enumerate(args):
        template = template.replace(f"{{{i}}}", arg)

    return template


def parse_skill_invocation(message: str) -> tuple[str, bool]:
    """
    Detect if message is a skill invocation (/skill-name [args]).

    Returns (expanded_prompt, True) if a skill was found.
    Returns (original_message, False) if not a skill invocation.
    """
    message = message.strip()
    if not message.startswith("/"):
        return message, False

    # Parse /skill-name arg1 arg2 ...
    # Quoted args are supported: /outreach "NYCEDC Green Fund"
    parts = _split_skill_args(message[1:])  # strip leading /
    if not parts:
        return message, False

    name = parts[0].lower()
    args = parts[1:]

    expanded = expand_skill(name, args)
    if expanded is None:
        # Unknown skill — list available skills
        available = [s["name"] for s in list_skills()]
        hint = f"Skill `/{name}` no encontrado. Skills disponibles: {', '.join(f'/{s}' for s in available)}"
        return hint, True  # still "is_skill" so it skips the plan loop

    return expanded, True


def _split_skill_args(text: str) -> list[str]:
    """Split command line respecting quoted strings."""
    parts = []
    current = []
    in_quote = False
    quote_char = ""
    for ch in text:
        if in_quote:
            if ch == quote_char:
                in_quote = False
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        elif ch in ('"', "'"):
            in_quote = True
            quote_char = ch
        elif ch == " ":
            if current:
                parts.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts
