"""
services/directives.py — Strategic directives management.

Persists David's current research focus and exclusions to data/directives.md.
Called by chat handler when David expresses strategic changes, and read by
research_engine.py to filter/prioritize what to investigate.

Public API:
  load_directives() -> str          — raw markdown content
  extract_and_save(user_msg, phi_response) -> str | None
      — if conversation contains a strategic directive, persist and return summary
  get_exclusions() -> list[str]     — topics/entities David asked to exclude
  get_focus_areas() -> list[str]    — current priority research areas
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from api.context import BASE_DIR
from llm.client import call_phi_sync

_log = logging.getLogger("phi.services.directives")

_DIRECTIVES_FILE = BASE_DIR / "data" / "directives.md"

# Trigger phrases that signal a strategic directive change
_DIRECTIVE_SIGNALS = [
    "ya no", "deja de", "para de", "no busques", "no investigues",
    "enfócate en", "prioriza", "cambia el foco", "olvida el sensor",
    "pausar", "pausa", "excluye", "ignora", "quita",
    "ahora quiero", "nuevo objetivo", "cambia la estrategia",
    "stop researching", "focus on", "no longer", "pivot to",
    "no me interesa", "no quiero más", "quiero enfocarme",
    "cambiar rumbo", "nueva dirección",
]


def load_directives() -> str:
    """Return current directives file content. Empty string if not found."""
    try:
        return _DIRECTIVES_FILE.read_text(encoding="utf-8")
    except Exception:
        return ""


def get_exclusions() -> list[str]:
    """Parse the Pausas/Exclusiones section and return list of excluded topics."""
    content = load_directives()
    exclusions = []
    in_section = False
    for line in content.splitlines():
        if "Pausas" in line or "Exclusiones" in line:
            in_section = True
            continue
        if in_section:
            if line.startswith("##"):
                break
            stripped = line.strip().lstrip("-").strip()
            if stripped and stripped != "(ninguna actualmente)":
                exclusions.append(stripped)
    return exclusions


def get_focus_areas() -> list[str]:
    """Parse the Foco actual section and return list of focus areas."""
    content = load_directives()
    areas = []
    in_section = False
    for line in content.splitlines():
        if "Foco actual" in line:
            in_section = True
            continue
        if in_section:
            if line.startswith("##"):
                break
            stripped = line.strip().lstrip("-").strip()
            if stripped:
                areas.append(stripped)
    return areas


def _contains_directive_signal(text: str) -> bool:
    """Quick check: does this text contain a strategic direction signal?"""
    lower = text.lower()
    return any(signal in lower for signal in _DIRECTIVE_SIGNALS)


def extract_and_save(user_message: str, phi_response: str) -> str | None:
    """
    Analyze a conversation turn. If David expressed a strategic directive,
    extract it and update directives.md. Returns a summary string if updated, None otherwise.
    """
    if not _contains_directive_signal(user_message):
        return None

    current = load_directives()

    prompt = f"""Analiza esta conversación. David Lagarejo le dice algo estratégico a Phi sobre qué investigar, qué pausar, o qué priorizar.

MENSAJE DE DAVID:
{user_message[:500]}

RESPUESTA DE PHI:
{phi_response[:300]}

DIRECTIVAS ACTUALES:
{current[:1500]}

TAREA: Si David expresó un cambio estratégico real (nuevo foco, exclusión, pausa, prioridad), genera un archivo de directivas actualizado en Markdown con estas secciones exactas:
- ## Foco actual
- ## Prioridades de investigación (orden)
- ## Pausas / Exclusiones
- ## Contexto de decisión

Si NO hubo cambio estratégico real (fue una pregunta, comentario casual, o solo pidió info), responde exactamente: NO_CHANGE

Reglas:
- Preserva todo lo existente que siga siendo válido
- Agrega o modifica solo lo que David cambió explícitamente
- Si dijo "ya no me interesa el sensor" → agregar "sensor ultrasónico / investigación de producto" a Pausas
- Si dijo "enfócate en inversión Series A" → actualizar Foco actual
- Responde con el markdown completo del nuevo archivo O con NO_CHANGE"""

    try:
        raw = call_phi_sync([
            {"role": "system", "content": "Output the updated directives markdown OR the exact text NO_CHANGE. Nothing else."},
            {"role": "user", "content": prompt},
        ], num_ctx=3000)
    except Exception as exc:
        _log.warning("extract_and_save LLM call failed: %s", exc)
        return None

    raw = raw.strip()
    if raw == "NO_CHANGE" or raw.startswith("NO_CHANGE"):
        return None

    # Strip markdown fences if present
    for fence in ("```markdown", "```md", "```"):
        if raw.startswith(fence):
            raw = raw[len(fence):]
            raw = raw.strip()
            break
    # Remove trailing code fence or commentary after last ## section
    if "```" in raw:
        raw = raw[:raw.rfind("```")].strip()

    if len(raw) < 50:
        return None

    # Ensure header line
    if not raw.startswith("#"):
        raw = "# Directivas estratégicas de David\n_Actualizado automáticamente desde conversación._\n\n" + raw

    # Trim any non-markdown prose after the last markdown section
    lines = raw.splitlines()
    last_section_end = 0
    for i, line in enumerate(lines):
        if line.startswith("#") or line.startswith("-") or line.startswith("_") or not line.strip():
            last_section_end = i + 1
    raw = "\n".join(lines[:last_section_end]).strip()

    try:
        _DIRECTIVES_FILE.write_text(raw, encoding="utf-8")
        _log.info("directives_updated from conversation")
        # Extract what changed for the summary
        new_exclusions = get_exclusions()
        new_focus = get_focus_areas()
        return f"Directivas actualizadas — foco: {len(new_focus)} áreas, exclusiones: {len(new_exclusions)}"
    except Exception as exc:
        _log.warning("directives write failed: %s", exc)
        return None
