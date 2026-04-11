"""
services/roadmap.py — Roadmap business logic.

Single responsibility: deadline scanning and LLM-driven step generation.
No HTTP concerns. No FastAPI imports.

Public API:
  load_deadlines() -> list[dict]                  — scan dossiers for upcoming deadlines
  generate_steps(dossier, from_email) -> dict     — call Phi to produce time-bound steps (blocking)
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime

from api.context import BASE_DIR
from llm.client import call_phi_sync
from tools.roadmap import get_by_entity as roadmap_get_by_entity, days_until_deadline

_log = logging.getLogger("phi.services.roadmap")


def load_deadlines() -> list[dict]:
    """
    Scan all dossiers and return entities that have an upcoming deadline.
    Skips deadlines more than 30 days in the past.
    Returns list sorted by deadline ascending.
    """
    dossier_dir = BASE_DIR / "workspace" / "dossiers"
    results = []

    for path in sorted(dossier_dir.glob("*.json")):
        try:
            dossier = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        deadline = dossier.get("deadline")
        if not deadline:
            continue

        days_left = days_until_deadline(deadline)
        if days_left is not None and days_left < -30:
            continue

        fit_score = (dossier.get("fit_assessment") or {}).get("fit_score", 0)
        has_roadmap = roadmap_get_by_entity(dossier["entity_id"]) is not None

        results.append({
            "entity_id":        dossier["entity_id"],
            "entity_name":      dossier.get("name", ""),
            "entity_type":      dossier.get("type", "ORG"),
            "deadline":         deadline,
            "deadline_label":   dossier.get("deadline_label") or deadline,
            "registration_url": dossier.get("registration_url"),
            "fit_score":        fit_score,
            "why_yes":          (dossier.get("fit_assessment") or {}).get("why_yes", [])[:2],
            "days_left":        days_left,
            "has_roadmap":      has_roadmap,
        })

    results.sort(key=lambda x: x.get("deadline") or "9999-99-99")
    return results


def generate_steps(dossier: dict, from_email: str) -> dict:
    """
    Blocking: call Phi to generate time-bound preparation steps for an entity.
    Steps work backwards from the entity's deadline.
    Returns {"steps": [...]} or {"steps": []} on failure.
    """
    entity_name    = dossier.get("name", "")
    entity_type    = dossier.get("type", "ORG")
    deadline       = dossier.get("deadline", "")
    deadline_label = dossier.get("deadline_label") or deadline
    fit            = (dossier.get("fit_assessment") or {})
    why_yes        = (fit.get("why_yes") or [])[:2]
    next_actions   = (dossier.get("next_actions") or [])[:3]

    type_context = {
        "GRANT":    "grant application",
        "EVENT":    "conference / event",
        "INVESTOR": "investor pitch",
    }.get(entity_type, "opportunity")

    prompt = f"""Create a concrete preparation roadmap for David Lagarejo to participate in this {type_context}.

ENTITY: {entity_name}
DEADLINE: {deadline} ({deadline_label})
TODAY: {datetime.now().strftime('%Y-%m-%d')}

WHY THIS MATTERS:
{chr(10).join('- ' + w for w in why_yes) if why_yes else 'Strategic fit for patent commercialization'}

KNOWN NEXT ACTIONS:
{chr(10).join('- ' + a for a in next_actions) if next_actions else 'N/A'}

DAVID'S PROFILE:
- Patent US2024/0077174, ultrasonic non-invasive steam sensor, validated Ecopetrol, -30% energy
- Seeking: pilot clients, grants, strategic partners

Generate 5-8 concrete preparation steps, WORKING BACKWARDS from the deadline.
Each step must have a specific due_date (YYYY-MM-DD) BEFORE the deadline.
Spread steps realistically — do not cluster them all in the final week.

Categories (use exactly one per step):
- research   : background reading, requirements gathering
- prep       : document writing, deck creation, application form
- outreach   : identifying and contacting people at the event/org
- submit     : final submission, registration, sending
- logistics  : travel, booking, scheduling

Return JSON only. No markdown:
{{
  "steps": [
    {{
      "id": "step_1",
      "title": "short action title",
      "description": "one sentence — what exactly to do and why",
      "due_date": "YYYY-MM-DD",
      "days_before_deadline": 60,
      "category": "research",
      "done": false
    }}
  ]
}}"""

    raw = call_phi_sync(
        [
            {"role": "system", "content": "Output JSON only. No markdown. No commentary."},
            {"role": "user", "content": prompt},
        ],
        num_ctx=4096,
    )

    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(raw[start:end])
            for step in parsed.get("steps", []):
                if not step.get("id") or step["id"] in ("step_1", "step_2", "step_3"):
                    step["id"] = f"step_{uuid.uuid4().hex[:6]}"
                step.setdefault("done", False)
                step.setdefault("description", "")
            return parsed
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        _log.warning("generate_steps parse failed: %s", exc)

    return {"steps": []}
