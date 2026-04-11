"""
api/routes/roadmap.py — Deadlines & roadmap HTTP boundary.

Single responsibility: parse request → call service → return JSONResponse.
No business logic here. All LLM calls and dossier scanning live in services/roadmap.py.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from api.context import BASE_DIR, request_executor
from services.roadmap import load_deadlines, generate_steps
from tools.roadmap import (
    create           as roadmap_create,
    list_active      as roadmap_list_active,
    get_by_entity    as roadmap_get_by_entity,
    update_step      as roadmap_update_step,
    archive          as roadmap_archive,
    days_until_deadline,
)

router = APIRouter()


class RoadmapGenerateRequest(BaseModel):
    entity_id: str
    from_email: str = ""


@router.get("/api/deadlines")
async def get_deadlines():
    """Return all entities with an upcoming deadline, sorted by date."""
    items = await asyncio.to_thread(load_deadlines)
    return JSONResponse({"deadlines": items})


@router.get("/api/roadmaps")
async def get_roadmaps():
    """Return all active roadmaps with computed days_left and progress."""
    roadmaps = await asyncio.to_thread(roadmap_list_active)
    for rm in roadmaps:
        rm["days_left"] = days_until_deadline(rm.get("deadline", ""))
        steps = rm.get("steps", [])
        done_count = sum(1 for s in steps if s.get("done"))
        rm["progress_pct"] = round(done_count / len(steps) * 100) if steps else 0
    return JSONResponse({"roadmaps": roadmaps})


@router.post("/api/roadmap")
async def generate_roadmap(req: RoadmapGenerateRequest):
    """
    Generate a roadmap for an entity by:
      1. Loading the entity's dossier (deadline required).
      2. Calling Phi to produce time-bound steps working backwards from the deadline.
      3. Persisting and returning the roadmap.

    Returns 404 if dossier not found, 422 if no deadline set.
    Returns existing roadmap (created=False) if one already exists.
    """
    dossier_dir = BASE_DIR / "workspace" / "dossiers"
    dossier = None
    for path in dossier_dir.glob("*.json"):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
            if d.get("entity_id") == req.entity_id:
                dossier = d
                break
        except (OSError, json.JSONDecodeError):
            continue

    if not dossier:
        return JSONResponse({"error": "dossier not found"}, status_code=404)

    deadline = dossier.get("deadline")
    if not deadline:
        return JSONResponse(
            {"error": "no deadline set for this entity — run a research cycle first"},
            status_code=422,
        )

    existing = await asyncio.to_thread(roadmap_get_by_entity, req.entity_id)
    if existing:
        existing["days_left"] = days_until_deadline(existing.get("deadline", ""))
        return JSONResponse({"roadmap": existing, "created": False})

    roadmap_data = await asyncio.get_event_loop().run_in_executor(
        request_executor, generate_steps, dossier, req.from_email
    )

    if not roadmap_data.get("steps"):
        return JSONResponse({"error": "Phi failed to generate steps"}, status_code=500)

    created = await asyncio.to_thread(
        roadmap_create,
        req.entity_id,
        dossier.get("name", req.entity_id),
        dossier.get("type", "ORG"),
        deadline,
        dossier.get("deadline_label") or deadline,
        roadmap_data["steps"],
        dossier.get("registration_url"),
    )
    created["days_left"] = days_until_deadline(deadline)
    return JSONResponse({"roadmap": created, "created": True})


@router.patch("/api/roadmap/{roadmap_id}/step/{step_id}")
async def patch_roadmap_step(roadmap_id: str, step_id: str, req: Request):
    """Toggle a roadmap step done/pending."""
    body = await req.json()
    done = bool(body.get("done", False))
    updated = await asyncio.to_thread(roadmap_update_step, roadmap_id, step_id, done)
    if not updated:
        return JSONResponse({"error": "roadmap or step not found"}, status_code=404)
    return JSONResponse({"ok": True})


@router.delete("/api/roadmap/{roadmap_id}")
async def delete_roadmap(roadmap_id: str):
    """Archive (soft-delete) a roadmap."""
    archived = await asyncio.to_thread(roadmap_archive, roadmap_id)
    if not archived:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"ok": True})
