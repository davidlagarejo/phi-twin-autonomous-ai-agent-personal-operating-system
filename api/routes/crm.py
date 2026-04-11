"""
api/routes/crm.py — CRM HTTP boundary.

Single responsibility: parse request → call service → return JSONResponse.
No business logic here. All LLM calls and data transforms live in services/crm.py.
"""
from __future__ import annotations

import asyncio
import subprocess

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from api.context import BASE_DIR, request_executor
from services.crm import enrich_contacts, build_draft, _STATUS_ORDER
from tools.crm import (
    get_all        as _crm_list,
    get            as _crm_get,
    upsert         as _crm_upsert,
    update_status  as _crm_update_status,
    set_followup   as _crm_set_followup,
    add_note       as _crm_add_note,
    save_draft     as _crm_save_draft,
    seed_from_dossiers as _crm_seed,
)

router = APIRouter()


@router.get("/api/crm")
async def crm_get_all(status: str = ""):
    contacts = await asyncio.to_thread(_crm_list, status)
    enriched = await asyncio.to_thread(enrich_contacts, contacts)
    enriched.sort(key=lambda c: (
        _STATUS_ORDER.get(c.get("status", "new"), 9),
        -(c.get("fit_score") or 0),
    ))
    return JSONResponse({"contacts": enriched, "total": len(enriched)})


@router.post("/api/crm")
async def crm_create(req: Request):
    body = await req.json()
    cid = await asyncio.to_thread(_crm_upsert, body)
    return JSONResponse({"ok": True, "id": cid})


@router.patch("/api/crm/{contact_id}")
async def crm_patch(contact_id: str, req: Request):
    body = await req.json()
    status      = body.get("status")
    note        = body.get("note", "")
    followup_at = body.get("followup_at")

    if status:
        await asyncio.to_thread(_crm_update_status, contact_id, status, note)
    elif note:
        await asyncio.to_thread(_crm_add_note, contact_id, note)

    if followup_at:
        await asyncio.to_thread(_crm_set_followup, contact_id, followup_at, note)

    return JSONResponse({"ok": True})


@router.post("/api/crm/{contact_id}/draft")
async def crm_draft(contact_id: str, req: Request):
    """Generate email + LinkedIn DM draft for a contact using Phi."""
    body = await req.json()
    from_email = body.get("from_email", "")

    contact = await asyncio.to_thread(_crm_get, contact_id)
    if not contact:
        return JSONResponse({"error": "not found"}, status_code=404)

    if not body.get("force") and contact.get("draft_body"):
        return JSONResponse({
            "subject":     contact.get("draft_subject", ""),
            "body":        contact.get("draft_body", ""),
            "linkedin_dm": contact.get("draft_linkedin", ""),
            "cached":      True,
        })

    result = await asyncio.get_event_loop().run_in_executor(
        request_executor, build_draft, contact, from_email
    )

    await asyncio.to_thread(
        _crm_save_draft,
        contact_id,
        result.get("subject", ""),
        result.get("body", ""),
        result.get("linkedin_dm", ""),
    )
    return JSONResponse({**result, "cached": False})


@router.post("/api/crm/{contact_id}/open_mail")
async def crm_open_mail(contact_id: str, req: Request):
    """Open Apple Mail compose window with draft pre-filled."""
    body = await req.json()

    contact = await asyncio.to_thread(_crm_get, contact_id)
    if not contact:
        return JSONResponse({"error": "not found"}, status_code=404)

    to_addr = (body.get("to") or contact.get("email") or "").replace('"', '\\"')
    subject = (body.get("subject") or contact.get("draft_subject") or "").replace('"', '\\"').replace("\n", " ")
    content = (body.get("body") or contact.get("draft_body") or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    if not subject or not content:
        return JSONResponse({"error": "no draft — call /draft first"}, status_code=400)

    script = f'''tell application "Mail"
    set newMsg to make new outgoing message with properties {{subject:"{subject}", content:"{content}", visible:true}}
    tell newMsg
        make new to recipient at end of to recipients with properties {{address:"{to_addr}"}}
    end tell
    activate
end tell'''
    try:
        subprocess.Popen(["osascript", "-e", script])
        await asyncio.to_thread(
            _crm_update_status, contact_id, "drafted", "Borrador abierto en Apple Mail"
        )
        return JSONResponse({"ok": True})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@router.post("/api/crm/seed")
async def crm_seed():
    """Seed CRM from all existing dossiers with fit_score >= 40."""
    dossier_dir = BASE_DIR / "workspace" / "dossiers"
    created = await asyncio.to_thread(_crm_seed, dossier_dir)
    return JSONResponse({"ok": True, "created": created})
