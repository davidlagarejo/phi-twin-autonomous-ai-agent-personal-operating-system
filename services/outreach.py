"""
services/outreach.py — Email draft business logic.

Builds outreach and reply drafts from structured data — no LLM calls.
Runs synchronously (intended for ThreadPoolExecutor workers).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

_log  = logging.getLogger("phi.services.outreach")
_BASE = Path(__file__).parent.parent
_DOSSIER_DIR = _BASE / "workspace" / "dossiers"


def _load_dossier(entity_id: str) -> dict:
    """Load dossier by entity_id (exact match, fallback to prefix without double ent_ bug)."""
    if not _DOSSIER_DIR.exists():
        return {}
    for df in _DOSSIER_DIR.glob("*.json"):
        try:
            d   = json.loads(df.read_text(encoding="utf-8"))
            eid = d.get("entity_id", "")
            if eid == entity_id:
                return d
            raw_in   = entity_id.replace("ent_", "")
            raw_doss = eid.replace("ent_", "")
            if raw_doss and raw_in and raw_doss.startswith(raw_in[:8]):
                return d
        except Exception:
            continue
    return {}


def _first_name(full_name: str) -> str:
    return full_name.strip().split()[0] if full_name.strip() else ""


def build_email_draft(req: dict) -> dict:
    """
    Build an outreach email from structured data — no LLM.

    Input keys: entity_id, contact_name, contact_role, action, from_email,
                opportunity_title, deadline_label, why_yes, why_contact.
    Returns: {subject, body, from, suggested_to}.
    """
    entity_id      = req.get("entity_id", "")
    contact_name   = (req.get("contact_name") or "").strip()
    contact_role   = (req.get("contact_role") or "").strip()
    from_email     = req.get("from_email") or "davidlagarejo@gmail.com"
    from_domain    = "Zircular" if "zircular" in from_email else "ZION ING"
    opp_title      = (req.get("opportunity_title") or "").strip()
    deadline_label = (req.get("deadline_label") or "").strip()
    req_why_yes    = req.get("why_yes") or []
    why_contact    = (req.get("why_contact") or req.get("action") or "").strip()

    dossier = _load_dossier(entity_id)
    name    = (dossier.get("name") or entity_id).strip()
    fa      = dossier.get("fit_assessment") or {}
    why_yes = [b.strip().rstrip(".") for b in (req_why_yes or fa.get("why_yes") or []) if b][:3]

    # ── Greeting ──────────────────────────────────────────────────────────────
    fn = _first_name(contact_name)
    greeting = f"Hi {fn}," if fn else "Hi,"

    # ── Opening line ──────────────────────────────────────────────────────────
    ref = opp_title if (opp_title and opp_title.lower() != name.lower()) else name
    opening = f"I'm writing about {ref} — I believe there's a strong fit with what we're building at {from_domain}."

    # ── Value proposition (first two why_yes bullets) ─────────────────────────
    if why_yes:
        value_block = why_yes[0] + "."
        if len(why_yes) > 1:
            value_block += f" {why_yes[1]}."
    else:
        value_block = (
            f"At {from_domain} we've built an IIoT platform for industrial energy efficiency "
            f"— patent-backed (US2024/0077174), validated with Ecopetrol (-30% steam energy loss)."
        )

    # ── Why this specific person ───────────────────────────────────────────────
    contact_line = ""
    if why_contact:
        contact_line = f"\n\n{why_contact.rstrip('.')}."

    # ── Deadline urgency ──────────────────────────────────────────────────────
    deadline_line = ""
    if deadline_label:
        deadline_line = f"\n\nWith the {deadline_label} deadline coming up, I wanted to reach out now."

    # ── Call to action ────────────────────────────────────────────────────────
    role_hint = f" as {contact_role}" if contact_role else ""
    cta = f"\n\nWould you{role_hint} have 20 minutes this week for a quick call?"

    # ── Signature ─────────────────────────────────────────────────────────────
    sig = f"\n\nBest,\nDavid Lagarejo\nCEO, {from_domain}\n{from_email}"

    body = f"{greeting}\n\n{opening}\n\n{value_block}{contact_line}{deadline_line}{cta}{sig}"

    # ── Subject ───────────────────────────────────────────────────────────────
    if opp_title:
        subject = f"{opp_title} × {from_domain}"
    else:
        subject = f"{name} × {from_domain}"

    return {
        "subject": subject,
        "body":    body,
        "from":    from_email,
        "suggested_to": contact_name,
    }


def build_reply_draft(req: dict) -> dict:
    """
    Build a reply email from structured data — no LLM.

    Input keys: entity_id, entity_name, contact_name, contact_role,
                original_subject, reply_snippet, from_email.
    Returns: {subject, body, from, suggested_to}.
    """
    entity_name   = (req.get("entity_name") or "").strip()
    contact_name  = (req.get("contact_name") or "").strip()
    contact_role  = (req.get("contact_role") or "").strip()
    orig_subject  = (req.get("original_subject") or "").strip()
    reply_snippet = (req.get("reply_snippet") or "").strip()
    from_email    = req.get("from_email") or "davidlagarejo@gmail.com"
    from_domain   = "Zircular" if "zircular" in from_email else "ZION ING"

    dossier = _load_dossier(req.get("entity_id", ""))
    fa      = dossier.get("fit_assessment") or {}
    why_yes = [b.strip().rstrip(".") for b in (fa.get("why_yes") or []) if b][:2]

    fn       = _first_name(contact_name)
    greeting = f"Hi {fn}," if fn else "Hi,"

    # Acknowledge their reply
    if reply_snippet:
        # Take first sentence of their reply as the reference point
        first_sent = reply_snippet.split(".")[0].strip()
        ack = f"Thanks for getting back to me{(' — ' + first_sent[:80]) if first_sent else ''}."
    else:
        ack = f"Thanks for getting back to me."

    # Next step using why_yes context
    if why_yes:
        context_line = f"\n\n{why_yes[0]}."
    else:
        context_line = ""

    # CTA: move toward concrete next step
    role_ref = f" you{' and the team' if not contact_role else ''}" if contact_name else " the team"
    cta = f"\n\nWould{role_ref} be available for a 20-min call this week to discuss next steps?"

    sig = f"\n\nBest,\nDavid Lagarejo\nCEO, {from_domain}\n{from_email}"

    body = f"{greeting}\n\n{ack}{context_line}{cta}{sig}"

    return {
        "subject":      f"Re: {orig_subject}" if orig_subject else f"Re: {entity_name}",
        "body":         body,
        "from":         from_email,
        "suggested_to": contact_name,
    }
