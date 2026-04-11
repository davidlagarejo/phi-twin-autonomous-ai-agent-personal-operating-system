"""
services/crm.py — CRM business logic.

Single responsibility: data enrichment and personalized draft generation.
No HTTP concerns. No FastAPI imports.

Public API:
  enrich_contacts(contacts)              — merge dossier data into contact records
  build_draft(contact, from_email) -> dict  — research + LLM draft (blocking)
"""
from __future__ import annotations

import json
import logging

from api.context import BASE_DIR
from llm.client import call_phi_sync
from tools.search import run_queries, format_for_prompt

_log = logging.getLogger("phi.services.crm")

_STATUS_ORDER = {
    "replied": 0, "meeting": 1, "followup_due": 2, "sent": 3,
    "drafted": 4, "ready": 5, "researching": 6, "new": 7, "closed": 8,
}


def enrich_contacts(contacts: list) -> list:
    """
    Merge dossier data (fit_score, entity_type, why_yes) into contact records.
    Reads each entity's dossier file at most once per call (cached in local dict).
    """
    dossier_dir = BASE_DIR / "workspace" / "dossiers"
    cache: dict = {}
    enriched = []
    for c in contacts:
        eid = c.get("entity_id", "")
        if eid and eid not in cache:
            dp = dossier_dir / f"{eid}.json"
            try:
                cache[eid] = json.loads(dp.read_text()) if dp.exists() else {}
            except Exception:
                cache[eid] = {}
        d = cache.get(eid, {})
        ec = dict(c)
        ec.setdefault("fit_score",   (d.get("fit_assessment") or {}).get("fit_score", 0))
        ec.setdefault("entity_type", d.get("type", "ORG"))
        ec.setdefault("company",     d.get("name", c.get("company", "")))
        ec["why_yes"] = ec.get("why_yes") or (d.get("fit_assessment") or {}).get("why_yes", [])
        enriched.append(ec)
    return enriched


def build_draft(contact: dict, from_email: str) -> dict:
    """
    Blocking: research the contact, then generate a personalized email + LinkedIn DM.

    Enforced pipeline (code, not prompt):
      1. Run 3 targeted web searches for this person/company.
      2. Feed real findings into the prompt.
      3. Phi drafts using only the researched context — no generic templates.

    Returns {"subject": str, "body": str, "linkedin_dm": str}.
    """
    company     = contact.get("company", "the company")
    name        = contact.get("name") or ""
    role        = contact.get("role") or ""
    why_yes     = (contact.get("why_yes") or [])[:3]
    angle       = contact.get("outreach_angle") or ""
    reason      = contact.get("outreach_reason") or ""
    from_domain = "Zircular" if "zircular" in from_email else "ZION ING"

    # ── Step 1: Research the contact (enforced — always runs) ─────────────────
    target = f"{name} {company}".strip() if name else company
    search_queries = [
        f"{target} CEO founder director LinkedIn",
        f"{company} technology energy industrial sector news 2024 2025",
        f"{target} {role} contact email".strip() if role else f"{company} partnership collaboration",
    ]
    search_results = run_queries(search_queries, max_per_query=3)
    research_text = format_for_prompt(search_results) if search_results else "No web results found."

    recipient_label = f"{name} ({role})" if name and role else (name or f"the team at {company}")

    prompt = f"""Generate TWO personalized outreach messages from David Lagarejo to {recipient_label} at {company}.

SENDER: David Lagarejo <{from_email}> — CEO, {from_domain}
- Physicist-engineer, NYC. LEED BD+C certified, IEEE reviewer
- Patent US2024/0077174 — ultrasonic non-invasive steam sensor (validated Ecopetrol, -30% energy reduction)
- Seeking: pilot clients, grants, strategic partners in US industrial/cleantech sector

WHY THIS CONTACT (internal research notes):
{reason or ('; '.join(why_yes) if why_yes else 'Strategic fit for patent commercialization')}

ANGLE:
{angle or 'Technology licensing or pilot program proposal'}

LIVE RESEARCH FINDINGS (what was found about this person/company right now — USE THIS):
{research_text}

RULES — HARD:
- Read the research findings above. Find ONE specific fact about this person or company and reference it explicitly in the email. If they recently published something, attended an event, or made a statement — use it.
- Do NOT write a generic email. If you cannot find a personal hook, use the most specific company fact from the research.
- Email: max 90 words, subject line must reference either the patent result OR a specific fact about the recipient's work.
- LinkedIn DM: max 55 words, even more direct, one concrete ask.
- No filler. No "I hope this finds you well". No "I'd love to". State facts, state the ask.
- End the email with a specific day + time slot for a call (e.g., "Thursday at 2pm EST").

Return JSON only — flat structure:
{{"subject": "...", "body": "...", "linkedin_dm": "..."}}"""

    raw = call_phi_sync([
        {"role": "system", "content": "Output JSON only. No markdown."},
        {"role": "user", "content": prompt},
    ], num_ctx=4096)

    result: dict = {"subject": "", "body": "", "linkedin_dm": ""}
    try:
        s = raw.find("{")
        e = raw.rfind("}") + 1
        if s >= 0 and e > s:
            parsed = json.loads(raw[s:e])
            if "subject" in parsed:
                result = parsed
            else:
                m1 = parsed.get("message_1") or {}
                m2 = parsed.get("message_2") or {}
                result = {
                    "subject":     m1.get("subject") or parsed.get("subject", ""),
                    "body":        m1.get("body") or parsed.get("body", ""),
                    "linkedin_dm": m2.get("linkedin_dm") or m2.get("body") or parsed.get("linkedin_dm", ""),
                }
    except Exception:
        result["body"] = raw[:600]
    return result
