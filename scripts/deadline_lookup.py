#!/usr/bin/env python3
"""
scripts/deadline_lookup.py — Deadline lookup via Claude API (structured data only).

For each dossier of type GRANT or EVENT without a deadline:
  - Calls Claude API asking ONLY for the deadline date as JSON
  - No prose, no explanations — just {deadline, deadline_label, confidence, source}
  - Writes result directly to dossier file
  - Skips ORG types (they don't have application deadlines)

Run standalone:  python3 scripts/deadline_lookup.py
Called from:     api/routes/outreach.py  POST /api/scan_deadlines
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

log = logging.getLogger("phi.deadline_lookup")

_DOSSIER_DIR  = BASE_DIR / "workspace" / "dossiers"
_DEADLINE_TYPES = {"GRANT", "EVENT"}

_MODEL   = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
_URL     = "https://api.anthropic.com/v1/messages"
_TODAY   = datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _ask_deadline(name: str, description: str, entity_type: str) -> dict:
    """
    Ask Claude for the next application deadline for this grant/event.
    Returns parsed dict or {} on failure.
    """
    if not _API_KEY:
        log.warning("CLAUDE_API_KEY not set — skipping deadline lookup")
        return {}

    system = (
        "You are a data extraction tool. Return ONLY a JSON object, no prose. "
        "If you don't know the answer with reasonable confidence, set confidence to 'low' "
        "and estimate based on typical cycles."
    )

    user = (
        f"What is the next upcoming application deadline for: {name}\n"
        f"Type: {entity_type}\n"
        f"Description: {description[:300]}\n"
        f"Today: {_TODAY}\n\n"
        "Return ONLY this JSON (no markdown, no explanation):\n"
        '{"deadline": "YYYY-MM-DD or null", '
        '"deadline_label": "human-readable deadline string or null", '
        '"confidence": "high|medium|low", '
        '"notes": "one sentence about the deadline cycle"}'
    )

    try:
        resp = httpx.post(
            _URL,
            headers={
                "x-api-key":         _API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      _MODEL,
                "max_tokens": 256,
                "system":     system,
                "messages":   [{"role": "user", "content": user}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"].strip()
        # Strip markdown fences if any
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        return json.loads(raw)
    except Exception as exc:
        log.warning("claude_deadline_failed entity=%s err=%s", name, exc)
        return {}


def lookup_deadlines(entity_ids: list[str] | None = None) -> list[dict]:
    """
    Lookup deadlines for all GRANT/EVENT dossiers without one.
    Returns list of result dicts.
    """
    if not _DOSSIER_DIR.exists():
        return []

    results = []
    for path in sorted(_DOSSIER_DIR.glob("*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        entity_id = d.get("entity_id", path.stem)
        name      = d.get("name", "")
        etype     = d.get("type", "")

        if entity_ids and entity_id not in entity_ids:
            continue

        # Skip if already has a deadline
        if d.get("deadline") or d.get("deadline_label"):
            results.append({"status": "skip", "entity_id": entity_id, "name": name,
                            "reason": "already_has_deadline"})
            continue

        # Only GRANTs and EVENTs have deadlines
        if etype not in _DEADLINE_TYPES:
            results.append({"status": "skip", "entity_id": entity_id, "name": name,
                            "reason": f"type={etype}_no_deadline"})
            continue

        log.info("looking_up entity=%s type=%s", name, etype)
        data = _ask_deadline(name, d.get("description", ""), etype)

        if not data:
            results.append({"status": "error", "entity_id": entity_id, "name": name})
            continue

        iso_date      = data.get("deadline")
        label         = data.get("deadline_label")
        confidence    = data.get("confidence", "low")
        notes         = data.get("notes", "")

        # Validate ISO date format
        if iso_date and iso_date != "null":
            try:
                dt = datetime.strptime(iso_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                if dt < datetime.now(timezone.utc):
                    # Past date — skip
                    log.info("past_deadline_skipped entity=%s date=%s", name, iso_date)
                    iso_date = None
            except ValueError:
                iso_date = None
        else:
            iso_date = None

        if not iso_date and not label:
            results.append({"status": "not_found", "entity_id": entity_id, "name": name,
                            "confidence": confidence, "notes": notes})
            continue

        # Write back to dossier
        if iso_date:
            d["deadline"]       = iso_date
        if label and label != "null":
            d["deadline_label"] = label
        elif iso_date:
            d["deadline_label"] = iso_date
        d["last_updated"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

        log.info("deadline_saved entity=%s date=%s label=%r confidence=%s",
                 name, iso_date, label, confidence)
        results.append({
            "status":       "found",
            "entity_id":    entity_id,
            "name":         name,
            "deadline":     iso_date,
            "deadline_label": label,
            "confidence":   confidence,
            "notes":        notes,
        })

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = lookup_deadlines()
    found     = [r for r in results if r["status"] == "found"]
    not_found = [r for r in results if r["status"] == "not_found"]
    print(f"\nFound: {len(found)}  Not found: {len(not_found)}")
    for r in found:
        print(f"  ✓ {r['name']}: {r['deadline']} — {r.get('deadline_label','')} [{r['confidence']}]")
        if r.get("notes"):
            print(f"    {r['notes']}")
    for r in not_found:
        print(f"  · {r['name']}: {r.get('notes','no info')}")
