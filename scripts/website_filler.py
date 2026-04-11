#!/usr/bin/env python3
"""
scripts/website_filler.py — Populate profile.website for dossiers missing it.

Strategy (code only, no LLM synthesis):
  1. Try _find_website_for_entity() — web search via SearXNG (if configured)
  2. Fallback: ask Claude API for just the URL as structured JSON
  3. Write URL directly to dossier profile.website

Run standalone:  python3 scripts/website_filler.py
Called from:     proactive_loop and /api/scan_deadlines
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

BASE_DIR     = Path(__file__).resolve().parent.parent
DOSSIER_DIR  = BASE_DIR / "workspace" / "dossiers"
sys.path.insert(0, str(BASE_DIR))

log = logging.getLogger("phi.website_filler")

_API_KEY = os.environ.get("CLAUDE_API_KEY", "")
_MODEL   = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
_URL     = "https://api.anthropic.com/v1/messages"


def _ask_claude_website(name: str, entity_type: str, description: str) -> str:
    """Ask Claude for just the official website URL. Returns URL or ''."""
    if not _API_KEY:
        return ""
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
                "max_tokens": 64,
                "system":     "Return ONLY a JSON object. No prose.",
                "messages":   [{
                    "role":    "user",
                    "content": (
                        f"What is the official website URL for: {name}\n"
                        f"Type: {entity_type}\n"
                        f"Description: {description[:200]}\n\n"
                        'Return ONLY: {"website": "https://..."} or {"website": null}'
                    ),
                }],
            },
            timeout=20,
        )
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"].strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
        data = json.loads(raw)
        url = data.get("website") or ""
        if url and url != "null" and url.startswith("http"):
            return url
    except Exception as exc:
        log.debug("claude_website_failed name=%s err=%s", name, exc)
    return ""


def fill_missing_websites(entity_ids: list[str] | None = None) -> list[dict]:
    """
    Populate profile.website for dossiers that are missing it.
    Returns list of result dicts.
    """
    if not DOSSIER_DIR.exists():
        return []

    # Try to use the research engine's built-in website finder first
    try:
        from tools.research_engine import _find_website_for_entity
        has_search = True
    except Exception:
        has_search = False

    results = []
    for path in sorted(DOSSIER_DIR.glob("*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        entity_id = d.get("entity_id", path.stem)
        name      = d.get("name", "")
        etype     = d.get("type", "")
        desc      = d.get("description", "")

        if entity_ids and entity_id not in entity_ids:
            continue

        profile = d.setdefault("profile", {})
        if profile.get("website"):
            results.append({"status": "skip", "entity_id": entity_id, "name": name})
            continue

        url = ""

        # Strategy 1: web search via SearXNG
        if has_search:
            try:
                url = _find_website_for_entity(name, etype)
            except Exception as e:
                log.debug("search_website_failed name=%s err=%s", name, e)

        # Strategy 2: Claude API structured lookup
        if not url:
            url = _ask_claude_website(name, etype, desc)

        if not url:
            results.append({"status": "not_found", "entity_id": entity_id, "name": name})
            continue

        profile["website"] = url
        d["last_updated"]  = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("website_saved entity=%s url=%s", name, url)
        results.append({"status": "found", "entity_id": entity_id, "name": name, "website": url})

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = fill_missing_websites()
    found     = [r for r in results if r["status"] == "found"]
    not_found = [r for r in results if r["status"] == "not_found"]
    skipped   = [r for r in results if r["status"] == "skip"]
    print(f"\nFound: {len(found)}  Not found: {len(not_found)}  Skipped: {len(skipped)}")
    for r in found:
        print(f"  ✓ {r['name']}: {r['website']}")
    for r in not_found:
        print(f"  · {r['name']}: no website found")
