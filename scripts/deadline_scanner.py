#!/usr/bin/env python3
"""
scripts/deadline_scanner.py — Deadline extractor (code only, no LLM).

For each dossier without a deadline:
  1. Fetch the entity's website (profile.website) or search SearXNG for
     "[name] application deadline [year]".
  2. Strip HTML, scan surrounding text for date patterns near
     deadline-related keywords.
  3. Parse best candidate to ISO 8601 date and write back to dossier.

Run standalone:   python3 scripts/deadline_scanner.py
API endpoint:     POST /api/scan_deadlines  (see api/routes/outreach.py)
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
import sys
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

log = logging.getLogger("phi.deadline_scanner")

_DOSSIER_DIR = BASE_DIR / "workspace" / "dossiers"

# ── Types eligible for deadline scanning ──────────────────────────────────────
_SCAN_TYPES = {"GRANT", "EVENT", "ORG"}

# ── Keyword patterns that signal a deadline is nearby ─────────────────────────
_DEADLINE_KEYWORDS = re.compile(
    r"(deadline|due date|due by|closes?|closing date|submission|applications? due|"
    r"apply by|apply before|submit by|last day|opens?.*closes?|rolling basis|"
    r"registration.*close|open.*call|letter of intent|LOI due)",
    re.IGNORECASE,
)

# ── Date patterns (month name or numeric) ─────────────────────────────────────
_MONTH_NAMES = (
    r"(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
)
_DATE_PATTERNS = [
    # "April 30, 2026" / "Apr 30 2026"
    re.compile(rf"\b({_MONTH_NAMES})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.IGNORECASE),
    # "30 April 2026"
    re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_NAMES}),?\s+(\d{{4}})\b", re.IGNORECASE),
    # "2026-04-30"
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
    # "04/30/2026" or "30/04/2026"
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"),
    # "Q1 2026", "Q2 2026" (fallback quarter)
    re.compile(r"\b(Q[1-4])\s+(\d{4})\b", re.IGNORECASE),
]

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_QUARTER_START = {"Q1": (1, 1), "Q2": (4, 1), "Q3": (7, 1), "Q4": (10, 1)}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
}


# ── HTML stripping ─────────────────────────────────────────────────────────────

def _strip_html(raw: str) -> str:
    """Remove tags, decode entities, collapse whitespace."""
    text = re.sub(r"<script[^>]*>.*?</script>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>",  " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Fetch URL ─────────────────────────────────────────────────────────────────

def _fetch(url: str, timeout: int = 10) -> str:
    """Fetch URL, return stripped text. Empty string on error."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = "utf-8"
            ct = resp.headers.get("Content-Type", "")
            m = re.search(r"charset=([^\s;]+)", ct)
            if m:
                charset = m.group(1)
            raw = resp.read(150_000).decode(charset, errors="replace")
        return _strip_html(raw)
    except Exception as exc:
        log.debug("fetch_failed url=%s err=%s", url, exc)
        return ""


# ── SearXNG search ─────────────────────────────────────────────────────────────

def _searxng_search(query: str, max_results: int = 5) -> list[str]:
    """Return list of URLs from local SearXNG. Empty list if unavailable."""
    base = os.environ.get("SEARXNG_URL", "").strip()
    if not base:
        return []
    try:
        params = urllib.parse.urlencode({"q": query, "format": "json", "language": "en"})
        req = urllib.request.Request(f"{base}/search?{params}", headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        return [r["url"] for r in (data.get("results") or [])[:max_results] if r.get("url")]
    except Exception as exc:
        log.debug("searxng_failed query=%r err=%s", query, exc)
        return []


# ── Date parsing ───────────────────────────────────────────────────────────────

def _parse_date_match(m: re.Match, pattern_idx: int) -> Optional[datetime]:
    """Convert a regex match to a datetime. Returns None if unparseable."""
    now = datetime.now(timezone.utc)
    try:
        g = m.groups()
        if pattern_idx == 0:   # "Month DD YYYY"
            month = _MONTH_MAP.get(g[0].lower())
            if not month:
                return None
            return datetime(int(g[2]), month, int(g[1]), tzinfo=timezone.utc)
        if pattern_idx == 1:   # "DD Month YYYY"
            month = _MONTH_MAP.get(g[1].lower())
            if not month:
                return None
            return datetime(int(g[2]), month, int(g[0]), tzinfo=timezone.utc)
        if pattern_idx == 2:   # "YYYY-MM-DD"
            return datetime(int(g[0]), int(g[1]), int(g[2]), tzinfo=timezone.utc)
        if pattern_idx == 3:   # "MM/DD/YYYY"
            y, a, b = int(g[2]), int(g[0]), int(g[1])
            # Guess MM/DD vs DD/MM
            if a > 12:
                a, b = b, a
            return datetime(y, a, b, tzinfo=timezone.utc)
        if pattern_idx == 4:   # "Q1 2026"
            q_str = g[0].upper()
            yr    = int(g[1])
            mo, day = _QUARTER_START.get(q_str, (1, 1))
            return datetime(yr, mo, day, tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return None
    return None


def _extract_deadline_from_text(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Scan text for dates near deadline keywords.
    Returns (iso_date_or_None, label_string_or_None).
    """
    now    = datetime.now(timezone.utc)
    window = 400  # chars to scan around each keyword hit

    candidates: list[tuple[datetime, str]] = []

    for km in _DEADLINE_KEYWORDS.finditer(text):
        start = max(0, km.start() - 50)
        end   = min(len(text), km.end() + window)
        snippet = text[start:end]

        for idx, pat in enumerate(_DATE_PATTERNS):
            for dm in pat.finditer(snippet):
                dt = _parse_date_match(dm, idx)
                if dt and dt > now:
                    label = dm.group(0).strip()
                    candidates.append((dt, label))

    if not candidates:
        return None, None

    # Pick the soonest future date
    candidates.sort(key=lambda x: x[0])
    best_dt, best_label = candidates[0]
    return best_dt.strftime("%Y-%m-%d"), best_label


# ── Per-dossier scan ───────────────────────────────────────────────────────────

def _scan_dossier(path: Path) -> dict:
    """
    Scan one dossier for a deadline. Returns result dict.
    Writes dossier in-place if a deadline is found.
    """
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "error", "file": path.name}

    entity_id = d.get("entity_id", path.stem)
    name      = d.get("name", "")
    etype     = d.get("type", "")
    year      = datetime.now(timezone.utc).year

    # Already has a deadline
    if d.get("deadline") or d.get("deadline_label"):
        return {"status": "skip", "entity_id": entity_id, "name": name, "reason": "already_has_deadline"}

    if etype not in _SCAN_TYPES:
        return {"status": "skip", "entity_id": entity_id, "name": name, "reason": f"type={etype}"}

    iso_date = label = None

    # ── Try website ─────────────────────────────────────────────────────────
    website = (d.get("profile") or {}).get("website") or ""
    if website:
        text = _fetch(website)
        if text:
            iso_date, label = _extract_deadline_from_text(text)
            log.debug("website_scan entity=%s found=%s", name, iso_date)

    # ── Try SearXNG ─────────────────────────────────────────────────────────
    if not iso_date:
        query = f'"{name}" deadline application {year}'
        urls  = _searxng_search(query, max_results=3)
        for url in urls:
            text = _fetch(url)
            if text:
                iso_date, label = _extract_deadline_from_text(text)
                if iso_date:
                    log.debug("search_scan entity=%s url=%s found=%s", name, url, iso_date)
                    break

    if not iso_date:
        return {"status": "not_found", "entity_id": entity_id, "name": name}

    # ── Write back ───────────────────────────────────────────────────────────
    d["deadline"]       = iso_date
    d["deadline_label"] = label or iso_date
    d["last_updated"]   = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("deadline_found entity=%s date=%s label=%r", name, iso_date, label)

    return {"status": "found", "entity_id": entity_id, "name": name,
            "deadline": iso_date, "deadline_label": label or iso_date}


# ── Public entry point ────────────────────────────────────────────────────────

def scan_all_deadlines(entity_ids: list[str] | None = None) -> list[dict]:
    """
    Scan all dossiers (or a subset by entity_id) for deadlines.
    Returns list of result dicts.
    """
    if not _DOSSIER_DIR.exists():
        return []

    results = []
    for path in sorted(_DOSSIER_DIR.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if entity_ids and d.get("entity_id") not in entity_ids:
            continue
        result = _scan_dossier(path)
        results.append(result)
        log.info("scan entity=%s status=%s", result.get("name"), result.get("status"))

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    results = scan_all_deadlines()
    found = [r for r in results if r["status"] == "found"]
    skipped = [r for r in results if r["status"] == "skip"]
    not_found = [r for r in results if r["status"] == "not_found"]
    print(f"\nFound:     {len(found)}")
    print(f"Skipped:   {len(skipped)}")
    print(f"Not found: {len(not_found)}")
    for r in found:
        print(f"  ✓ {r['name']}: {r['deadline']} ({r['deadline_label']})")
    for r in not_found:
        print(f"  · {r['name']}: no deadline detected")
