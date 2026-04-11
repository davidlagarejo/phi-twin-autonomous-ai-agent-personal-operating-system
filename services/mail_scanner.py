"""
services/mail_scanner.py — Scan Apple Mail inbox for relevant opportunities.

Reads recent emails via osascript, matches against:
  - Known timeline entities (investors, grants, orgs, jobs) by name
  - Keyword categories: INVESTMENT, GRANT, JOB, REPLY

Returns structured matches the chat can use to suggest replies.

No LLM — pure script matching. Safe to run frequently.
"""
from __future__ import annotations

import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

_log = logging.getLogger("phi.mail_scanner")

_BASE = Path(__file__).parent.parent

# ── Keyword sets by category ───────────────────────────────────────────────────

_KW_INVESTMENT = {
    "investor", "invest", "funding", "fund", "venture", "capital", "vc", "angel",
    "pre-seed", "preseed", "seed", "series", "pitch", "term sheet", "due diligence",
    "portfolio", "accelerator", "incubator", "startup", "raise", "round",
    "cleantech", "climatetech", "climate tech", "energy",
}

_KW_GRANT = {
    "grant", "sbir", "sttr", "doe", "nsf", "nist", "award", "solicitation",
    "proposal", "application", "rfp", "funding opportunity", "government",
    "department of energy", "epa", "arpa-e", "arpa", "phase i", "phase ii",
    "contract", "cooperative agreement",
}

_KW_JOB = {
    "job", "position", "role", "opportunity", "hiring", "interview", "application",
    "resume", "cv", "recruiter", "talent", "offer", "salary", "compensation",
    "applied", "thank you for applying", "next steps", "phone screen", "technical",
}

_KW_REPLY_SIGNALS = {
    "re:", "reply", "following up", "following-up", "checking in", "follow up",
    "as discussed", "per our conversation", "thank you for your email",
    "thanks for reaching out",
}


def _is_mail_running() -> bool:
    """Check if Apple Mail is open."""
    try:
        r = subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to (name of processes) contains "Mail"'],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip().lower() == "true"
    except Exception:
        return False


def _run_applescript(script: str, timeout: int = 25) -> str:
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:
        _log.warning("applescript failed: %s", e)
        return ""


def _fetch_inbox_messages(days: int = 7, max_messages: int = 60) -> list[dict]:
    """
    Read recent inbox messages via AppleScript.
    Returns list of {subject, sender, snippet, date_str}.
    """
    script = f"""
tell application "Mail"
    set cutoff to (current date) - ({days} * days)
    set hits to {{}}
    set msgCount to 0
    try
        set inbox to mailbox "INBOX" of first account
        set msgs to (messages of inbox whose date received >= cutoff)
        repeat with m in msgs
            if msgCount >= {max_messages} then exit repeat
            try
                set snip to text 1 thru 400 of (content of m)
            on error
                set snip to ""
            end try
            set end of hits to ((subject of m) & "|||" & (sender of m) & "|||" & snip & "|||" & ((date received of m) as string))
            set msgCount to msgCount + 1
        end repeat
    end try
    return hits
end tell
"""
    raw = _run_applescript(script)
    messages = []
    if not raw:
        return messages

    for line in raw.split(",\n"):
        line = line.strip().strip("{}")
        if "|||" not in line:
            continue
        parts = line.split("|||")
        messages.append({
            "subject":  parts[0].strip() if len(parts) > 0 else "",
            "sender":   parts[1].strip() if len(parts) > 1 else "",
            "snippet":  parts[2].strip()[:400] if len(parts) > 2 else "",
            "date_str": parts[3].strip() if len(parts) > 3 else "",
        })
    return messages


def _load_entity_names() -> list[tuple[str, str]]:
    """
    Return [(entity_id, name)] for all dossiers in data/dossiers/.
    Used to match email senders/subjects against known entities.
    """
    dossiers_dir = _BASE / "data" / "dossiers"
    if not dossiers_dir.exists():
        return []
    pairs = []
    import json
    for p in dossiers_dir.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            name = d.get("name") or ""
            eid  = d.get("entity_id") or p.stem
            if name and len(name) >= 4:
                pairs.append((eid, name.lower()))
                # Also add aliases
                for alias in d.get("aliases", []):
                    if alias and len(alias) >= 4:
                        pairs.append((eid, alias.lower()))
        except Exception:
            pass
    return pairs


def _categorize(subject: str, sender: str, snippet: str) -> str | None:
    """
    Return category string or None if not relevant.
    Priority: REPLY > INVESTMENT > GRANT > JOB
    """
    combined = f"{subject} {sender} {snippet}".lower()

    # Check for known entity match first (handled separately)
    # REPLY signals
    if any(kw in combined for kw in _KW_REPLY_SIGNALS):
        return "REPLY"
    if any(kw in combined for kw in _KW_INVESTMENT):
        return "INVESTMENT"
    if any(kw in combined for kw in _KW_GRANT):
        return "GRANT"
    if any(kw in combined for kw in _KW_JOB):
        return "JOB"
    return None


def _match_entity(combined_lower: str, entity_pairs: list[tuple[str, str]]) -> str | None:
    """Return entity_id if any entity name appears in the text, else None."""
    for eid, name in entity_pairs:
        # Require word-boundary-like match for short names
        if len(name) >= 6 and name in combined_lower:
            return eid
        elif len(name) >= 4 and re.search(r'\b' + re.escape(name) + r'\b', combined_lower):
            return eid
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def scan_inbox(days: int = 7, max_messages: int = 60) -> dict:
    """
    Scan Apple Mail inbox for emails relevant to timeline opportunities.

    Returns:
    {
        "mail_available": bool,
        "total_scanned": int,
        "matches": [
            {
                "subject": str,
                "sender": str,
                "snippet": str,
                "date_str": str,
                "category": "INVESTMENT"|"GRANT"|"JOB"|"REPLY"|"ENTITY",
                "entity_id": str|None,   # linked timeline entity if found
                "entity_name": str|None,
            }, ...
        ]
    }
    """
    if not _is_mail_running():
        return {"mail_available": False, "total_scanned": 0, "matches": []}

    messages  = _fetch_inbox_messages(days=days, max_messages=max_messages)
    entity_pairs = _load_entity_names()
    # Build reverse map entity_id → display name
    import json
    _ename_map: dict[str, str] = {}
    dossiers_dir = _BASE / "data" / "dossiers"
    if dossiers_dir.exists():
        for p in dossiers_dir.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                _ename_map[d.get("entity_id", p.stem)] = d.get("name", "")
            except Exception:
                pass

    matches = []
    for msg in messages:
        subject = msg["subject"]
        sender  = msg["sender"]
        snippet = msg["snippet"]
        combined_lower = f"{subject} {sender} {snippet}".lower()

        # Entity match first (highest confidence)
        entity_id = _match_entity(combined_lower, entity_pairs)
        if entity_id:
            matches.append({
                **msg,
                "category":    "ENTITY",
                "entity_id":   entity_id,
                "entity_name": _ename_map.get(entity_id, ""),
            })
            continue

        # Category match
        cat = _categorize(subject, sender, snippet)
        if cat:
            matches.append({
                **msg,
                "category":    cat,
                "entity_id":   None,
                "entity_name": None,
            })

    return {
        "mail_available": True,
        "total_scanned":  len(messages),
        "matches":        matches,
    }


def format_for_chat(scan_result: dict) -> str:
    """
    Format scan results as a readable Phi chat message (in English, to be translated).
    """
    if not scan_result.get("mail_available"):
        return "Apple Mail is not running — can't scan inbox."

    matches = scan_result.get("matches", [])
    total   = scan_result.get("total_scanned", 0)

    if not matches:
        return f"Scanned {total} recent emails — no relevant matches found."

    lines = [f"Found **{len(matches)} relevant email(s)** in your inbox (scanned {total} total):\n"]
    for m in matches[:8]:
        cat_label = {
            "ENTITY":     "📌 Known contact",
            "INVESTMENT": "💰 Investment",
            "GRANT":      "📋 Grant",
            "JOB":        "💼 Job",
            "REPLY":      "↩️ Reply",
        }.get(m["category"], "📧")
        entity_note = f" — **{m['entity_name']}**" if m.get("entity_name") else ""
        lines.append(
            f"- {cat_label}{entity_note}: **{m['subject'][:70]}** "
            f"(from: {m['sender'][:50]})\n"
            f"  _{m['snippet'][:120].strip()}..._"
        )

    lines.append("\nTo draft a reply, say: **\"draft reply to [sender/topic]\"** — in English or Spanish.")
    return "\n".join(lines)
