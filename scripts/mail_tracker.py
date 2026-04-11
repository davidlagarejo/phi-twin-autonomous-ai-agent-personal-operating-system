#!/usr/bin/env python3
"""
scripts/mail_tracker.py — Apple Mail reply tracker for CRM contacts.

Enforced pipeline (code, not prompt):
  1. Load CRM contacts in status 'sent' or 'followup_due'.
  2. For each contact with a known email, query Apple Mail via AppleScript
     for any message from that address received after the contact's last_sent_at.
  3. If a reply is found → update CRM status to 'replied', save reply snippet as note.
  4. Notify macOS.

Called from proactive_loop.py every cycle.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

log = logging.getLogger("mail_tracker")

# Statuses that mean "we sent something and are waiting for a reply"
_WAITING_STATUSES = {"sent", "followup_due"}


def _applescript(script: str, timeout: int = 10) -> str:
    """Run an AppleScript and return stdout, empty string on failure."""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout.strip()
    except Exception as exc:
        log.debug("applescript_failed: %s", exc)
        return ""


def _mail_is_running() -> bool:
    out = _applescript('tell application "System Events" to (name of processes) contains "Mail"')
    return out.strip().lower() == "true"


def _find_replies_from(sender_email: str, after_iso: str) -> list[dict]:
    """
    Search Apple Mail for messages FROM sender_email received after after_iso.
    Returns list of {subject, date, snippet}.
    """
    # AppleScript: search all mailboxes for messages from this sender
    script = f"""
tell application "Mail"
    set resultList to {{}}
    set allAccounts to every account
    repeat with acct in allAccounts
        set allBoxes to every mailbox of acct
        repeat with mb in allBoxes
            try
                set msgs to (messages of mb whose sender contains "{sender_email}")
                repeat with m in msgs
                    set msgDate to date received of m
                    set resultList to resultList & {{(subject of m) & "|||" & (date string of msgDate) & "|||" & (extract name from (sender of m))}}
                end repeat
            end try
        end repeat
    end repeat
    return resultList
end tell
"""
    raw = _applescript(script, timeout=20)
    if not raw:
        return []

    replies = []
    for line in raw.split(","):
        line = line.strip()
        if "|||" not in line:
            continue
        parts = line.split("|||")
        if len(parts) >= 2:
            replies.append({"subject": parts[0].strip(), "date": parts[1].strip()})
    return replies


def check_mail_replies() -> int:
    """
    Main entry: check Apple Mail for replies from CRM contacts.
    Returns count of contacts updated to 'replied'.
    """
    if not _mail_is_running():
        log.info("mail_not_running — skipping reply check")
        return 0

    from tools.crm import get_all, update_status, add_note

    waiting = [c for c in get_all() if c.get("status") in _WAITING_STATUSES]
    if not waiting:
        log.info("no_contacts_waiting_for_reply")
        return 0

    updated = 0
    for contact in waiting:
        email = contact.get("email") or ""
        if not email or "@" not in email:
            continue

        after = contact.get("last_sent_at") or contact.get("created_at") or "2024-01-01"
        replies = _find_replies_from(email, after)

        if replies:
            contact_id = contact["id"]
            name = contact.get("name") or contact.get("company", "?")
            first_reply = replies[0]
            note = f"Reply received: '{first_reply['subject']}' on {first_reply['date']}"
            log.info("reply_found contact=%s email=%s", name, email)
            update_status(contact_id, "replied", note)
            add_note(contact_id, note)
            _notify_reply(name, contact.get("company", ""), first_reply["subject"])
            updated += 1

    log.info("mail_check_done updated=%d", updated)
    return updated


def _notify_reply(name: str, company: str, subject: str):
    label = f"{name} @ {company}" if company else name
    subject_safe = subject.replace('"', "'")[:80]
    script = (
        f'display notification "Respondió: {subject_safe}" '
        f'with title "Phi · Respuesta de {label[:50]}" '
        f'sound name "Glass"'
    )
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
    except Exception:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    n = check_mail_replies()
    print(f"Updated {n} contacts to 'replied'")
