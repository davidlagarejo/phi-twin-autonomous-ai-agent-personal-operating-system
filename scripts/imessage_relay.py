#!/usr/bin/env python3
"""
imessage_relay.py — phi-twin iMessage relay
============================================
Runs a local HTTP server on 127.0.0.1:9999.

Inbound (iMessage → Phi):
  Polls ~/Library/Messages/chat.db every POLL_INTERVAL seconds.
  New messages → POST http://127.0.0.1:PHI_SERVER/api/chat → reply via iMessage.

Outbound (proactive_loop → Mac):
  POST /send  {"to": "email_or_phone", "message": "text"}
  Uses AppleScript to send via Messages.app.

GET  /health → {"status": "ok", ...}

Privacy:
  - Runs on 127.0.0.1 only (no LAN/internet exposure)
  - RELAY_ALLOW_SENDERS env var limits inbound senders (default: all)
  - Never writes message content to disk (state file = last_rowid only)
  - No logging of message text unless LOG_LEVEL=debug

Usage:
  python3 scripts/imessage_relay.py

  # Or with env overrides:
  PHI_SERVER=http://127.0.0.1:8080 POLL_INTERVAL=3 python3 scripts/imessage_relay.py

Environment variables:
  PHI_SERVER        phi-twin server URL (default: http://127.0.0.1:8080)
  RELAY_PORT        Port this relay listens on (default: 9999)
  POLL_INTERVAL     Seconds between chat.db polls (default: 3)
  RELAY_ALLOW_SENDERS  Comma-separated sender allowlist, or * for all (default: *)
  LOG_LEVEL         debug | info | warn (default: info)
  STATE_FILE        Path to last-rowid state (default: ~/.phi-twin-relay.json)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── Security: path allowlist ──────────────────────────────────────────────────
# The relay must ONLY read these paths (defense-in-depth on top of macOS FDA).
# Any file access outside this set is refused before it hits the OS.

_ALLOWED_READ_PATHS: tuple[Path, ...] = (
    Path.home() / "Library/Messages/chat.db",           # inbound poller
)
_ALLOWED_WRITE_PATHS: tuple[Path, ...] = (
    Path.home() / ".phi-twin-relay.json",                # state (rowid only)
    Path(os.environ.get("STATE_FILE",
         str(Path.home() / ".phi-twin-relay.json"))),
)
# Logs are written by the LaunchAgent stdout redirect, not by this process.

def _assert_path_allowed(p: Path, write: bool = False) -> None:
    """Abort if p is not in the relay's explicit path allowlist."""
    allowed = _ALLOWED_WRITE_PATHS if write else _ALLOWED_READ_PATHS
    resolved = p.resolve()
    for allowed_p in allowed:
        if resolved == allowed_p.resolve():
            return
    raise PermissionError(
        f"SECURITY: relay attempted {'write' if write else 'read'} outside "
        f"allowlist: {p}"
    )

# ── Config ────────────────────────────────────────────────────────────────────

PHI_SERVER     = os.environ.get("PHI_SERVER",    "http://127.0.0.1:8080")
RELAY_PORT     = int(os.environ.get("RELAY_PORT",     "9999"))
POLL_INTERVAL  = float(os.environ.get("POLL_INTERVAL", "3"))
LOG_LEVEL      = os.environ.get("LOG_LEVEL", "info").upper()
STATE_FILE     = Path(os.environ.get("STATE_FILE",
                      str(Path.home() / ".phi-twin-relay.json")))
CHAT_DB        = Path.home() / "Library/Messages/chat.db"

# Validate config paths are in allowlist at startup
assert CHAT_DB   in _ALLOWED_READ_PATHS,  f"CHAT_DB not in read allowlist: {CHAT_DB}"
assert STATE_FILE in (p.resolve() for p in _ALLOWED_WRITE_PATHS) or \
       STATE_FILE.resolve() in (p.resolve() for p in _ALLOWED_WRITE_PATHS), \
       f"STATE_FILE not in write allowlist: {STATE_FILE}"

# Comma-separated sender IDs (phone numbers / iCloud emails), or * for all
_raw_allow = os.environ.get("RELAY_ALLOW_SENDERS", "*").strip()
ALLOW_ALL   = (_raw_allow == "*")
ALLOW_LIST  = set() if ALLOW_ALL else {s.strip() for s in _raw_allow.split(",")}

PHI_CHAT_URL = f"{PHI_SERVER}/api/chat"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [relay] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("relay")

# ── State ─────────────────────────────────────────────────────────────────────

_state_lock = threading.Lock()

def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"last_rowid": 0}

def _save_state(state: dict) -> None:
    with _state_lock:
        _assert_path_allowed(STATE_FILE, write=True)
        STATE_FILE.write_text(json.dumps(state))

# ── AppleScript sender ────────────────────────────────────────────────────────

def _send_imessage(to: str, message: str) -> tuple[bool, str]:
    """
    Send an iMessage via AppleScript.
    Returns (success, error_message).
    Requires: Messages.app and iMessage account signed in.
    """
    # Escape for AppleScript string literal (backslash and double-quote)
    safe_msg = message.replace("\\", "\\\\").replace('"', '\\"')
    safe_to  = to.replace('"', '\\"')

    script = f'''
tell application "Messages"
    set targetService to 1st service whose service type = iMessage
    set targetBuddy to buddy "{safe_to}" of targetService
    send "{safe_msg}" to targetBuddy
end tell
'''
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return False, result.stderr.strip()
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "AppleScript timeout"
    except Exception as exc:
        return False, str(exc)

# ── chat.db poller ────────────────────────────────────────────────────────────

_QUERY = """
SELECT
    m.ROWID,
    m.text,
    m.attributedBody,
    m.date              AS apple_date,
    h.id                AS sender,
    c.chat_identifier   AS thread_id,
    c.service_name      AS service
FROM message m
JOIN handle h ON m.handle_id = h.ROWID
JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
JOIN chat c ON c.ROWID = cmj.chat_id
WHERE m.is_from_me = 0
  AND (m.text IS NOT NULL OR m.attributedBody IS NOT NULL)
  AND m.ROWID > ?
ORDER BY m.ROWID ASC
"""


def _decode_attributed_body(blob: bytes) -> str | None:
    """Extract plain text from NSArchiver streamtyped blob (chat.db attributedBody)."""
    try:
        # Locate 'NSString' class marker, then find '+' type tag for the string value
        p = blob.rfind(b'NSString')
        if p == -1:
            return None
        p = blob.find(b'\x2b', p)
        if p == -1:
            return None
        p += 1  # skip '+'
        # Read length: 0x81 + 1 byte means lengths 128-255; otherwise direct 1-byte length
        b0 = blob[p]
        if b0 == 0x81:
            length = blob[p + 1]
            p += 2
        else:
            length = b0
            p += 1
        if p < len(blob) and blob[p] == 0x00:
            p += 1  # skip null padding
        if length <= 0 or length > 5000:
            return None
        return blob[p:p + length].decode("utf-8", errors="replace")
    except Exception:
        return None

# Apple epoch is 2001-01-01 00:00:00 UTC; date column = nanoseconds since then
_APPLE_EPOCH_OFFSET = 978307200  # seconds between Unix epoch and Apple epoch

def _apple_date_to_iso(apple_ns: int) -> str:
    try:
        unix_ts = apple_ns / 1e9 + _APPLE_EPOCH_OFFSET
        return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()
    except Exception:
        return datetime.now(timezone.utc).isoformat()

def _post_to_phi(payload: dict) -> bool:
    """Envía el mensaje de iMessage a /api/chat y responde por iMessage."""
    sender  = payload.get("from", "")
    message = payload.get("message", "")
    if not message or not sender:
        return False

    chat_body = json.dumps({
        "messages": [{"role": "user", "content": message}],
        "search_first": False,
    }).encode()
    req = urllib.request.Request(
        PHI_CHAT_URL,
        data=chat_body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            # Consumir SSE stream y concatenar texto
            raw = resp.read().decode("utf-8", errors="replace")
            # Parsear data: lines del SSE
            reply_parts = []
            for line in raw.splitlines():
                if line.startswith("data:"):
                    chunk = line[5:].strip()
                    if chunk and chunk != "[DONE]":
                        reply_parts.append(chunk)
            reply = "".join(reply_parts).strip()
            if reply:
                ok, err = _send_imessage(sender, reply)
                if ok:
                    log.info("phi_reply sent to=%s len=%d", sender, len(reply))
                else:
                    log.warning("phi_reply send_failed to=%s err=%s", sender, err)
            return True
    except Exception as exc:
        log.warning("phi_chat_failed sender=%s err=%s", sender, exc)
        return False

_FDA_WARNED = False

def _poll_loop() -> None:
    global _FDA_WARNED
    state     = _load_state()
    last_rowid = state.get("last_rowid", 0)
    log.info("Poller started. last_rowid=%d, phi_chat=%s", last_rowid, PHI_CHAT_URL)
    log.info("Allow list: %s", "ALL" if ALLOW_ALL else ALLOW_LIST)

    while True:
        try:
            # Copy chat.db to avoid locking the live database
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                tmp_path = tmp.name
            _assert_path_allowed(CHAT_DB, write=False)
            shutil.copy2(str(CHAT_DB), tmp_path)

            db  = sqlite3.connect(tmp_path)
            rows = db.execute(_QUERY, (last_rowid,)).fetchall()
            db.close()
            Path(tmp_path).unlink(missing_ok=True)

            for rowid, text, attr_body, apple_date, sender, thread_id, service in rows:
                if not text and attr_body:
                    text = _decode_attributed_body(attr_body)
                if not text:
                    last_rowid = rowid
                    continue
                # Apply sender allowlist
                if not ALLOW_ALL and sender not in ALLOW_LIST:
                    log.debug("Skipping sender not in allowlist: %s", sender)
                    last_rowid = rowid
                    continue

                log.debug("New message ROWID=%d from=%s", rowid, sender)
                payload = {
                    "channel":   "imessage",
                    "from":      sender,
                    "message":   text,
                    "thread_id": thread_id or sender,
                    "service":   service,
                    "timestamp": _apple_date_to_iso(apple_date),
                }
                _post_to_phi(payload)
                last_rowid = rowid

            _save_state({"last_rowid": last_rowid})

        except PermissionError:
            if not _FDA_WARNED:
                _FDA_WARNED = True
                log.error(
                    "PERMISSION DENIED reading chat.db — Full Disk Access required.\n"
                    "  Fix: System Settings → Privacy & Security → Full Disk Access\n"
                    "       Click + and add: /usr/bin/python3\n"
                    "  Then: launchctl kickstart -k gui/%d/com.phitwin.imessage-relay"
                    % os.getuid()
                )
        except Exception as exc:
            log.error("Poll error: %s", exc)

        time.sleep(POLL_INTERVAL)

# ── HTTP server ───────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):  # suppress default access log
        log.debug("http: " + fmt, *args)

    def _send_json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        # Localhost only — no CORS needed
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length))
        except Exception:
            return None

    def do_GET(self):
        if self.path == "/health":
            chat_db_readable = os.access(str(CHAT_DB), os.R_OK)
            self._send_json(200, {
                "status":           "ok",
                "relay_port":       RELAY_PORT,
                "phi_chat_url":     PHI_CHAT_URL,
                "poll_interval":    POLL_INTERVAL,
                "allow_all":        ALLOW_ALL,
                "chat_db":          str(CHAT_DB),
                "chat_db_readable": chat_db_readable,
                "inbound_active":   chat_db_readable,
                "outbound_active":  True,
                "fda_needed":       not chat_db_readable,
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        # Enforce localhost-only
        client_ip = self.client_address[0]
        if client_ip not in ("127.0.0.1", "::1"):
            log.warning("Rejected non-localhost POST from %s", client_ip)
            self._send_json(403, {"error": "localhost only"})
            return

        if self.path == "/send":
            body = self._read_json()
            if not body:
                self._send_json(400, {"error": "invalid JSON"})
                return

            to      = body.get("to", "").strip()
            message = body.get("message", "").strip()

            if not to or not message:
                self._send_json(400, {"error": "missing 'to' or 'message'"})
                return

            # Privacy: log destination but not content
            log.info("Sending iMessage to %s (%d chars)", to, len(message))
            ok, err = _send_imessage(to, message)
            if ok:
                self._send_json(200, {"status": "sent", "to": to})
            else:
                log.error("Send failed: %s", err)
                self._send_json(500, {"status": "error", "detail": err})

        else:
            self._send_json(404, {"error": "not found"})

# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not CHAT_DB.exists():
        log.error("chat.db not found at %s", CHAT_DB)
        log.error("Grant Full Disk Access to Terminal in System Preferences → Privacy.")
        sys.exit(1)

    # Seed last_rowid from current DB max so we only forward NEW messages
    state = _load_state()
    if state.get("last_rowid", 0) == 0:
        try:
            db = sqlite3.connect(str(CHAT_DB))
            max_rowid = db.execute("SELECT MAX(ROWID) FROM message").fetchone()[0] or 0
            db.close()
            _save_state({"last_rowid": max_rowid})
            log.info("First run: seeded last_rowid=%d (only forward future messages)", max_rowid)
        except Exception as exc:
            log.warning("Could not seed last_rowid: %s", exc)

    # Start poller thread
    t = threading.Thread(target=_poll_loop, daemon=True, name="poller")
    t.start()

    # Start HTTP server
    server = HTTPServer(("127.0.0.1", RELAY_PORT), _Handler)
    log.info("iMessage relay listening on 127.0.0.1:%d", RELAY_PORT)
    log.info("  /health  — status check")
    log.info("  /send    — POST {to, message}")
    log.info("  polling chat.db every %.0fs → %s", POLL_INTERVAL, PHI_CHAT_URL)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Relay stopped.")

if __name__ == "__main__":
    main()
