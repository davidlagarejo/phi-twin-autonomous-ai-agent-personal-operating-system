"""
tools/crm.py — Contact Relationship Manager

Single responsibility: CRUD for contacts with status tracking and draft caching.

Backed by AtomicJSONStore (tools/store_base.py):
  - O(1) get by contact_id (in-memory index).
  - O(1) get by entity_id (secondary index).
  - O(1) dedup by entity_id + name (name index).
  - Atomic disk writes — no partial state on crash.
  - Loads from disk once; all subsequent reads are in-memory.

Public interface is identical to the previous version — no callers need to change.

Status flow:
  new → researching → ready → drafted → sent → followup_due → replied → meeting | closed
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from tools.store_base import AtomicJSONStore

_CONTACTS_FILE = Path(__file__).parent.parent / "data" / "contacts.json"

VALID_STATUSES = {
    "new",           # Entity found, no contact person yet
    "researching",   # Actively searching for contact info
    "ready",         # Has email or LinkedIn — ready to draft
    "drafted",       # Draft email/DM generated
    "sent",          # Email/DM sent
    "followup_due",  # Follow-up date reached
    "replied",       # Got a reply
    "meeting",       # Meeting scheduled
    "closed",        # Deal done or not viable
}

STATUS_ORDER = {
    "replied": 0, "meeting": 1, "followup_due": 2,
    "sent": 3, "drafted": 4, "ready": 5,
    "researching": 6, "new": 7, "closed": 8,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()[:19] + "Z"


# ── Store with secondary indices ──────────────────────────────────────────────

class _CRMStore(AtomicJSONStore):
    """
    CRM store with two secondary indices maintained in memory:
      _entity_index: entity_id → set of contact_ids  (get_by_entity is O(k))
      _name_index:   "entity_id:name_lower" → contact_id  (dedup is O(1))
    Both indices stay in sync with every put() and delete() via the hooks.
    """

    def __init__(self, path: Path) -> None:
        self._entity_index: dict[str, set[str]] = {}
        self._name_index: dict[str, str] = {}
        super().__init__(path, id_field="id")

    def _on_load(self, records: list[dict]) -> None:
        self._entity_index = {}
        self._name_index = {}
        for r in records:
            self._index_add(r)

    def _on_record_added(self, record: dict) -> None:
        self._index_add(record)

    def _on_record_removed(self, record: dict) -> None:
        self._index_remove(record)

    def _index_add(self, r: dict) -> None:
        cid = r.get("id")
        eid = r.get("entity_id")
        if cid and eid:
            self._entity_index.setdefault(eid, set()).add(cid)
        name = (r.get("name") or "").strip().lower()
        if cid and eid and name:
            self._name_index[f"{eid}:{name}"] = cid

    def _index_remove(self, r: dict) -> None:
        cid = r.get("id")
        eid = r.get("entity_id")
        if cid and eid:
            s = self._entity_index.get(eid, set())
            s.discard(cid)
            if not s:
                self._entity_index.pop(eid, None)
        name = (r.get("name") or "").strip().lower()
        if eid and name:
            self._name_index.pop(f"{eid}:{name}", None)

    def ids_for_entity(self, entity_id: str) -> list[str]:
        with self._lock:
            self._ensure_loaded()
            return list(self._entity_index.get(entity_id, set()))

    def id_for_name(self, entity_id: str, name: str) -> str | None:
        with self._lock:
            self._ensure_loaded()
            return self._name_index.get(f"{entity_id}:{name.strip().lower()}")


# Module-level singleton — loaded once per process
_store = _CRMStore(_CONTACTS_FILE)


# ── Public API (identical interface to previous version) ──────────────────────

def get_all(status: str = "") -> list:
    """Return all contacts, optionally filtered by status, sorted by priority."""
    if status:
        contacts = _store.filter(lambda c: c.get("status") == status)
    else:
        contacts = _store.all()
    contacts.sort(key=lambda c: (
        STATUS_ORDER.get(c.get("status", "new"), 9),
        -(c.get("fit_score") or 0),
        c.get("updated_at") or "",
    ))
    return contacts


def get(contact_id: str) -> dict | None:
    """O(1) lookup by contact id."""
    return _store.get(contact_id)


def get_by_entity(entity_id: str) -> list:
    """Return all contacts linked to an entity_id. O(k) where k = contacts for entity."""
    ids = _store.ids_for_entity(entity_id)
    return [c for cid in ids if (c := _store.get(cid)) is not None]


def upsert(contact: dict) -> str:
    """
    Create or update a contact. Returns contact_id.

    Dedup logic (in-memory, no disk read):
      - If contact has an 'id' → update that record.
      - Else if entity_id + name already exists → update that record.
      - Else → create new contact.

    Merge rule: existing non-None fields are not overwritten by incoming None values.
    """
    with _store._lock:
        _store._ensure_loaded()

        cid = contact.get("id")

        # Dedup: same entity_id + name (case-insensitive) → treat as update
        if not cid and contact.get("entity_id") and contact.get("name"):
            cid = _store.id_for_name(contact["entity_id"], contact["name"])
            if cid:
                contact["id"] = cid

        if not cid:
            cid = "cid_" + uuid.uuid4().hex[:8]
            contact["id"] = cid
            contact.setdefault("created_at", _now())

        contact["updated_at"] = _now()
        contact.setdefault("status", "new")
        contact.setdefault("notes", [])
        contact.setdefault("outreach_history", [])
        contact.setdefault("email_confidence", "none")

        existing = _store._records.get(cid)
        if existing:
            # Merge: keep existing values where incoming is None
            merged = dict(existing)
            for k, v in contact.items():
                if v is not None or k not in merged:
                    merged[k] = v
            contact = merged

        old = _store._records.get(cid)
        if old:
            _store._on_record_removed(old)
        _store._records[cid] = contact
        _store._on_record_added(contact)
        _store._flush()

    return cid


def update_status(contact_id: str, status: str, note: str = "") -> bool:
    """Update contact status. Returns True if found."""
    if status not in VALID_STATUSES:
        return False
    with _store._lock:
        _store._ensure_loaded()
        c = _store._records.get(contact_id)
        if c is None:
            return False
        old_status = c.get("status", "new")
        c["status"] = status
        c["updated_at"] = _now()
        if note:
            c.setdefault("notes", []).append({"text": note, "ts": _now()})
        c.setdefault("outreach_history", []).append({
            "event": f"{old_status} → {status}",
            "ts": _now(),
            "note": note,
        })
        _store._flush()
    return True


def set_followup(contact_id: str, followup_at: str, note: str = "") -> bool:
    """Set a follow-up date. followup_at is ISO datetime string."""
    with _store._lock:
        _store._ensure_loaded()
        c = _store._records.get(contact_id)
        if c is None:
            return False
        c["followup_at"] = followup_at
        c["updated_at"] = _now()
        if note:
            c.setdefault("notes", []).append({"text": note, "ts": _now()})
        _store._flush()
    return True


def add_note(contact_id: str, note: str) -> bool:
    """Append a note to a contact."""
    with _store._lock:
        _store._ensure_loaded()
        c = _store._records.get(contact_id)
        if c is None:
            return False
        c.setdefault("notes", []).append({"text": note, "ts": _now()})
        c["updated_at"] = _now()
        _store._flush()
    return True


def save_draft(contact_id: str, subject: str, body: str,
               linkedin_dm: str = "") -> bool:
    """Cache a generated email/DM draft on the contact record."""
    with _store._lock:
        _store._ensure_loaded()
        c = _store._records.get(contact_id)
        if c is None:
            return False
        c["draft_subject"]  = subject
        c["draft_body"]     = body
        c["draft_linkedin"] = linkedin_dm
        c["draft_at"]       = _now()
        c["updated_at"]     = _now()
        if c.get("status") in ("new", "researching", "ready"):
            c["status"] = "drafted"
        _store._flush()
    return True


def get_followup_due() -> list:
    """Return contacts where followup_at <= now and status is sent or followup_due."""
    now = _now()
    return _store.filter(
        lambda c: (
            bool(c.get("followup_at"))
            and c["followup_at"] <= now
            and c.get("status") in ("sent", "followup_due")
        )
    )


def seed_from_dossiers(dossier_dir: Path) -> int:
    """
    Create CRM contacts for every dossier with fit_score >= 40 that has
    a named contact and no existing CRM entry for that entity.
    Returns number of contacts created.
    """
    created = 0
    for f in sorted(dossier_dir.glob("*.json")):
        try:
            dossier = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        entity_id = dossier.get("entity_id", "")
        if not entity_id:
            continue

        fit = (dossier.get("fit_assessment") or {}).get("fit_score", 0)
        if fit < 40:
            continue

        if get_by_entity(entity_id):
            continue

        ro = dossier.get("recommended_outreach") or {}
        contact = {
            "entity_id":        entity_id,
            "company":          dossier.get("name", ""),
            "entity_type":      dossier.get("type", "ORG"),
            "fit_score":        fit,
            "name":             ro.get("contact_name"),
            "role":             ro.get("contact_role"),
            "email":            None,
            "email_confidence": "none",
            "linkedin_url":     None,
            "status":           "new",
            "outreach_reason":  ro.get("reason"),
            "outreach_angle":   ro.get("angle"),
            "why_yes":          (dossier.get("fit_assessment") or {}).get("why_yes", []),
            "next_actions":     dossier.get("next_actions", []),
            "notes":            [],
            "outreach_history": [],
            "followup_at":      None,
        }
        upsert(contact)
        created += 1

    return created
