"""
tools/store_base.py — Atomic, memory-indexed JSON array store.

Single responsibility: reliable, fast CRUD for a flat list of dicts stored as JSON.
Knows nothing about business domains (CRM, roadmaps, contacts, etc.).

Design choices:
  - Records in memory as dict[id → record] — O(1) get by id, no repeated disk reads.
  - Disk writes are atomic: write to .tmp then os.replace() — no partial writes,
    no corrupt files if the process is killed mid-write.
  - Load from disk once, lazily on first access.
  - RLock allows the same thread to re-enter (safe for read-then-write patterns
    within the same call, e.g. get + update + save in upsert).
  - Subclasses extend _on_record_added / _on_record_removed to maintain
    secondary indices without touching the base write path.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable


class AtomicJSONStore:
    """
    Thread-safe, memory-indexed store for a list of JSON objects.

    Each record must have a unique string field (default: 'id').
    Subclasses define secondary indices by overriding _on_load and
    _on_record_added / _on_record_removed.
    """

    def __init__(self, path: Path, id_field: str = "id") -> None:
        self._path = Path(path)
        self._id_field = id_field
        self._lock = threading.RLock()
        self._records: dict[str, dict] = {}   # id → record, ordered
        self._loaded = False

    # ── Internal: load / save ─────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load from disk exactly once. Idempotent after first load."""
        if self._loaded:
            return
        raw: list[dict] = []
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if not isinstance(raw, list):
                    raw = []
            except (OSError, json.JSONDecodeError):
                raw = []
        self._records = {}
        for record in raw:
            rid = record.get(self._id_field)
            if rid:
                self._records[rid] = record
        self._on_load(list(self._records.values()))
        self._loaded = True

    def _flush(self) -> None:
        """Write current records to disk atomically. Caller must hold _lock."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        payload = list(self._records.values())
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)   # atomic on POSIX (same filesystem)

    # ── Hooks for secondary indices ───────────────────────────────────────────

    def _on_load(self, records: list[dict]) -> None:
        """Called once after loading all records. Override to build indices."""

    def _on_record_added(self, record: dict) -> None:
        """Called after a record is inserted or updated. Override to update indices."""

    def _on_record_removed(self, record: dict) -> None:
        """Called after a record is removed. Override to clean up indices."""

    # ── Public interface ──────────────────────────────────────────────────────

    def get(self, record_id: str) -> dict | None:
        """O(1) lookup by id. Returns a copy to prevent external mutation."""
        with self._lock:
            self._ensure_loaded()
            rec = self._records.get(record_id)
            return dict(rec) if rec else None

    def all(self) -> list[dict]:
        """Return all records as a list of copies."""
        with self._lock:
            self._ensure_loaded()
            return [dict(r) for r in self._records.values()]

    def filter(self, predicate: Callable[[dict], bool]) -> list[dict]:
        """Return records matching predicate. Runs in memory — no disk I/O."""
        with self._lock:
            self._ensure_loaded()
            return [dict(r) for r in self._records.values() if predicate(r)]

    def put(self, record: dict) -> str:
        """
        Insert or replace a record by id. Generates id if missing.
        Returns the record id.
        Caller is responsible for setting all fields before calling put().
        """
        with self._lock:
            self._ensure_loaded()
            rid = record.get(self._id_field)
            if not rid:
                raise ValueError(f"Record must have '{self._id_field}' field set before put()")
            old = self._records.get(rid)
            if old:
                self._on_record_removed(old)
            self._records[rid] = dict(record)
            self._on_record_added(self._records[rid])
            self._flush()
        return rid

    def delete(self, record_id: str) -> bool:
        """Remove a record by id. Returns True if found."""
        with self._lock:
            self._ensure_loaded()
            old = self._records.pop(record_id, None)
            if old is None:
                return False
            self._on_record_removed(old)
            self._flush()
        return True

    def count(self) -> int:
        with self._lock:
            self._ensure_loaded()
            return len(self._records)
