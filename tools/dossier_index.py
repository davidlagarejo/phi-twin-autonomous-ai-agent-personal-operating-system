"""
tools/dossier_index.py — Dossier index cache.

Problem solved: services/timeline.py was doing N file reads (glob + read_text) on
every /api/timeline request. With 29+ dossiers this means 29+ disk reads per request.

Solution: maintain data/dossier_index.json — a single JSON array with all the fields
the timeline needs. The index is rebuilt only when a dossier file has changed (stale
check via mtime comparison, 0 JSON reads).

Single responsibility: maintain and expose the dossier index. No HTTP, no LLM.

Usage:
    from tools.dossier_index import load_dossiers
    dossiers = load_dossiers()   # returns list[dict], 1 file read if fresh

Invalidation:
    from tools.dossier_index import invalidate
    invalidate()   # forces rebuild on next load_dossiers() call

Write-through (optional, for lower latency):
    from tools.dossier_index import upsert
    upsert(dossier_dict)   # update one entry without full rebuild
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import List

_log = logging.getLogger("phi.tools.dossier_index")

_BASE         = Path(__file__).parent.parent
_DOSSIER_DIR  = _BASE / "workspace" / "dossiers"
_INDEX_FILE   = _BASE / "data" / "dossier_index.json"

# Fields to store in the index (everything timeline needs)
_KEEP_FIELDS = frozenset({
    "entity_id", "name", "type", "status", "last_updated",
    "description", "fit_assessment", "profile",
    "evidence_ids", "next_actions",
    "recommended_outreach", "key_contacts",
    "deadline", "deadline_label", "registration_url",
    "sub_programs",
})

_lock    = threading.Lock()
_forced  = False   # set by invalidate() to force rebuild on next load


# ── Stale check ───────────────────────────────────────────────────────────────

def _newest_dossier_mtime() -> float:
    """Return the mtime of the most recently modified dossier file. 0 if none."""
    latest = 0.0
    if _DOSSIER_DIR.exists():
        for p in _DOSSIER_DIR.glob("*.json"):
            try:
                m = p.stat().st_mtime
                if m > latest:
                    latest = m
            except Exception:
                pass
    return latest


def _index_mtime() -> float:
    try:
        return _INDEX_FILE.stat().st_mtime if _INDEX_FILE.exists() else 0.0
    except Exception:
        return 0.0


def _is_stale() -> bool:
    global _forced
    if _forced:
        return True
    return _newest_dossier_mtime() > _index_mtime()


# ── Rebuild ───────────────────────────────────────────────────────────────────

def _rebuild() -> List[dict]:
    """Read all dossier files and write a fresh index. Returns the index list."""
    global _forced
    entries: list = []
    if not _DOSSIER_DIR.exists():
        _write_index(entries)
        _forced = False
        return entries

    for df in sorted(_DOSSIER_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            d = json.loads(df.read_text(encoding="utf-8"))
        except Exception:
            continue
        entry = {k: d[k] for k in _KEEP_FIELDS if k in d}
        entry["_file"]  = str(df.relative_to(_BASE))
        entry["_mtime"] = df.stat().st_mtime
        entries.append(entry)

    _write_index(entries)
    _forced = False
    _log.debug("dossier_index rebuilt entries=%d", len(entries))
    return entries


def _write_index(entries: list) -> None:
    _INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    _INDEX_FILE.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def load_dossiers() -> List[dict]:
    """
    Return all dossiers as a list of dicts.
    Rebuilds index only when a dossier file is newer than the index (stale check:
    N stat() calls, 0 JSON reads if fresh).
    Thread-safe.
    """
    with _lock:
        if _is_stale():
            return _rebuild()
        if _INDEX_FILE.exists():
            try:
                return json.loads(_INDEX_FILE.read_text(encoding="utf-8"))
            except Exception:
                return _rebuild()
        return _rebuild()


def invalidate() -> None:
    """Force index rebuild on next load_dossiers() call."""
    global _forced
    with _lock:
        _forced = True


def upsert(dossier: dict) -> None:
    """
    Update one dossier entry in the index without full rebuild.
    Call after writing a new dossier file to keep the index current.
    """
    with _lock:
        entity_id = dossier.get("entity_id")
        if not entity_id:
            _forced = True  # can't match — force rebuild
            return
        try:
            entries = json.loads(_INDEX_FILE.read_text(encoding="utf-8")) if _INDEX_FILE.exists() else []
        except Exception:
            entries = []
        entry = {k: dossier[k] for k in _KEEP_FIELDS if k in dossier}
        entry["_mtime"] = time.time()
        # Replace existing or append
        idx = next((i for i, e in enumerate(entries) if e.get("entity_id") == entity_id), None)
        if idx is not None:
            entries[idx] = entry
        else:
            entries.insert(0, entry)
        _write_index(entries)
