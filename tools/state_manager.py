"""
state_manager.py — Workspace state persistence for the proactive research engine.
====================================================================================
Manages all state files under workspace/:
  state/strategy_state.json   — hypotheses, open_loops, backoff, run counter
  state/evidence.jsonl        — append-only evidence log
  state/queue.jsonl           — append-only task queue
  state/dedupe.json           — fingerprint index (in-memory + disk)
  state/timeline.jsonl        — append-only research event log
  dossiers/<entity_id>.json   — one dossier per investigated entity
  briefs/daily_<date>.json    — daily brief snapshots
  library_cache/<doc_id>/     — locally cached PDFs (written by library_fetch)

All writes are thread-safe and atomic (write-to-tmp then os.replace).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional

BASE_DIR_DEFAULT = Path(__file__).parent.parent


# ── Exceptions ────────────────────────────────────────────────────────────────

class StateWriteError(RuntimeError):
    pass


class DuplicateError(RuntimeError):
    def __init__(self, fingerprint: str, existing_id: str):
        super().__init__(f"Duplicate fingerprint {fingerprint!r} (existing: {existing_id!r})")
        self.fingerprint = fingerprint
        self.existing_id = existing_id


# ── Fingerprint helpers (pure, no I/O) ────────────────────────────────────────

def fingerprint_opportunity(type_: str, org: str, title: str, url: str) -> str:
    raw = f"{type_}|{org.lower().strip()}|{title.lower().strip()}|{url.rstrip('/')}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def fingerprint_task(strategy: str, stable_keys: dict) -> str:
    raw = f"{strategy}|{json.dumps(stable_keys, sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def fingerprint_evidence(url: str, title: str) -> str:
    raw = f"{url.rstrip('/')}|{title.lower().strip()[:80]}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def fingerprint_timeline(type_: str, ref_id: str, date_bucket: str) -> str:
    raw = f"{type_}|{ref_id}|{date_bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Default strategy state ────────────────────────────────────────────────────

def _default_strategy_state() -> dict:
    return {
        "schema_version": "1.0",
        "updated_at": _now(),
        "run_counter": 0,
        "last_checkpoint_id": None,
        "consecutive_empty_runs": 0,
        "backoff_until": None,
        "open_loops": [],
        "hypotheses": [],
        "active_entity_ids": [],
        "pending_approval_ids": [],
        "stats": {
            "total_runs": 0,
            "total_evidence_items": 0,
            "total_dossiers": 0,
            "last_value_notification_at": None,
        },
    }


def _default_dedupe() -> dict:
    return {
        "schema_version": "1.0",
        "updated_at": _now(),
        "opportunity_fingerprints": {},
        "task_fingerprints": {},
        "evidence_fingerprints": {},
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── WorkspaceState ────────────────────────────────────────────────────────────

class WorkspaceState:
    """Thread-safe workspace state manager."""

    def __init__(self, base_dir: Optional[Path | str] = None) -> None:
        self.BASE_DIR = Path(base_dir) if base_dir else BASE_DIR_DEFAULT
        self.STATE_DIR = self.BASE_DIR / "workspace" / "state"
        self.DOSSIER_DIR = self.BASE_DIR / "workspace" / "dossiers"
        self.BRIEF_DIR = self.BASE_DIR / "workspace" / "briefs"
        self.LIBRARY_DIR = self.BASE_DIR / "workspace" / "library_cache"
        self._lock = threading.RLock()
        self._dedupe_cache: dict | None = None

        # Ensure directories exist
        for d in (self.STATE_DIR, self.DOSSIER_DIR, self.BRIEF_DIR, self.LIBRARY_DIR):
            d.mkdir(parents=True, exist_ok=True)

        # Eagerly load dedupe cache into memory
        self._dedupe_cache = self.load_dedupe()

    # ── Paths ─────────────────────────────────────────────────────────────────

    @property
    def _strategy_state_path(self) -> Path:
        return self.STATE_DIR / "strategy_state.json"

    @property
    def _evidence_path(self) -> Path:
        return self.STATE_DIR / "evidence.jsonl"

    @property
    def _queue_path(self) -> Path:
        return self.STATE_DIR / "queue.jsonl"

    @property
    def _dedupe_path(self) -> Path:
        return self.STATE_DIR / "dedupe.json"

    @property
    def _timeline_path(self) -> Path:
        return self.STATE_DIR / "timeline.jsonl"

    # ── State loading ─────────────────────────────────────────────────────────

    def load_strategy_state(self) -> dict:
        """Read strategy_state.json; create with defaults if missing."""
        if not self._strategy_state_path.exists():
            state = _default_strategy_state()
            self._write_json_atomic(self._strategy_state_path, state)
            return state
        try:
            return json.loads(self._strategy_state_path.read_text(encoding="utf-8"))
        except Exception:
            state = _default_strategy_state()
            self._write_json_atomic(self._strategy_state_path, state)
            return state

    def load_dedupe(self) -> dict:
        """Read dedupe.json; create empty if missing."""
        if not self._dedupe_path.exists():
            d = _default_dedupe()
            self._write_json_atomic(self._dedupe_path, d)
            return d
        try:
            return json.loads(self._dedupe_path.read_text(encoding="utf-8"))
        except Exception:
            d = _default_dedupe()
            self._write_json_atomic(self._dedupe_path, d)
            return d

    # ── Atomic write helpers ──────────────────────────────────────────────────

    def _write_json_atomic(self, path: Path, data: dict) -> None:
        """Write to <path>.tmp then os.replace(). Holds _lock."""
        with self._lock:
            tmp = path.with_suffix(path.suffix + ".tmp")
            try:
                tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(str(tmp), str(path))
            except Exception as exc:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
                raise StateWriteError(f"Atomic write failed for {path}: {exc}") from exc

    def _append_jsonl(self, path: Path, record: dict) -> None:
        """Append one JSON line to a .jsonl file. Holds _lock."""
        with self._lock:
            try:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as exc:
                raise StateWriteError(f"JSONL append failed for {path}: {exc}") from exc

    def _read_jsonl(self, path: Path) -> list[dict]:
        """Read all records from a .jsonl file."""
        if not path.exists():
            return []
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        return records

    def _save_dedupe_cache(self) -> None:
        """Persist in-memory dedupe cache to disk. Caller holds _lock."""
        if self._dedupe_cache is not None:
            self._dedupe_cache["updated_at"] = _now()
            tmp = self._dedupe_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._dedupe_cache, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(str(tmp), str(self._dedupe_path))

    # ── Dedup check-and-register ──────────────────────────────────────────────

    def is_duplicate_opportunity(self, fingerprint: str) -> bool:
        with self._lock:
            cache = self._dedupe_cache or {}
            return fingerprint in cache.get("opportunity_fingerprints", {})

    def register_opportunity(self, fingerprint: str, seen_at: Optional[str] = None) -> None:
        """Raises DuplicateError if already registered."""
        with self._lock:
            cache = self._dedupe_cache or {}
            fps = cache.setdefault("opportunity_fingerprints", {})
            if fingerprint in fps:
                raise DuplicateError(fingerprint, fps[fingerprint])
            fps[fingerprint] = seen_at or _now()
            self._save_dedupe_cache()

    def is_duplicate_task(self, fingerprint: str) -> bool:
        with self._lock:
            cache = self._dedupe_cache or {}
            return fingerprint in cache.get("task_fingerprints", {})

    def register_task(self, fingerprint: str, task_id: str) -> None:
        """Raises DuplicateError if already registered."""
        with self._lock:
            cache = self._dedupe_cache or {}
            fps = cache.setdefault("task_fingerprints", {})
            if fingerprint in fps:
                raise DuplicateError(fingerprint, fps[fingerprint])
            fps[fingerprint] = task_id
            self._save_dedupe_cache()

    def is_duplicate_evidence(self, fingerprint: str) -> bool:
        with self._lock:
            cache = self._dedupe_cache or {}
            return fingerprint in cache.get("evidence_fingerprints", {})

    def register_evidence(self, fingerprint: str, evidence_id: str) -> None:
        with self._lock:
            cache = self._dedupe_cache or {}
            fps = cache.setdefault("evidence_fingerprints", {})
            if fingerprint in fps:
                raise DuplicateError(fingerprint, fps[fingerprint])
            fps[fingerprint] = evidence_id
            self._save_dedupe_cache()

    # ── Evidence ──────────────────────────────────────────────────────────────

    def append_evidence(self, record: dict) -> str:
        """Validate, dedup-check, append to evidence.jsonl. Returns evidence_id."""
        url = record.get("url") or ""
        title = record.get("title") or ""
        fp = fingerprint_evidence(url, title)

        with self._lock:
            if self.is_duplicate_evidence(fp):
                raise DuplicateError(fp, self._dedupe_cache["evidence_fingerprints"][fp])
            ev_id = record.get("evidence_id") or f"ev_{hashlib.sha256((url + _now()).encode()).hexdigest()[:12]}"
            record["evidence_id"] = ev_id
            record["fingerprint"] = fp
            if "retrieved_at" not in record:
                record["retrieved_at"] = _now()
            self._append_jsonl(self._evidence_path, record)
            self.register_evidence(fp, ev_id)
            return ev_id

    def read_evidence(self, evidence_ids: Optional[list[str]] = None) -> list[dict]:
        """Read all or filtered evidence records."""
        records = self._read_jsonl(self._evidence_path)
        if evidence_ids is not None:
            id_set = set(evidence_ids)
            records = [r for r in records if r.get("evidence_id") in id_set]
        return records

    # ── Queue ─────────────────────────────────────────────────────────────────

    def enqueue_task(self, task: dict) -> str:
        """Dedup-check then append to queue.jsonl. Returns task_id."""
        strategy = task.get("strategy", "DISCOVER")
        stable_keys = {k: task.get("payload", {}).get(k)
                       for k in ("entity_id", "query_hint", "hypothesis_ids")}
        fp = fingerprint_task(strategy, stable_keys)

        with self._lock:
            if self.is_duplicate_task(fp):
                raise DuplicateError(fp, self._dedupe_cache["task_fingerprints"][fp])
            task_id = task.get("task_id") or f"task_{uuid.uuid4().hex[:8]}"
            task["task_id"] = task_id
            task["fingerprint"] = fp
            if "enqueued_at" not in task:
                task["enqueued_at"] = _now()
            if "status" not in task:
                task["status"] = "PENDING"
            if "attempts" not in task:
                task["attempts"] = 0
            if "priority" not in task:
                task["priority"] = 4
            self._append_jsonl(self._queue_path, task)
            self.register_task(fp, task_id)
            return task_id

    def peek_next_tasks(self, n: int = 2) -> list[dict]:
        """Return up to n PENDING tasks sorted by priority then enqueued_at."""
        all_tasks = self._read_jsonl(self._queue_path)
        pending = [t for t in all_tasks if t.get("status") == "PENDING"]
        pending.sort(key=lambda t: (t.get("priority", 99), t.get("enqueued_at", "")))
        return pending[:n]

    def mark_task_status(
        self,
        task_id: str,
        status: str,
        checkpoint_id: Optional[str] = None,
    ) -> None:
        """Rewrite queue.jsonl with updated task status."""
        with self._lock:
            tasks = self._read_jsonl(self._queue_path)
            updated = False
            for t in tasks:
                if t.get("task_id") == task_id:
                    t["status"] = status
                    t["last_attempt_at"] = _now()
                    if status in ("IN_PROGRESS", "FAILED", "FROZEN"):
                        t["attempts"] = t.get("attempts", 0) + 1
                    if checkpoint_id is not None:
                        t["checkpoint_id"] = checkpoint_id
                    updated = True
            if not updated:
                return
            tmp = self._queue_path.with_suffix(".jsonl.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for t in tasks:
                    f.write(json.dumps(t, ensure_ascii=False) + "\n")
            os.replace(str(tmp), str(self._queue_path))

    def read_queue(self, status_filter: Optional[list[str]] = None) -> list[dict]:
        """Read all queue entries, optionally filtered by status list."""
        tasks = self._read_jsonl(self._queue_path)
        if status_filter:
            tasks = [t for t in tasks if t.get("status") in status_filter]
        return tasks

    # ── Strategy state mutations ──────────────────────────────────────────────

    def update_strategy_state(self, delta: dict) -> dict:
        """Merge delta into strategy_state.json atomically. Returns new state."""
        with self._lock:
            state = self.load_strategy_state()
            state["updated_at"] = _now()

            if "run_counter" in delta:
                state["run_counter"] = delta["run_counter"]
            if "last_checkpoint_id" in delta:
                state["last_checkpoint_id"] = delta["last_checkpoint_id"]
            if "consecutive_empty_runs" in delta:
                state["consecutive_empty_runs"] = delta["consecutive_empty_runs"]
            if "backoff_until" in delta:
                state["backoff_until"] = delta["backoff_until"]

            # open_loops: append new ones, update existing by id
            if "open_loops" in delta:
                existing_ids = {ol["id"] for ol in state.get("open_loops", [])}
                for ol in delta["open_loops"]:
                    if ol["id"] in existing_ids:
                        state["open_loops"] = [
                            ol if l["id"] == ol["id"] else l
                            for l in state["open_loops"]
                        ]
                    else:
                        state.setdefault("open_loops", []).append(ol)

            # hypotheses: upsert by id
            if "hypotheses" in delta:
                existing_hyp_ids = {h["id"] for h in state.get("hypotheses", [])}
                for h in delta["hypotheses"]:
                    if h["id"] in existing_hyp_ids:
                        state["hypotheses"] = [
                            h if existing["id"] == h["id"] else existing
                            for existing in state["hypotheses"]
                        ]
                    else:
                        state.setdefault("hypotheses", []).append(h)

            if "active_entity_ids" in delta:
                current = set(state.get("active_entity_ids", []))
                current.update(delta["active_entity_ids"])
                state["active_entity_ids"] = sorted(current)

            if "pending_approval_ids" in delta:
                current = set(state.get("pending_approval_ids", []))
                current.update(delta["pending_approval_ids"])
                state["pending_approval_ids"] = sorted(current)

            # stats: merge dict
            if "stats" in delta:
                state.setdefault("stats", {}).update(delta["stats"])

            self._write_json_atomic(self._strategy_state_path, state)
            return state

    # ── Timeline ──────────────────────────────────────────────────────────────

    def append_timeline(self, event_type: str, summary: str, **metadata) -> str:
        """Append event to timeline.jsonl. Returns event_id."""
        state = self.load_strategy_state()
        event_id = f"evt_{uuid.uuid4().hex[:8]}"
        record = {
            "event_id": event_id,
            "timestamp": _now(),
            "run_counter": state.get("run_counter", 0),
            "event_type": event_type,
            "summary": summary,
            "task_id": metadata.pop("task_id", None),
            "entity_id": metadata.pop("entity_id", None),
            "metadata": metadata,
        }
        self._append_jsonl(self._timeline_path, record)
        return event_id

    def read_timeline(self, limit: int = 50) -> list[dict]:
        """Read recent timeline events."""
        events = self._read_jsonl(self._timeline_path)
        return events[-limit:]

    # ── Dossiers ──────────────────────────────────────────────────────────────

    def save_dossier(self, dossier: dict) -> str:
        """Write to workspace/dossiers/<entity_id>.json. Returns entity_id."""
        entity_id = dossier.get("entity_id")
        if not entity_id:
            raise ValueError("dossier must have entity_id")
        dossier["last_updated"] = _now()
        if "created_at" not in dossier:
            dossier["created_at"] = _now()
        path = self.DOSSIER_DIR / f"{entity_id}.json"
        self._write_json_atomic(path, dossier)
        return entity_id

    def load_dossier(self, entity_id: str) -> Optional[dict]:
        """Load dossier or return None if not found."""
        path = self.DOSSIER_DIR / f"{entity_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def list_dossiers(self, status_filter: Optional[list[str]] = None) -> list[dict]:
        """List dossier summaries."""
        summaries = []
        for p in self.DOSSIER_DIR.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if status_filter and d.get("status") not in status_filter:
                    continue
                summaries.append({
                    "entity_id": d.get("entity_id"),
                    "name": d.get("name"),
                    "fit_score": d.get("fit_assessment", {}).get("fit_score", 0),
                    "status": d.get("status"),
                    "type": d.get("type"),
                    "last_updated": d.get("last_updated"),
                })
            except Exception:
                pass
        return summaries

    # ── Daily briefs ──────────────────────────────────────────────────────────

    def save_brief(self, brief: dict) -> Path:
        """Write workspace/briefs/daily_<date>.json. Returns path."""
        date_str = brief.get("date") or date.today().isoformat()
        brief["date"] = date_str
        if "generated_at" not in brief:
            brief["generated_at"] = _now()
        path = self.BRIEF_DIR / f"daily_{date_str}.json"
        self._write_json_atomic(path, brief)
        return path

    def load_brief(self, date_str: Optional[str] = None) -> Optional[dict]:
        """Load brief for date (default: today). Returns None if not found."""
        date_str = date_str or date.today().isoformat()
        path = self.BRIEF_DIR / f"daily_{date_str}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    # ── Library cache ─────────────────────────────────────────────────────────

    def library_cache_path(self, doc_id: str) -> Path:
        """workspace/library_cache/<doc_id>/. Creates dir if missing."""
        p = self.LIBRARY_DIR / doc_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def list_library_cache(self) -> list[dict]:
        """Return [{doc_id, filename, size_bytes, cached_at}] for all cached PDFs."""
        results = []
        for doc_dir in self.LIBRARY_DIR.iterdir():
            if not doc_dir.is_dir():
                continue
            for f in doc_dir.glob("*.pdf"):
                stat = f.stat()
                results.append({
                    "doc_id": doc_dir.name,
                    "filename": f.name,
                    "size_bytes": stat.st_size,
                    "cached_at": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(),
                })
        return results

    # ── Workspace summary ─────────────────────────────────────────────────────

    def get_state_summary(self) -> dict:
        """Return serializable summary of current workspace state."""
        state = self.load_strategy_state()
        queue = self.read_queue()
        dossiers = self.list_dossiers()

        queue_summary = {"PENDING": 0, "IN_PROGRESS": 0, "DONE": 0, "FAILED": 0, "FROZEN": 0}
        for t in queue:
            s = t.get("status", "PENDING")
            queue_summary[s] = queue_summary.get(s, 0) + 1

        hyp_summary = {"ACTIVE": 0, "CONFIRMED": 0, "REJECTED": 0}
        for h in state.get("hypotheses", []):
            s = h.get("status", "ACTIVE")
            hyp_summary[s] = hyp_summary.get(s, 0) + 1

        dossier_summary = {"DRAFT": 0, "COMPLETE": 0, "ARCHIVED": 0}
        for d in dossiers:
            s = d.get("status", "DRAFT")
            dossier_summary[s] = dossier_summary.get(s, 0) + 1

        today_brief = self.load_brief()

        # Derive a human-readable context label from the most recently updated dossier
        context_label = "PHI TWIN"
        if dossiers:
            recent = sorted(dossiers, key=lambda d: d.get("last_updated") or "", reverse=True)
            name = recent[0].get("name")
            if name:
                context_label = name.upper()

        return {
            "schema_version": "1.0",
            "updated_at": state.get("updated_at"),
            "run_counter": state.get("run_counter", 0),
            "last_checkpoint_id": state.get("last_checkpoint_id"),
            "backoff_until": state.get("backoff_until"),
            "consecutive_empty_runs": state.get("consecutive_empty_runs", 0),
            "queue_summary": queue_summary,
            "hypothesis_summary": hyp_summary,
            "dossier_summary": dossier_summary,
            "pending_approvals": state.get("pending_approval_ids", []),
            "open_loops_count": len(state.get("open_loops", [])),
            "today_brief_available": today_brief is not None,
            "context_label": context_label,
        }
