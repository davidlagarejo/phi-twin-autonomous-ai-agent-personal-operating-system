"""
memory.py — Persistent conversation memory for Phi-twin
========================================================
Uses ChromaDB on the external SSD (/Volumes/ZLab_Run) to store every
conversation turn as a vector embedding. Before each phi call, retrieves
the most semantically relevant past exchanges and injects them as context.

Storage: /Volumes/ZLab_Run/Zlab_Run/Active/phi-twin/memory/
Fallback: ~/phi-twin/data/memory/ (if SSD not mounted)

Embedding model: chromadb's built-in (no external API, fully local).
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# ── Storage path ──────────────────────────────────────────────────────────────

_SSD_PATH   = Path("/Volumes/ZLab_Run/Zlab_Run/Active/phi-twin/memory")
_LOCAL_PATH = Path.home() / "phi-twin" / "data" / "memory"

def _memory_path() -> Path:
    if Path("/Volumes/ZLab_Run").exists():
        _SSD_PATH.mkdir(parents=True, exist_ok=True)
        return _SSD_PATH
    _LOCAL_PATH.mkdir(parents=True, exist_ok=True)
    return _LOCAL_PATH

# ── ChromaDB client (lazy init) ───────────────────────────────────────────────

_client: chromadb.PersistentClient | None = None
_collection = None

def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection
    path = _memory_path()
    _client = chromadb.PersistentClient(path=str(path))
    ef = embedding_functions.DefaultEmbeddingFunction()
    _collection = _client.get_or_create_collection(
        name="phi_memory",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


# ── Public API ────────────────────────────────────────────────────────────────

def store(user_msg: str, phi_response: str, session_id: str = "default") -> None:
    """Store a conversation exchange as a memory entry."""
    try:
        col = _get_collection()
        ts  = datetime.now(timezone.utc).isoformat()
        # Use a hash of content + timestamp as stable ID
        entry_id = hashlib.sha256(f"{ts}{user_msg}".encode()).hexdigest()[:16]

        # Store both turns as a single document for retrieval coherence
        document = f"David: {user_msg}\nPhi: {phi_response}"
        col.add(
            documents=[document],
            metadatas=[{
                "user_msg":    user_msg[:500],
                "phi_response": phi_response[:500],
                "session_id":  session_id,
                "timestamp":   ts,
            }],
            ids=[entry_id],
        )
    except Exception:
        pass  # Never block the main chat flow


def retrieve(query: str, n: int = 5, exclude_session: str | None = None) -> list[dict]:
    """
    Retrieve the n most semantically relevant past exchanges for a given query.
    Returns list of dicts with keys: user_msg, phi_response, timestamp, distance.
    """
    try:
        col = _get_collection()
        count = col.count()
        if count == 0:
            return []

        results = col.query(
            query_texts=[query],
            n_results=min(n, count),
            include=["documents", "metadatas", "distances"],
        )

        memories = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Skip very distant results (cosine distance > 0.85 = not relevant)
            if dist > 0.85:
                continue
            memories.append({
                "user_msg":     meta.get("user_msg", ""),
                "phi_response": meta.get("phi_response", ""),
                "timestamp":    meta.get("timestamp", ""),
                "distance":     round(dist, 3),
            })
        return memories
    except Exception:
        return []


def format_for_context(memories: list[dict]) -> str:
    """Format retrieved memories as a context block for phi's prompt."""
    if not memories:
        return ""
    lines = ["[MEMORIA RELEVANTE DE CONVERSACIONES ANTERIORES — usa esto como contexto:]"]
    for m in memories:
        ts = m["timestamp"][:10] if m.get("timestamp") else "?"
        lines.append(f"\n— {ts}")
        lines.append(f"  David: {m['user_msg']}")
        lines.append(f"  Phi:   {m['phi_response']}")
    lines.append("\n[FIN MEMORIA — responde al mensaje actual de David]")
    return "\n".join(lines)


def stats() -> dict:
    """Return memory statistics."""
    try:
        col = _get_collection()
        count = col.count()
        path  = _memory_path()
        # Approximate size on disk
        size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1_048_576
        return {
            "total_memories": count,
            "storage_path":   str(path),
            "size_mb":        round(size_mb, 2),
            "ssd_active":     Path("/Volumes/ZLab_Run").exists(),
        }
    except Exception as e:
        return {"error": str(e)}
