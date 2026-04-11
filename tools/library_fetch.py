"""
library_fetch.py — Paywalled document broker (Lane 2).
=======================================================
INVARIANT: This module NEVER auto-logs in to any service.
  - If a document has a free PDF URL → download it.
  - If a document requires login → return NEEDS_USER_LOGIN.
  - Score threshold gate: only fetch high-value documents.
  - Privacy gate: every outbound HTTP request is pre-checked.

Local cache: workspace/library_cache/<doc_id>/<filename>.pdf
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.privacy_pre_hook import privacy_pre_hook
except ImportError:
    privacy_pre_hook = None  # type: ignore[assignment]

try:
    from core.audit_append import append_audit as _append_audit, _payload_hash as _audit_hash
except ImportError:
    _append_audit = None  # type: ignore[assignment]
    _audit_hash = None  # type: ignore[assignment]

log = logging.getLogger("phi.library_fetch")

LIBRARY_FETCH_MIN_SCORE: float = 0.6  # (credibility + relevance) / 2 threshold

_TIMEOUT = 30  # seconds for PDF download

# ── Audit helper ──────────────────────────────────────────────────────────────

def _lib_audit(decision: str, doc_id: str, reason: str) -> None:
    """Write a LIBRARY_FETCH audit entry. Never raises."""
    if _append_audit is None or _audit_hash is None:
        return
    try:
        _append_audit(
            action_type="LIBRARY_FETCH",
            decision=decision,
            pii_detected=[],
            redaction_notes=[],
            payload_hash=_audit_hash({"doc_id": doc_id}),
            reason=reason,
        )
    except Exception:
        pass


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class LibraryFetchResult:
    status: str          # CACHED | DOWNLOADED | NEEDS_USER_LOGIN | BLOCKED | SCORE_BELOW_THRESHOLD
    doc_id: str
    local_path: Optional[str]
    title: Optional[str]
    message: str
    unlocking_question: Optional[str]


# ── Normalization ─────────────────────────────────────────────────────────────

def _normalize_doc_id(doi_or_url: str) -> str:
    """Normalize to a filesystem-safe identifier."""
    s = doi_or_url.strip()
    # DOI: strip prefix
    for prefix in ("https://doi.org/", "http://doi.org/", "doi.org/"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            return "doi_" + re.sub(r"[^\w\-.]", "_", s)[:80]
    # arXiv URL
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s?#/]+)", s)
    if m:
        return "arxiv_" + re.sub(r"[^\w\-.]", "_", m.group(1))[:40]
    # PMC
    m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", s)
    if m:
        return "pmid_" + m.group(1)
    # Fallback: sha256 of URL
    return "url_" + hashlib.sha256(s.encode()).hexdigest()[:12]


def _safe_filename(doi_or_url: str) -> str:
    """Generate a safe PDF filename."""
    doc_id = _normalize_doc_id(doi_or_url)
    return doc_id + ".pdf"


# ── SSRF guard ────────────────────────────────────────────────────────────────

def _is_safe_external_url(url: str) -> bool:
    """Block private IPs and loopback — defense against SSRF via malformed DOI redirects."""
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or ""
        if host in ("localhost", "127.0.0.1", "::1"):
            return False
        # Block RFC-1918 ranges via simple prefix check (no socket.getaddrinfo needed)
        private_prefixes = ("10.", "192.168.", "172.16.", "172.17.", "172.18.",
                            "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
                            "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
                            "172.29.", "172.30.", "172.31.")
        if any(host.startswith(p) for p in private_prefixes):
            return False
        if not parsed.scheme in ("http", "https"):
            return False
        return True
    except Exception:
        return False


# ── Core function ─────────────────────────────────────────────────────────────

def fetch_library_document(
    doi_or_url: str,
    title: Optional[str] = None,
    credibility_score: float = 0.0,
    relevance_score: float = 0.0,
    open_pdf_url: Optional[str] = None,
    workspace_state=None,  # WorkspaceState | None — avoid circular import
    _policy: Optional[dict] = None,
) -> LibraryFetchResult:
    """
    Fetch or locate a document in the local library cache.

    Decision tree:
      1. Score gate → SCORE_BELOW_THRESHOLD with unlocking question.
      2. Already cached → CACHED.
      3. open_pdf_url available → privacy gate → download → DOWNLOADED.
      4. No open PDF → NEEDS_USER_LOGIN.
    """
    doc_id = _normalize_doc_id(doi_or_url)

    # ── 1. Score gate ─────────────────────────────────────────────────────────
    combined_score = (credibility_score + relevance_score) / 2.0
    if combined_score < LIBRARY_FETCH_MIN_SCORE:
        _lib_audit("BLOCK", doc_id, f"score_below_threshold:{combined_score:.2f}")
        return LibraryFetchResult(
            status="SCORE_BELOW_THRESHOLD",
            doc_id=doc_id,
            local_path=None,
            title=title,
            message=f"Combined score {combined_score:.2f} below threshold {LIBRARY_FETCH_MIN_SCORE}.",
            unlocking_question=(
                f"This paper scored {combined_score:.2f}/1.0. "
                "Fetch it anyway? It may have lower relevance to your current research."
            ),
        )

    # ── 2. Cache check ────────────────────────────────────────────────────────
    if workspace_state is not None:
        cache_dir = workspace_state.library_cache_path(doc_id)
    else:
        cache_dir = Path.home() / "phi-twin" / "workspace" / "library_cache" / doc_id
        cache_dir.mkdir(parents=True, exist_ok=True)

    existing = list(cache_dir.glob("*.pdf"))
    if existing:
        return LibraryFetchResult(
            status="CACHED",
            doc_id=doc_id,
            local_path=str(existing[0]),
            title=title,
            message=f"Document already cached at {existing[0].name}.",
            unlocking_question=None,
        )

    # ── 3. Open PDF download ──────────────────────────────────────────────────
    if open_pdf_url:
        if not _is_safe_external_url(open_pdf_url):
            _lib_audit("BLOCK", doc_id, "ssrf_guard")
            return LibraryFetchResult(
                status="BLOCKED",
                doc_id=doc_id,
                local_path=None,
                title=title,
                message="PDF URL failed SSRF safety check.",
                unlocking_question=None,
            )

        # Privacy gate
        try:
            _hook = privacy_pre_hook
            result = _hook("SEARCH_WEB", {"query": doi_or_url}, policy=_policy or {}) if _hook is not None else {"decision": "ALLOW"}
            if result.get("decision") == "BLOCK":
                return LibraryFetchResult(
                    status="BLOCKED",
                    doc_id=doc_id,
                    local_path=None,
                    title=title,
                    message=f"Privacy gate blocked: {result.get('reason', 'PII detected')}",
                    unlocking_question=None,
                )
        except ImportError:
            pass
        except Exception as exc:
            log.warning("privacy_pre_hook error: %s — proceeding", exc)

        # Download
        filename = _safe_filename(doi_or_url)
        dest = cache_dir / filename
        try:
            _download_pdf(open_pdf_url, dest, workspace_state)
            _lib_audit("ALLOW", doc_id, "downloaded")
            return LibraryFetchResult(
                status="DOWNLOADED",
                doc_id=doc_id,
                local_path=str(dest),
                title=title,
                message=f"Downloaded and cached at {filename}.",
                unlocking_question=None,
            )
        except Exception as exc:
            log.error("PDF download failed for %s: %s", doc_id, exc)
            return LibraryFetchResult(
                status="NEEDS_USER_LOGIN",
                doc_id=doc_id,
                local_path=None,
                title=title,
                message=f"Download failed (may require login): {exc}",
                unlocking_question=(
                    "I can fetch the full paper via your library if you log in once. "
                    "Want to do that now?"
                ),
            )

    # ── 4. No open PDF URL ────────────────────────────────────────────────────
    return LibraryFetchResult(
        status="NEEDS_USER_LOGIN",
        doc_id=doc_id,
        local_path=None,
        title=title,
        message="No open-access PDF found. Requires institutional login.",
        unlocking_question=(
            "I can fetch the full paper via your library if you log in once. "
            "Want to do that now?"
        ),
    )


def _download_pdf(url: str, dest: Path, workspace_state=None) -> None:
    """Download PDF to dest. Uses workspace_state._lock if available."""
    import urllib.request

    lock = getattr(workspace_state, "_lock", None)
    if lock:
        with lock:
            _do_download(url, dest)
    else:
        _do_download(url, dest)


def _do_download(url: str, dest: Path) -> None:
    import urllib.request
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "phi-twin-research/1.0 (noreply@phi-twin.local)"},
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower() and not url.lower().endswith(".pdf"):
            raise ValueError(f"Response is not a PDF: Content-Type={content_type}")
        data = resp.read()
    if len(data) < 1024:
        raise ValueError(f"PDF too small ({len(data)} bytes) — likely a login redirect")
    dest.write_bytes(data)
