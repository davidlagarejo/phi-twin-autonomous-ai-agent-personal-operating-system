"""
Carril A — Web search (local SearXNG only).

DuckDuckGo is DISABLED. The only permitted egress path is:
  LOCAL SearXNG → privacy_pre_hook gate → search_web()

Every query is screened by privacy_pre_hook before any network call.
If SearXNG is unreachable or not configured → BLOCK (empty results, no exception).
"""

from __future__ import annotations

import ipaddress
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.privacy_pre_hook import privacy_pre_hook
from core.pii_rules import text_is_safe


# ── Exceptions ───────────────────────────────────────────────────────────────

class DDGDisabledError(RuntimeError):
    """Raised when any code path attempts to use the disabled DDG client."""


class SearchBlockedError(RuntimeError):
    """Raised internally when a query is blocked by the privacy gate."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ── Data types ───────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    source_id: str
    title: str
    url: str
    snippet: str
    source: str = "web"
    query: str = ""
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Disabled DDG stubs ───────────────────────────────────────────────────────

def search_ddg(query: str, max_results: int = 6) -> list[SearchResult]:  # noqa: ARG001
    """DDG is permanently disabled. Raises DDGDisabledError on any call."""
    raise DDGDisabledError(
        "DuckDuckGo search is disabled for privacy. Use search_web() with local SearXNG."
    )


def search_news_ddg(query: str, max_results: int = 5) -> list[SearchResult]:  # noqa: ARG001
    """DDG news is permanently disabled. Raises DDGDisabledError on any call."""
    raise DDGDisabledError(
        "DuckDuckGo news is disabled for privacy. Use search_web() with local SearXNG."
    )


# ── Network guardrails ───────────────────────────────────────────────────────

_ALLOWED_HOSTNAMES = frozenset({
    "localhost",
    "searxng",
    "host.docker.internal",
})

# RFC 1918 private ranges + loopback
_PRIVATE_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),        # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),       # IPv6 ULA
]


def _is_local_url(url: str) -> bool:
    """
    Return True only if the URL's host is localhost, a known docker hostname,
    or a private/loopback IP address (RFC 1918 / loopback).
    Any public IP or domain → False (will be blocked).
    """
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        return False

    if not host:
        return False

    if host.lower() in _ALLOWED_HOSTNAMES:
        return True

    try:
        addr = ipaddress.ip_address(host)
        return any(addr in net for net in _PRIVATE_NETWORKS)
    except ValueError:
        # Not an IP address — it's a domain name; treat as public
        return False


# ── SearXNG client ───────────────────────────────────────────────────────────

_SEARXNG_TIMEOUT = 10  # seconds


def _call_searxng(
    safe_query: str,
    mode: str = "web",
    max_results: int = 6,
    base_url: str | None = None,
) -> list[SearchResult]:
    """
    HTTP GET to local SearXNG JSON endpoint.
    base_url is validated to be a local address before this function is called.
    Returns empty list on any HTTP/network error.
    """
    url = f"{base_url.rstrip('/')}/search"
    params = {
        "q": safe_query,
        "format": "json",
        "categories": "news" if mode == "news" else "general",
        "language": "en",
    }
    headers = {
        "Accept": "text/html, application/json, */*",
        "User-Agent": "Mozilla/5.0 (compatible; phi-twin/1.0; local)",
    }
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=_SEARXNG_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    results = []
    for item in data.get("results", [])[:max_results]:
        results.append(SearchResult(
            source_id=f"src_{uuid.uuid4().hex[:8]}",
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=(item.get("content") or item.get("snippet") or "")[:400],
            source=item.get("engine", "searxng"),
            query=safe_query,
        ))
    return results


# ── Public search API (the ONLY permitted web search path) ───────────────────

def search_web(
    query: str,
    mode: str = "web",
    max_results: int = 6,
    _policy: dict | None = None,
) -> list[SearchResult]:
    """
    The ONLY permitted web search function.

    Pipeline:
      1. Read SEARXNG_URL env var. Missing → BLOCK.
      2. Validate URL is local (RFC1918 / loopback / docker). Public → BLOCK.
      3. Run privacy_pre_hook("SEARCH_WEB", ...). BLOCK → return [].
      4. Re-verify sanitized query with text_is_safe(). Not safe → BLOCK.
      5. Call local SearXNG.

    Never raises; BLOCK is silent (returns empty list with no external call).
    """
    base_url = os.environ.get("SEARXNG_URL", "").strip()

    # ── Step 1: URL configured? ──────────────────────────────────────────────
    if not base_url:
        _audit_block("SEARXNG_URL not set — web search blocked by default.", query)
        return []

    # ── Step 2: URL must be local ────────────────────────────────────────────
    if not _is_local_url(base_url):
        _audit_block(
            f"SEARXNG_URL '{base_url}' is not a local address — blocked.",
            query,
        )
        return []

    # ── Step 3: Privacy gate ─────────────────────────────────────────────────
    hook = privacy_pre_hook(
        action_type="SEARCH_WEB",
        payload={"query": query, "mode": mode, "tool": "searxng"},
        policy=_policy,
    )

    if hook.decision == "BLOCK":
        return []

    # Resolve the sanitized query
    if hook.decision == "REDACT_AND_PROCEED":
        safe_query = (hook.redacted_payload or {}).get("query", query)
    else:
        safe_query = query

    # ── Step 4: Re-verify after redaction ────────────────────────────────────
    is_clean, remaining = text_is_safe(safe_query)
    if not is_clean:
        _audit_block(
            f"Residual PII after redaction ({[f.kind for f in remaining]}) — blocked.",
            safe_query,
        )
        return []

    # ── Step 5: Call SearXNG ─────────────────────────────────────────────────
    return _call_searxng(safe_query, mode=mode, max_results=max_results, base_url=base_url)


def _audit_block(reason: str, query: str) -> None:
    """Write a BLOCK audit entry without crashing."""
    try:
        from core.audit_append import append_audit, _payload_hash
        append_audit(
            action_type="SEARCH_WEB",
            decision="BLOCK",
            pii_detected=[],
            redaction_notes=[],
            payload_hash=_payload_hash({"query": query}),
            reason=reason,
        )
    except Exception:
        pass  # audit must never break the caller


# ── Batch queries ─────────────────────────────────────────────────────────────

def run_queries(queries: list[str], max_per_query: int = 4) -> list[SearchResult]:
    """Run multiple queries through local SearXNG, deduplicate by URL."""
    all_results: list[SearchResult] = []
    seen_urls: set[str] = set()

    for query in queries[:6]:
        for result in search_web(query, mode="news", max_results=max_per_query):
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                all_results.append(result)
        time.sleep(0.5)

    return all_results[:30]


# ── Formatting helpers (unchanged) ───────────────────────────────────────────

def format_for_prompt(results: list[SearchResult]) -> str:
    """Format search results as context block for the LLM."""
    if not results:
        return "No search results available."
    lines = []
    for r in results:
        lines.append(
            f"[{r.source_id}] [{r.source}] {r.title}\n"
            f"  URL: {r.url}\n"
            f"  {r.snippet}"
        )
    return "\n\n".join(lines)


def results_to_sources(results: list[SearchResult]) -> list[dict]:
    """Convert search results to sources[] format for JSON schemas."""
    return [
        {
            "source_id": r.source_id,
            "url": r.url,
            "title": r.title,
            "date": r.retrieved_at[:10],
            "relevance": r.query,
        }
        for r in results
    ]
