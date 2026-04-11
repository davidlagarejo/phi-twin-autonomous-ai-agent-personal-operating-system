"""
literature.py — Open-access scientific literature search pipeline.
==================================================================
Lane 1 (fully automatic, no login required):
  Sources: OpenAlex, Crossref, Semantic Scholar, arXiv, PubMed

All queries pass through privacy_pre_hook before any outbound HTTP.
Individual source failures are non-fatal: logged as warnings, skipped.
Returns unified LiteratureResult objects with relevance + credibility scores.
"""
from __future__ import annotations

import logging
import math
import re
import time
import urllib.parse
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import sys
from pathlib import Path

# Try to import httpx; fall back to urllib for the simple GETs
try:
    import httpx as _httpx
    _HAS_HTTPX = True
except ImportError:
    import urllib.request
    _HAS_HTTPX = False

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.privacy_pre_hook import privacy_pre_hook
except ImportError:
    privacy_pre_hook = None  # type: ignore[assignment]

log = logging.getLogger("phi.literature")

# ── Exceptions ────────────────────────────────────────────────────────────────

class LiteratureBlockedError(RuntimeError):
    """privacy_pre_hook blocked the query."""
    def __init__(self, reason: str):
        super().__init__(f"Literature search blocked: {reason}")
        self.reason = reason


class LiteratureAPIError(RuntimeError):
    """Non-fatal: one source failed."""
    def __init__(self, source: str, reason: str):
        super().__init__(f"[{source}] {reason}")
        self.source = source
        self.reason = reason


# ── Data type ─────────────────────────────────────────────────────────────────

@dataclass
class LiteratureResult:
    source_id: str
    source_api: str
    title: str
    authors: list[str]
    year: Optional[int]
    venue: Optional[str]
    doi: Optional[str]
    url: str
    open_pdf_url: Optional[str]
    abstract: str
    citations_count: int
    relevance_score: float
    credibility_score: float
    retrieved_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ── HTTP helper ───────────────────────────────────────────────────────────────

_TIMEOUT = 10
_MAILTO = "noreply@phi-twin.local"


def _get_json(url: str, params: dict | None = None, headers: dict | None = None) -> dict | list:
    """Simple GET → JSON. Raises on error."""
    full_url = url
    if params:
        full_url = url + "?" + urllib.parse.urlencode(params)
    if _HAS_HTTPX:
        r = _httpx.get(full_url, headers=headers or {}, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    else:
        req = urllib.request.Request(full_url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            import json
            return json.loads(resp.read().decode("utf-8"))


def _get_text(url: str, params: dict | None = None) -> str:
    """Simple GET → text. Raises on error."""
    full_url = url
    if params:
        full_url = url + "?" + urllib.parse.urlencode(params)
    if _HAS_HTTPX:
        r = _httpx.get(full_url, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.text
    else:
        req = urllib.request.Request(full_url)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return resp.read().decode("utf-8")


# ── Relevance + credibility scoring ──────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    stopwords = {"the", "a", "an", "and", "or", "of", "in", "for", "to", "on", "with", "at", "by", "is", "are", "was", "be"}
    return set(tokens) - stopwords


def _relevance_score(query: str, title: str, abstract: str) -> float:
    """Token overlap between query terms and title (2x weight) + abstract."""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return 0.0
    title_tokens = _tokenize(title)
    abstract_tokens = _tokenize(abstract[:1000])
    title_overlap = len(q_tokens & title_tokens)
    abstract_overlap = len(q_tokens & abstract_tokens)
    score = (title_overlap * 2 + abstract_overlap) / (len(q_tokens) * 3)
    return min(1.0, round(score, 3))


def _credibility_score(doi: Optional[str], source_api: str, open_pdf_url: Optional[str]) -> float:
    if doi:
        return 1.0
    if source_api == "arxiv":
        return 0.7
    if open_pdf_url:
        return 0.6
    return 0.5


def _dedup_by_doi_then_title(results: list[LiteratureResult]) -> list[LiteratureResult]:
    """Remove duplicates; prefer higher credibility_score."""
    seen_dois: dict[str, LiteratureResult] = {}
    seen_titles: dict[str, LiteratureResult] = {}
    deduped = []
    for r in results:
        if r.doi:
            norm_doi = r.doi.lower().strip()
            if norm_doi in seen_dois:
                existing = seen_dois[norm_doi]
                if r.credibility_score > existing.credibility_score:
                    seen_dois[norm_doi] = r
                    deduped = [r if x is existing else x for x in deduped]
                continue
            seen_dois[norm_doi] = r
        else:
            norm_title = r.title.lower().strip()[:60]
            if norm_title in seen_titles:
                existing = seen_titles[norm_title]
                if r.credibility_score > existing.credibility_score:
                    seen_titles[norm_title] = r
                    deduped = [r if x is existing else x for x in deduped]
                continue
            seen_titles[norm_title] = r
        deduped.append(r)
    return deduped


# ── Per-source fetchers ───────────────────────────────────────────────────────

def _fetch_openalex(query: str, max_results: int) -> list[LiteratureResult]:
    """OpenAlex API — no login required."""
    try:
        data = _get_json("https://api.openalex.org/works", params={
            "search": query,
            "per-page": str(min(max_results, 25)),
            "mailto": _MAILTO,
        })
        results = []
        for item in data.get("results", [])[:max_results]:
            title = item.get("title") or ""
            if not title:
                continue
            doi = item.get("doi")
            if doi and doi.startswith("https://doi.org/"):
                doi = doi[len("https://doi.org/"):]
            authors = [
                a.get("author", {}).get("display_name", "")
                for a in (item.get("authorships") or [])[:5]
            ]
            abstract_inv = item.get("abstract_inverted_index") or {}
            abstract = _reconstruct_inverted_abstract(abstract_inv)
            open_pdf = None
            oa = item.get("open_access") or {}
            if oa.get("is_oa") and oa.get("oa_url"):
                open_pdf = oa["oa_url"]
            year = item.get("publication_year")
            venue = (item.get("primary_location") or {}).get("source", {}) or {}
            venue_name = venue.get("display_name") if isinstance(venue, dict) else None
            url = item.get("doi") or f"https://openalex.org/{item.get('id', '')}"
            cred = _credibility_score(doi, "openalex", open_pdf)
            results.append(LiteratureResult(
                source_id=f"lit_{uuid.uuid4().hex[:8]}",
                source_api="openalex",
                title=title,
                authors=authors,
                year=year,
                venue=venue_name,
                doi=doi,
                url=url,
                open_pdf_url=open_pdf,
                abstract=abstract[:500],
                citations_count=item.get("cited_by_count", 0),
                relevance_score=0.0,
                credibility_score=cred,
            ))
        return results
    except Exception as exc:
        raise LiteratureAPIError("openalex", str(exc)) from exc


def _reconstruct_inverted_abstract(inv: dict) -> str:
    """Reconstruct abstract from OpenAlex inverted index."""
    if not inv:
        return ""
    try:
        max_pos = max(pos for positions in inv.values() for pos in positions)
        words = [""] * (max_pos + 1)
        for word, positions in inv.items():
            for pos in positions:
                if pos <= max_pos:
                    words[pos] = word
        return " ".join(words).strip()[:500]
    except Exception:
        return ""


def _fetch_crossref(query: str, max_results: int) -> list[LiteratureResult]:
    """Crossref API — no login required."""
    try:
        data = _get_json("https://api.crossref.org/works", params={
            "query": query,
            "rows": str(min(max_results, 20)),
            "mailto": _MAILTO,
        })
        items = data.get("message", {}).get("items", [])
        results = []
        for item in items[:max_results]:
            title_list = item.get("title") or []
            title = title_list[0] if title_list else ""
            if not title:
                continue
            doi = item.get("DOI")
            authors = []
            for a in (item.get("author") or [])[:5]:
                given = a.get("given", "")
                family = a.get("family", "")
                name = f"{given} {family}".strip()
                if name:
                    authors.append(name)
            year = None
            issued = item.get("issued", {}).get("date-parts")
            if issued and issued[0]:
                year = issued[0][0]
            venue_list = item.get("container-title") or []
            venue = venue_list[0] if venue_list else None
            url = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
            abstract = item.get("abstract") or ""
            # Strip JATS XML tags
            abstract = re.sub(r"<[^>]+>", "", abstract).strip()[:500]
            cred = _credibility_score(doi, "crossref", None)
            results.append(LiteratureResult(
                source_id=f"lit_{uuid.uuid4().hex[:8]}",
                source_api="crossref",
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                doi=doi,
                url=url,
                open_pdf_url=None,
                abstract=abstract,
                citations_count=item.get("is-referenced-by-count", 0),
                relevance_score=0.0,
                credibility_score=cred,
            ))
        return results
    except Exception as exc:
        raise LiteratureAPIError("crossref", str(exc)) from exc


def _fetch_semantic_scholar(query: str, max_results: int) -> list[LiteratureResult]:
    """Semantic Scholar API — no API key required for moderate usage."""
    try:
        data = _get_json(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": str(min(max_results, 10)),
                "fields": "title,authors,year,venue,externalIds,openAccessPdf,abstract,citationCount",
            },
            headers={"User-Agent": "phi-twin-research/1.0 (noreply@phi-twin.local)"},
        )
        results = []
        for item in (data.get("data") or [])[:max_results]:
            title = item.get("title") or ""
            if not title:
                continue
            ext_ids = item.get("externalIds") or {}
            doi = ext_ids.get("DOI")
            arxiv_id = ext_ids.get("ArXiv")
            authors = [a.get("name", "") for a in (item.get("authors") or [])[:5]]
            abstract = (item.get("abstract") or "")[:500]
            oa_pdf = item.get("openAccessPdf") or {}
            open_pdf = oa_pdf.get("url") if oa_pdf else None
            year = item.get("year")
            venue = item.get("venue")
            paper_id = item.get("paperId", "")
            url = f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else (
                f"https://doi.org/{doi}" if doi else f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
            )
            source = "arxiv" if arxiv_id and not doi else "semantic_scholar"
            cred = _credibility_score(doi, source, open_pdf)
            results.append(LiteratureResult(
                source_id=f"lit_{uuid.uuid4().hex[:8]}",
                source_api="semantic_scholar",
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                doi=doi,
                url=url,
                open_pdf_url=open_pdf,
                abstract=abstract,
                citations_count=item.get("citationCount", 0),
                relevance_score=0.0,
                credibility_score=cred,
            ))
        return results
    except Exception as exc:
        raise LiteratureAPIError("semantic_scholar", str(exc)) from exc


def _fetch_arxiv(query: str, max_results: int) -> list[LiteratureResult]:
    """arXiv Atom API — no login required."""
    try:
        xml_text = _get_text(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": f"all:{query}",
                "max_results": str(min(max_results, 10)),
                "sortBy": "relevance",
            },
        )
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        root = ET.fromstring(xml_text)
        results = []
        for entry in root.findall("atom:entry", ns)[:max_results]:
            title = (entry.findtext("atom:title", "", ns) or "").strip().replace("\n", " ")
            if not title:
                continue
            abstract = (entry.findtext("atom:summary", "", ns) or "").strip().replace("\n", " ")[:500]
            arxiv_id = ""
            id_text = entry.findtext("atom:id", "", ns) or ""
            if "arxiv.org/abs/" in id_text:
                arxiv_id = id_text.split("arxiv.org/abs/")[-1].strip()
            doi = entry.findtext("arxiv:doi", None, ns)
            authors = [
                (a.findtext("atom:name", "", ns) or "").strip()
                for a in entry.findall("atom:author", ns)[:5]
            ]
            published = entry.findtext("atom:published", "", ns)
            year = int(published[:4]) if published and len(published) >= 4 else None
            url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else id_text
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None
            # Find journal_ref if any
            journal_ref = entry.findtext("arxiv:journal_ref", None, ns)
            cred = _credibility_score(doi, "arxiv", pdf_url)
            results.append(LiteratureResult(
                source_id=f"lit_{uuid.uuid4().hex[:8]}",
                source_api="arxiv",
                title=title,
                authors=authors,
                year=year,
                venue=journal_ref,
                doi=doi,
                url=url,
                open_pdf_url=pdf_url,
                abstract=abstract,
                citations_count=0,
                relevance_score=0.0,
                credibility_score=cred,
            ))
        return results
    except Exception as exc:
        raise LiteratureAPIError("arxiv", str(exc)) from exc


def _fetch_pubmed(query: str, max_results: int) -> list[LiteratureResult]:
    """PubMed E-utilities — no login required."""
    try:
        # Step 1: esearch for IDs
        search_data = _get_json(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": str(min(max_results, 10)),
                "retmode": "json",
            },
        )
        ids = search_data.get("esearchresult", {}).get("idlist", [])[:max_results]
        if not ids:
            return []

        # Step 2: efetch for summaries
        summary_data = _get_json(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(ids),
                "retmode": "json",
            },
        )
        results_map = summary_data.get("result", {})
        results = []
        for pmid in ids:
            item = results_map.get(pmid)
            if not item or isinstance(item, list):
                continue
            title = item.get("title") or ""
            if not title:
                continue
            authors = [a.get("name", "") for a in (item.get("authors") or [])[:5]]
            year = None
            pubdate = item.get("pubdate") or ""
            if pubdate:
                parts = pubdate.split()
                try:
                    year = int(parts[0])
                except Exception:
                    pass
            doi = None
            for id_obj in (item.get("articleids") or []):
                if id_obj.get("idtype") == "doi":
                    doi = id_obj.get("value")
                    break
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            venue = item.get("fulljournalname") or item.get("source")
            cred = _credibility_score(doi, "pubmed", None)
            results.append(LiteratureResult(
                source_id=f"lit_{uuid.uuid4().hex[:8]}",
                source_api="pubmed",
                title=title,
                authors=authors,
                year=year,
                venue=venue,
                doi=doi,
                url=url,
                open_pdf_url=None,
                abstract="",  # efetch with rettype=abstract requires separate call
                citations_count=0,
                relevance_score=0.0,
                credibility_score=cred,
            ))
        return results
    except Exception as exc:
        raise LiteratureAPIError("pubmed", str(exc)) from exc


# ── Source registry ───────────────────────────────────────────────────────────

_SOURCE_FETCHERS = {
    "openalex":         _fetch_openalex,
    "crossref":         _fetch_crossref,
    "semantic_scholar": _fetch_semantic_scholar,
    "arxiv":            _fetch_arxiv,
    "pubmed":           _fetch_pubmed,
}


# ── Public API ────────────────────────────────────────────────────────────────

def search_literature(
    query: str,
    max_results_per_source: int = 5,
    sources: Optional[list[str]] = None,
    min_relevance: float = 0.1,
    _policy: Optional[dict] = None,
) -> list[LiteratureResult]:
    """
    Privacy-gated multi-source literature search.

    1. Run privacy_pre_hook("SEARCH_WEB", {"query": query}).
    2. Fan out to all enabled sources in parallel.
    3. Dedup by DOI then title.
    4. Score relevance + credibility.
    5. Filter by min_relevance. Sort by (credibility DESC, relevance DESC).
    """
    # ── Privacy gate ──────────────────────────────────────────────────────────
    safe_query = query
    try:
        _hook = privacy_pre_hook
        if _hook is not None:
            result = _hook("SEARCH_WEB", {"query": query}, policy=_policy or {})
            decision = result.get("decision", "ALLOW")
            if decision == "BLOCK":
                raise LiteratureBlockedError(result.get("reason", "PII detected"))
            if decision == "REDACT_AND_PROCEED":
                safe_query = result.get("sanitized_payload", {}).get("query", query)
    except LiteratureBlockedError:
        raise
    except Exception as exc:
        log.warning("privacy_pre_hook unavailable: %s — proceeding with original query", exc)

    active_sources = sources if sources is not None else list(_SOURCE_FETCHERS.keys())

    # ── Fan out in parallel ───────────────────────────────────────────────────
    all_results: list[LiteratureResult] = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {
            executor.submit(_SOURCE_FETCHERS[s], safe_query, max_results_per_source): s
            for s in active_sources if s in _SOURCE_FETCHERS
        }
        for future in as_completed(future_map, timeout=20):
            src = future_map[future]
            try:
                results = future.result()
                all_results.extend(results)
            except LiteratureAPIError as exc:
                log.warning("Literature source %s failed: %s", exc.source, exc.reason)
            except Exception as exc:
                log.warning("Literature source %s unexpected error: %s", src, exc)

    # ── Dedup ─────────────────────────────────────────────────────────────────
    all_results = _dedup_by_doi_then_title(all_results)

    # ── Score ─────────────────────────────────────────────────────────────────
    for r in all_results:
        r.relevance_score = _relevance_score(safe_query, r.title, r.abstract)
        # Recalculate credibility in case dedup swapped the result
        r.credibility_score = _credibility_score(r.doi, r.source_api, r.open_pdf_url)

    # ── Filter + sort ─────────────────────────────────────────────────────────
    filtered = [r for r in all_results if r.relevance_score >= min_relevance]
    # Fall back to all results if nothing passes threshold
    if not filtered and all_results:
        filtered = all_results
    filtered.sort(key=lambda r: (-r.credibility_score, -r.relevance_score))

    # Hard cap: 50
    return filtered[:50]


def format_literature_for_prompt(results: list[LiteratureResult]) -> str:
    """Format as context block for LLM prompt."""
    if not results:
        return ""
    lines = ["[LITERATURE SEARCH RESULTS — cite source_id for any claim]"]
    for i, r in enumerate(results, 1):
        authors_str = ", ".join(r.authors[:3])
        if len(r.authors) > 3:
            authors_str += " et al."
        year_str = str(r.year) if r.year else "n.d."
        doi_str = f" DOI:{r.doi}" if r.doi else ""
        lines.append(
            f"\n[{i}] source_id={r.source_id}\n"
            f"    Title: {r.title}\n"
            f"    Authors: {authors_str} ({year_str}){doi_str}\n"
            f"    Venue: {r.venue or 'n/a'} | Citations: {r.citations_count}\n"
            f"    Credibility: {r.credibility_score:.1f} | Relevance: {r.relevance_score:.2f}\n"
            f"    URL: {r.url}\n"
            f"    Abstract: {r.abstract[:200]}..."
        )
    lines.append("\n[END LITERATURE]")
    return "\n".join(lines)


def results_to_evidence_records(
    results: list[LiteratureResult],
    hypothesis_ids: Optional[list[str]] = None,
) -> list[dict]:
    """Convert LiteratureResult list to evidence.jsonl-compatible dicts."""
    records = []
    for r in results:
        records.append({
            "type": "literature",
            "title": r.title,
            "url": r.url,
            "source_id": r.source_id,
            "snippet": r.abstract[:400],
            "retrieved_at": r.retrieved_at,
            "credibility_score": r.credibility_score,
            "relevance_score": r.relevance_score,
            "hypothesis_ids": hypothesis_ids or [],
            "tags": [r.source_api, f"year:{r.year}" if r.year else "year:unknown"],
            "doi": r.doi,
            "citations_count": r.citations_count,
        })
    return records
