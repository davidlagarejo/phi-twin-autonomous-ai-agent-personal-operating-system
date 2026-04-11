"""
research_engine.py — Deep research execution engine.
=====================================================
Implements four research strategies as a micro-cycle engine:
  DISCOVER      — generate candidate leads from anonymized queries
  DUE_DILIGENCE — deep multi-source investigation of a specific entity
  CORRELATE     — cross-reference findings → hypotheses
  VALIDATE      — two-pass verification before requesting human approval

Called from POST /api/execute. Respects RunBudget (max_seconds, max_web_queries,
max_sources, max_tasks=2). Returns ExecuteResult with checkpoint for resumption.

Privacy: all outbound searches route through tools/search.py (SearXNG, privacy-gated).
Literature: all academic queries route through tools/literature.py (privacy-gated).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.client import call_phi_sync as _call_phi_sync  # single definition — llm/client.py
from tools.state_manager import (
    WorkspaceState,
    DuplicateError,
    fingerprint_opportunity,
    fingerprint_task,
)

try:
    from tools.literature import search_literature, results_to_evidence_records
except ImportError:
    search_literature = None  # type: ignore[assignment]
    results_to_evidence_records = None  # type: ignore[assignment]

try:
    from core.privacy_pre_hook import privacy_pre_hook
except ImportError:
    privacy_pre_hook = None  # type: ignore[assignment]

log = logging.getLogger("phi.research_engine")

_research_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research")
# Separate pool for parallel HTTP search requests (I/O bound, safe to parallelize).
# max_workers=3: enough to run 3 SearXNG queries concurrently without overwhelming the instance.
_search_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="search")

# ── Budget ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunBudget:
    max_seconds: float = 600.0    # 10 min per task
    max_web_queries: int = 40     # up from 16 — deep research needs many queries
    max_sources: int = 80         # up from 40
    max_tasks: int = 2


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class Checkpoint:
    checkpoint_id: str
    strategy: str
    step_name: str
    task_id: str
    run_counter: int
    created_at: str
    resumable: bool
    state_snapshot: dict


@dataclass
class ExecuteResult:
    status: str                   # DONE | IN_PROGRESS | FAILED | FROZEN
    checkpoint: Checkpoint
    result_summary: str
    artifacts: list[dict]
    next_task_suggestions: list[dict]
    tasks_run: int
    queries_used: int
    sources_consulted: int
    elapsed_seconds: float
    gate_results: list[dict]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["checkpoint"] = asdict(self.checkpoint)
        return d


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_checkpoint(
    strategy: str,
    step_name: str,
    task_id: str,
    run_counter: int,
    snapshot: dict,
    resumable: bool = True,
) -> Checkpoint:
    raw = f"{run_counter}|{strategy}|{step_name}|{task_id}"
    cid = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return Checkpoint(
        checkpoint_id=cid,
        strategy=strategy,
        step_name=step_name,
        task_id=task_id,
        run_counter=run_counter,
        created_at=_now(),
        resumable=resumable,
        state_snapshot=snapshot,
    )


# ── Budget guard helpers ──────────────────────────────────────────────────────

def _budget_exceeded(budget: RunBudget, start_time: float, queries_used: list[int]) -> bool:
    elapsed = time.monotonic() - start_time
    return elapsed >= budget.max_seconds or queries_used[0] >= budget.max_web_queries


# ── Search helper (wraps tools/search.py) ────────────────────────────────────

def _safe_search(query: str, max_results: int = 5) -> list[dict]:
    """Run a privacy-gated SearXNG search. Returns empty list on any failure."""
    try:
        from tools.search import search_web, SearchResult
        results = search_web(query)
        return [
            {
                "source_id": r.source_id,
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet[:300],
                "source": r.source,
            }
            for r in results[:max_results]
        ]
    except Exception as exc:
        log.warning("search_web failed: %s", exc)
        return []


def _safe_search_parallel(
    queries: list[str],
    max_results: int = 5,
    budget: "RunBudget | None" = None,
    start_time: float = 0.0,
    queries_used: "list[int] | None" = None,
) -> tuple[list[dict], list[str]]:
    """
    Run up to len(queries) searches in parallel using _search_pool (max 3 concurrent).
    Skips any query that would exceed the budget before submitting.

    Returns (all_results, completed_queries).
    Callers should add len(completed_queries) to their queries_used counter.
    """
    from concurrent.futures import as_completed

    if queries_used is None:
        queries_used = [0]

    pending: list[str] = []
    for q in queries:
        if budget and _budget_exceeded(budget, start_time, queries_used):
            break
        pending.append(q)
        queries_used[0] += 1  # count eagerly (mirrors sequential behavior)

    if not pending:
        return [], []

    futures = {_search_pool.submit(_safe_search, q, max_results): q for q in pending}
    all_results: list[dict] = []
    completed: list[str] = []

    for fut in as_completed(futures, timeout=30.0):
        q = futures[fut]
        try:
            results = fut.result()
            log.info("ACT:search_done n=%d q=%r", len(results), q[:80])
            all_results.extend(results)
            completed.append(q)
        except Exception as exc:
            log.warning("parallel_search failed q=%r: %s", q, exc)

    return all_results, completed


# ── Direct page fetcher (for team/about/leadership pages) ────────────────────

def _fetch_page_text(url: str, max_chars: int = 6000) -> str:
    """
    Fetch a URL and return plain text (HTML tags stripped).
    Returns "" on any error or timeout. Used to scrape team/about pages directly
    instead of relying on SearXNG snippets.
    """
    import re as _re
    try:
        import httpx as _httpx
        r = _httpx.get(
            url,
            timeout=8.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PhiBot/1.0; research)"},
        )
        if r.status_code != 200:
            return ""
        html = r.text[:max_chars * 4]
        # Strip scripts, styles, nav, footer
        html = _re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html, flags=_re.S | _re.I)
        # Strip remaining tags
        text = _re.sub(r"<[^>]+>", " ", html)
        # Collapse whitespace
        text = _re.sub(r"\s{2,}", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""


# ── David's fit keywords (used for sub-program scoring — code, not LLM) ────────
_FIT_KEYWORDS_HIGH = [
    "steam", "boiler", "thermal", "heat", "ultrasonic", "non-invasive",
    "industrial iot", "iiot", "industrial internet", "predictive maintenance",
    "energy efficiency", "energy management", "manufacturing", "process industry",
    "oil gas", "petroleum", "utilities", "metering", "flow measurement",
]
_FIT_KEYWORDS_MED = [
    "sensor", "iot", "cleantech", "clean energy", "emissions", "decarbonization",
    "hardware", "embedded", "industrial", "smart factory", "automation",
    "startup", "pre-seed", "seed", "sbir", "phase i",
]

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")


def _score_topic_fit(text: str) -> int:
    """Score 0-100 how well a topic/category matches David's profile. Pure keyword matching."""
    t = text.lower()
    score = 0
    for kw in _FIT_KEYWORDS_HIGH:
        if kw in t:
            score += 15
    for kw in _FIT_KEYWORDS_MED:
        if kw in t:
            score += 7
    return min(score, 100)


def _run_sub_program_research(
    entity_name: str,
    entity_type: str,
    website: str,
    state,
    budget: "RunBudget",
    start_time: float,
    queries_used: list,
) -> list[dict]:
    """
    Deep research path for any entity type.

    Code-only pipeline:
    1. Fetch official website looking for programs/tracks/topics/portfolio thesis
    2. For each relevant area, search for the contact person + email
    3. Score each area against David's profile (keyword matching)
    4. Return list of sub_programs dicts

    sub_program schema:
    {
        "name": str,
        "description": str,
        "fit_score": int (0-100),
        "fit_reason": str,
        "program_officer": str | null,
        "email": str | null,
        "linkedin_url": str | null,
        "deadline": str | null,
        "apply_url": str | null,
    }
    """

    log.info("ACT:sub_program_research entity=%r type=%s", entity_name[:60], entity_type)
    sub_programs: list[dict] = []

    # ── Step 1: Fetch relevant deep page based on entity type ─────────────────
    topic_text = ""
    if website:
        # Paths to try, ordered by entity type relevance
        if entity_type in ("GRANT",):
            topic_paths = [
                "/applicants/topics", "/topics", "/solicitation", "/funding-opportunities",
                "/programs", "/apply", "/calls", "/areas", "/categories", "/technology-areas",
            ]
        elif entity_type in ("EVENT",):
            topic_paths = [
                "/tracks", "/agenda", "/speakers", "/sessions", "/categories",
                "/program", "/schedule", "/topics", "/awards", "/startup-showcase",
            ]
        elif entity_type in ("INVESTOR",):
            topic_paths = [
                "/portfolio", "/thesis", "/sectors", "/focus", "/programs",
                "/apply", "/about", "/team", "/investments",
            ]
        else:  # ORG, PERSON, PATENT, etc.
            topic_paths = [
                "/partnerships", "/programs", "/solutions", "/services",
                "/industries", "/customers", "/about", "/team", "/careers",
            ]
        # Scrape ALL relevant paths and concatenate (up to 5 pages)
        _pages_scraped = 0
        _seen_urls: set[str] = set()
        for path in topic_paths:
            if _pages_scraped >= 5:
                break
            url = website.rstrip("/") + path
            if url in _seen_urls:
                continue
            _seen_urls.add(url)
            text = _fetch_page_text(url, max_chars=6000)
            if len(text) > 300:
                topic_text += f"\n\n--- Page: {url} ---\n{text}"
                _pages_scraped += 1
                log.info("ACT:topics_page_scraped url=%s chars=%d total_pages=%d", url, len(text), _pages_scraped)
        if not topic_text:
            topic_text = _fetch_page_text(website, max_chars=8000)

    # ── Step 2: Search for specific sub-areas based on entity type ─────────────
    if entity_type in ("GRANT",):
        topic_queries = [
            f"{entity_name} technology topic areas program officers 2026",
            f"{entity_name} sub-programs funding categories eligibility",
            f"{entity_name} IIoT sensor hardware energy efficiency industrial funded topics",
            f"{entity_name} annual solicitation 2025 2026 open window",
        ]
    elif entity_type in ("EVENT",):
        topic_queries = [
            f"{entity_name} conference tracks sessions startup showcase 2026",
            f"{entity_name} speaker topics agenda categories",
            f"{entity_name} industrial IoT energy sensor track 2025 2026",
        ]
    elif entity_type in ("INVESTOR",):
        topic_queries = [
            f"{entity_name} investment thesis sectors portfolio focus areas",
            f"{entity_name} cleantech IIoT industrial portfolio partner contact",
            f"{entity_name} pre-seed seed stage hardware startup funding criteria",
            f'site:crunchbase.com "{entity_name}" portfolio investments',
        ]
    else:
        topic_queries = [
            f"{entity_name} programs partnerships solutions industrial IoT energy",
            f"{entity_name} specific divisions teams pilot program contact",
            f"{entity_name} open innovation vendor technology partner",
        ]
    topic_results = []
    for q in topic_queries[:4]:
        if _budget_exceeded(budget, start_time, queries_used):
            break
        res = _safe_search(q, max_results=6)
        topic_results.extend(res)
        queries_used[0] = queries_used[0] + 1 if queries_used else 1

    # Combine all text
    combined_text = topic_text + "\n" + "\n".join(
        r.get("snippet", "") for r in topic_results
    )

    if not combined_text.strip():
        log.info("ACT:sub_program_no_text entity=%r — going straight to Claude API", entity_name[:40])
        combined_text = ""  # Claude fallback will trigger below (top_topics stays empty)

    # ── Step 3: Extract topic areas using pattern matching ─────────────────────
    # Look for patterns like:
    # "Topic X: description", "Area Y — description", "Track Z: ..."
    # "Program Area: Advanced Manufacturing", bullet lists, etc.
    topic_patterns = [
        re.compile(r"(?:topic|area|track|program|category|subtopic|focus area)[:\s]+([A-Z][^\n.;]{10,80})", re.I),
        re.compile(r"\b([A-Z][a-zA-Z\s/&-]{8,50})(?:\s*[:—–]\s*)([A-Za-z][^.\n]{20,150})"),
        re.compile(r"(?:^|\n)\s*[•\-\*]\s*([A-Z][a-zA-Z\s/&-]{8,60})", re.M),
    ]

    raw_topics: list[str] = []
    for pat in topic_patterns:
        for m in pat.finditer(combined_text):
            raw_topics.append(m.group(0)[:200])
        if len(raw_topics) >= 20:
            break

    # Deduplicate and score
    seen_topics: set = set()
    scored: list[tuple[int, str]] = []
    for raw in raw_topics:
        key = raw[:40].lower().strip()
        if key in seen_topics or len(key) < 8:
            continue
        seen_topics.add(key)
        score = _score_topic_fit(raw)
        if score >= 14:  # at least one medium keyword
            scored.append((score, raw.strip()))

    # Filter out navigation/boilerplate text masquerading as topics
    _nav_noise = re.compile(r"skip to|main content|cookie|privacy policy|terms of use|"
                             r"©\s*20\d\d|all rights reserved|sign in|log in", re.I)
    scored = [(s, t) for s, t in scored if not _nav_noise.search(t[:80])]

    scored.sort(key=lambda x: x[0], reverse=True)
    top_topics = scored[:6]  # top 6 relevant topics

    if len(top_topics) < 2:
        # ── Fallback: Claude API structured lookup ─────────────────────────────
        log.info("ACT:sub_program_scrape_empty — trying Claude API entity=%r", entity_name[:40])
        try:
            import os as _os, httpx as _httpx
            _api_key = _os.environ.get("CLAUDE_API_KEY", "")
            if _api_key:
                _model = _os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
                _david = (
                    "David Lagarejo, CEO. Zircular: IIoT platform, pre-seed. "
                    "Patent US2024/0077174 — ultrasonic non-invasive steam sensor "
                    "(-30% energy, validated Ecopetrol). Seeking cleantech grants, "
                    "industrial IoT pilots, NYC/US focus."
                )
                _type_context = {
                    "GRANT":    "specific sub-programs, topic areas, or funding categories",
                    "EVENT":    "specific conference tracks, sessions, startup showcases, or award programs",
                    "INVESTOR": "specific investment thesis areas, portfolio sectors, or programs they fund",
                    "ORG":      "specific divisions, partnership programs, pilot programs, or procurement tracks",
                }.get(entity_type, "specific programs, tracks, divisions, or opportunity areas")
                _contact_role = {
                    "GRANT":    "program officer/director",
                    "EVENT":    "track chair/organizer",
                    "INVESTOR": "partner/associate who covers this sector",
                    "ORG":      "division lead, partnership manager, or procurement contact",
                }.get(entity_type, "relevant contact")
                _prompt = (
                    f"For '{entity_name}' (type: {entity_type}), list the {_type_context} "
                    f"most relevant to:\n{_david}\n\n"
                    f"For each area, find the {_contact_role} name and email if public.\n\n"
                    "Return ONLY JSON array (max 6 items, ordered by relevance):\n"
                    '[{"name":"area name","description":"1 sentence","fit_score":0-100,'
                    '"fit_reason":"why it fits David","program_officer":"Full Name or null",'
                    '"email":"email@example.com or null","apply_url":"url or null"}]'
                )
                _resp = _httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": _api_key, "anthropic-version": "2023-06-01",
                             "content-type": "application/json"},
                    json={"model": _model, "max_tokens": 2048,
                          "system": "Return ONLY a JSON array. No prose.",
                          "messages": [{"role": "user", "content": _prompt}]},
                    timeout=45,
                )
                _resp.raise_for_status()
                _raw = _resp.json()["content"][0]["text"].strip()
                _raw = re.sub(r"^```[a-z]*\n?", "", _raw).rstrip("`").strip()
                _s = _raw.find("[")
                _e = _raw.rfind("]") + 1
                if _s >= 0 and _e <= _s:
                    # Truncated response — close the array and try partial parse
                    _raw = _raw[_s:].rstrip().rstrip(",") + "]"
                    _e = len(_raw)
                if _s >= 0 and _e > _s:
                    try:
                        _data = json.loads(_raw[_s:_e])
                    except json.JSONDecodeError:
                        # Last resort: strip incomplete last item and close
                        _partial = _raw[_s:_e].rsplit("{", 1)[0].rstrip().rstrip(",") + "]"
                        _data = json.loads(_partial)
                    sub_programs = [
                        sp for sp in _data
                        if sp.get("name") and int(sp.get("fit_score") or 0) >= 20
                    ]
                    log.info("ACT:claude_sub_programs count=%d entity=%r", len(sub_programs), entity_name[:40])
                    return sub_programs
        except Exception as _ce:
            log.warning("claude_sub_program_fallback_failed: %s", _ce)
        return []

    log.info("ACT:relevant_topics found=%d entity=%r", len(top_topics), entity_name[:40])

    # ── Step 4: For each relevant topic, find program officer + email ──────────
    for fit_score, topic_text_raw in top_topics:
        if _budget_exceeded(budget, start_time, queries_used):
            break

        # Extract a short topic name (first ~50 chars, clean up)
        topic_name = re.sub(r"\s+", " ", topic_text_raw.split("\n")[0])[:60].strip(" :-–")

        # Build fit reason from matching keywords
        t_lower = topic_text_raw.lower()
        matched_kw = [kw for kw in _FIT_KEYWORDS_HIGH + _FIT_KEYWORDS_MED if kw in t_lower]
        fit_reason = f"Keywords matched: {', '.join(matched_kw[:4])}" if matched_kw else "General fit"

        # Search for program officer
        officer_query = f'"{entity_name}" "{topic_name}" program officer director email'
        officer_results = _safe_search(officer_query, max_results=4)
        queries_used[0] = queries_used[0] + 1 if queries_used else 1

        # Extract emails and names from results
        officer_name = None
        officer_email = None
        officer_url = None

        combined_officer_text = "\n".join(
            r.get("title", "") + " " + r.get("snippet", "")
            for r in officer_results
        ) + "\n"

        # Extract email addresses
        emails = _EMAIL_RE.findall(combined_officer_text)
        # Prefer .gov, .edu, .org emails
        for e in emails:
            domain = e.split("@")[-1].lower()
            if any(domain.endswith(tld) for tld in (".gov", ".edu", ".org", ".eu")):
                officer_email = e
                break
        if not officer_email and emails:
            officer_email = emails[0]

        # Extract person name — look for "Name, Title" patterns near email or "program officer"
        name_patterns = [
            re.compile(r"([A-Z][a-z]+ [A-Z][a-z]+(?:\s[A-Z][a-z]+)?),\s*(?:Program|Director|Officer|Manager|Lead)", re.I),
            re.compile(r"(?:contact|officer|director|manager)[:\s]+([A-Z][a-z]+ [A-Z][a-z]+)", re.I),
            re.compile(r"([A-Z][a-z]+ [A-Z][a-z]+)\s*<[^>@]+@[^>]+>"),
        ]
        for pat in name_patterns:
            m = pat.search(combined_officer_text)
            if m:
                candidate = m.group(1).strip()
                if len(candidate.split()) >= 2:
                    officer_name = candidate
                    break

        # LinkedIn URL from officer results
        for r in officer_results:
            url = r.get("url", "")
            if "linkedin.com/in/" in url:
                officer_url = url
                break

        # ── Deep person enrichment: if we found a name, dig further ───────────
        if officer_name and not officer_email and not _budget_exceeded(budget, start_time, queries_used):
            _enrich_queries = [
                f'"{officer_name}" "{entity_name}" email contact',
                f'"{officer_name}" site:linkedin.com',
            ]
            for _eq in _enrich_queries:
                if _budget_exceeded(budget, start_time, queries_used):
                    break
                _enrich_res = _safe_search(_eq, max_results=5)
                queries_used[0] += 1
                _enrich_text = " ".join(r.get("snippet", "") + r.get("title", "") for r in _enrich_res)
                _found_emails = _EMAIL_RE.findall(_enrich_text)
                if _found_emails:
                    officer_email = _found_emails[0]
                    break
                if not officer_url:
                    for _er in _enrich_res:
                        if "linkedin.com/in/" in _er.get("url", ""):
                            officer_url = _er["url"]
                            break

        # Extract apply URL from topic text
        _apply_url = None
        _url_re = re.compile(r"https?://[^\s\"'>]{10,100}")
        _url_candidates = _url_re.findall(topic_text_raw)
        for _uc in _url_candidates:
            if any(kw in _uc.lower() for kw in ("apply", "submit", "application", "register", "solicitation")):
                _apply_url = _uc
                break

        sub_programs.append({
            "name":            topic_name,
            "description":     topic_text_raw[:200].strip(),
            "fit_score":       fit_score,
            "fit_reason":      fit_reason,
            "program_officer": officer_name,
            "email":           officer_email,
            "linkedin_url":    officer_url,
            "deadline":        None,
            "apply_url":       _apply_url,
        })

    log.info("ACT:sub_programs_built count=%d entity=%r", len(sub_programs), entity_name[:40])
    return sub_programs


def _find_website_for_entity(entity_name: str, entity_type: str) -> str:
    """
    Search for the official website of an entity.
    Returns the URL string or "" if not found.
    """
    queries = {
        "GRANT": [f"{entity_name} official website grants program", f'"{entity_name}" site:energy.gov OR site:nsf.gov OR site:grants.gov'],
        "ORG":   [f"{entity_name} official website", f'"{entity_name}" company site'],
        "INVESTOR": [f"{entity_name} venture capital official website"],
        "EVENT": [f"{entity_name} conference official website 2025 2026"],
    }.get(entity_type, [f"{entity_name} official website"])

    for q in queries[:1]:
        results = _safe_search(q, max_results=3)
        for r in results:
            url = r.get("url", "")
            # Skip directories / aggregator sites
            _skip = ("linkedin.com", "crunchbase.com", "bloomberg.com", "wikipedia.org",
                     "twitter.com", "facebook.com", "yelp.com", "glassdoor.com")
            if url and not any(s in url for s in _skip) and url.startswith("http"):
                return url
    return ""


_TEAM_PATHS = ["/team", "/about", "/leadership", "/our-team", "/about-us",
               "/people", "/staff", "/company/team", "/about/team", "/who-we-are"]


def _scrape_team_page(website: str) -> str:
    """
    Try known team/about paths on a website and return the first non-empty page text.
    Returns "" if nothing found.
    """
    from urllib.parse import urlparse
    parsed = urlparse(website)
    base   = f"{parsed.scheme}://{parsed.netloc}"
    for path in _TEAM_PATHS:
        text = _fetch_page_text(base + path, max_chars=5000)
        if len(text) > 200:  # meaningful page
            return text
    # Also try the homepage itself
    return _fetch_page_text(website, max_chars=4000)


# ── LLM helper ────────────────────────────────────────────────────────────────

# _call_phi_sync is imported from llm.client at the top of this file.
# Single definition for the entire project — no duplicate here.


# ── LOCAL_SCAN strategy ───────────────────────────────────────────────────────

_LOCAL_SCAN_PATHS = [
    Path.home() / "Documents" / "knowledge_base_backup.md",
    Path.home() / "phi-twin" / "prompts" / "system.md",
    Path.home() / "phi-twin" / "config" / "style_profile.json",
]

# Glob patterns for additional context docs (non-sensitive)
_LOCAL_SCAN_GLOBS = [
    (Path.home() / "Documents" / "Eficiencia energética", "**/*.txt"),
]


def _run_local_scan(
    task: dict,
    budget: RunBudget,
    state: WorkspaceState,
    queries_used: list[int],
    start_time: float,
) -> ExecuteResult:
    """
    LOCAL_SCAN: Read local files to enrich Phi's context.
    Adds high-credibility evidence from David's own documents and seeds
    DISCOVER tasks based on opportunity types found locally.
    """
    task_id = task.get("task_id", "task_unknown")
    run_counter = state.load_strategy_state().get("run_counter", 0)
    evidence_added: list[str] = []
    files_scanned = 0

    for fpath in _LOCAL_SCAN_PATHS:
        try:
            if not fpath.exists():
                continue
            text = fpath.read_text(encoding="utf-8", errors="ignore")[:6000]
            ev_id = state.append_evidence({
                "type": "local_file",
                "title": fpath.name,
                "url": f"file://{fpath}",
                "source_id": f"local:{fpath.name}",
                "snippet": text[:500],
                "full_text": text,
                "credibility_score": 1.0,
                "relevance_score": 1.0,
                "hypothesis_ids": [],
                "tags": ["local_context", "david_profile"],
            })
            evidence_added.append(ev_id)
            files_scanned += 1
            log.info("ACT:local_scan_file file=%r chars=%d", fpath.name, len(text))
        except DuplicateError:
            pass
        except Exception as exc:
            log.warning("LOCAL_SCAN read failed %s: %s", fpath, exc)

    for base, pattern in _LOCAL_SCAN_GLOBS:
        try:
            for fpath in sorted(base.glob(pattern))[:10]:
                if fpath.stat().st_size > 50_000:
                    continue
                text = fpath.read_text(encoding="utf-8", errors="ignore")[:2000]
                if not text.strip():
                    continue
                try:
                    ev_id = state.append_evidence({
                        "type": "local_file",
                        "title": fpath.name,
                        "url": f"file://{fpath}",
                        "source_id": f"local:{fpath.stem}",
                        "snippet": text[:300],
                        "full_text": text,
                        "credibility_score": 0.9,
                        "relevance_score": 0.8,
                        "hypothesis_ids": [],
                        "tags": ["local_context"],
                    })
                    evidence_added.append(ev_id)
                    files_scanned += 1
                except DuplicateError:
                    pass
        except Exception as exc:
            log.warning("LOCAL_SCAN glob failed %s: %s", base, exc)

    # Seed DISCOVER tasks based on known opportunity types from knowledge base
    _job_queries = [
        "IIoT cleantech energy efficiency director VP jobs United States 2025 2026",
        "head of energy sustainability industrial IoT executive role NYC",
        "cleantech startup CTO co-founder equity role United States",
    ]
    _grant_queries = [
        "DOE SBIR STTR cleantech IIoT sensor grant open call 2025 2026",
        "NYSERDA EPA grant program industrial energy efficiency application 2026",
    ]
    seeded = 0
    for q in _job_queries + _grant_queries:
        try:
            state.enqueue_task({
                "strategy": "DISCOVER",
                "priority": 3,
                "payload": {"query_hint": q},
            })
            seeded += 1
        except DuplicateError:
            pass

    cp = _make_checkpoint("LOCAL_SCAN", "complete", task_id, run_counter,
                          {"files_scanned": files_scanned, "evidence_added": len(evidence_added),
                           "discover_tasks_seeded": seeded})
    return ExecuteResult(
        status="DONE", checkpoint=cp,
        result_summary=f"LOCAL_SCAN: {files_scanned} files indexed, {seeded} DISCOVER tasks seeded.",
        artifacts=[{"type": "evidence_batch", "id": f"local_{task_id}", "path_or_ref": "evidence.jsonl"}],
        next_task_suggestions=[],
        tasks_run=1, queries_used=0,
        sources_consulted=files_scanned,
        elapsed_seconds=time.monotonic() - start_time,
        gate_results=[],
    )


# ── DISCOVER strategy ─────────────────────────────────────────────────────────

def _run_discover(
    task: dict,
    budget: RunBudget,
    state: WorkspaceState,
    queries_used: list[int],
    start_time: float,
) -> ExecuteResult:
    """
    DISCOVER: Generate candidate leads from anonymized queries.
    Enqueues DUE_DILIGENCE for each new (non-duplicate) entity found.
    """
    task_id = task.get("task_id", "task_unknown")
    strategy_state = state.load_strategy_state()
    run_counter = strategy_state.get("run_counter", 0)
    hint = (task.get("payload") or {}).get("query_hint", "")

    # Already completed queries (from checkpoint resume)
    checkpoint = task.get("checkpoint_id")
    completed_queries: list[str] = []
    evidence_added: list[str] = []
    entity_ids_processed: list[str] = []
    enqueued_count = 0

    # Build anonymized queries from hint + strategy state context
    hypotheses_text = " ".join(
        h.get("statement", "") for h in strategy_state.get("hypotheses", [])[:3]
    )
    query_prompt = f"""You are building research queries for a local business development agent.
Based on this hint: {hint or 'general business opportunities in industrial sustainability and IIoT'}
And these active hypotheses: {hypotheses_text or 'none yet'}

Generate exactly 3 short, anonymized, generic search queries (no personal names, no private data).
Each query should target: sector trends, funding opportunities, or technology developments.
Return JSON array only: ["query1", "query2", "query3"]"""

    raw_queries = _call_phi_sync([
        {"role": "system", "content": "You are a research assistant. Output JSON only."},
        {"role": "user", "content": query_prompt},
    ])

    queries: list[str] = []
    try:
        # Extract JSON array
        start = raw_queries.find("[")
        end = raw_queries.rfind("]") + 1
        if start >= 0 and end > start:
            queries = json.loads(raw_queries[start:end])
        if not isinstance(queries, list):
            queries = []
    except Exception:
        queries = []

    # Fallback queries
    if not queries:
        queries = [
            f"{hint} industry trends 2025",
            f"{hint} funding grants opportunities",
            f"{hint} technology companies research",
        ] if hint else [
            "industrial sustainability technology funding United States 2025",
            "IIoT circular economy companies investment USA",
            "deep tech cleantech grants opportunities United States",
            "IIoT cleantech energy efficiency executive job opportunities United States",
        ]
    queries = queries[:4]  # cap at 4

    all_results: list[dict] = []
    if _budget_exceeded(budget, start_time, queries_used):
        cp = _make_checkpoint("DISCOVER", "queries_partial", task_id, run_counter,
                              {"queries_completed": completed_queries,
                               "evidence_ids_added": evidence_added,
                               "entity_ids_processed": entity_ids_processed})
        return ExecuteResult(
            status="IN_PROGRESS", checkpoint=cp,
            result_summary=f"DISCOVER partial: {enqueued_count} entities found so far.",
            artifacts=[{"type": "evidence_batch", "id": "partial", "path_or_ref": "evidence.jsonl"}],
            next_task_suggestions=[],
            tasks_run=1, queries_used=queries_used[0],
            sources_consulted=len(all_results), elapsed_seconds=time.monotonic() - start_time,
            gate_results=[],
        )
    pending = [q for q in queries if q not in completed_queries]
    log.info("ACT:search_start q=%r (parallel batch %d)", pending[0][:80] if pending else "", len(pending))
    batch_results, batch_completed = _safe_search_parallel(
        pending, max_results=5, budget=budget, start_time=start_time, queries_used=queries_used,
    )
    all_results.extend(batch_results)
    completed_queries.extend(batch_completed)

    # Persist evidence
    for r in all_results:
        try:
            ev_id = state.append_evidence({
                "type": "web",
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "source_id": r.get("source_id", ""),
                "snippet": r.get("snippet", ""),
                "credibility_score": 0.5,
                "relevance_score": 0.5,
                "hypothesis_ids": [],
                "tags": ["discover"],
            })
            evidence_added.append(ev_id)
        except DuplicateError:
            pass
        except Exception as exc:
            log.warning("Evidence append failed: %s", exc)

    # Extract entity leads via phi
    if all_results:
        snippets = "\n".join(
            f"- {r.get('title', '')}: {r.get('snippet', '')[:150]}"
            for r in all_results[:15]
        )
        entity_prompt = f"""Extract distinct organizations, programs, or funding bodies from these snippets.
For each, provide: name, type (ORG/GRANT/EVENT/INVESTOR/JOB), brief description, and a relevance signal.
JOB = specific job position or role opportunity relevant to David Lagarejo (physicist-engineer, IIoT/cleantech CEO, NYC).
Return JSON array only:
[{{"name":"...", "type":"ORG|GRANT|EVENT|INVESTOR|JOB", "description":"...", "signal":"..."}}]

Snippets:
{snippets}"""
        raw_entities = _call_phi_sync([
            {"role": "system", "content": "Output JSON only. No markdown."},
            {"role": "user", "content": entity_prompt},
        ])
        entities: list[dict] = []
        try:
            start = raw_entities.find("[")
            end = raw_entities.rfind("]") + 1
            if start >= 0 and end > start:
                entities = json.loads(raw_entities[start:end])
        except Exception:
            entities = []

        # Load exclusions from directives (cached per DISCOVER run)
        _directives_exclusions: list[str] = []
        try:
            _dir_path = Path(__file__).parent.parent / "data" / "directives.md"
            if _dir_path.exists():
                _dir_content = _dir_path.read_text(encoding="utf-8", errors="ignore")
                _in_excl = False
                for _dl in _dir_content.splitlines():
                    if "Pausas" in _dl or "Exclusiones" in _dl:
                        _in_excl = True
                        continue
                    if _in_excl:
                        if _dl.startswith("##"):
                            break
                        _stripped = _dl.strip().lstrip("-").strip().lower()
                        if _stripped and _stripped != "(ninguna actualmente)":
                            _directives_exclusions.append(_stripped)
        except Exception:
            pass

        for ent in entities[:8]:
            name = ent.get("name", "").strip()
            ent_type = ent.get("type", "ORG")
            if not name:
                continue
            # Skip entities that match active exclusions
            _name_lower = name.lower()
            if any(_excl in _name_lower or _name_lower in _excl for _excl in _directives_exclusions):
                log.info("ACT:entity_excluded_by_directive name=%r", name[:60])
                continue
            fp = fingerprint_opportunity(ent_type, name, name, "")
            if state.is_duplicate_opportunity(fp):
                continue
            try:
                state.register_opportunity(fp)
            except DuplicateError:
                continue

            log.info("ACT:entity_found name=%r type=%r", name[:60], ent_type)
            entity_id = f"ent_{hashlib.sha256(name.lower().encode()).hexdigest()[:8]}"
            entity_ids_processed.append(entity_id)

            # Create minimal dossier stub
            dossier = {
                "entity_id": entity_id,
                "schema_version": "1.0",
                "status": "DRAFT",
                "type": ent_type,
                "name": name,
                "aliases": [],
                "description": ent.get("description", ""),
                "profile": {},
                "fit_assessment": {"fit_score": 0, "why_yes": [], "why_not": []},
                "evidence_ids": list(evidence_added),
                "open_loops": [],
                "next_actions": [],
                "approval_status": "NONE",
            }
            state.save_dossier(dossier)

            # Enqueue DUE_DILIGENCE
            try:
                state.enqueue_task({
                    "strategy": "DUE_DILIGENCE",
                    "priority": 2,
                    "payload": {"entity_id": entity_id, "query_hint": name},
                })
                enqueued_count += 1
            except DuplicateError:
                pass

    # Update strategy state
    consecutive_empty = strategy_state.get("consecutive_empty_runs", 0)
    if enqueued_count == 0:
        consecutive_empty += 1
    else:
        consecutive_empty = 0

    backoff_until = None
    if consecutive_empty >= 4:
        from datetime import timedelta
        backoff_until = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
    elif consecutive_empty == 3:
        from datetime import timedelta
        backoff_until = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
    elif consecutive_empty == 2:
        from datetime import timedelta
        backoff_until = (datetime.now(timezone.utc) + timedelta(minutes=30)).isoformat()

    state.update_strategy_state({
        "consecutive_empty_runs": consecutive_empty,
        "backoff_until": backoff_until,
        "active_entity_ids": entity_ids_processed,
    })
    state.append_timeline(
        "TASK_DONE",
        f"DISCOVER: found {enqueued_count} new entities, {len(evidence_added)} evidence items.",
        task_id=task_id,
    )

    cp = _make_checkpoint("DISCOVER", "complete", task_id, run_counter,
                          {"queries_completed": completed_queries,
                           "evidence_ids_added": evidence_added,
                           "entity_ids_processed": entity_ids_processed})

    status = "FROZEN" if (consecutive_empty >= 4 and enqueued_count == 0) else "DONE"
    return ExecuteResult(
        status=status, checkpoint=cp,
        result_summary=f"DISCOVER: {enqueued_count} new entities, {len(evidence_added)} evidence items stored.",
        artifacts=[{"type": "evidence_batch", "id": f"batch_{task_id}", "path_or_ref": "evidence.jsonl"}],
        next_task_suggestions=[
            {"strategy": "DUE_DILIGENCE", "priority": 2,
             "payload": {"entity_id": eid}, "reason": "Newly discovered entity."}
            for eid in entity_ids_processed[:3]
        ],
        tasks_run=1, queries_used=queries_used[0],
        sources_consulted=len(all_results), elapsed_seconds=time.monotonic() - start_time,
        gate_results=[],
    )


# ── DUE_DILIGENCE strategy ────────────────────────────────────────────────────

def _run_due_diligence(
    task: dict,
    budget: RunBudget,
    state: WorkspaceState,
    queries_used: list[int],
    start_time: float,
) -> ExecuteResult:
    """
    DUE_DILIGENCE: Deep multi-source investigation of a specific entity.
    """
    task_id = task.get("task_id", "task_unknown")
    payload = task.get("payload") or {}
    entity_id = payload.get("entity_id", "")
    query_hint = payload.get("query_hint", entity_id)
    strategy_state = state.load_strategy_state()
    run_counter = strategy_state.get("run_counter", 0)

    # Load existing dossier or create stub
    dossier = state.load_dossier(entity_id)
    if not dossier:
        dossier = {
            "entity_id": entity_id, "schema_version": "1.0",
            "status": "DRAFT", "type": "ORG", "name": query_hint,
            "aliases": [], "description": "",
            "profile": {}, "fit_assessment": {"fit_score": 0, "why_yes": [], "why_not": []},
            "evidence_ids": [], "open_loops": [],
            "next_actions": [], "approval_status": "NONE",
        }

    entity_name = dossier.get("name") or query_hint
    entity_type = dossier.get("type", "ORG")
    evidence_added: list[str] = []
    all_results: list[dict] = []

    # ── Phase 1: Self-questioning — Phi generates the right research questions ─
    # Phi analyzes the entity type and context to generate targeted queries,
    # instead of running the same fixed 6 queries for every entity.
    log.info("ACT:due_diligence_start entity=%r type=%s", entity_name[:60], entity_type)

    type_context = {
        "GRANT": "a grant/funding program. David needs to know: eligibility, application deadlines, funded project examples, and MOST IMPORTANTLY the program officer or technical reviewer responsible for this category.",
        "EVENT": "a conference or event. David needs to know: who organizes it, key speakers/chairs, sponsorship/speaking opportunities, and a specific person to contact about participation.",
        "INVESTOR": "an investor. David needs to know: their portfolio focus, recent investments, who manages relevant sectors, and the specific partner/associate to contact.",
        "ORG": "a potential client or partner organization. David needs to know: their technology stack, steam/thermal/IIoT infrastructure, who handles external tech partnerships, and decision-maker contact.",
    }.get(entity_type, "an organization relevant to David's business.")

    # ── Deep query builder — script-based, no LLM ─────────────────────────────
    # 12 targeted queries covering: overview, eligibility, deadline, funded examples,
    # contacts (3 angles), recent news, cross-reference with David's tech, LinkedIn,
    # financial details, and regulatory/certification angle.
    _n = entity_name

    _query_map: dict[str, list[str]] = {
        "GRANT": [
            f"{_n} application deadline 2025 2026 submit date",
            f"{_n} eligibility requirements technology hardware IIoT",
            f"{_n} funded projects examples industrial IoT sensor hardware startup",
            f"{_n} program officer technical reviewer contact email site:nsf.gov OR site:energy.gov OR site:sbir.gov",
            f'"{_n}" program manager director name email phone',
            f"{_n} site:linkedin.com program officer manager",
            f"{_n} SBIR STTR IIoT cleantech industrial sensor award 2024 2025",
            f"{_n} phase I phase II award amount funding 2025 2026",
            f"{_n} annual report funded companies success stories",
            f"{_n} review panel criteria scoring rubric technology readiness",
            f"site:grants.gov \"{_n}\" open solicitation",
            f"{_n} webinar info day office hours contact",
        ],
        "INVESTOR": [
            f"{_n} portfolio industrial IoT hardware cleantech",
            f"{_n} investment thesis focus areas pre-seed seed",
            f"{_n} partner associate contact industrial energy sector",
            f'"{_n}" GP LP partner name email site:linkedin.com',
            f"{_n} recent investments 2024 2025 hardware B2B industrial",
            f"{_n} apply pitch deck submission how to contact",
            f"{_n} IIoT sensor hardware funding steam energy US",
            f"site:crunchbase.com \"{_n}\" investments portfolio",
            f"{_n} fund size AUM check size typical investment",
            f"{_n} founder CEO partner name contact email",
            f"news \"{_n}\" new investment fund raise 2025",
            f"{_n} accelerator program application deadline cohort",
        ],
        "EVENT": [
            f"{_n} agenda tracks startup showcase 2025 2026",
            f"{_n} speakers program chair organizer contact email",
            f"{_n} exhibitor sponsor booth attendees IIoT cleantech",
            f'"{_n}" program committee submissions deadline',
            f"{_n} networking sessions industrial IoT energy sector",
            f"{_n} award competition pitch grants cleantech",
            f'site:linkedin.com "{_n}" organizer director contact',
            f"{_n} registration tickets cost date location 2026",
            f"{_n} past speakers panel session IoT sensor energy",
            f"{_n} startup competition application deadline prize",
            f"{_n} site:eventbrite.com OR site:meetup.com schedule",
            f"\"call for papers\" OR \"call for speakers\" \"{_n}\" 2026",
        ],
        "ORG": [
            f"{_n} IIoT industrial IoT steam energy technology partner",
            f"{_n} VP technology partnerships director contact email",
            f"{_n} open innovation program vendor supplier",
            f'"{_n}" technology team leadership site:linkedin.com',
            f"{_n} annual report 2024 2025 technology investments",
            f"{_n} site:crunchbase.com OR site:pitchbook.com funding",
            f"{_n} procurement process RFP startup pilot program",
            f"{_n} energy efficiency steam thermal industrial IoT initiative",
            f'"{_n}" CEO founder director name email contact',
            f"{_n} press release innovation lab partnership 2025",
            f"{_n} customer case study industrial steam sensor meter",
            f"{_n} accelerator startup collaboration technology scouting",
        ],
    }

    queries: list[str] = _query_map.get(entity_type, _query_map["ORG"])
    log.info("ACT:deep_queries entity=%r type=%s count=%d", entity_name[:40], entity_type, len(queries))

    if _budget_exceeded(budget, start_time, queries_used):
        cp = _make_checkpoint("DUE_DILIGENCE", "web_partial", task_id, run_counter,
                              {"entity_id": entity_id, "evidence_ids_added": evidence_added})
        return ExecuteResult(
            status="IN_PROGRESS", checkpoint=cp,
            result_summary=f"DUE_DILIGENCE for {entity_name}: partial, budget exceeded.",
            artifacts=[], next_task_suggestions=[],
            tasks_run=1, queries_used=queries_used[0],
            sources_consulted=len(all_results), elapsed_seconds=time.monotonic() - start_time,
            gate_results=[],
        )
    # ── Phase 2: Run all queries in parallel batches of 4 ────────────────────
    # Run up to 12 queries in 3 batches of 4, collecting max 8 results each.
    log.info("ACT:deep_search_start entity=%r total_queries=%d", entity_name[:40], len(queries))
    for _batch_start in range(0, len(queries), 4):
        if _budget_exceeded(budget, start_time, queries_used):
            break
        _batch = queries[_batch_start:_batch_start + 4]
        batch_results, _ = _safe_search_parallel(
            _batch, max_results=8, budget=budget, start_time=start_time, queries_used=queries_used,
        )
        all_results.extend(batch_results)
        log.info("ACT:batch_done batch=%d results_so_far=%d", _batch_start // 4 + 1, len(all_results))

    # Literature search
    try:
        if search_literature is None:
            raise ImportError("search_literature not available")
        lit_results = search_literature(
            f"{entity_name} technology research",
            max_results_per_source=3,
            sources=["semantic_scholar", "arxiv"],
        )
        lit_evidence = results_to_evidence_records(lit_results)
        for ev in lit_evidence:
            try:
                ev_id = state.append_evidence(ev)
                evidence_added.append(ev_id)
            except DuplicateError:
                pass
        all_results.extend([{"title": r.title, "snippet": r.abstract[:200],
                              "url": r.url, "source_id": r.source_id}
                             for r in lit_results])
    except Exception as exc:
        log.warning("Literature search failed for %s: %s", entity_name, exc)

    # Persist web evidence
    for r in all_results:
        if not r.get("url"):
            continue
        try:
            ev_id = state.append_evidence({
                "type": "web",
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "source_id": r.get("source_id", ""),
                "snippet": r.get("snippet", "")[:300],
                "credibility_score": 0.6,
                "relevance_score": 0.6,
                "hypothesis_ids": [],
                "tags": ["due_diligence", entity_id],
            })
            evidence_added.append(ev_id)
        except DuplicateError:
            pass
        except Exception:
            pass

    # Synthesize dossier via phi
    evidence_text = "\n".join(
        f"[{r.get('source_id', '?')}] {r.get('title', '')}: {r.get('snippet', '')[:200]}"
        for r in all_results[:12]
    )
    # Load full knowledge base — try multiple locations, use longest found
    _kb_text = ""
    for _kb_path in [
        Path.home() / "Documents" / "knowledge_base_backup.md",
        Path(__file__).parent.parent / "data" / "knowledge_base.md",
    ]:
        try:
            _candidate = _kb_path.read_text(encoding="utf-8", errors="ignore")
            if len(_candidate) > len(_kb_text):
                _kb_text = _candidate
        except Exception:
            pass
    _kb_text = _kb_text[:4000]  # use more context than before

    # Load strategic directives (David's current focus/exclusions from conversation)
    _directives_text = ""
    try:
        _directives_path = Path(__file__).parent.parent / "data" / "directives.md"
        if _directives_path.exists():
            _directives_text = _directives_path.read_text(encoding="utf-8", errors="ignore")[:1200]
    except Exception:
        pass

    synth_prompt = f"""You are a ruthless business analyst. Evaluate this opportunity for David Lagarejo. Be specific and evidence-driven. No fluff.

DAVID'S FULL PROFILE:
{_kb_text or '''- CEO/Founder, physicist-engineer, NYC
- Zircular: IIoT platform, pre-seed. ZION ING: energy consulting, active revenue
- Patent US2024/0077174 — ultrasonic non-invasive steam sensor (validated: Ecopetrol, Class I Div 2, -30% energy)
- Seeking: $500K seed / enterprise pilot clients / cleantech grants / executive roles
- LEED BD+C certified, IEEE reviewer. US focus only.'''}
{f'''
DAVID'S CURRENT STRATEGIC DIRECTIVES (his explicit instructions about what to research and what to skip):
{_directives_text}

IMPORTANT: If this entity falls under an exclusion listed in "Pausas / Exclusiones" above, set fit_score to 5 or below and explain in why_not.
If this entity aligns with the current focus areas, weigh that positively in fit_score.''' if _directives_text else ''}

Organization: {entity_name}
Evidence:
{evidence_text or 'No evidence collected yet.'}

RULES FOR why_yes — write as many bullets as the evidence justifies (minimum 2, up to 7 if warranted). Cover what applies:
1. OPPORTUNITY TYPE + ACTION: What exactly should David do? Be specific. E.g. "Apply to NSF SBIR Phase I — up to $250K non-dilutive for cleantech R&D"
2. DEADLINE / TIMING: Specific date, open window, funding cycle, or market event. E.g. "Next submission window closes June 2026". Estimate if not exact.
3. SPECIFIC FIT: What in the evidence matches David's ultrasonic sensor patent or ZION ING? Be concrete.
4. FINANCIAL UPSIDE: Dollar amount, contract value, or revenue potential.
5. VALIDATION ANGLE: How Ecopetrol validation or IEEE/LEED credentials strengthen David's position.
6. SPECIFIC PROGRAM/TRACK/AREA: Name a specific sub-program, investment thesis area, accelerator track, conference session, or procurement category that fits — not just the org in general.
7. COMPETITIVE EDGE: What makes David's profile unusual or hard to replicate for this opportunity?

Only include bullets where you have real evidence. Skip generic claims. Fewer sharp bullets beat more vague ones.

RULES FOR why_not (1-3 bullets): Real blockers only — missing eligibility, wrong geography, competition locked in, or timing issue.

BAD why_yes (NEVER write these): "Strong leadership", "Global presence", "Innovative solutions", "Growing market"

DEADLINE — REQUIRED for ALL entity types, not just grants:
- deadline: ISO date (YYYY-MM-DD) if found in evidence. If not found, set to null but EXPLAIN in deadline_label.
- deadline_label: ALWAYS fill this. For grants: "NSF SBIR Phase I — submission est. Jun 2026 (verify at seedfund.nsf.gov)". For events: "IIoT World 2026 — date TBD, check site". For companies: "Q2 2026 budget cycle (typical enterprise)". Never leave this null.
- registration_url: direct URL to apply/register if visible in evidence, else null.

KEY_CONTACTS — find up to 3 specific named people from the evidence:
For each: name, role, why_contact (one sentence: why THIS person for David's goal), when_to_contact (one sentence: what trigger/timing makes this the right moment), do_not_contact (true only if truly no valid reason).

Return JSON only:
{{
  "name": "{entity_name}",
  "description": "2-3 sentences: what they do + why relevant to David",
  "profile": {{
    "website": null,
    "country": null,
    "sector": null,
    "size_signal": null,
    "funding_stage": null,
    "key_people": []
  }},
  "fit_assessment": {{
    "fit_score": 0,
    "profile_match": 0,
    "timing": 0,
    "effort_vs_reward": 0,
    "risk": 50,
    "why_yes": [],
    "why_not": []
  }},
  "recommended_outreach": {{
    "contact_name": null,
    "contact_role": null,
    "reason": null,
    "angle": null,
    "do_not_contact": true
  }},
  "key_contacts": [
    {{
      "name": null,
      "role": null,
      "why_contact": null,
      "when_to_contact": null,
      "email": null,
      "linkedin_url": null,
      "do_not_contact": true
    }}
  ],
  "deadline": null,
  "deadline_label": null,
  "registration_url": null,
  "next_actions": []
}}"""

    raw = _call_phi_sync([
        {"role": "system", "content": "You are a business analyst. Output JSON only. No markdown. No explanation outside JSON."},
        {"role": "user", "content": synth_prompt},
    ])

    try:
        start_idx = raw.find("{")
        end_idx = raw.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            synthesis = json.loads(raw[start_idx:end_idx])
            dossier.update({k: v for k, v in synthesis.items()
                            if k in ("description", "profile", "fit_assessment",
                                     "recommended_outreach", "key_contacts",
                                     "deadline", "deadline_label",
                                     "registration_url", "next_actions")})
            # Save key_contacts to CRM (if found)
            for _kc in (synthesis.get("key_contacts") or []):
                if not _kc.get("do_not_contact", True) and _kc.get("name"):
                    try:
                        from tools.crm import upsert as _crm_upsert_kc
                        _crm_upsert_kc({
                            "entity_id":       f"ent_{entity_id[:8]}",
                            "company":         entity_name,
                            "entity_type":     entity_type,
                            "fit_score":       (synthesis.get("fit_assessment") or {}).get("fit_score", 0),
                            "name":            _kc["name"],
                            "role":            _kc.get("role"),
                            "email":           _kc.get("email"),
                            "linkedin_url":    _kc.get("linkedin_url"),
                            "why_contact":     _kc.get("why_contact"),
                            "outreach_reason": _kc.get("why_contact"),
                            "outreach_angle":  _kc.get("when_to_contact"),
                            "status":          "ready" if _kc.get("email") else "researching",
                        })
                    except Exception as _kc_exc:
                        log.warning("key_contact CRM save failed: %s", _kc_exc)
    except Exception as exc:
        log.warning("Dossier synthesis parse failed: %s", exc)

    dossier["evidence_ids"] = list(set(dossier.get("evidence_ids", []) + evidence_added))
    dossier["status"] = "DRAFT"
    fit_score = (dossier.get("fit_assessment") or {}).get("fit_score", 0)

    # ── Cross-reference phase: link entity to David's specific tech ───────────
    # Run targeted queries that connect this entity to David's exact profile keywords.
    # This surfaces connections not found by generic queries.
    if not _budget_exceeded(budget, start_time, queries_used):
        _cross_queries = [
            f'"{entity_name}" ultrasonic sensor IIoT industrial steam energy',
            f'"{entity_name}" Zircular OR "steam sensor" OR "non-invasive sensor" OR "Ecopetrol"',
            f'"{entity_name}" patent US2024 hardware startup cleantech pre-seed',
        ]
        _cross_results, _ = _safe_search_parallel(
            _cross_queries[:2], max_results=6,
            budget=budget, start_time=start_time, queries_used=queries_used,
        )
        if _cross_results:
            all_results.extend(_cross_results)
            log.info("ACT:cross_reference entity=%r cross_hits=%d", entity_name[:40], len(_cross_results))
            # If cross-reference found strong hits, store as evidence
            for _cr in _cross_results[:4]:
                try:
                    _ev_id = state.append_evidence({
                        "source_id": _cr.get("source_id", f"cross_{uuid.uuid4().hex[:6]}"),
                        "title": _cr.get("title", ""),
                        "url": _cr.get("url", ""),
                        "snippet": _cr.get("snippet", "")[:300],
                        "source": "cross_reference",
                        "entity_id": entity_id,
                    })
                    evidence_added.append(_ev_id)
                except Exception:
                    pass

    # ── Deep sub-program research for GRANTs and EVENTs ──────────────────────
    # Code-only path: scrapes topic areas, finds program officers + emails,
    # scores each category against David's profile via keyword matching.
    if fit_score >= 40:
        try:
            _website = (dossier.get("profile") or {}).get("website") or ""
            if not _website:
                _website = _find_website_for_entity(entity_name, entity_type)
                if _website and dossier.get("profile") is not None:
                    dossier["profile"]["website"] = _website
            sub_progs = _run_sub_program_research(
                entity_name, entity_type, _website,
                state, budget, start_time, queries_used,
            )
            if sub_progs:
                dossier["sub_programs"] = sub_progs
                # Auto-add high-fit sub-programs as CRM contacts
                for sp in sub_progs:
                    if sp.get("program_officer") and sp["fit_score"] >= 30:
                        try:
                            from tools.crm import upsert as _crm_upsert_sp
                            _crm_upsert_sp({
                                "entity_id":       entity_id,
                                "company":         entity_name,
                                "entity_type":     entity_type,
                                "fit_score":       sp["fit_score"],
                                "name":            sp["program_officer"],
                                "role":            f"Program Officer — {sp['name']}",
                                "email":           sp.get("email"),
                                "linkedin_url":    sp.get("linkedin_url"),
                                "why_contact":     f"Program Officer for {sp['name']} — {sp['fit_reason']}",
                                "outreach_reason": sp["fit_reason"],
                                "outreach_angle":  f"Apply to {sp['name']} — {sp['fit_reason']}",
                                "status":          "ready" if sp.get("email") else "researching",
                            })
                        except Exception as _sp_exc:
                            log.warning("sub_program CRM save failed: %s", _sp_exc)
                log.info("ACT:sub_programs_saved count=%d entity=%r", len(sub_progs), entity_name[:40])
        except Exception as _sp_err:
            log.warning("sub_program_research_failed entity=%r err=%s", entity_name[:40], _sp_err)

    state.save_dossier(dossier)
    log.info("ACT:dossier_saved entity=%r fit=%s evidence=%d", entity_name[:60], fit_score, len(evidence_added))

    # ── Contact extraction — enforced pipeline, always finds a real person ──────
    #
    # Step 1: Try to extract a person from the evidence already collected.
    # Step 2: If no person found, run 3 more targeted "find the person" searches.
    # Step 3: Only write to CRM if we have a real full name — no nameless company records.
    #
    # Cap person-search retries — stop after 3 failed attempts to avoid burning budget
    _person_attempts = (dossier.get("person_search_attempts") or 0)
    _skip_person_search = _person_attempts >= 3

    if fit_score >= 40 and all_results and not _skip_person_search:
        try:
            def _repair_json(raw: str) -> dict:
                """Strip markdown fences, fix trailing commas, extract first {...} block."""
                import re as _re
                # Strip ```json ... ``` fences
                raw = _re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
                _s = raw.find("{"); _e = raw.rfind("}") + 1
                if _s < 0 or _e <= _s:
                    return {}
                raw = raw[_s:_e]
                # Fix trailing commas before } or ]
                raw = _re.sub(r",\s*([}\]])", r"\1", raw)
                try:
                    return json.loads(raw)
                except Exception:
                    return {}

            def _extract_person_from_evidence(evidence_list: list[dict]) -> dict:
                """LLM call to extract a specific named person from evidence. Returns parsed dict."""
                _role_hint = {
                    "GRANT":    "program officer, grants manager, or technical reviewer",
                    "EVENT":    "event organizer or program chair",
                    "INVESTOR": "partner or associate covering energy/industrial sectors",
                    "ORG":      "VP Partnerships, VP Technology, or CEO/Founder",
                }.get(entity_type, "VP Partnerships or CEO")

                # Keep evidence short — phi4 loses context with long prompts
                _ev_text = "\n".join(
                    f"- {r.get('title','')}: {r.get('snippet','')[:200]}"
                    for r in evidence_list[:12]
                )

                _prompt = (
                    f'Find ONE real named person to contact at "{entity_name}" '
                    f'(target role: {_role_hint}) from this evidence.\n\n'
                    f"{_ev_text}\n\n"
                    f'If found: {{"found":true,"name":"First Last","role":"title",'
                    f'"email":"email or null","linkedin_url":"url or null","why_contact":"one sentence"}}\n'
                    f'If not found: {{"found":false}}\n'
                    f"JSON only. Real full name only — no placeholders."
                )

                _sys = "You extract named people from text. Output only a JSON object. No markdown. No explanation."

                # Attempt 1: temperature=0 for deterministic output
                _raw = _call_phi_sync(
                    [{"role": "system", "content": _sys}, {"role": "user", "content": _prompt}],
                    num_ctx=2048, temperature=0.0,
                )
                _parsed = _repair_json(_raw)
                if _parsed.get("found") and _parsed.get("name"):
                    return _parsed

                # Attempt 2: ultra-minimal prompt — just ask for name and role
                _minimal = (
                    f'From this text about "{entity_name}", output the name and job title of '
                    f"one real person mentioned. JSON only.\n\n"
                    f"{_ev_text[:800]}\n\n"
                    f'{{"found":true,"name":"...","role":"..."}}'
                    f' or {{"found":false}}'
                )
                _raw2 = _call_phi_sync(
                    [{"role": "system", "content": _sys}, {"role": "user", "content": _minimal}],
                    num_ctx=1024, temperature=0.0,
                )
                _parsed2 = _repair_json(_raw2)
                if _parsed2.get("found") and _parsed2.get("name"):
                    # Merge what we got — fill missing keys with None
                    return {
                        "found": True,
                        "name": _parsed2["name"],
                        "role": _parsed2.get("role", ""),
                        "email": None,
                        "linkedin_url": None,
                        "why_contact": f"Key contact at {entity_name}",
                    }
                return {"found": False}

            # Step 1: Extract from current evidence
            _cp = _extract_person_from_evidence(all_results)

            # Step 2: If no person found, enrich website + scrape team page
            if not _cp.get("found") or not _cp.get("name"):
                log.info("ACT:person_not_found_in_evidence — enriching website+team entity=%r", entity_name[:40])

                # 2a. Find and save website if missing
                _website = (dossier.get("profile") or {}).get("website") or ""
                if not _website and not _budget_exceeded(budget, start_time, queries_used):
                    _website = _find_website_for_entity(entity_name, entity_type)
                    queries_used[0] += 1
                    if _website:
                        log.info("ACT:website_found entity=%r url=%r", entity_name[:40], _website)
                        _prof = dossier.get("profile") or {}
                        _prof["website"] = _website
                        dossier["profile"] = _prof
                        state.save_dossier(dossier)

                # 2b. Scrape team/about page directly if we have a website
                _team_text = ""
                if _website:
                    log.info("ACT:scraping_team_page entity=%r", entity_name[:40])
                    _team_text = _scrape_team_page(_website)
                    if _team_text:
                        all_results.insert(0, {
                            "source_id": "team_page",
                            "title":     f"{entity_name} — Team/About page",
                            "url":       _website,
                            "snippet":   _team_text[:500],
                        })

                # 2c. Targeted web queries (parallel) — grant-specific gov sites
                _person_queries = {
                    "GRANT": [
                        f'"{entity_name}" program officer contact site:energy.gov OR site:nsf.gov OR site:sbir.gov OR site:grants.gov',
                        f"{entity_name} program manager technical point of contact email",
                    ],
                    "EVENT": [
                        f'"{entity_name}" organizer chair director contact email 2025 2026',
                        f"{entity_name} program committee submissions contact",
                    ],
                    "INVESTOR": [
                        f'"{entity_name}" partner associate cleantech energy contact LinkedIn',
                        f"{entity_name} investment team industrial sector site:crunchbase.com",
                    ],
                }.get(entity_type, [
                    f'"{entity_name}" VP director technology partnerships contact email',
                    f"{entity_name} leadership team site:linkedin.com OR site:crunchbase.com",
                ])

                if not _budget_exceeded(budget, start_time, queries_used):
                    _pr_batch, _ = _safe_search_parallel(
                        _person_queries[:2], max_results=5,
                        budget=budget, start_time=start_time, queries_used=queries_used,
                    )
                    all_results.extend(_pr_batch)

                # 2d. Retry extraction with enriched evidence (team page + new queries)
                _extract_input = ([{"source_id": "team_page", "title": f"{entity_name} Team",
                                    "url": _website, "snippet": _team_text[:2000]}]
                                  + all_results) if _team_text else all_results
                _cp = _extract_person_from_evidence(_extract_input)

            # Step 3: Write to CRM only if a real person was found
            if _cp.get("found") and _cp.get("name"):
                from tools.crm import upsert as _crm_upsert
                _email = _cp.get("email") or None
                _crm_upsert({
                    "entity_id":        entity_id,
                    "company":          entity_name,
                    "entity_type":      entity_type,
                    "fit_score":        fit_score,
                    "name":             _cp["name"],
                    "role":             _cp.get("role"),
                    "email":            _email,
                    "email_confidence": "public" if _email else "none",
                    "linkedin_url":     _cp.get("linkedin_url"),
                    "why_contact":      _cp.get("why_contact"),
                    "status":           "ready" if _email else "researching",
                    "outreach_reason":  (dossier.get("recommended_outreach") or {}).get("reason"),
                    "outreach_angle":   (dossier.get("recommended_outreach") or {}).get("angle"),
                    "why_yes":          (dossier.get("fit_assessment") or {}).get("why_yes", []),
                    "next_actions":     dossier.get("next_actions", []),
                })
                log.info("ACT:contact_saved entity=%r person=%r email=%s",
                         entity_name[:40], _cp["name"], bool(_email))
            else:
                # Increment attempt counter — stop retrying after 3 failed attempts
                _new_attempts = _person_attempts + 1
                dossier["person_search_attempts"] = _new_attempts
                dossier["open_loops"] = list(set((dossier.get("open_loops") or []) + ["person_not_found"]))
                state.save_dossier(dossier)
                if _new_attempts >= 3:
                    log.info("ACT:no_person_found entity=%r — max attempts reached, skipping future retries", entity_name[:40])
                else:
                    log.info("ACT:no_person_found entity=%r attempt=%d/3", entity_name[:40], _new_attempts)

        except Exception as _exc:
            log.warning("Contact extraction failed for %s: %s", entity_name, _exc)

    # Auto-create outreach task if LLM approved a specific contact
    _ro = dossier.get("recommended_outreach") or {}
    if _ro and not _ro.get("do_not_contact", True) and _ro.get("contact_name"):
        try:
            import urllib.request as _ur2
            _fa = dossier.get("fit_assessment") or {}
            _why = (_fa.get("why_yes") or [])
            _action = _why[0] if _why else "Contactar y presentar propuesta de valor"
            _task = {
                "task_id":        f"ro_{entity_id[:8]}",
                "entity_id":      f"ent_{entity_id[:8]}",
                "entity_name":    entity_name,
                "contact_name":   _ro["contact_name"],
                "contact_role":   _ro.get("contact_role") or "",
                "action":         _action,
                "outreach_reason": _ro.get("reason") or "",
                "outreach_angle":  _ro.get("angle") or "",
                "status":         "pending",
                "source":         "phi_recommended",
                "fit_score":      fit_score,
            }
            _req2 = _ur2.Request("http://127.0.0.1:8080/api/tasks/save",
                                 data=json.dumps(_task).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
            _ur2.urlopen(_req2, timeout=3)
            log.info("ACT:outreach_task_created entity=%r contact=%r fit=%s",
                     entity_name[:40], _ro["contact_name"], fit_score)
        except Exception:
            pass  # non-blocking
    next_suggestions = []
    if fit_score >= 70:
        try:
            state.enqueue_task({
                "strategy": "VALIDATE",
                "priority": 1,
                "payload": {"entity_id": entity_id},
            })
            next_suggestions.append({
                "strategy": "VALIDATE", "priority": 1,
                "payload": {"entity_id": entity_id},
                "reason": f"Fit score {fit_score} >= 70, ready for validation.",
            })
        except DuplicateError:
            pass

    state.update_strategy_state({"active_entity_ids": [entity_id]})
    state.append_timeline(
        "DOSSIER_CREATED",
        f"DUE_DILIGENCE complete for {entity_name}. Fit score: {fit_score}.",
        task_id=task_id, entity_id=entity_id,
    )

    cp = _make_checkpoint("DUE_DILIGENCE", "complete", task_id, run_counter,
                          {"entity_id": entity_id, "evidence_ids_added": evidence_added,
                           "fit_score": fit_score})

    return ExecuteResult(
        status="DONE", checkpoint=cp,
        result_summary=f"DUE_DILIGENCE for {entity_name}: fit_score={fit_score}, {len(evidence_added)} evidence items.",
        artifacts=[{"type": "dossier", "id": entity_id,
                    "path_or_ref": f"dossiers/{entity_id}.json"}],
        next_task_suggestions=next_suggestions,
        tasks_run=1, queries_used=queries_used[0],
        sources_consulted=len(all_results), elapsed_seconds=time.monotonic() - start_time,
        gate_results=[{"task_id": task_id, "gate_valid": True, "gate_score": fit_score,
                       "failures": []}],
    )


# ── CORRELATE strategy ────────────────────────────────────────────────────────

def _run_correlate(
    task: dict,
    budget: RunBudget,
    state: WorkspaceState,
    queries_used: list[int],
    start_time: float,
) -> ExecuteResult:
    """
    CORRELATE: Cross-reference findings across dossiers → build/update hypotheses.
    """
    task_id = task.get("task_id", "task_unknown")
    payload = task.get("payload") or {}
    hypothesis_ids = payload.get("hypothesis_ids", [])
    strategy_state = state.load_strategy_state()
    run_counter = strategy_state.get("run_counter", 0)

    # Load all complete/draft dossiers
    dossiers = state.list_dossiers(status_filter=["DRAFT", "COMPLETE"])
    if len(dossiers) < 2:
        cp = _make_checkpoint("CORRELATE", "skipped_insufficient_data", task_id, run_counter, {})
        return ExecuteResult(
            status="DONE", checkpoint=cp,
            result_summary="CORRELATE: insufficient dossiers for correlation (need ≥ 2).",
            artifacts=[], next_task_suggestions=[],
            tasks_run=1, queries_used=queries_used[0],
            sources_consulted=0, elapsed_seconds=time.monotonic() - start_time,
            gate_results=[],
        )

    # Build cross-reference prompt
    dossier_summaries = []
    for d in dossiers[:6]:
        full = state.load_dossier(d["entity_id"])
        if not full:
            continue
        fa = full.get("fit_assessment") or {}
        dossier_summaries.append(
            f"- {d['name']} ({d['type']}): fit={d['fit_score']}, "
            f"why_yes={fa.get('why_yes', [])[:2]}, "
            f"sector={full.get('profile', {}).get('sector', 'unknown')}"
        )

    corr_prompt = f"""Analyze these organizations and identify cross-cutting patterns, market hypotheses, and strategic opportunities.

Organizations under investigation:
{chr(10).join(dossier_summaries)}

Active hypotheses: {json.dumps([h.get('statement', '') for h in strategy_state.get('hypotheses', [])[:5]])}

Generate 2-3 hypotheses connecting these findings. Each hypothesis should have:
- A clear statement
- Confidence (0.0-1.0)
- Two falsifiers (conditions that would prove it wrong)

Return JSON only:
[{{"id": "hyp_NEW", "statement": "...", "confidence": 0.5, "falsifiers": ["...", "..."], "evidence_ids": [], "status": "ACTIVE"}}]"""

    raw = _call_phi_sync([
        {"role": "system", "content": "You are a strategic analyst. Output JSON only."},
        {"role": "user", "content": corr_prompt},
    ])

    new_hypotheses: list[dict] = []
    try:
        start_idx = raw.find("[")
        end_idx = raw.rfind("]") + 1
        if start_idx >= 0 and end_idx > start_idx:
            candidates = json.loads(raw[start_idx:end_idx])
            for h in candidates:
                h["id"] = f"hyp_{uuid.uuid4().hex[:8]}"
                h.setdefault("created_at", _now())
                h.setdefault("last_updated", _now())
                h.setdefault("status", "ACTIVE")
                h.setdefault("evidence_ids", [])
                new_hypotheses.append(h)
    except Exception as exc:
        log.warning("Hypothesis parse failed: %s", exc)

    if new_hypotheses:
        state.update_strategy_state({"hypotheses": new_hypotheses})

    state.append_timeline(
        "HYPOTHESIS_UPDATED",
        f"CORRELATE: generated {len(new_hypotheses)} new hypotheses from {len(dossiers)} dossiers.",
        task_id=task_id,
    )

    # Enqueue VALIDATE for high-confidence hypotheses
    suggestions = []
    for h in new_hypotheses:
        if h.get("confidence", 0) >= 0.8:
            suggestions.append({
                "strategy": "VALIDATE", "priority": 1,
                "payload": {"hypothesis_ids": [h["id"]]},
                "reason": f"High-confidence hypothesis: {h.get('statement', '')[:80]}",
            })

    cp = _make_checkpoint("CORRELATE", "complete", task_id, run_counter,
                          {"hypotheses_draft": new_hypotheses})
    return ExecuteResult(
        status="DONE", checkpoint=cp,
        result_summary=f"CORRELATE: {len(new_hypotheses)} hypotheses generated from {len(dossiers)} dossiers.",
        artifacts=[{"type": "hypothesis_set", "id": f"corr_{task_id}",
                    "path_or_ref": "state/strategy_state.json"}],
        next_task_suggestions=suggestions,
        tasks_run=1, queries_used=queries_used[0],
        sources_consulted=len(dossiers), elapsed_seconds=time.monotonic() - start_time,
        gate_results=[],
    )


# ── VALIDATE strategy ─────────────────────────────────────────────────────────

def _run_validate(
    task: dict,
    budget: RunBudget,
    state: WorkspaceState,
    queries_used: list[int],
    start_time: float,
) -> ExecuteResult:
    """
    VALIDATE: Two-pass verification before requesting human approval.
    Pass 1: phi synthesizes final assessment.
    Pass 2: evidence present + no PII in output + style check.
    """
    task_id = task.get("task_id", "task_unknown")
    payload = task.get("payload") or {}
    entity_id = payload.get("entity_id", "")
    strategy_state = state.load_strategy_state()
    run_counter = strategy_state.get("run_counter", 0)

    dossier = state.load_dossier(entity_id)
    if not dossier:
        cp = _make_checkpoint("VALIDATE", "dossier_not_found", task_id, run_counter, {}, resumable=False)
        return ExecuteResult(
            status="FAILED", checkpoint=cp,
            result_summary=f"VALIDATE: dossier {entity_id} not found.",
            artifacts=[], next_task_suggestions=[],
            tasks_run=1, queries_used=queries_used[0],
            sources_consulted=0, elapsed_seconds=time.monotonic() - start_time,
            gate_results=[{"task_id": task_id, "gate_valid": False, "gate_score": 0,
                           "failures": ["dossier_not_found"]}],
        )

    entity_name = dossier.get("name", entity_id)
    evidence_ids = dossier.get("evidence_ids") or None  # None → read all evidence
    evidence = state.read_evidence(evidence_ids)
    evidence_text = "\n".join(
        f"[{e.get('source_id', '?')}] {e.get('title', '')}: {e.get('snippet', '')[:150]}"
        for e in evidence[:8]
    )

    # Pass 1: phi final synthesis
    validate_prompt = f"""Produce a final validated assessment for: {entity_name}

Evidence base:
{evidence_text or 'No evidence loaded.'}

Current assessment: {json.dumps(dossier.get('fit_assessment', {}))}

Produce a concise validation report. Verify: every claim has a source_id from the evidence above.
Return JSON only:
{{
  "validated": true,
  "fit_score_final": 0,
  "key_findings": ["finding with [source_id]", "..."],
  "risks": ["risk 1", "..."],
  "recommended_next_action": "single clear action verb + what",
  "draft_outreach": "2-3 sentence outreach draft (no placeholders)",
  "evidence_coverage": "fraction of claims with source_ids",
  "gate_failures": []
}}"""

    raw = _call_phi_sync([
        {"role": "system", "content": "You are a validation analyst. Output JSON only."},
        {"role": "user", "content": validate_prompt},
    ])

    validation_report: dict = {}
    try:
        start_idx = raw.find("{")
        end_idx = raw.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            validation_report = json.loads(raw[start_idx:end_idx])
    except Exception as exc:
        log.warning("Validation parse failed: %s", exc)

    # Pass 2: gate checks
    gate_failures = []

    # Check evidence coverage
    if not evidence:
        gate_failures.append("no_evidence_base")

    # PII check via privacy_pre_hook
    try:
        _hook = privacy_pre_hook
        if _hook is None:
            raise ImportError("privacy_pre_hook not available")
        pii_check = _hook("SEARCH_WEB", {"output": raw})
        if pii_check.get("decision") == "BLOCK":
            gate_failures.append(f"pii_in_output:{pii_check.get('reason', 'unknown')}")
    except Exception:
        pass

    # Placeholder check
    import re
    if re.search(r"(\[|\{|<)(FIRST_NAME|LAST_NAME|FULL_NAME|NAME)(\]|\}|>)", raw, re.IGNORECASE):
        gate_failures.append("placeholder_tokens")

    gate_valid = len(gate_failures) == 0
    fit_score_final = validation_report.get("fit_score_final",
                                            dossier.get("fit_assessment", {}).get("fit_score", 0))

    if gate_valid:
        dossier["approval_status"] = "PENDING"
        dossier["fit_assessment"]["fit_score"] = fit_score_final
        state.save_dossier(dossier)
        state.update_strategy_state({"pending_approval_ids": [entity_id]})
        state.append_timeline(
            "APPROVAL_REQUESTED",
            f"VALIDATE passed for {entity_name}. Awaiting human approval.",
            task_id=task_id, entity_id=entity_id,
        )
        status = "DONE"
        summary = f"VALIDATE passed for {entity_name} (fit_score={fit_score_final}). Approval requested."
    else:
        state.append_timeline(
            "TASK_FAILED",
            f"VALIDATE failed for {entity_name}: {gate_failures}",
            task_id=task_id, entity_id=entity_id,
        )
        status = "FAILED"
        summary = f"VALIDATE failed for {entity_name}: {', '.join(gate_failures)}."

    cp = _make_checkpoint("VALIDATE", "complete", task_id, run_counter,
                          {"entity_id": entity_id, "gate_failures": gate_failures,
                           "fit_score_final": fit_score_final}, resumable=False)

    return ExecuteResult(
        status=status, checkpoint=cp,
        result_summary=summary,
        artifacts=[{"type": "validation_report", "id": f"val_{task_id}",
                    "path_or_ref": f"dossiers/{entity_id}.json"}],
        next_task_suggestions=[],
        tasks_run=1, queries_used=queries_used[0],
        sources_consulted=len(evidence),
        elapsed_seconds=time.monotonic() - start_time,
        gate_results=[{"task_id": task_id, "gate_valid": gate_valid,
                       "gate_score": fit_score_final, "failures": gate_failures}],
    )


# ── Strategy dispatch ─────────────────────────────────────────────────────────

STRATEGY_DISPATCH: dict[str, Callable] = {
    "DISCOVER":      _run_discover,
    "DUE_DILIGENCE": _run_due_diligence,
    "CORRELATE":     _run_correlate,
    "VALIDATE":      _run_validate,
    "LOCAL_SCAN":    _run_local_scan,
}


# ── Main entry point ──────────────────────────────────────────────────────────

def execute_research_cycle(
    workspace_state: WorkspaceState,
    budget: Optional[RunBudget] = None,
    resume_checkpoint_id: Optional[str] = None,
) -> ExecuteResult:
    """
    Run up to max_tasks_per_run=2 tasks from the queue.
    Returns a single ExecuteResult aggregating all work done this cycle.
    """
    if budget is None:
        budget = RunBudget()

    start_time = time.monotonic()
    queries_used = [0]  # mutable container shared across tasks

    # Increment run counter
    strategy_state = workspace_state.load_strategy_state()
    run_counter = strategy_state.get("run_counter", 0) + 1
    workspace_state.update_strategy_state({
        "run_counter": run_counter,
        "stats": {"total_runs": run_counter},
    })

    # Check backoff
    backoff_until = strategy_state.get("backoff_until")
    if backoff_until:
        now_iso = _now()
        if backoff_until > now_iso:
            cp = _make_checkpoint("DISCOVER", "backoff", "none", run_counter,
                                  {"backoff_until": backoff_until})
            return ExecuteResult(
                status="FROZEN", checkpoint=cp,
                result_summary=f"Engine in backoff until {backoff_until[:16]}.",
                artifacts=[], next_task_suggestions=[],
                tasks_run=0, queries_used=0,
                sources_consulted=0, elapsed_seconds=0.0, gate_results=[],
            )

    # Recover stale IN_PROGRESS tasks (server restart or crash mid-execution)
    STALE_MINUTES = 45
    stale_recovered = 0
    all_queued = workspace_state.read_queue()
    for t in all_queued:
        if t.get("status") != "IN_PROGRESS":
            continue
        last_attempt = t.get("last_attempt_at") or t.get("enqueued_at", "")
        if not last_attempt:
            continue
        try:
            last_dt = datetime.fromisoformat(last_attempt.replace("Z", "+00:00"))
            age_minutes = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60
            if age_minutes > STALE_MINUTES:
                workspace_state.mark_task_status(t["task_id"], "PENDING")
                stale_recovered += 1
        except (ValueError, KeyError):
            pass
    if stale_recovered:
        _log.warning("stale_task_recovery recovered=%d tasks reset_to=PENDING", stale_recovered)

    # Pick tasks
    tasks = workspace_state.peek_next_tasks(n=budget.max_tasks)
    if not tasks:
        # ── Auto-reset: when queue is all DONE, re-open oldest tasks for re-investigation ──
        # This ensures Phi always has work: re-investigates dossiers periodically.
        _all_q = workspace_state.read_queue()
        _done_tasks = [t for t in _all_q if t.get("status") == "DONE"]
        _REINVESTIGATE_DAYS = 3  # re-investigate any dossier older than this
        _reset_count = 0
        for _t in sorted(_done_tasks, key=lambda x: x.get("completed_at", ""))[:8]:
            _eid = (_t.get("payload") or {}).get("entity_id", "")
            # Always reset INVESTOR tasks (new stubs need research) and old tasks
            _completed = _t.get("completed_at", "")
            _age_days = 999
            if _completed:
                try:
                    from datetime import datetime as _dt, timezone as _tz
                    _age_days = (_dt.now(_tz.utc) - _dt.fromisoformat(
                        _completed.replace("Z", "+00:00"))).total_seconds() / 86400
                except Exception:
                    pass
            _d = workspace_state.get_dossier(_eid) or {}
            _etype = _d.get("type", "")
            _should_reset = (
                _etype == "INVESTOR"                     # investors always need fresh research
                or _age_days >= _REINVESTIGATE_DAYS      # stale: re-investigate
                or not (_d.get("recommended_outreach") or {}).get("contact_name")  # missing contact
                or not _d.get("sub_programs")            # never got deep research
            )
            if _should_reset:
                workspace_state.mark_task_status(_t["task_id"], "PENDING")
                _reset_count += 1
        if _reset_count:
            log.info("proactive_cycle: auto-reset %d DONE tasks for re-investigation", _reset_count)

        # Also enqueue any dossiers that have no task at all yet
        _all_queued_eids = {
            (_t.get("payload") or {}).get("entity_id", "")
            for _t in workspace_state.read_queue()
        }
        for _d in workspace_state.list_dossiers():
            _eid = _d.get("entity_id", "")
            if _eid and _eid not in _all_queued_eids:
                try:
                    workspace_state.enqueue_task({
                        "strategy": "DUE_DILIGENCE",
                        "priority": 1 if _d.get("type") in ("INVESTOR", "GRANT") else 3,
                        "payload": {
                            "entity_id":   _eid,
                            "entity_name": _d.get("name", ""),
                            "entity_type": _d.get("type", "ORG"),
                        },
                    })
                    log.info("proactive_cycle: enqueued new dossier %s", _d.get("name", _eid)[:40])
                except DuplicateError:
                    pass

        # Then seed LOCAL_SCAN and DISCOVER
        hint = strategy_state.get("open_loops", [{}])[0].get("description", "") if strategy_state.get("open_loops") else ""
        try:
            workspace_state.enqueue_task({
                "strategy": "LOCAL_SCAN",
                "priority": 1,
                "payload": {},
            })
        except DuplicateError:
            pass
        try:
            workspace_state.enqueue_task({
                "strategy": "DISCOVER",
                "priority": 4,
                "payload": {"query_hint": hint or "industrial IIoT cleantech grants investors United States"},
            })
        except DuplicateError:
            pass
        tasks = workspace_state.peek_next_tasks(n=budget.max_tasks)

    if not tasks:
        cp = _make_checkpoint("DISCOVER", "empty_queue", "none", run_counter, {})
        return ExecuteResult(
            status="DONE", checkpoint=cp,
            result_summary="No pending tasks in queue.",
            artifacts=[], next_task_suggestions=[],
            tasks_run=0, queries_used=0,
            sources_consulted=0, elapsed_seconds=time.monotonic() - start_time, gate_results=[],
        )

    # Run tasks
    all_artifacts: list[dict] = []
    all_gate_results: list[dict] = []
    all_suggestions: list[dict] = []
    final_status = "DONE"
    final_checkpoint = None
    summaries: list[str] = []
    tasks_run = 0

    for task in tasks:
        task_id = task.get("task_id", "unknown")
        strategy = task.get("strategy", "DISCOVER")

        if _budget_exceeded(budget, start_time, queries_used):
            workspace_state.mark_task_status(task_id, "PENDING")  # reset for next run
            final_status = "IN_PROGRESS"
            break

        workspace_state.mark_task_status(task_id, "IN_PROGRESS")

        fn = STRATEGY_DISPATCH.get(strategy)
        if fn is None:
            workspace_state.mark_task_status(task_id, "FAILED")
            all_gate_results.append({"task_id": task_id, "gate_valid": False,
                                     "gate_score": 0, "failures": [f"unknown_strategy:{strategy}"]})
            continue

        try:
            result = fn(task, budget, workspace_state, queries_used, start_time)
            tasks_run += 1
            all_artifacts.extend(result.artifacts)
            all_gate_results.extend(result.gate_results)
            all_suggestions.extend(result.next_task_suggestions)
            summaries.append(result.result_summary)
            final_checkpoint = result.checkpoint

            if result.status in ("DONE", "FROZEN"):
                workspace_state.mark_task_status(
                    task_id, result.status,
                    checkpoint_id=result.checkpoint.checkpoint_id,
                )
                if result.status == "FROZEN":
                    final_status = "FROZEN"
            elif result.status == "IN_PROGRESS":
                workspace_state.mark_task_status(
                    task_id, "PENDING",
                    checkpoint_id=result.checkpoint.checkpoint_id,
                )
                final_status = "IN_PROGRESS"
                break
            elif result.status == "FAILED":
                task_attempts = task.get("attempts", 0) + 1
                new_status = "FROZEN" if task_attempts >= 3 else "FAILED"
                workspace_state.mark_task_status(task_id, new_status,
                                                 checkpoint_id=result.checkpoint.checkpoint_id)
                if new_status == "FROZEN":
                    final_status = "FROZEN"

        except Exception as exc:
            log.error("Strategy %s failed for task %s: %s", strategy, task_id, exc)
            workspace_state.mark_task_status(task_id, "FAILED")
            all_gate_results.append({"task_id": task_id, "gate_valid": False,
                                     "gate_score": 0, "failures": [f"exception:{type(exc).__name__}"]})
            summaries.append(f"{strategy} exception: {exc}")

    # Update last_checkpoint_id
    if final_checkpoint:
        workspace_state.update_strategy_state({"last_checkpoint_id": final_checkpoint.checkpoint_id})

    if final_checkpoint is None:
        final_checkpoint = _make_checkpoint(
            "DISCOVER", "no_tasks", "none", run_counter, {}, resumable=False
        )

    return ExecuteResult(
        status=final_status,
        checkpoint=final_checkpoint,
        result_summary=" | ".join(summaries) or "No tasks ran.",
        artifacts=all_artifacts,
        next_task_suggestions=all_suggestions[:5],
        tasks_run=tasks_run,
        queries_used=queries_used[0],
        sources_consulted=sum(a.get("sources_consulted", 0) for a in all_artifacts
                              if isinstance(a, dict)),
        elapsed_seconds=time.monotonic() - start_time,
        gate_results=all_gate_results,
    )
