#!/usr/bin/env python3
"""
scripts/job_tracker.py — Job opportunity scanner daemon
=========================================================
Busca automáticamente ofertas de trabajo que encajen con el perfil de David
en múltiples plataformas usando SearXNG, las analiza con job_analyzer.py
y notifica cuando encuentra buenos matches.

Comportamiento:
  - Corre como daemon. Cada JOB_SCAN_INTERVAL horas escanea todas las queries.
  - Deduplica por URL (data/jobs_seen.json).
  - Analiza cada nueva oferta con el pipeline completo (3 passes LLM).
  - Envía notificación macOS cuando fit >= NOTIFY_THRESHOLD.
  - Puede dispararse manualmente vía POST /api/jobs/scan.

Inicio:
  python3 scripts/job_tracker.py &
  O desde start.sh (ya incluido).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR            = Path(__file__).resolve().parent.parent
DATA_DIR            = BASE_DIR / "data"
JOBS_DIR            = DATA_DIR / "jobs"
SEEN_FILE           = DATA_DIR / "jobs_seen.json"
STATE_FILE          = DATA_DIR / "job_tracker_state.json"

SERVER_URL          = os.environ.get("PHI_SERVER",  "http://127.0.0.1:8080")
SEARXNG_URL         = os.environ.get("SEARXNG_URL", "http://127.0.0.1:8888")
JOB_SCAN_INTERVAL   = int(os.environ.get("JOB_SCAN_INTERVAL", 14400))  # 4 horas
NOTIFY_THRESHOLD    = int(os.environ.get("JOB_NOTIFY_THRESHOLD", 60))  # fit >= 60%
ANALYZE_TIMEOUT     = int(os.environ.get("JOB_ANALYZE_TIMEOUT", 180))  # 3 min por job
MAX_JOBS_PER_CYCLE  = int(os.environ.get("MAX_JOBS_PER_CYCLE", 8))     # máx nuevos por ciclo
FETCH_TIMEOUT       = 12
SEARCH_RESULTS      = 6   # resultados por query

DATA_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [job_tracker] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log = logging.getLogger("job_tracker")

# ── Queries de búsqueda (perfil de David) ─────────────────────────────────────
#
# Ordenadas por prioridad. Cubren los ángulos del perfil:
#   - Energy efficiency engineer/PM NYC
#   - IIoT / industrial IoT
#   - Building systems / ECM / retrofit
#   - Cleantech / sustainability
#   - Analytics/data en energy
#   - Technical PM / program manager
#
JOB_QUERIES = [
    # ── Core energy NYC / remote ────────────────────────────────────────────
    {
        "query":    "energy efficiency engineer project manager New York job",
        "priority": 1,
        "angle":    "energy_pm",
    },
    {
        "query":    "energy efficiency consultant industrial buildings New York job",
        "priority": 1,
        "angle":    "energy_consulting",
    },
    {
        "query":    "building energy performance analyst engineer NYC job",
        "priority": 1,
        "angle":    "energy_analytics",
    },
    # ── IIoT / industrial ──────────────────────────────────────────────────
    {
        "query":    "IIoT industrial IoT energy engineer New York job",
        "priority": 2,
        "angle":    "iot",
    },
    {
        "query":    "industrial energy optimization engineer steam HVAC job United States",
        "priority": 2,
        "angle":    "industrial_energy",
    },
    # ── Building systems / retrofit ────────────────────────────────────────
    {
        "query":    "building systems MEP retrofit project manager New York job",
        "priority": 2,
        "angle":    "building_systems",
    },
    {
        "query":    "energy conservation measures ECM program manager NYC job",
        "priority": 2,
        "angle":    "ecm_pm",
    },
    {
        "query":    "HVAC energy efficiency engineer QA QC New York job",
        "priority": 2,
        "angle":    "hvac_qa",
    },
    # ── Cleantech / sustainability ─────────────────────────────────────────
    {
        "query":    "cleantech sustainability engineer New York job startup",
        "priority": 2,
        "angle":    "cleantech",
    },
    {
        "query":    "electrification decarbonization engineer analyst NYC job",
        "priority": 3,
        "angle":    "decarbonization",
    },
    # ── Data/analytics angle ────────────────────────────────────────────────
    {
        "query":    "energy data analyst Python financial modeling NPV IRR job",
        "priority": 2,
        "angle":    "data_analytics",
    },
    {
        "query":    "energy efficiency program analyst reporting New York job",
        "priority": 3,
        "angle":    "program_analytics",
    },
    # ── Technical PM / program manager ─────────────────────────────────────
    {
        "query":    "technical program manager energy efficiency remote job",
        "priority": 2,
        "angle":    "technical_pm",
    },
    {
        "query":    "energy project manager vendor coordination QA New York job",
        "priority": 2,
        "angle":    "vendor_pm",
    },
    # ── Specific employers / boards (no LinkedIn/Indeed — filtered out) ─────────
    {
        "query":    "site:climatebase.org energy engineer project manager USA",
        "priority": 1,
        "angle":    "climatebase",
    },
    {
        "query":    "site:wellfound.com energy IIoT engineer startup New York",
        "priority": 2,
        "angle":    "wellfound",
    },
    {
        "query":    "site:nyserda.ny.gov careers energy engineer",
        "priority": 2,
        "angle":    "nyserda",
    },
    {
        "query":    "site:energyjobline.com energy engineer project manager New York",
        "priority": 2,
        "angle":    "energyjobline",
    },
    {
        "query":    "energy engineer project manager jobs USA 2026 site:careers",
        "priority": 3,
        "angle":    "us_careers",
    },
]

# ── Patterns para detectar si una URL es una oferta de trabajo ────────────────
_JOB_URL_PATTERNS = re.compile(
    r'/(job|jobs|career|careers|position|opening|vacancy|viewjob|apply|posting'
    r'|empleos?|oferta|work-with-us|join-us)/|'
    r'linkedin\.com/jobs/|'
    r'indeed\.com/viewjob|'
    r'glassdoor\.com/job-listing|'
    r'climatebase\.org/job/|'
    r'wellfound\.com/jobs|'
    r'workatastartup\.com/jobs',
    re.IGNORECASE,
)

_SKIP_DOMAINS = re.compile(
    r'\b(news|blog|press|article|wiki|reddit|twitter|facebook|instagram|'
    r'youtube|medium|substack|quora|stackoverflow|github|scholar|arxiv|'
    r'hbr\.org|forbes\.com|wsj\.com|nytimes\.com|bloomberg\.com)\b',
    re.IGNORECASE,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_seen() -> set:
    try:
        return set(json.loads(SEEN_FILE.read_text(encoding="utf-8")))
    except Exception:
        return set()


def _save_seen(seen: set):
    try:
        SEEN_FILE.write_text(
            json.dumps(sorted(seen), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        _log.warning("save_seen_failed: %s", e)


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict):
    try:
        STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        _log.warning("save_state_failed: %s", e)


def _url_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]


def _notify(title: str, message: str):
    title_s   = title.replace('"', "'")[:80]
    message_s = message.replace('"', "'")[:200]
    script = (
        f'display notification "{message_s}" '
        f'with title "{title_s}" '
        f'sound name "Funk"'
    )
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
    except Exception:
        pass


# ── SearXNG search ────────────────────────────────────────────────────────────

def _search_searxng(query: str, max_results: int = SEARCH_RESULTS) -> list[dict]:
    """Returns list of {url, title, content} from SearXNG."""
    params = urllib.parse.urlencode({
        "q":          query,
        "format":     "json",
        "categories": "general",
        "language":   "en",
    })
    url = f"{SEARXNG_URL}/search?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "phi-twin/1.0"})
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        _log.debug("searxng_error query=%r err=%s", query[:60], e)
        return []

    results = []
    for item in data.get("results", [])[:max_results]:
        item_url = item.get("url", "")
        if not item_url:
            continue
        results.append({
            "url":     item_url,
            "title":   item.get("title", ""),
            "content": item.get("content", ""),
        })
    return results


# ── Page fetcher ──────────────────────────────────────────────────────────────

def _fetch_page_text(url: str) -> str:
    """Fetch a URL and return stripped plain text. Returns empty on error."""
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as r:
            content_type = r.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return ""
            raw = r.read(80_000).decode("utf-8", errors="replace")
    except Exception:
        return ""

    # Strip HTML
    raw = re.sub(r'<style[^>]*>.*?</style>', ' ', raw, flags=re.DOTALL | re.I)
    raw = re.sub(r'<script[^>]*>.*?</script>', ' ', raw, flags=re.DOTALL | re.I)
    raw = re.sub(r'<[^>]+>', ' ', raw)
    for entity, char in [('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'),
                          ('&nbsp;', ' '), ('&#39;', "'"), ('&quot;', '"')]:
        raw = raw.replace(entity, char)
    raw = re.sub(r'[ \t]{2,}', ' ', raw)
    raw = re.sub(r'\n{3,}', '\n\n', raw)
    return raw.strip()[:8000]


# ── Job text builder ──────────────────────────────────────────────────────────

def _build_job_text(result: dict) -> str:
    """Combine search snippet + page content into analyzable text."""
    title   = result.get("title", "")
    snippet = result.get("content", "")
    url     = result.get("url", "")

    # Try fetching the full page
    page_text = _fetch_page_text(url)

    if len(page_text) > 400:
        # Use page text as primary, with title prepended
        combined = f"{title}\n\n{page_text}"
    else:
        # Fallback to snippet only
        combined = f"{title}\n\n{snippet}"

    return combined.strip()


# ── Analyze via API ───────────────────────────────────────────────────────────

def _analyze_job_via_api(text: str, url: str, source: str) -> dict | None:
    """POST to /api/jobs/analyze and return the result dict."""
    if len(text.strip()) < 150:
        return None
    body = json.dumps({
        "text":   text,
        "url":    url,
        "source": source,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{SERVER_URL}/api/jobs/analyze",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=ANALYZE_TIMEOUT) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        _log.warning("analyze_http_error url=%s code=%d", url[:60], e.code)
        return None
    except Exception as e:
        _log.warning("analyze_error url=%s err=%s", url[:60], e)
        return None


# ── URL filter ────────────────────────────────────────────────────────────────

def _is_likely_job_url(url: str, title: str, snippet: str) -> bool:
    """Heuristic: decide if this result is likely a job posting."""
    # Skip obviously non-job domains
    if _SKIP_DOMAINS.search(url):
        return False

    # Must be http/https
    if not url.startswith("http"):
        return False

    # Check URL pattern
    if _JOB_URL_PATTERNS.search(url):
        return True

    # Check title/snippet for job signals
    job_signals = re.compile(
        r'\b(job|position|opening|career|hiring|apply|vacancy|'
        r'full.time|part.time|salary|compensation|responsibilities|'
        r'qualifications|requirements|we.re looking|join our team)\b',
        re.IGNORECASE,
    )
    text = f"{title} {snippet}"
    if job_signals.search(text):
        return True

    return False


# ── One scan cycle ────────────────────────────────────────────────────────────

def run_scan_cycle(force: bool = False) -> dict:
    """
    Execute one full scan cycle across all queries.
    Returns summary dict: {scanned, analyzed, high_fit, errors}.
    """
    seen        = _load_seen()
    candidates  = []   # list of {url, title, content, source, angle}
    scanned_q   = 0

    # Sort queries by priority
    queries = sorted(JOB_QUERIES, key=lambda q: q.get("priority", 9))

    _log.info("scan_cycle_start queries=%d seen_urls=%d", len(queries), len(seen))

    for q_info in queries:
        query  = q_info["query"]
        angle  = q_info.get("angle", "general")
        results = _search_searxng(query, max_results=SEARCH_RESULTS)
        scanned_q += 1

        for res in results:
            url     = res.get("url", "")
            title   = res.get("title", "")
            snippet = res.get("content", "")
            uid     = _url_id(url)

            if uid in seen:
                continue
            if not _is_likely_job_url(url, title, snippet):
                continue

            candidates.append({
                "url":     url,
                "title":   title,
                "content": snippet,
                "source":  f"auto_{angle}",
                "uid":     uid,
            })

        # Polite delay between searches
        time.sleep(1.0)

    # Deduplicate candidates (same uid can appear from multiple queries)
    seen_uids = set()
    unique_candidates = []
    for c in candidates:
        if c["uid"] not in seen_uids:
            seen_uids.add(c["uid"])
            unique_candidates.append(c)

    _log.info("scan_candidates_found new=%d (from %d queries)", len(unique_candidates), scanned_q)

    analyzed   = 0
    high_fit   = 0
    errors     = 0
    high_fit_jobs = []

    # Limit per cycle to avoid overloading the LLM
    for candidate in unique_candidates[:MAX_JOBS_PER_CYCLE]:
        url    = candidate["url"]
        source = candidate["source"]
        uid    = candidate["uid"]

        _log.info("analyzing url=%s source=%s", url[:80], source)
        text = _build_job_text(candidate)

        result = _analyze_job_via_api(text, url, source)

        # Always mark as seen (even on error) to avoid retrying bad URLs
        seen.add(uid)

        if result is None:
            errors += 1
            continue

        if result.get("error"):
            _log.info("analyze_skipped url=%s reason=%s", url[:60], result["error"])
            continue

        analyzed += 1
        overall = result.get("overall_fit") or (result.get("scores") or {}).get("overall_fit", 0)
        verdict = result.get("verdict", "—")
        title   = result.get("title", candidate["title"])[:60]
        company = result.get("company", "")[:40]

        _log.info(
            "job_analyzed title=%r company=%r fit=%d verdict=%s",
            title, company, overall, verdict,
        )

        if overall >= NOTIFY_THRESHOLD:
            high_fit += 1
            high_fit_jobs.append({
                "title":   title,
                "company": company,
                "fit":     overall,
                "verdict": verdict,
            })

    # Save seen URLs
    _save_seen(seen)

    # Notify macOS for high-fit jobs
    if high_fit_jobs:
        best = max(high_fit_jobs, key=lambda j: j["fit"])
        others = len(high_fit_jobs) - 1
        msg = f"{best['title']} @ {best['company']} — {best['fit']}% fit"
        if others:
            msg += f" (+{others} más)"
        _notify(f"Phi · {high_fit_jobs[0]['verdict']} — {high_fit} empleo(s) relevante(s)", msg)

    summary = {
        "queries_run": scanned_q,
        "candidates":  len(unique_candidates),
        "analyzed":    analyzed,
        "high_fit":    high_fit,
        "errors":      errors,
        "ran_at":      datetime.now(timezone.utc).isoformat()[:19] + "Z",
    }
    _log.info("scan_cycle_done %s", summary)

    # Update state
    state = _load_state()
    state["last_scan_at"]  = time.time()
    state["last_summary"]  = summary
    _save_state(state)

    return summary


# ── Standalone daemon ─────────────────────────────────────────────────────────

def _wait_for_server(max_wait: int = 120) -> bool:
    for _ in range(max_wait // 5):
        try:
            req = urllib.request.Request(f"{SERVER_URL}/health")
            with urllib.request.urlopen(req, timeout=5) as r:
                data = json.loads(r.read())
                if data.get("status") == "ok":
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def main():
    _log.info(
        "job_tracker started server=%s searxng=%s interval=%ds notify_threshold=%d%%",
        SERVER_URL, SEARXNG_URL, JOB_SCAN_INTERVAL, NOTIFY_THRESHOLD,
    )

    # Check if --once flag passed (single scan then exit)
    once = "--once" in sys.argv

    # Wait for server
    if not _wait_for_server():
        _log.error("server_not_ready — exiting")
        sys.exit(1)

    # Extra grace period on first run
    if not once:
        _log.info("waiting 120s grace period before first scan")
        time.sleep(120)

    state = _load_state()

    while True:
        now               = time.time()
        last_scan         = float(state.get("last_scan_at", 0))
        since_last        = now - last_scan

        if since_last >= JOB_SCAN_INTERVAL or once:
            try:
                run_scan_cycle()
            except Exception as e:
                _log.error("scan_cycle_failed: %s", e)
            state = _load_state()

        if once:
            break

        time.sleep(300)  # check every 5 min if it's time to run


if __name__ == "__main__":
    main()
