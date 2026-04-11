"""
api/routes/jobs.py — Job posting analysis endpoints.

Endpoints:
  POST   /api/jobs/analyze      — analyze a raw job posting text
  POST   /api/jobs/scan         — trigger a background job board scan
  GET    /api/jobs              — list all analyzed jobs (summary)
  GET    /api/jobs/scan/status  — scan status (last run, summary)
  GET    /api/jobs/{job_id}     — full analysis for one job
  DELETE /api/jobs/{job_id}     — remove a job analysis
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from tools.job_analyzer import analyze_job, list_jobs, get_job, delete_job, generate_resume_and_email, resolve_company_url

router = APIRouter()

_BASE            = Path(__file__).parent.parent.parent
_SCAN_STATE_FILE = _BASE / "data" / "job_tracker_state.json"
_scan_running    = False   # in-process guard (one scan at a time)


@router.post("/api/jobs/analyze")
async def jobs_analyze(req: Request):
    """Run the full analysis pipeline on a job posting."""
    body = await req.json()
    raw_text = (body.get("text") or "").strip()
    if not raw_text:
        return JSONResponse({"error": "text field required"}, status_code=400)

    source = body.get("source", "manual")
    url    = body.get("url", "")

    result = await asyncio.to_thread(analyze_job, raw_text, source, url)
    if result.get("error"):
        return JSONResponse({"error": result["error"]}, status_code=422)

    # Kick off company URL resolution in background so the job becomes
    # visible in the list once the source URL is confirmed.
    job_id = result.get("job_id")
    if job_id and not result.get("company_url"):
        async def _resolve_bg():
            try:
                await asyncio.to_thread(resolve_company_url, job_id)
            except Exception:
                pass
        asyncio.create_task(_resolve_bg())

    return JSONResponse(result)


@router.post("/api/jobs/resolve-urls")
async def jobs_resolve_urls():
    """
    Background: try to resolve company URLs for all jobs that don't have one yet.
    Returns count of jobs newly resolved.
    """
    jobs = list_jobs(100, only_with_url=False)
    pending = [j for j in jobs if not j.get("company_url")]

    async def _resolve_all():
        resolved = 0
        for j in pending:
            try:
                url = await asyncio.to_thread(resolve_company_url, j["job_id"])
                if url:
                    resolved += 1
            except Exception:
                pass
        return resolved

    asyncio.create_task(_resolve_all())
    return JSONResponse({"ok": True, "pending": len(pending)})


@router.post("/api/jobs/scan")
async def jobs_scan():
    """Trigger a background job board scan cycle."""
    global _scan_running
    if _scan_running:
        return JSONResponse({"ok": False, "message": "Scan already running"})

    async def _run():
        global _scan_running
        _scan_running = True
        try:
            import sys
            sys.path.insert(0, str(_BASE / "scripts"))
            from job_tracker import run_scan_cycle
            await asyncio.to_thread(run_scan_cycle)
        except Exception as e:
            import logging
            logging.getLogger("phi.jobs").error("scan_failed: %s", e)
        finally:
            _scan_running = False

    asyncio.create_task(_run())
    return JSONResponse({"ok": True, "message": "Scan started"})


@router.get("/api/jobs/scan/status")
async def jobs_scan_status():
    """Return last scan summary and whether a scan is running."""
    state = {}
    try:
        state = json.loads(_SCAN_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return JSONResponse({
        "running":     _scan_running,
        "last_scan_at": state.get("last_scan_at"),
        "last_summary": state.get("last_summary", {}),
    })


@router.get("/api/jobs")
async def jobs_list():
    """Return all analyzed jobs, sorted newest first (summary fields only)."""
    jobs = list_jobs(50)
    return JSONResponse({"jobs": jobs, "total": len(jobs)})


@router.post("/api/jobs/{job_id}/generate")
async def jobs_generate(job_id: str):
    """Generate tailored resume text + short email for an analyzed job."""
    result = await asyncio.to_thread(generate_resume_and_email, job_id)
    if result.get("error"):
        return JSONResponse({"error": result["error"]}, status_code=404)
    return JSONResponse(result)


@router.get("/api/jobs/{job_id}/company-url")
async def jobs_company_url(job_id: str):
    """
    Resolve and return the company's official website URL (not a job board).
    Tries URL pattern matching + content verification first.
    If not found, returns a Google search fallback URL the user can open.
    """
    import urllib.parse
    url = await asyncio.to_thread(resolve_company_url, job_id)
    if url:
        return JSONResponse({"url": url, "found": True})
    # Build Google search fallback so user can find it themselves
    job = await asyncio.to_thread(get_job, job_id)
    company = (job or {}).get("company", "")
    if company:
        search_url = "https://www.google.com/search?q=" + urllib.parse.quote(f'"{company}" official website')
        return JSONResponse({"url": search_url, "found": False, "search_fallback": True})
    return JSONResponse({"url": "", "found": False})


@router.get("/api/jobs/{job_id}")
async def jobs_get(job_id: str):
    """Full analysis detail for a single job."""
    job = await asyncio.to_thread(get_job, job_id)
    if job is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(job)


@router.delete("/api/jobs/{job_id}")
async def jobs_delete(job_id: str):
    """Delete a job analysis by ID."""
    ok = await asyncio.to_thread(delete_job, job_id)
    if not ok:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"ok": True})
