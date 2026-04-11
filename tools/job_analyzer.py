"""
tools/job_analyzer.py — Job posting analyzer for DavidSan timeline.

Pipeline:
  1. Ingest    — accept text / HTML / markdown
  2. Normalize — strip HTML, clean whitespace, detect sections
  3. Extract   — deterministic regex/rules for structured fields
  4. LLM pass 1 — semantic extraction (role type, implicit needs, requirements)
  5. LLM pass 2 — scoring, gaps, verdict narrative
  6. Combine   — merge deterministic + LLM, apply rule-based scoring
  7. Output    — structured JSON + human summary

Phi is used for semantic inference only; verdict logic is deterministic.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_log = logging.getLogger("phi.tools.job_analyzer")

_BASE        = Path(__file__).parent.parent
_PROFILE_PATH = _BASE / "data" / "profile.json"
_JOBS_DIR    = _BASE / "data" / "jobs"
_JOBS_DIR.mkdir(parents=True, exist_ok=True)

# ── Profile (loaded once at module import) ────────────────────────────────────

def _load_profile() -> dict:
    try:
        return json.loads(_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        _log.error("Cannot load profile.json: %s", e)
        return {}

_PROFILE: dict = _load_profile()

# ── Contact info from profile (gitignored) ────────────────────────────────────
_CONTACT     = _PROFILE.get("contact", {})
_NAME        = _PROFILE.get("name", "Candidate")
_EMAIL       = _CONTACT.get("email", "")
_PHONE       = _CONTACT.get("phone", "")
_LINKEDIN    = _CONTACT.get("linkedin", "")
_LOCATION    = _PROFILE.get("location", "")
_SIGNATURE   = f"{_NAME}\n{_EMAIL} | {_PHONE} | {_LINKEDIN}"
_HEADER_LINE = f"{_LOCATION}  •  {_PHONE}  •  {_EMAIL}  •  {_LINKEDIN}"

# ── Known tool sets for deterministic matching ────────────────────────────────

_TOOLS_STRONG   = {t.lower() for t in _PROFILE.get("tools_software", {}).get("strong",   [])}
_TOOLS_MODERATE = {t.lower() for t in _PROFILE.get("tools_software", {}).get("moderate", [])}
_TOOLS_LIGHT    = {t.lower() for t in _PROFILE.get("tools_software", {}).get("light",    [])}
_TOOLS_ALL      = _TOOLS_STRONG | _TOOLS_MODERATE | _TOOLS_LIGHT

_DOMAIN_KEYWORDS = set(k.lower() for k in _PROFILE.get("core_domains", []))
_TECH_KEYWORDS   = set(k.lower() for k in _PROFILE.get("technical_skills", []))
_MGMT_KEYWORDS   = set(k.lower() for k in _PROFILE.get("management_skills", []))
_BIZ_KEYWORDS    = set(k.lower() for k in _PROFILE.get("business_skills", []))
_GAP_KEYWORDS    = set(k.lower() for k in _PROFILE.get("known_honest_gaps", []))
_INDUSTRIES      = set(k.lower() for k in _PROFILE.get("industries_worked", []))

# Flatten tool names for deterministic matching (strip parenthetical descriptions)
def _bare_tool(t: str) -> str:
    return re.split(r'\s*[\(\[]', t)[0].strip().lower()

_TOOLS_STRONG_BARE   = {_bare_tool(t) for t in _PROFILE.get("tools_software", {}).get("strong",   [])}
_TOOLS_MODERATE_BARE = {_bare_tool(t) for t in _PROFILE.get("tools_software", {}).get("moderate", [])}
_TOOLS_LIGHT_BARE    = {_bare_tool(t) for t in _PROFILE.get("tools_software", {}).get("light",    [])}
_TOOLS_ALL_BARE      = _TOOLS_STRONG_BARE | _TOOLS_MODERATE_BARE | _TOOLS_LIGHT_BARE

# ── Regex patterns ────────────────────────────────────────────────────────────

_RE_SALARY = re.compile(
    r'\$[\d,]+(?:k|K)?(?:\s*[-–]\s*\$?[\d,]+(?:k|K)?)?'
    r'|(?:USD\s*)?[\d,]+(?:k|K)\s*(?:[-–]\s*[\d,]+(?:k|K))?(?:\s*(?:per\s+year|/yr|annually|/year|/hr|/hour))?',
    re.IGNORECASE,
)
_RE_YEARS = re.compile(
    r'(\d+)\+?\s*(?:to|-)\s*(\d+)\s*years?'
    r'|(\d+)\+\s*years?'
    r'|(\d+)\s*years?\s*(?:of\s+)?(?:experience|exp)',
    re.IGNORECASE,
)
_RE_DEGREE = re.compile(
    r"bachelor(?:'?s)?|master(?:'?s)?|ph\.?d|mba|bs|ms|be|b\.eng|m\.eng|m\.sc",
    re.IGNORECASE,
)
_RE_TRAVEL = re.compile(
    r'travel(?:ing)?\s+(?:required|up\s+to|may\s+be|\d+%)|up\s+to\s+\d+%\s+travel',
    re.IGNORECASE,
)
_RE_CERT = re.compile(
    r'\b(?:PMP|PE|LEED\s*(?:AP|GA|BD\+C|ID\+C|ND|O\+M)?|CEM|CEA|CBCP|'
    r'ASHRAE|AWS|GCP|Azure|Certified|Licensed|Registration)\b',
    re.IGNORECASE,
)
_RE_TOOLS = re.compile(
    r'\b(?:AutoCAD|Revit|Python|Excel|SQL|Power\s*BI|Tableau|MATLAB|'
    r'SAP|Oracle|Maximo|ArcGIS|ESRI|EnergyPlus|eQUEST|HAP|Trane\s*TRACE|'
    r'R\b|Java|C\+\+|Salesforce|HubSpot|Jira|Asana|MS\s*Project|'
    r'SketchUp|Bentley|MicroStation|Primavera|Procore|PlanGrid|Bluebeam)\b',
    re.IGNORECASE,
)

# ── Step 1: Ingest & normalize ────────────────────────────────────────────────

def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', ' ', text)
    for entity, char in [('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'),
                          ('&nbsp;', ' '), ('&#39;', "'"), ('&quot;', '"')]:
        text = text.replace(entity, char)
    return text


def normalize_text(raw: str) -> str:
    """Clean and normalize job posting text."""
    text = _strip_html(raw)
    text = re.sub(r'[\r\n\t]+', '\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ── Step 2: Deterministic extraction ─────────────────────────────────────────

def extract_deterministic(text: str) -> dict:
    """Extract structured fields using regex and keyword rules."""
    salary_matches = _RE_SALARY.findall(text)
    years_matches  = _RE_YEARS.findall(text)
    cert_matches   = list(set(_RE_CERT.findall(text)))
    tool_matches   = list(set(_RE_TOOLS.findall(text)))

    # Years of experience: take the maximum mentioned
    min_years = 0
    for m in years_matches:
        nums = [int(x) for x in m if x and x.isdigit()]
        if nums:
            min_years = max(min_years, min(nums))

    # Tool overlap with profile
    tools_lower  = [t.strip().lower() for t in tool_matches]
    matched_tools = {
        "known":   [t for t in tools_lower if t in _TOOLS_ALL_BARE],
        "unknown": [t for t in tools_lower if t not in _TOOLS_ALL_BARE],
    }

    # Domain overlap
    text_lower = text.lower()
    matched_domains = [kw for kw in _DOMAIN_KEYWORDS if kw in text_lower]
    matched_gaps    = [kw for kw in _GAP_KEYWORDS if re.search(r'\b' + re.escape(kw[:15]) + r'\b', text_lower)]

    # Industry detection
    matched_industry = [ind for ind in _INDUSTRIES if ind in text_lower]

    # Red flag signals (deterministic)
    salary_str = salary_matches[0] if salary_matches else ""
    low_salary = False
    if salary_str:
        nums = [int(x.replace(',', '').replace('k', '000').replace('K', '000'))
                for x in re.findall(r'[\d,]+(?:k|K)?', salary_str) if x]
        if nums:
            max_sal = max(nums)
            low_salary = max_sal < _PROFILE.get("salary_floor_usd", 80000)

    # Detect if job has many requirements relative to seniority signals
    req_count = len(cert_matches) + len(tool_matches)
    has_degree = bool(_RE_DEGREE.search(text))
    has_travel = bool(_RE_TRAVEL.search(text))

    return {
        "salary_raw":       salary_str,
        "low_salary_flag":  low_salary,
        "min_years_exp":    min_years,
        "degree_required":  has_degree,
        "certifications":   cert_matches,
        "tools_required":   tool_matches,
        "tools_matched":    matched_tools,
        "domains_matched":  matched_domains,
        "gaps_triggered":   matched_gaps,
        "industries_match": matched_industry,
        "has_travel":       has_travel,
        "req_count":        req_count,
    }


# ── Step 3: LLM pass 1 — semantic extraction ─────────────────────────────────

_EXTRACTION_SYSTEM = (
    "You are a job posting parser. Return ONLY a valid JSON object. "
    "No explanation, no markdown, no extra text."
)

_EXTRACTION_TEMPLATE = """Parse this job posting and return a JSON object.

JOB POSTING:
{job_text}

Return exactly this JSON structure (fill every field, use null if unknown):
{{
  "title": "",
  "company": "",
  "location": "",
  "modality": "remote|hybrid|onsite|unknown",
  "seniority": "junior|mid|senior|lead|director|unknown",
  "industry": "",
  "role_type": "project_management|program_management|operations|analytics|data_automation|construction|energy_engineer|business_development|sales|compliance|startup|corporate_execution|other",
  "job_nature": "executor|coordinator|analyst|builder|consultant|field|office|customer_facing|administrative|technical",
  "explicit_requirements": {{
    "hard_skills": [],
    "soft_skills": [],
    "domains": [],
    "tools": [],
    "certifications": [],
    "years_experience": null,
    "degree": "",
    "languages": [],
    "travel": false,
    "legal_status": ""
  }},
  "implicit_requirements": [],
  "responsibilities_summary": ""
}}"""


def _call_llm_extraction(job_text: str) -> dict:
    """LLM call 1: semantic extraction of job requirements."""
    from llm.client import call_phi_sync
    truncated = job_text[:3000]
    prompt = _EXTRACTION_TEMPLATE.format(job_text=truncated)
    raw = call_phi_sync(
        [{"role": "system", "content": _EXTRACTION_SYSTEM},
         {"role": "user",   "content": prompt}],
        num_ctx=4096, temperature=0.0,
    )
    return _parse_json_response(raw, {})


# ── Step 4: LLM pass 2 — scoring and narrative ───────────────────────────────

_SCORING_SYSTEM = (
    "You are a job match analyst. Be honest, critical, and direct. "
    "Never inflate scores. Distinguish between direct experience and theoretical knowledge. "
    "Return ONLY valid JSON. No explanation outside the JSON."
)

_SCORING_TEMPLATE = """Score how well this candidate matches this job. Be honest and critical.

CANDIDATE PROFILE (David Lagarejo):
- Background: Applied physics engineer, IIoT/cleantech CEO
- Core domains: {domains}
- Technical skills: {tech_skills}
- Tools STRONG: {tools_strong}
- Tools MODERATE: {tools_moderate}
- Project experience: {projects}
- Management: project management, vendor coordination, QA/QC, budget tracking
- Business: ROI/NPV/IRR, proposal writing, startup operations
- Honest gaps (do not inflate these): {gaps}
- Preferred roles: {preferred}
- NOT preferred: {not_preferred}

JOB REQUIREMENTS:
Title: {title}
Industry: {industry}
Role type: {role_type}
Hard skills required: {hard_skills}
Domains required: {domains_req}
Tools required: {tools_req}
Certifications: {certs}
Implicit needs: {implicit}
Responsibilities: {responsibilities}

Score each dimension 0-100. For risk scores (learning_curve_risk, credibility_risk): 0=no risk, 100=maximum risk.
Be especially strict on credibility_risk if requirements fall in candidate's honest gaps.

Return exactly this JSON:
{{
  "scores": {{
    "technical_fit": 0,
    "industry_fit": 0,
    "tools_fit": 0,
    "execution_fit": 0,
    "management_fit": 0,
    "communication_fit": 0,
    "regulatory_fit": 0,
    "learning_curve_risk": 0,
    "credibility_risk": 0
  }},
  "strengths_against_role": [],
  "gaps": [
    {{"type": "critical|moderate|narrative|easy_to_close|not_recommended", "description": ""}}
  ],
  "red_flags": [],
  "reasoning_summary": "",
  "positioning_strategy": "",
  "resume_angle": ""
}}"""


def _call_llm_scoring(extracted: dict, det: dict) -> dict:
    """LLM call 2: scoring, gaps, and narrative."""
    from llm.client import call_phi_sync
    p = _PROFILE

    prompt = _SCORING_TEMPLATE.format(
        domains        = ", ".join(p.get("core_domains", [])[:10]),
        tech_skills    = ", ".join(p.get("technical_skills", [])[:10]),
        tools_strong   = ", ".join(p.get("tools_software", {}).get("strong", [])),
        tools_moderate = ", ".join(p.get("tools_software", {}).get("moderate", [])),
        projects       = "; ".join(p.get("project_experience", [])[:5]),
        gaps           = "; ".join(p.get("known_honest_gaps", [])[:6]),
        preferred      = ", ".join(p.get("preferred_roles", [])[:6]),
        not_preferred  = ", ".join(p.get("not_preferred_roles", [])[:5]),
        title          = extracted.get("title", "Unknown"),
        industry       = extracted.get("industry", "Unknown"),
        role_type      = extracted.get("role_type", "Unknown"),
        hard_skills    = ", ".join((extracted.get("explicit_requirements") or {}).get("hard_skills", [])[:12]),
        domains_req    = ", ".join((extracted.get("explicit_requirements") or {}).get("domains", [])[:8]),
        tools_req      = ", ".join(det.get("tools_required", [])[:10]),
        certs          = ", ".join(det.get("certifications", [])[:6]),
        implicit       = "; ".join(extracted.get("implicit_requirements", [])[:6]),
        responsibilities = (extracted.get("responsibilities_summary") or "")[:400],
    )

    raw = call_phi_sync(
        [{"role": "system", "content": _SCORING_SYSTEM},
         {"role": "user",   "content": prompt}],
        num_ctx=4096, temperature=0.0,
    )
    return _parse_json_response(raw, {})


# ── Step 4b: LLM pass 3 — cover letter + email summary ───────────────────────

_COVERLETTER_SYSTEM = (
    "You are David Lagarejo's professional writing assistant. "
    "Write in first person as David. Be direct, confident, and specific. "
    "Never use generic openers. Use real project details and numbers when relevant. "
    "Return ONLY valid JSON. No extra text outside the JSON."
)

_COVERLETTER_TEMPLATE = """Write a tailored cover letter and email pitch for David Lagarejo applying to this job.

DAVID'S PROFILE SUMMARY:
- Physics Engineer with 10+ years in energy efficiency, IIoT, building systems, industrial projects
- Certifications: LEED Green Associate
- Key patents: U.S. Patent US2024/0077174 (steam monitoring system), 5 PCT patents
- Real project outcomes: $3M annual energy cost impact, 30% energy reduction achieved, $500K projects managed
- Projects: Ecopetrol (Oil & Gas, steam diagnostics), Alucol (power quality/IEEE 519), Malaysian Consulate (building retrofit), CW Contractor (NYC construction PM), Zircular (IIoT startup, IDEA 2023)
- Tools: Excel (advanced financial modeling), Python (data/automation), AutoCAD, ASANA, Power Quality Analyzers, Ultrasonic ToF
- Cover letter style: Direct, action-oriented, 3-4 paragraphs, NO generic openers, uses real numbers
- Signature: {signature}

JOB DETAILS:
Title: {title}
Company: {company}
Industry: {industry}
Role type: {role_type}
Key requirements: {hard_skills}
Key responsibilities: {responsibilities}
Salary: {salary}

VERDICT & POSITIONING:
Overall fit: {overall_fit}%
Strengths for this role: {strengths}
Resume angle: {resume_angle}
Positioning strategy: {positioning}

Write a targeted cover letter that:
1. Opens with the most relevant specific achievement (not generic)
2. Connects 2-3 real proof points to specific job requirements
3. Addresses why this company/role specifically
4. Ends with a direct, confident call to action

Also write a short email subject line and 2-sentence email pitch.

Return exactly this JSON:
{{
  "cover_letter": "Full cover letter text here (use \\n for line breaks, 3-4 paragraphs)",
  "email_subject": "Subject line for email application",
  "email_pitch": "2 sentences max. Opening line for the email body before attaching resume."
}}"""


def _call_llm_coverletter(extracted: dict, llm_scores_raw: dict, scores: dict) -> dict:
    """LLM call 3: generate tailored cover letter and email pitch."""
    from llm.client import call_phi_sync
    p = _PROFILE

    strengths = "; ".join((llm_scores_raw.get("strengths_against_role") or [])[:4])
    prompt = _COVERLETTER_TEMPLATE.format(
        title           = extracted.get("title", "Unknown"),
        company         = extracted.get("company", "Unknown"),
        industry        = extracted.get("industry", "Unknown"),
        role_type       = extracted.get("role_type", "Unknown"),
        hard_skills     = ", ".join((extracted.get("explicit_requirements") or {}).get("hard_skills", [])[:10]),
        responsibilities= (extracted.get("responsibilities_summary") or "")[:400],
        salary          = extracted.get("salary", "not specified"),
        overall_fit     = scores.get("overall_fit", 0),
        strengths       = strengths or "energy efficiency, project management, data-driven modeling",
        resume_angle    = (llm_scores_raw.get("resume_angle") or "")[:300],
        positioning     = (llm_scores_raw.get("positioning_strategy") or "")[:300],
        signature       = _SIGNATURE,
    )

    raw = call_phi_sync(
        [{"role": "system", "content": _COVERLETTER_SYSTEM},
         {"role": "user",   "content": prompt}],
        num_ctx=4096, temperature=0.2,
    )
    return _parse_json_response(raw, {
        "cover_letter": "",
        "email_subject": f"Application – {extracted.get('title', 'Position')}",
        "email_pitch": "",
    })


# ── Step 5: Rule-based scoring overlay ───────────────────────────────────────

def _apply_rule_scoring(llm_scores: dict, det: dict, extracted: dict) -> dict:
    """
    Apply deterministic corrections on top of LLM scores.
    Rules override when we have hard evidence (tools matched, gaps triggered, etc.).
    """
    scores = dict(llm_scores)

    # Tools fit: override if we have deterministic data
    tools_req = det.get("tools_required", [])
    if tools_req:
        known  = len(det.get("tools_matched", {}).get("known",   []))
        total  = len(tools_req)
        strong_count = sum(1 for t in det.get("tools_matched", {}).get("known", [])
                           if t.lower() in _TOOLS_STRONG_BARE)
        if total > 0:
            # Weighted: strong=1.0, moderate=0.7, light=0.4
            score = min(100, int((strong_count * 1.0 + (known - strong_count) * 0.6) / total * 100))
            # Blend with LLM (60% rule, 40% LLM)
            llm_t = scores.get("tools_fit", 50)
            scores["tools_fit"] = int(score * 0.6 + llm_t * 0.4)

    # Credibility risk: boost if gaps triggered
    gaps_hit = len(det.get("gaps_triggered", []))
    if gaps_hit >= 2:
        scores["credibility_risk"] = min(100, scores.get("credibility_risk", 20) + gaps_hit * 12)
    elif gaps_hit == 1:
        scores["credibility_risk"] = min(100, scores.get("credibility_risk", 20) + 10)

    # Industry fit: boost if direct industry match
    if det.get("industries_match"):
        scores["industry_fit"] = max(scores.get("industry_fit", 50), 65)

    # Clamp all scores to 0-100
    for k in scores:
        scores[k] = max(0, min(100, int(scores[k])))

    return scores


# ── Step 6: Red flags (deterministic) ────────────────────────────────────────

_SALES_KEYWORDS    = re.compile(r'\b(quota|commission|OTE|pipeline|closing deals|book of business|hunter|BD revenue target)\b', re.I)
_ADMIN_KEYWORDS    = re.compile(r'\b(data entry|filing|scheduling meetings|administrative support|receptionist|clerical)\b', re.I)
_DISGUISED_KEYWORDS = re.compile(r'\b(sales engineer|account executive|business development representative|BDR|SDR)\b', re.I)


def detect_red_flags(text: str, det: dict, extracted: dict, scores: dict) -> list[str]:
    """Deterministic red flag detection."""
    flags = []

    if det.get("low_salary_flag"):
        flags.append(f"Salary below floor (${_PROFILE.get('salary_floor_usd', 80000):,}/yr minimum)")

    if _SALES_KEYWORDS.search(text):
        flags.append("Sales/quota-driven elements detected despite technical framing")

    if _DISGUISED_KEYWORDS.search(text):
        role = extracted.get("role_type", "")
        if role not in ("business_development", "sales"):
            flags.append("Title may disguise a primarily sales role")

    if _ADMIN_KEYWORDS.search(text):
        flags.append("Administrative tasks dominate — below strategic value threshold")

    gaps_hit = det.get("gaps_triggered", [])
    if len(gaps_hit) >= 2:
        flags.append(f"Multiple honest gap areas required: {', '.join(gaps_hit[:3])}")

    certs = det.get("certifications", [])
    hard_certs = [c for c in certs if re.search(r'\bPE\b|LEED\s*AP|CEM\b|Licensed', c, re.I)]
    if hard_certs:
        flags.append(f"Requires licensure/certification not held: {', '.join(hard_certs[:3])}")

    yrs = det.get("min_years_exp", 0)
    if yrs >= 10 and det.get("low_salary_flag"):
        flags.append("Expects 10+ years experience with below-market compensation")

    req_count = det.get("req_count", 0)
    if req_count > 12:
        flags.append(f"Unusually high number of requirements ({req_count}) — may be unrealistic or catch-all posting")

    if scores.get("credibility_risk", 0) >= 60:
        flags.append("High credibility risk: role requires domain expertise outside direct experience")

    return flags


# ── Step 7: Verdict (deterministic rules) ────────────────────────────────────

def determine_verdict(overall: int, credibility_risk: int, flags: list[str]) -> str:
    """
    Apply deterministic thresholds from profile.json.
    Red flags can veto verdict upgrades.
    """
    thresholds = _PROFILE.get("verdict_thresholds", {
        "APPLY_NOW":     {"overall_min": 72, "credibility_max": 35},
        "APPLY_IF_TIME": {"overall_min": 55, "credibility_max": 55},
        "HOLD":          {"overall_min": 38, "credibility_max": 70},
    })

    high_risk_flags = sum(1 for f in flags if any(kw in f.lower() for kw in
                          ["credibility", "licensure", "gap area", "quota", "below"]))

    t_now = thresholds.get("APPLY_NOW", {})
    if (overall >= t_now.get("overall_min", 72) and
            credibility_risk <= t_now.get("credibility_max", 35) and
            high_risk_flags == 0):
        return "APPLY_NOW"

    t_ift = thresholds.get("APPLY_IF_TIME", {})
    if (overall >= t_ift.get("overall_min", 55) and
            credibility_risk <= t_ift.get("credibility_max", 55) and
            high_risk_flags <= 1):
        return "APPLY_IF_TIME"

    t_hold = thresholds.get("HOLD", {})
    if (overall >= t_hold.get("overall_min", 38) and
            credibility_risk <= t_hold.get("credibility_max", 70)):
        return "HOLD"

    return "DO_NOT_APPLY"


# ── Step 8: Overall fit ───────────────────────────────────────────────────────

def compute_overall_fit(scores: dict) -> int:
    """
    Weighted composite of positive dimensions.
    learning_curve_risk and credibility_risk reduce the score.
    """
    w = _PROFILE.get("scoring_weights", {})
    positive = (
        scores.get("technical_fit",    50) * w.get("technical_fit",    0.20) +
        scores.get("industry_fit",     50) * w.get("industry_fit",     0.15) +
        scores.get("tools_fit",        50) * w.get("tools_fit",        0.10) +
        scores.get("execution_fit",    50) * w.get("execution_fit",    0.15) +
        scores.get("management_fit",   50) * w.get("management_fit",   0.10) +
        scores.get("communication_fit",50) * w.get("communication_fit",0.10) +
        scores.get("regulatory_fit",   50) * w.get("regulatory_fit",   0.05)
    )
    # Risk scores reduce overall (higher risk = lower overall)
    risk_penalty = (
        (scores.get("learning_curve_risk", 20) / 100) * 100 * w.get("learning_curve_risk", 0.075) +
        (scores.get("credibility_risk",    20) / 100) * 100 * w.get("credibility_risk",    0.075)
    )
    # Normalize positive to same base (sum of positive weights = 0.85)
    pos_weight_sum = sum(v for k, v in w.items()
                         if k not in ("learning_curve_risk", "credibility_risk"))
    normalized = (positive / pos_weight_sum) if pos_weight_sum > 0 else positive
    return max(0, min(100, int(normalized - risk_penalty)))


# ── Utilities ─────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str, fallback: dict) -> dict:
    """Robustly extract first JSON object from LLM output."""
    if not raw:
        return fallback
    # Strip markdown fences
    raw = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`').strip()
    s = raw.find('{')
    e = raw.rfind('}') + 1
    if s < 0 or e <= s:
        return fallback
    chunk = raw[s:e]
    # Fix trailing commas
    chunk = re.sub(r',\s*([}\]])', r'\1', chunk)
    try:
        return json.loads(chunk)
    except json.JSONDecodeError:
        _log.warning("JSON parse failed on LLM output, returning fallback")
        return fallback


def _extract_metadata_from_text(text: str) -> dict:
    """Heuristic extraction of title/company/location from first ~500 chars."""
    lines = [l.strip() for l in text[:500].split('\n') if l.strip()]
    title   = lines[0] if lines else ""
    company = lines[1] if len(lines) > 1 else ""
    location_match = re.search(
        r'\b(?:New York|NYC|Remote|Hybrid|San Francisco|Boston|Chicago|Austin|Los Angeles|LA)\b',
        text[:600], re.I
    )
    location = location_match.group(0) if location_match else ""
    return {"title_hint": title[:100], "company_hint": company[:80], "location_hint": location}


# ── Main entry point ──────────────────────────────────────────────────────────

def analyze_job(raw_text: str, source: str = "", url: str = "", link: str = "") -> dict:
    """
    Full analysis pipeline. Returns structured job result dict.
    Saves result to data/jobs/{job_id}.json.
    """
    t0 = time.monotonic()

    # 1. Normalize
    text = normalize_text(raw_text)
    if len(text) < 100:
        return {"error": "Job text too short to analyze", "job_id": None}

    # 2. Deterministic extraction
    det = extract_deterministic(text)

    # 3. LLM extraction
    _log.info("job_analyzer: LLM extraction pass (len=%d)", len(text))
    extracted = _call_llm_extraction(text)
    if not extracted:
        # Fallback to heuristic hints
        hints      = _extract_metadata_from_text(text)
        extracted  = {
            "title": hints["title_hint"],
            "company": hints["company_hint"],
            "location": hints["location_hint"],
            "seniority": "unknown",
            "industry": "",
            "role_type": "other",
            "job_nature": "unknown",
            "explicit_requirements": {},
            "implicit_requirements": [],
            "responsibilities_summary": "",
        }

    # Merge salary from deterministic if LLM missed it
    if not extracted.get("salary") and det.get("salary_raw"):
        extracted["salary"] = det["salary_raw"]

    # 4. LLM scoring
    _log.info("job_analyzer: LLM scoring pass")
    llm_scores_raw = _call_llm_scoring(extracted, det)
    llm_scores     = llm_scores_raw.get("scores", {}) or {}

    # 5. Rule overlay
    scores = _apply_rule_scoring(llm_scores, det, extracted)

    # 6. Overall fit
    overall = compute_overall_fit(scores)
    scores["overall_fit"] = overall

    # 7. Red flags (combine deterministic + LLM)
    det_flags = detect_red_flags(text, det, extracted, scores)
    llm_flags = llm_scores_raw.get("red_flags", []) or []
    all_flags = list(dict.fromkeys(det_flags + [f for f in llm_flags if f not in det_flags]))

    # 8. Verdict
    verdict = determine_verdict(overall, scores.get("credibility_risk", 0), all_flags)

    # 9. Cover letter + email pitch
    _log.info("job_analyzer: LLM cover letter pass")
    coverletter_raw = _call_llm_coverletter(extracted, llm_scores_raw, scores)

    # 10. Assemble result
    job_id = "job_" + hashlib.md5((text[:500] + source).encode()).hexdigest()[:10]

    result = {
        "job_id":       job_id,
        "analyzed_at":  datetime.now(timezone.utc).isoformat()[:19] + "Z",
        "source":       source or "manual",
        "url":          url or link or "",
        "title":        extracted.get("title")    or "",
        "company":      extracted.get("company")  or "",
        "location":     extracted.get("location") or det.get("location_hint", ""),
        "salary":       extracted.get("salary")   or det.get("salary_raw", ""),
        "modality":     extracted.get("modality") or "unknown",
        "seniority":    extracted.get("seniority") or "unknown",
        "industry":     extracted.get("industry")  or "",
        "role_type":    extracted.get("role_type") or "other",
        "job_nature":   extracted.get("job_nature") or "unknown",
        "explicit_requirements": extracted.get("explicit_requirements") or {},
        "implicit_requirements": extracted.get("implicit_requirements") or [],
        "tools_required":   det.get("tools_required", []),
        "domain_required":  (extracted.get("explicit_requirements") or {}).get("domains", []),
        "scores":           scores,
        "overall_fit":      overall,
        "strengths_against_role": llm_scores_raw.get("strengths_against_role", []),
        "gaps":             llm_scores_raw.get("gaps", []),
        "red_flags":        all_flags,
        "verdict":          verdict,
        "reasoning_summary":    llm_scores_raw.get("reasoning_summary", ""),
        "positioning_strategy": llm_scores_raw.get("positioning_strategy", ""),
        "resume_angle":         llm_scores_raw.get("resume_angle", ""),
        "cover_letter":         coverletter_raw.get("cover_letter", ""),
        "email_subject":        coverletter_raw.get("email_subject", ""),
        "email_pitch":          coverletter_raw.get("email_pitch", ""),
        "confidence":           max(30, min(95, overall - len(all_flags) * 5)),
        "elapsed_seconds":      round(time.monotonic() - t0, 1),
        "company_url":          "",   # resolved on demand via /api/jobs/{id}/company-url
    }

    # 10. Persist
    job_path = _JOBS_DIR / f"{job_id}.json"
    try:
        job_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        _log.info("job_analyzer: saved %s verdict=%s fit=%d elapsed=%.1fs",
                  job_id, verdict, overall, result["elapsed_seconds"])
    except Exception as e:
        _log.warning("job_analyzer: failed to save job %s: %s", job_id, e)

    return result


# ── US location filter ────────────────────────────────────────────────────────

# Two-letter US state abbreviations
_US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC",
}

_US_KEYWORDS = frozenset({
    "united states", "usa", "u.s.a", "u.s.", " us ", "america",
    "new york", "nyc", "manhattan", "brooklyn", "bronx", "queens",
    "los angeles", "san francisco", "chicago", "houston", "boston",
    "seattle", "austin", "denver", "miami", "washington dc",
    "remote",  # include unqualified remote — likely US-targeting
})

# Phrases that clearly indicate non-US locations
_NON_US_INDICATORS = re.compile(
    r"\b(india|india\b|canada|uk|london|united kingdom|australia|germany|france|"
    r"spain|mexico|brazil|china|singapore|dubai|uae|europe|asia|africa|"
    r"latin america|remote.*worldwide|worldwide.*remote|global.*remote)\b",
    re.IGNORECASE,
)


def _is_us_location(location: str) -> bool:
    """
    Return True if the location is US-based, US-remote, or unspecified.
    Returns False only when a clearly non-US location is detected.
    """
    if not location or not location.strip():
        return True  # unspecified → assume US (search queries already target US)

    loc = location.strip()

    # Reject clearly non-US locations
    if _NON_US_INDICATORS.search(loc):
        return False

    loc_lower = loc.lower()

    # Accept if contains a known US keyword
    if any(kw in loc_lower for kw in _US_KEYWORDS):
        return True

    # Accept if ends with a US state abbreviation (e.g. "New York, NY")
    parts = re.split(r"[,\s]+", loc)
    if any(p.upper() in _US_STATES for p in parts):
        return True

    # Anything else — optimistically include (search queries target US)
    return True


# ── Company URL resolver ──────────────────────────────────────────────────────

# Sites that are job boards / aggregators — NOT the company's own site
_JOB_BOARD_DOMAINS = {
    # Job boards & ATS
    "linkedin.com", "indeed.com", "glassdoor.com", "wellfound.com",
    "angel.co", "monster.com", "ziprecruiter.com", "dice.com",
    "simplyhired.com", "careerbuilder.com", "greenhouse.io", "lever.co",
    "workday.com", "taleo.net", "icims.com", "smartrecruiters.com",
    "jobvite.com", "bamboohr.com", "recruitee.com", "workable.com",
    "breezy.hr", "ashbyhq.com", "rippling.com", "gusto.com",
    # Social / directories
    "twitter.com", "x.com", "facebook.com", "instagram.com",
    "crunchbase.com", "pitchbook.com", "bloomberg.com", "reuters.com",
    "builtinnyc.com", "builtin.com", "ycombinator.com",
    "techcrunch.com", "venturebeat.com", "wired.com", "forbes.com",
    # Document / aggregator / wiki
    "dokumen.pub", "scribd.com", "academia.edu", "researchgate.net",
    "wikipedia.org", "wikimedia.org", "archive.org",
    "slideshare.net", "issuu.com", "docslib.org",
    # Generic info / news sites
    "bing.com", "google.com", "yahoo.com", "duckduckgo.com",
    "startpage.com", "quora.com", "reddit.com", "medium.com",
    "substack.com", "tumblr.com", "wordpress.com",
}

_URL_RE = re.compile(r"https?://(?:www\.)?([^/\s\"'<>]+)")


def _is_job_board(url: str) -> bool:
    """Return True if the URL belongs to a job board / aggregator."""
    m = _URL_RE.match(url)
    if not m:
        return True
    domain = m.group(1).lower()
    return any(domain == d or domain.endswith("." + d) for d in _JOB_BOARD_DOMAINS)


def _build_url_candidates(company_name: str) -> list[str]:
    """
    Script-based: generate likely URL candidates for a company name.
    No network calls. Returns list of https URLs to try, ordered by likelihood.
    """
    words = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in company_name.split() if len(w) >= 2]
    words_no_generic = [w for w in words if w not in (
        "inc", "llc", "corp", "ltd", "co", "the", "and", "of",
        "ventures", "capital", "fund", "group", "partners", "labs",
    )]

    slug_full  = "".join(words)
    slug_main  = "".join(words_no_generic)
    slug_dash  = "-".join(words_no_generic)
    slug_first = words[0] if words else ""
    slug_fn    = words_no_generic[0] if words_no_generic else slug_first

    candidates = []
    for tld in (".com", ".org", ".io", ".net"):
        if slug_main:
            candidates.append(f"https://www.{slug_main}{tld}")
            candidates.append(f"https://{slug_main}{tld}")
        if slug_dash and slug_dash != slug_main:
            candidates.append(f"https://www.{slug_dash}{tld}")
        if slug_full and slug_full != slug_main:
            candidates.append(f"https://www.{slug_full}{tld}")
    # Abbreviation patterns: e.g. "Congruent Ventures" → congruentvc.com
    if len(words_no_generic) >= 2:
        abbr = words_no_generic[0] + words_no_generic[1][0] + "c"  # e.g. congruentvc
        candidates.append(f"https://www.{abbr}.com")
        abbr2 = words_no_generic[0] + words_no_generic[1][:3]  # congruentven
        candidates.append(f"https://www.{abbr2}.com")
    if slug_fn:
        candidates.append(f"https://www.{slug_fn}.com")

    return candidates


def _check_url_exists(url: str, timeout: int = 3) -> bool:
    """HEAD request to verify a URL responds with 2xx or 3xx."""
    import urllib.request
    try:
        req = urllib.request.Request(url, method="HEAD", headers={
            "User-Agent": "Mozilla/5.0 (compatible; phi-twin/1.0)",
            "Accept": "text/html",
        })
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.status < 400
    except Exception:
        return False


def _verify_url_is_company(url: str, company_name: str, timeout: int = 4) -> bool:
    """
    Fetch the first 2KB of the page and check if the company name appears.
    Returns True if the page looks like the company's own site.
    """
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; phi-twin/1.0)",
            "Accept": "text/html",
        })
        resp = urllib.request.urlopen(req, timeout=timeout)
        content = resp.read(2048).decode("utf-8", errors="ignore").lower()
        # Check if any significant word from company name appears in page
        _tokens = [
            re.sub(r"[^a-z0-9]", "", w.lower())
            for w in company_name.split()
            if len(w) >= 4
        ]
        return any(tok in content for tok in _tokens)
    except Exception:
        return False


def find_company_url(company_name: str, job_title: str = "", existing_url: str = "") -> str:
    """
    Script-based company website resolver.

    Strategy (no LLM):
    1. Generate candidate URLs from company name patterns
    2. Verify each with a HEAD request — return first that responds
    3. Fall back to SearXNG search with name-match filter

    Returns the company URL or "" if not found.
    """
    if not company_name or company_name in ("Unknown", "—", ""):
        return ""

    # Step 1: Try generated URL candidates + verify content matches company
    candidates = _build_url_candidates(company_name)
    for url in candidates[:12]:  # limit HEAD requests
        if _check_url_exists(url) and _verify_url_is_company(url, company_name):
            _log.info("company_url_found_verified company=%r url=%r", company_name[:40], url[:80])
            return url

    # Step 2: SearXNG search with strict name-domain matching
    try:
        from tools.search import search_web
    except ImportError:
        return ""

    _name_tokens = [
        re.sub(r"[^a-z0-9]", "", w.lower())
        for w in company_name.split()
        if len(w) >= 4
    ]

    queries = [
        f"{company_name} official website",
        f"{company_name} about company",
    ]

    for query in queries:
        try:
            results = search_web(query)
        except Exception:
            continue
        for r in results:
            url = r.url or ""
            if not url.startswith("http") or _is_job_board(url):
                continue
            _dm = _URL_RE.match(url)
            _domain = _dm.group(1).lower() if _dm else ""
            # Only accept if a company name token appears in the domain
            if any(tok in _domain for tok in _name_tokens if len(tok) >= 4):
                _log.info("company_url_found_search company=%r url=%r", company_name[:40], url[:80])
                return url

    _log.info("company_url_not_found company=%r", company_name[:40])
    return ""


def resolve_company_url(job_id: str) -> str:
    """
    Resolve and cache the company URL for a saved job.
    Reads the job file, finds the URL if missing, saves it back.
    Returns the company URL or "".
    """
    job = get_job(job_id)
    if job is None:
        return ""

    # Already resolved to a real company site
    existing = job.get("company_url", "")
    if existing and not _is_job_board(existing):
        return existing
    # Clear bad (job board) company_url so we can resolve properly
    if existing:
        job["company_url"] = ""

    company = job.get("company", "")
    title   = job.get("title", "")
    url     = find_company_url(company, title)

    if url:
        job["company_url"] = url
        try:
            (_JOBS_DIR / f"{job_id}.json").write_text(
                json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            _log.info("company_url_cached job=%s company=%r url=%r", job_id, company[:40], url[:80])
        except Exception as e:
            _log.warning("company_url_save_failed: %s", e)

    return url


# ── List / get / delete ───────────────────────────────────────────────────────

def list_jobs(limit: int = 50, only_with_url: bool = True) -> list[dict]:
    """
    Return analyzed jobs sorted by analyzed_at descending (summary fields only).

    only_with_url=True (default): only returns jobs where a verified company URL
    was found. Jobs without a source URL are hidden until resolved — this prevents
    recruiter/job-board aggregator postings from appearing without a direct source.
    Jobs sourced from or linking to job boards (ZipRecruiter, Indeed, etc.) are
    always hidden regardless of the only_with_url flag.
    """
    jobs = []
    for p in sorted(_JOBS_DIR.glob("job_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit * 3]:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            company_url = d.get("company_url", "")
            listing_url = d.get("url", "")

            # Always hide jobs whose listing URL is a job board aggregator
            if listing_url and _is_job_board(listing_url):
                continue

            # If company_url was incorrectly set to a job board, clear it
            if company_url and _is_job_board(company_url):
                company_url = ""

            # Skip jobs without a real company URL when filter is on
            if only_with_url and not company_url:
                continue

            # US-only filter — skip jobs with clearly non-US locations
            if not _is_us_location(d.get("location", "")):
                continue

            jobs.append({
                "job_id":      d.get("job_id"),
                "title":       d.get("title"),
                "company":     d.get("company"),
                "location":    d.get("location"),
                "verdict":     d.get("verdict"),
                "overall_fit": d.get("overall_fit") or (d.get("scores") or {}).get("overall_fit", 0),
                "analyzed_at": d.get("analyzed_at"),
                "role_type":   d.get("role_type"),
                "salary":      d.get("salary"),
                "company_url": company_url,
            })
            if len(jobs) >= limit:
                break
        except Exception:
            pass
    return jobs


def get_job(job_id: str) -> dict | None:
    p = _JOBS_DIR / f"{job_id}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def delete_job(job_id: str) -> bool:
    p = _JOBS_DIR / f"{job_id}.json"
    if p.exists():
        p.unlink()
        return True
    return False


# ── Resume & email generator (on-demand) ─────────────────────────────────────

_RESUME_SYSTEM = (
    "You are David Lagarejo's resume writer. "
    "Write in first person implied (no 'I'). Be specific, results-oriented, and direct. "
    "Tailor EVERY bullet and the summary tightly to the specific job. "
    "Never invent experience David doesn't have. "
    "Return ONLY valid JSON. No text outside the JSON."
)

_RESUME_TEMPLATE = """Write a tailored resume for David Lagarejo applying to this specific job.
Follow the EXACT format template below — adapt content, never change the structure.

DAVID'S REAL EXPERIENCE (use only what's real):
- Physics Engineer, 10+ years energy efficiency + IIoT + construction PM
- LEED Green Associate · U.S. Patent US2024/0077174 (steam monitoring)
- 5 PCT patents · IDEA 2023 presenter
- ZION ING / ZIRCULAR (2014-2023): Founded, industrial energy consulting, $3M impact, 30% energy reduction
- Ecopetrol: ultrasonic steam trap diagnostics, Class I Div 2, patent deployment
- Alucol: power quality analysis per IEEE 519, harmonic distortion, kW loss modeling
- CW Contractor / GoPro Restoration (2022-2025): PM construction + energy retrofits NYC, 5+ vendors, HVAC/LED/steam/insulation
- Malaysian Consulate: full building retrofit
- Class One Construction: site supervisor, safety compliance
- Tools STRONG: Excel (advanced financial models), Python (data/automation), ASANA, AutoCAD, Power Quality Analyzers, Ultrasonic ToF
- Tools MODERATE: MATLAB, Power BI, Scikit-learn, Make.com, C++
- Key metrics: $3M annual cost impact, 50K+ liters water savings, 3.4 tCO2e/yr, projects up to $500K/site

JOB DETAILS:
Title: {title}
Company: {company}
Industry: {industry}
Role type: {role_type}
Key hard skills required: {hard_skills}
Key responsibilities: {responsibilities}
Tools required: {tools_req}
Implicit needs: {implicit}

POSITIONING: {positioning}
RESUME ANGLE: {resume_angle}
STRENGTHS FOR THIS ROLE: {strengths}

FORMAT RULES (follow exactly):
1. Summary: 2-4 sentences tailored to this specific role and company. No generic openers.
2. SKILLS section: group into 2-3 categories, 3-5 items each, pick skills most relevant to this job.
3. PROFESSIONAL EXPERIENCE: include only the 3-4 most relevant roles. Tailor EACH bullet to match job requirements. 4-6 bullets per role. Lead with strongest results.
4. SELECTED ACHIEVEMENTS: 3-4 bullets. Only include if they add credibility for this specific role.
5. EDUCATION: always include.
6. CERTIFICATIONS: only include certs relevant to this role.
7. Never invent companies, projects, or outcomes David didn't actually have.

Return exactly this JSON:
{{
  "summary": "2-4 sentence tailored summary for this specific role",
  "skills": [
    {{"category": "Category Name", "items": ["skill1", "skill2", "skill3"]}}
  ],
  "experience": [
    {{
      "company": "Company Name",
      "role": "Job Title",
      "location_dates": "Location | Start – End",
      "context": "Optional 1-2 sentence context (omit if not needed)",
      "bullets": ["Bullet 1", "Bullet 2", "Bullet 3", "Bullet 4"]
    }}
  ],
  "achievements": ["Achievement 1", "Achievement 2", "Achievement 3"],
  "certifications": ["LEED Green Associate"],
  "include_patent": true
}}"""

_SHORT_EMAIL_TEMPLATE = """Write a SHORT intro email for David Lagarejo applying to this job.

Rules:
- Maximum 5 lines total (not counting signature)
- NO generic openers ("I am excited to apply", "I hope this email finds you well")
- Start with the single strongest connection between David's work and this role
- 1 specific proof point with a number
- End with 1 direct sentence asking for a call/meeting
- Sign as: {signature}

Job: {title} at {company}
Key requirements: {hard_skills}
David's strongest fit: {strengths}
Salary: {salary}

Return exactly this JSON:
{{
  "subject": "Email subject line (concise, specific, NOT 'Application for...')",
  "body": "Full email body text (use \\n for line breaks, max 5 lines before signature)",
  "signature": "{signature}"
}}"""


def _render_resume_text(data: dict, job: dict) -> str:
    """Convert resume JSON dict → plain text in the standard format."""
    lines = []

    # Header
    lines += [
        _NAME.upper(),
        _HEADER_LINE,
        "",
        data.get("summary", ""),
        "",
    ]

    # Skills
    lines += ["SKILLS", ""]
    for grp in data.get("skills", []):
        lines.append(f"• {grp['category']}")
        for item in grp.get("items", []):
            lines.append(f"    ○ {item}")
    lines.append("")

    # Experience
    lines += ["PROFESSIONAL EXPERIENCE", ""]
    for exp in data.get("experience", []):
        lines.append(f"{exp.get('company', '')}  —  {exp.get('role', '')}")
        lines.append(exp.get("location_dates", ""))
        ctx = exp.get("context", "").strip()
        if ctx:
            lines += ["", ctx]
        for b in exp.get("bullets", []):
            lines.append(f"• {b}")
        lines.append("")

    # Achievements
    achievements = data.get("achievements", [])
    if achievements:
        lines += ["SELECTED ACHIEVEMENTS", ""]
        for a in achievements:
            lines.append(f"• {a}")
        lines.append("")

    # Patent
    if data.get("include_patent", False):
        lines += [
            "SELECTED TECHNICAL WORK",
            "",
            "U.S. Patent – Steam Flow Monitoring System  (USPTO Application: US2024/0077174)",
            "• Co-invented a non-invasive monitoring method using ultrasonic sensor data, signal processing, and system-level analysis.",
            "• Designed data processing logic to translate raw signals into validated performance metrics.",
            "• Applied predictive modeling to support maintenance and optimization decisions.",
            "",
        ]

    # Education
    lines += [
        "EDUCATION",
        "",
        "Bachelor of Physics Engineering – Technological University of Pereira, Colombia, 2014",
        "Additional Training",
        "• Machine Learning & AI Algorithms — Platzi (Python-based, 2020)",
        "",
    ]

    # Certifications
    certs = data.get("certifications", [])
    if certs:
        lines += ["CERTIFICATIONS", ""]
        for c in certs:
            lines.append(f"• {c}")
        lines.append("")

    # Languages
    lines += [
        "LANGUAGES",
        "",
        "• English — Professional working proficiency",
        "• Spanish — Native",
    ]

    return "\n".join(lines)


def generate_resume_and_email(job_id: str) -> dict:
    """
    Generate a tailored resume + short email for a previously analyzed job.
    Returns {"resume_text": str, "email_subject": str, "email_body": str, "email_signature": str}.
    """
    from llm.client import call_phi_sync

    job = get_job(job_id)
    if not job:
        return {"error": "Job not found"}

    p             = _PROFILE
    extracted     = {"title": job.get("title"), "company": job.get("company"),
                     "industry": job.get("industry"), "role_type": job.get("role_type"),
                     "explicit_requirements": job.get("explicit_requirements") or {}}
    hard_skills   = ", ".join((job.get("explicit_requirements") or {}).get("hard_skills", [])[:10])
    responsibilities = (job.get("explicit_requirements") or {}).get("responsibilities_summary",
                        job.get("reasoning_summary", ""))[:400]
    tools_req     = ", ".join(job.get("tools_required", [])[:8])
    implicit      = "; ".join((job.get("implicit_requirements") or [])[:5])
    strengths     = "; ".join((job.get("strengths_against_role") or [])[:4])
    positioning   = (job.get("positioning_strategy") or "")[:250]
    resume_angle  = (job.get("resume_angle") or "")[:250]

    # ── Resume pass ──────────────────────────────────────────────────────────
    _log.info("generate_resume job_id=%s title=%r", job_id, job.get("title"))
    resume_prompt = _RESUME_TEMPLATE.format(
        title          = job.get("title", ""),
        company        = job.get("company", ""),
        industry       = job.get("industry", ""),
        role_type      = job.get("role_type", ""),
        hard_skills    = hard_skills,
        responsibilities = responsibilities,
        tools_req      = tools_req,
        implicit       = implicit,
        positioning    = positioning,
        resume_angle   = resume_angle,
        strengths      = strengths,
    )
    resume_raw = call_phi_sync(
        [{"role": "system", "content": _RESUME_SYSTEM},
         {"role": "user",   "content": resume_prompt}],
        num_ctx=6144, temperature=0.15,
    )
    resume_data = _parse_json_response(resume_raw, {})
    resume_text = _render_resume_text(resume_data, job) if resume_data else ""

    # ── Short email pass ──────────────────────────────────────────────────────
    _log.info("generate_email job_id=%s", job_id)
    email_prompt = _SHORT_EMAIL_TEMPLATE.format(
        title       = job.get("title", ""),
        company     = job.get("company", ""),
        hard_skills = hard_skills,
        strengths   = strengths,
        salary      = job.get("salary", "not specified"),
        signature   = _SIGNATURE,
    )
    email_raw = call_phi_sync(
        [{"role": "system", "content": _RESUME_SYSTEM},
         {"role": "user",   "content": email_prompt}],
        num_ctx=3072, temperature=0.2,
    )
    email_data = _parse_json_response(email_raw, {})

    return {
        "resume_text":      resume_text,
        "resume_data":      resume_data,
        "email_subject":    email_data.get("subject", f"Application – {job.get('title', '')}"),
        "email_body":       email_data.get("body", ""),
        "email_signature":  email_data.get("signature", _SIGNATURE),
        "job_id":           job_id,
        "job_title":        job.get("title", ""),
        "job_company":      job.get("company", ""),
    }
