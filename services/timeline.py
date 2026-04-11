"""
services/timeline.py — Timeline data assembly.

Single responsibility: aggregate dossiers, curated cards, task cards, OG images,
and audit metadata into the payload returned by GET /api/timeline.

Performance:
  - In-memory response cache (TTL=30s). Invalidated explicitly by route handlers
    that mutate timeline data (like, add, delete, cleanup). Research engine cycles
    are covered by TTL.
  - Dossier data read via tools/dossier_index (1 file read when index is fresh,
    N stat() calls + 1 rebuild when stale). Eliminates N JSON reads per request.
  - OG images fetched in parallel via asyncio.gather, cached to disk.

This is async because OG image fetching uses httpx.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx

from tools.dossier_index import load_dossiers

_log  = logging.getLogger("phi.services.timeline")
_BASE = Path(__file__).parent.parent

_LIKES_FILE    = _BASE / "data" / "likes.json"
_OG_CACHE_FILE = _BASE / "data" / "og_cache.json"
_CARDS_FILE    = _BASE / "data" / "timeline_cards.json"
_EVIDENCE_FILE = _BASE / "workspace" / "state" / "evidence.jsonl"
_DOSSIER_DIR   = _BASE / "workspace" / "dossiers"
_CONTACTS_FILE = _BASE / "data" / "contacts.json"

# ── In-memory response cache ──────────────────────────────────────────────────
_CACHE_TTL = 30.0   # seconds
_cache: dict = {"data": None, "ts": 0.0}
_cache_lock = asyncio.Lock()

# ── Translation cache — persisted to disk, survives restarts ─────────────────
_TX_CACHE_FILE = _BASE / "data" / "tx_cache.json"
_tx_cache: dict[str, str] = {}
_tx_bg_running = False   # guard: only one bg_translate at a time

def _load_tx_cache() -> None:
    global _tx_cache
    try:
        if _TX_CACHE_FILE.exists():
            _tx_cache = json.loads(_TX_CACHE_FILE.read_text(encoding="utf-8"))
            _log.info("tx_cache loaded: %d entries", len(_tx_cache))
    except Exception as _e:
        _log.warning("tx_cache load error: %s", _e)

def _save_tx_cache() -> None:
    try:
        _TX_CACHE_FILE.write_text(
            json.dumps(_tx_cache, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as _e:
        _log.warning("tx_cache save error: %s", _e)

_load_tx_cache()  # load on import


def invalidate_cache() -> None:
    """Force rebuild on next build_timeline_response() call."""
    _cache["ts"] = 0.0


async def _get_cached() -> Optional[dict]:
    async with _cache_lock:
        if _cache["data"] is not None and (time.monotonic() - _cache["ts"]) < _CACHE_TTL:
            return _cache["data"]
    return None


async def _set_cached(data: dict) -> None:
    async with _cache_lock:
        _cache["data"] = data
        _cache["ts"]   = time.monotonic()

_EU_KEYWORDS = frozenset({
    "europe", "eu", "germany", "ukraine", "france", "spain",
    "italy", "netherlands", "austria", "belgium", "poland",
    "sweden", "denmark", "finland", "norway", "portugal",
    "switzerland", "czech", "hungary", "romania", "slovakia",
})

_TYPE_LABEL = {
    "ORG": "empresa", "GRANT": "grant", "PERSON": "persona",
    "PATENT": "patente", "EVENT": "evento", "INVESTOR": "inversor",
    "JOB": "empleo",
}

_ACTION_KEYWORDS = (
    "reach out", "contact", "email", "schedule", "send", "proposal",
    "pitch", "meeting", "follow", "apply", "submit", "prepare",
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_logo_url(website: Optional[str]) -> Optional[str]:
    if not website:
        return None
    try:
        host = urlparse(website).netloc.lstrip("www.")
        return f"https://logo.clearbit.com/{host}" if host else None
    except Exception:
        return None


def _get_top_sources(evidence_ids: list, evidence_by_id: dict, limit: int = 4) -> list:
    seen_domains: set = set()
    sources: list = []
    candidates: list = []

    for eid in evidence_ids:
        ev = evidence_by_id.get(eid)
        if not ev or not ev.get("url") or not ev.get("title"):
            continue
        score = (ev.get("relevance_score") or 0) + (ev.get("credibility_score") or 0)
        candidates.append((score, ev))

    candidates.sort(key=lambda x: -x[0])

    for _, ev in candidates:
        url = ev.get("url", "")
        try:
            domain = urlparse(url).netloc.lstrip("www.")
        except Exception:
            domain = url
        if domain in seen_domains:
            continue
        seen_domains.add(domain)

        title   = ev.get("title", "")[:90]
        snippet = (ev.get("snippet") or "").strip()[:160]
        is_yt   = "youtube.com" in domain or "youtu.be" in domain
        yt_id   = None
        if is_yt:
            m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
            yt_id = m.group(1) if m else None

        sources.append({
            "title": title,
            "url": url,
            "domain": domain,
            "snippet": snippet,
            "is_youtube": is_yt,
            "yt_id": yt_id,
        })
        if len(sources) >= limit:
            break
    return sources


async def _fetch_og_image(url: str) -> Optional[str]:
    """Fetch og:image from a URL with disk cache. Returns image URL or None."""
    og_cache: dict = {}
    try:
        if _OG_CACHE_FILE.exists():
            og_cache = json.loads(_OG_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass

    if url in og_cache:
        return og_cache[url]

    result = None
    try:
        async with httpx.AsyncClient(timeout=2.5, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; PhiBot/1.0)"})
            if resp.status_code == 200:
                chunk = resp.text[:12000]
                for pattern in [
                    r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']{10,})["\']',
                    r'<meta[^>]+content=["\']([^"\']{10,})["\'][^>]+property=["\']og:image["\']',
                    r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']{10,})["\']',
                    r'<meta[^>]+content=["\']([^"\']{10,})["\'][^>]+name=["\']twitter:image["\']',
                ]:
                    m = re.search(pattern, chunk, re.I)
                    if m:
                        img = m.group(1).strip()
                        if img.startswith("http"):
                            result = img
                            break
    except Exception:
        pass

    try:
        og_cache[url] = result
        _OG_CACHE_FILE.write_text(json.dumps(og_cache, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    return result


# ── Main assembler ────────────────────────────────────────────────────────────

async def build_timeline_response(audit_log: Path) -> dict:
    """
    Assemble the full /api/timeline response.

    Cache: returns cached payload if within TTL (30s). Callers that mutate
    timeline-relevant data must call invalidate_cache() first.

    Steps:
      1. Read audit log metadata.
      2. Load likes and evidence index from disk.
      3. Assemble dossier cards (one per workspace/dossiers/*.json).
      4. Fetch OG images in parallel for top-6 cards (cached).
      5. Load curated timeline_cards.json (non-research).
      6. Build task cards from next_actions + key_people in top-12 dossiers.

    Returns the dict that should be passed to JSONResponse.
    """
    cached = await _get_cached()
    if cached is not None:
        return cached

    now_iso = datetime.now(timezone.utc).isoformat()

    # ── Audit metadata ────────────────────────────────────────────────────────
    audit_count = 0
    last_audit  = None
    if audit_log.exists():
        lines = [l.strip() for l in audit_log.open(encoding="utf-8") if l.strip()]
        audit_count = len(lines)
        if lines:
            try:
                last_audit = json.loads(lines[-1]).get("timestamp", "")[:19]
            except Exception:
                pass

    # ── Likes ─────────────────────────────────────────────────────────────────
    likes: dict = {}
    try:
        if _LIKES_FILE.exists():
            likes = json.loads(_LIKES_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass

    # ── Evidence index ────────────────────────────────────────────────────────
    evidence_by_id: dict = {}
    try:
        if _EVIDENCE_FILE.exists():
            for line in _EVIDENCE_FILE.read_text(encoding="utf-8").splitlines():
                try:
                    ev = json.loads(line)
                    evidence_by_id[ev["evidence_id"]] = ev
                except Exception:
                    pass
    except Exception:
        pass

    # ── Dossier cards (uses index — 1 file read if fresh, 0 JSON reads per dossier) ──
    dossier_cards: list = []
    for d in load_dossiers():
        desc = (d.get("description") or "").strip()
        if not desc:
            continue

        fa      = d.get("fit_assessment") or {}
        pr      = d.get("profile") or {}
        fit     = fa.get("fit_score", 0) or 0
        pm      = fa.get("profile_match", 0) or 0
        tm      = fa.get("timing", 0) or 0
        er      = fa.get("effort_vs_reward", 0) or 0
        risk    = fa.get("risk", 0) or 0
        why_yes = (fa.get("why_yes") or [])
        why_not = (fa.get("why_not") or [])[:3]

        ev_count = len(d.get("evidence_ids") or [])
        name     = d.get("name", "?")
        etype    = d.get("type", "ORG")
        updated  = (d.get("last_updated") or "")[:10]
        website  = pr.get("website") or None
        country  = pr.get("country") or None
        sector   = pr.get("sector") or None

        # Fallback bullets when LLM hasn't generated why_yes yet
        if not why_yes:
            etype_map = {"ORG": "empresa/aliado", "GRANT": "grant", "INVESTOR": "inversor",
                         "PERSON": "contacto clave", "EVENT": "evento", "PATENT": "patente"}
            opp_type     = etype_map.get(etype, "oportunidad")
            next_actions = d.get("next_actions") or []
            next_action  = ""
            if next_actions and isinstance(next_actions, list):
                na = next_actions[0]
                next_action = (na if isinstance(na, str) else na.get("action", ""))[:100]
            bullets = []
            if next_action:
                bullets.append(f"Tipo: {opp_type} — acción: {next_action}")
            elif sector:
                bullets.append(f"Tipo: {opp_type} en {sector}")
            else:
                bullets.append(f"Tipo: {opp_type}")
            if fit >= 70:
                bullets.append(f"Score {fit}% — alto alineamiento con sensor ultrasónico y perfil cleantech de David")
            elif fit >= 50:
                bullets.append(f"Score {fit}% — fit moderado, validar encaje antes de invertir tiempo")
            first_sent = (d.get("description") or "").strip().split(".")[0].strip()
            if first_sent:
                bullets.append(first_sent)
            why_yes = bullets

        card_id    = d.get("entity_id") or f"ent_{d.get('entity_id','')[:8]}"
        like_count = likes.get(card_id, 0)

        country_lower = (country or "").lower()
        geo_flag = "out_of_scope" if any(k in country_lower for k in _EU_KEYWORDS) else None

        if geo_flag == "out_of_scope":
            prio = 3
        else:
            prio = 1 if (like_count > 0 or fit >= 75) else (2 if fit >= 50 else 3)

        fit_bars = []
        if fit > 0:
            fit_bars = [
                {"name": "Perfil",  "pct": pm},
                {"name": "Timing",  "pct": tm},
                {"name": "ROI",     "pct": er},
                {"name": "Riesgo",  "pct": risk, "inverted": True},
            ]

        type_label  = _TYPE_LABEL.get(etype, etype.lower())
        top_sources = _get_top_sources(d.get("evidence_ids") or [], evidence_by_id)
        logo_url    = _get_logo_url(website)

        deadline       = d.get("deadline") or None
        deadline_label = d.get("deadline_label") or None
        reg_url        = d.get("registration_url") or None

        dossier_cards.append({
            "id":               card_id,
            "entity_id":        card_id,
            "project":          name,
            "priority":         prio,
            "tag":              type_label,
            "title":            name,
            "desc":             desc,
            "meta":             f"{updated} · {ev_count} fuentes",
            "why_yes":          why_yes,
            "why_not":          why_not,
            "fit":              fit,
            "fit_bars":         fit_bars,
            "risk":             risk,
            "website":          website,
            "logo_url":         logo_url,
            "country":          country,
            "geo_flag":         geo_flag,
            "sector":           sector,
            "like_count":       like_count,
            "sources":          ev_count,
            "ref_sources":      top_sources,
            "top_pick":         False,
            "deadline":         deadline,
            "deadline_label":   deadline_label,
            "registration_url": reg_url,
            "metrics":          [],
            "actions":          [],
            "viz":              None,
        })

    # Sort: liked/high-fit first; out_of_scope geo pushed to bottom
    dossier_cards.sort(key=lambda c: (
        1 if c.get("geo_flag") == "out_of_scope" else 0,
        -c["like_count"] * 20 - c["fit"],
        c["title"],
    ))
    if dossier_cards:
        dossier_cards[0]["top_pick"] = True

    # ── OG images (parallel fetch, top-6 cards only) ──────────────────────────
    try:
        existing_cache: dict = {}
        if _OG_CACHE_FILE.exists():
            existing_cache = json.loads(_OG_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        existing_cache = {}

    og_tasks: list = []  # (card_idx, src_idx, url)
    for ci, card in enumerate(dossier_cards[:6]):
        for si, src in enumerate(card.get("ref_sources", [])):
            url = src.get("url", "")
            if url and not src.get("is_youtube") and url not in existing_cache:
                og_tasks.append((ci, si, url))
            elif url in existing_cache:
                src["image_url"] = existing_cache[url]

    if og_tasks:
        fetch_results = await asyncio.gather(
            *[_fetch_og_image(url) for _, _, url in og_tasks],
            return_exceptions=True,
        )
        for (ci, si, _url), result in zip(og_tasks, fetch_results):
            dossier_cards[ci]["ref_sources"][si]["image_url"] = result if isinstance(result, str) else None

    # ── Curated timeline cards (non-research) ────────────────────────────────
    curated_cards: list = []
    if _CARDS_FILE.exists():
        try:
            all_saved = json.loads(_CARDS_FILE.read_text(encoding="utf-8"))
            curated_cards = [c for c in all_saved if c.get("tag") != "investigación"]
        except Exception:
            pass

    # ── Load CRM contacts indexed by entity_id (both full and short forms) ───
    _crm_by_entity: dict[str, list] = {}
    try:
        _crm_raw = json.loads(_CONTACTS_FILE.read_text(encoding="utf-8")) if _CONTACTS_FILE.exists() else []
        for _c in _crm_raw:
            _eid = (_c.get("entity_id") or "").strip()
            if not _eid:
                continue
            # Index by full CRM entity_id (e.g. "ent_af7ddda9")
            _crm_by_entity.setdefault(_eid, []).append(_c)
            # Also index by short 8-char suffix (e.g. "af7d") for card_id lookups
            _short = _eid.replace("ent_", "")[:4]
            _crm_by_entity.setdefault(_short, []).append(_c)
    except Exception:
        pass

    # ── Task cards: CRM contacts (primary) + next_actions + key_people ───────
    task_cards: list = []
    _seen_contact_ids: set = set()

    _all_dossiers = load_dossiers()
    # Sort: GRANT > INVESTOR > EVENT > ORG, then by fit score descending
    _TYPE_PRIO = {"GRANT": 0, "INVESTOR": 1, "EVENT": 2, "ORG": 3, "PERSON": 4}
    _all_dossiers.sort(key=lambda d: (
        _TYPE_PRIO.get(d.get("type", "ORG"), 5),
        -((d.get("fit_assessment") or {}).get("fit_score", 0) or 0),
    ))
    def _strategy(etype: str, name: str, role: str, sp_name: str = "", why: str = "") -> str:
        """One-sentence presentation strategy tailored to entity type and contact role."""
        sp_ref = f" para '{sp_name}'" if sp_name else ""
        if etype == "GRANT":
            return (f"Aplicar{sp_ref}: presentar patent US2024/0077174 + validación Ecopetrol "
                    f"(-30% energía) como evidencia de tracción real. "
                    f"Contactar a {role or 'el Program Officer'} para confirmar encaje antes de someter.")
        if etype == "INVESTOR":
            return (f"Pitch a {role or 'el partner'}{sp_ref}: abrir con el patent + Ecopetrol "
                    f"(Tier 1 reference client), luego ROI (-30% vapor), luego ask ($300–750K pre-seed). "
                    f"Alinear con su tesis antes del primer email.")
        if etype == "EVENT":
            return (f"Participar{sp_ref}: proponer demo/ponencia con el sensor ultrasónico, "
                    f"contactar a {role or 'el organizador'} para solicitar espacio de startup showcase.")
        # ORG / default
        return (f"Abrir conversación con {role or 'el contacto'} en {name}: "
                f"proponer piloto IIoT no-invasivo — instala sin parar planta, "
                f"ROI medible en 90 días. {why[:80] if why else ''}").strip()

    for d in _all_dossiers:
        name    = d.get("name", "?")
        fa      = d.get("fit_assessment") or {}
        fit     = fa.get("fit_score", 0) or 0
        pr      = d.get("profile") or {}
        website = pr.get("website") or None
        raw_eid = d.get('entity_id', '')
        card_id = raw_eid or f"ent_{raw_eid[:8]}"
        etype   = d.get("type", "ORG")

        # 1. CRM contacts — look up by full entity_id or short suffix
        _crm_contacts = (
            _crm_by_entity.get(raw_eid) or          # full: "ent_af7ddda9"
            _crm_by_entity.get(card_id) or           # double-prefixed (legacy): "ent_ent_af7d"
            _crm_by_entity.get(raw_eid.replace("ent_","")[:4]) or  # short: "af7d"
            []
        )
        for crm in _crm_contacts[:3]:
            pname = crm.get("name") or ""
            prole = crm.get("role") or ""
            if not pname or pname.startswith("Dr. ["):
                continue
            uid = f"{card_id}_{pname[:12]}"
            if uid in _seen_contact_ids:
                continue
            _seen_contact_ids.add(uid)

            why    = crm.get("why_contact") or crm.get("outreach_reason") or ""
            angle  = crm.get("outreach_angle") or ""
            # Build a concise "why contact" sentence
            why_contact = (why.strip().rstrip(".") + ". " + angle.strip()).strip(" .") or f"Contacto clave en {name}"
            if len(why_contact) > 180:
                why_contact = why_contact[:177] + "…"

            task_cards.append({
                "id":           f"{card_id}_crm_{pname[:8].replace(' ','_')}",
                "type":         "task",
                "entity_name":  name,
                "entity_id":    card_id,
                "action":       why_contact,
                "strategy":     _strategy(etype, name, prole, why=why_contact),
                "fit":          crm.get("fit_score") or fit,
                "website":      website,
                "contact_name": pname,
                "contact_role": prole,
                "why_contact":  why_contact,
                "email":        crm.get("email") or None,
                "linkedin_url": crm.get("linkedin_url") or None,
            })

        # 2. recommended_outreach from dossier (single contact synthesized by Phi)
        _ro = d.get("recommended_outreach") or {}
        if _ro and not _ro.get("do_not_contact", True):
            pname = _ro.get("contact_name") or ""
            prole = _ro.get("contact_role") or ""
            uid   = f"{card_id}_{pname[:12]}"
            if pname and not pname.startswith("Dr. [") and uid not in _seen_contact_ids:
                _seen_contact_ids.add(uid)
                why   = _ro.get("reason") or ""
                angle = _ro.get("angle") or ""
                why_contact = (why.strip().rstrip(".") + ". " + angle.strip()).strip(" .") or f"Contacto clave en {name}"
                if len(why_contact) > 180:
                    why_contact = why_contact[:177] + "…"
                task_cards.append({
                    "id":           f"{card_id}_ro_{pname[:8].replace(' ','_')}",
                    "type":         "task",
                    "entity_name":  name,
                    "entity_id":    card_id,
                    "action":       why_contact,
                    "strategy":     _strategy(etype, name, prole, why=why_contact),
                    "fit":          fit,
                    "website":      website,
                    "contact_name": pname,
                    "contact_role": prole,
                    "why_contact":  why_contact,
                    "email":        None,
                    "linkedin_url": None,
                })

        # 3. sub_programs — one contact per relevant sub-program (GRANT/EVENT deep research)
        for sp_idx, sp in enumerate((d.get("sub_programs") or [])[:6]):
            sp_name    = sp.get("name", "")
            sp_officer = sp.get("program_officer") or ""
            sp_fit     = sp.get("fit_score", 0)
            sp_email   = sp.get("email") or None
            sp_reason  = sp.get("fit_reason") or ""
            sp_desc    = sp.get("description", "")[:120]
            if not sp_name:
                continue
            uid = f"{card_id}_sp_{sp_idx}_{sp_name[:10].replace(' ','_')}"
            if uid in _seen_contact_ids:
                continue
            _seen_contact_ids.add(uid)
            why_contact = f"Aplicar a {sp_name}: {sp_reason}" if sp_reason else f"Aplicar a {sp_name}"
            if sp_officer:
                why_contact += f" — contacto: {sp_officer}"
            task_cards.append({
                "id":              uid,
                "type":            "task",
                "entity_name":     name,
                "entity_id":       card_id,
                "action":          why_contact,
                "strategy":        _strategy(etype, name, sp_officer or "Program Officer", sp_name=sp_name),
                "fit":             sp_fit or fit,
                "website":         sp.get("apply_url") or website,
                "contact_name":    sp_officer or f"Program Officer ({sp_name})",
                "contact_role":    f"Program Officer — {sp_name}",
                "why_contact":     why_contact,
                "email":           sp_email,
                "linkedin_url":    sp.get("linkedin_url") or None,
                "sub_program":     sp_name,
                "sub_description": sp_desc,
            })

        # 4. key_people from profile (fallback)
        for person in (pr.get("key_people") or [])[:2]:
            pname = person.get("name", "")
            prole = person.get("role", "")
            uid   = f"{card_id}_{pname[:12]}"
            if not pname or pname.startswith("Dr. [") or uid in _seen_contact_ids:
                continue
            _seen_contact_ids.add(uid)
            task_cards.append({
                "id":           f"{card_id}_kp_{pname[:8].replace(' ','_')}",
                "type":         "task",
                "entity_name":  name,
                "entity_id":    card_id,
                "action":       f"Evaluar contacto: {pname} ({prole})",
                "strategy":     _strategy(etype, name, prole),
                "fit":          fit,
                "website":      website,
                "contact_name": pname,
                "contact_role": prole,
                "why_contact":  f"Figura clave en {name}. Investigar razón de contacto específica.",
                "email":        None,
                "linkedin_url": None,
            })

    # Sort by fit, then cap at 4 tasks per entity so no single org floods the list
    task_cards.sort(key=lambda c: -c["fit"])
    _per_entity: dict = {}
    _capped: list = []
    for tc in task_cards:
        eid = tc.get("entity_id", "")
        if _per_entity.get(eid, 0) >= 4:
            continue
        _per_entity[eid] = _per_entity.get(eid, 0) + 1
        _capped.append(tc)
        if len(_capped) >= 60:
            break
    task_cards = _capped

    # ── Apply any already-cached translations (instant — no blocking) ───────────
    try:
        from services.translator import translate_to_spanish, is_english  # noqa: F401

        def _apply_tx_cache(dc: list, tc: list) -> None:
            """Apply _tx_cache entries already computed; does NOT translate new texts."""
            for _card in dc:
                _v = _card.get("desc") or ""
                if _v in _tx_cache:
                    _card["desc"] = _tx_cache[_v]
                _card["why_yes"] = [_tx_cache.get(_b, _b) for _b in (_card.get("why_yes") or [])]
                _card["why_not"] = [_tx_cache.get(_b, _b) for _b in (_card.get("why_not") or [])]
            for _tc in tc:
                for _fld in ("action", "why_contact"):
                    _v = _tc.get(_fld) or ""
                    if _v in _tx_cache:
                        _tc[_fld] = _tx_cache[_v]

        _apply_tx_cache(dossier_cards, task_cards)

        # Collect texts not yet cached that need translation
        def _collect_untranslated(dc: list, tc: list) -> list[str]:
            out: list[str] = []
            seen: set[str] = set()
            def _add(t: str) -> None:
                if t and t not in _tx_cache and t not in seen and is_english(t):
                    seen.add(t)
                    out.append(t)
            for _card in dc:
                _add(_card.get("desc") or "")
                for _b in (_card.get("why_yes") or []):
                    _add(_b)
                for _b in (_card.get("why_not") or []):
                    _add(_b)
            for _tc in tc:
                for _fld in ("action", "why_contact"):
                    _add(_tc.get(_fld) or "")
            return out

        _untranslated = _collect_untranslated(dossier_cards, task_cards)
        if _untranslated and not _tx_bg_running:
            import threading as _threading

            def _bg_translate_thread(texts: list) -> None:
                global _tx_bg_running
                _tx_bg_running = True
                try:
                    for t in texts:
                        if t not in _tx_cache:
                            _tx_cache[t] = translate_to_spanish(t)
                    _save_tx_cache()
                    invalidate_cache()
                    _log.info("bg_translate done: %d texts → disk saved, timeline invalidated", len(texts))
                except Exception as _e:
                    _log.warning("bg_translate error: %s", _e)
                finally:
                    _tx_bg_running = False

            _threading.Thread(
                target=_bg_translate_thread, args=(_untranslated,), daemon=True, name="tx-bg"
            ).start()
            _log.info("bg_translate started: %d new texts to translate", len(_untranslated))

    except ImportError:
        _log.warning("services.translator not available — omitting EN→ES translation in timeline")
    except Exception as _exc:
        _log.warning("Translation error in build_timeline_response: %s", _exc)

    payload = {
        "cards":      dossier_cards + curated_cards,
        "task_cards": task_cards,
        "updated_at": now_iso,
    }
    await _set_cached(payload)
    return payload
