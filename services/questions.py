"""
services/questions.py — Active question system.

One question at a time. Phi analyzes David's documents + research context
to generate a single high-value strategic question. The next question is
only generated after the current one is answered.

Public API:
  get_active_question()           -> dict | None
  answer_active(qid, answer)      -> dict   (saves + generates next)
  load_profile_qa(n)              -> list[dict]
  scan_documents_folder()         -> str    (summary text for LLM context)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from api.context import BASE_DIR
from llm.client import call_phi_sync

_log = logging.getLogger("phi.services.questions")

_ACTIVE_Q_FILE  = BASE_DIR / "data" / "active_question.json"
_PROFILE_QA_FILE = BASE_DIR / "data" / "profile_qa.jsonl"
_DOCUMENTS_DIR  = Path("/Volumes/ZLab_Documents/Zlab_Document/Documents")

# ── Document scanner (script-based, no LLM) ───────────────────────────────────

# Folders to include from the Documents directory
_SCAN_FOLDERS = [
    "Proyecto Zircul1ar", "ZION ING", "Zircular accounting",
    "SCALE", "SOLIGASS Plasma Technology", "Vault",
]

# File extensions we can read as text
_TEXT_EXTS = {".txt", ".md", ".csv"}


def _read_pdf_text(path: Path, max_chars: int = 800) -> str:
    """Extract first N chars from a PDF using pdftotext if available."""
    try:
        result = subprocess.run(
            ["pdftotext", "-l", "2", str(path), "-"],
            capture_output=True, text=True, timeout=8,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()[:max_chars]
    except Exception:
        pass
    return ""


def scan_documents_folder() -> str:
    """
    Script-based scan of David's Documents folder.
    Returns a structured text summary for use in question generation.
    No LLM calls — pure filesystem + text extraction.
    """
    if not _DOCUMENTS_DIR.exists():
        return ""

    lines: list[str] = ["## Documentos de David (resumen automático)\n"]
    total_files = 0

    for folder_name in _SCAN_FOLDERS:
        folder = _DOCUMENTS_DIR / folder_name
        if not folder.exists():
            continue

        folder_files: list[str] = []
        for fpath in sorted(folder.rglob("*"), key=lambda p: p.stat().st_mtime if p.is_file() else 0, reverse=True):
            if not fpath.is_file():
                continue
            ext = fpath.suffix.lower()
            name = fpath.name

            # Skip system/temp files
            if name.startswith(".") or name.startswith("~") or "conflicto" in name.lower():
                continue

            if ext == ".pdf":
                text_preview = _read_pdf_text(fpath, max_chars=400)
                entry = f"  - {fpath.relative_to(_DOCUMENTS_DIR)} (PDF)"
                if text_preview:
                    entry += f"\n    Preview: {text_preview[:200].replace(chr(10), ' ')}"
                folder_files.append(entry)
                total_files += 1

            elif ext in _TEXT_EXTS:
                try:
                    content = fpath.read_text(encoding="utf-8", errors="ignore")[:300]
                    folder_files.append(f"  - {fpath.relative_to(_DOCUMENTS_DIR)}: {content[:200].replace(chr(10), ' ')}")
                    total_files += 1
                except Exception:
                    pass

            elif ext in (".xlsx", ".docx"):
                folder_files.append(f"  - {fpath.relative_to(_DOCUMENTS_DIR)} ({ext[1:].upper()})")
                total_files += 1

            if total_files >= 60:
                break

        if folder_files:
            lines.append(f"\n### {folder_name}")
            lines.extend(folder_files[:15])  # max 15 per folder

    if total_files == 0:
        return ""

    lines.append(f"\n_Total archivos escaneados: {total_files}_")
    return "\n".join(lines)


# ── Q&A persistence ────────────────────────────────────────────────────────────

def load_profile_qa(n: int = 20) -> list[dict]:
    """Return last n David answers from profile_qa.jsonl."""
    if not _PROFILE_QA_FILE.exists():
        return []
    lines = _PROFILE_QA_FILE.read_text(encoding="utf-8").strip().splitlines()
    entries = []
    for line in lines[-n:]:
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    return entries


def _save_answer(question: str, answer: str, question_type: str = "strategic") -> None:
    """Append answer to profile_qa.jsonl."""
    entry = {
        "question":    question,
        "answer":      answer,
        "type":        question_type,
        "answered_at": datetime.now(timezone.utc).isoformat()[:19] + "Z",
    }
    _PROFILE_QA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_PROFILE_QA_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Active question state ──────────────────────────────────────────────────────

def get_active_question() -> dict | None:
    """
    Return the current unanswered question, or None if no active question.
    Generates a new one if file is missing or expired (>72h).
    """
    if _ACTIVE_Q_FILE.exists():
        try:
            q = json.loads(_ACTIVE_Q_FILE.read_text(encoding="utf-8"))
            if not q.get("answered", False):
                # Regenerate if older than 3 days without answer
                age_h = (time.time() - q.get("generated_at_ts", 0)) / 3600
                if age_h < 72:
                    return q
        except Exception:
            pass

    # Generate a fresh question
    return _generate_and_save_question()


def answer_active(question_id: str, answer: str) -> dict:
    """
    Mark the active question as answered, save to profile_qa.jsonl,
    apply any category changes implied by the answer, then generate next question.
    Returns the new active question.
    Pass question_id='skip' / answer='__skip__' to skip without saving.
    """
    # Skip: just force regeneration without saving answer
    if question_id == "skip" or answer == "__skip__":
        return _generate_and_save_question()

    if _ACTIVE_Q_FILE.exists():
        try:
            q = json.loads(_ACTIVE_Q_FILE.read_text(encoding="utf-8"))
            if q.get("question_id") == question_id:
                _save_answer(q["question"], answer, q.get("type", "strategic"))
                # Apply category/directive changes if this was a meta question
                _apply_answer_to_context(q, answer)
                q["answered"] = True
                q["answer"] = answer
                q["answered_at"] = datetime.now(timezone.utc).isoformat()[:19] + "Z"
                _ACTIVE_Q_FILE.write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            _log.warning("answer_active error: %s", e)

    # Generate next question
    return _generate_and_save_question()


def _apply_answer_to_context(question: dict, answer: str) -> None:
    """
    If the question was about priorities or job categories, apply changes
    to directives.md or job filters based on the answer.
    Pure script: keyword matching only, no LLM.
    """
    q_type = question.get("type", "")
    q_text = question.get("question", "").lower()
    ans_lower = answer.lower()

    # Priority/research directive changes
    if q_type in ("priority", "directive") or any(
        kw in q_text for kw in ("prioridad", "investigar", "foco", "strategy", "priorit")
    ):
        try:
            from services.directives import extract_and_save
            extract_and_save(answer, "")
        except Exception as e:
            _log.warning("directive apply failed: %s", e)

    # Job filter changes
    if q_type == "jobs" or any(kw in q_text for kw in ("empleo", "trabajo", "job", "rol", "puesto")):
        _apply_job_filter_from_answer(answer)


def _apply_job_filter_from_answer(answer: str) -> None:
    """
    Parse an answer about job preferences and update job filter config.
    Script-based: looks for patterns like "no me interesa X", "solo quiero Y".
    """
    _JOB_FILTERS_FILE = BASE_DIR / "data" / "job_filters.json"

    try:
        filters = json.loads(_JOB_FILTERS_FILE.read_text(encoding="utf-8")) if _JOB_FILTERS_FILE.exists() else {
            "excluded_roles": [],
            "preferred_locations": [],
            "excluded_industries": [],
            "notes": "",
        }
    except Exception:
        filters = {"excluded_roles": [], "preferred_locations": [], "excluded_industries": [], "notes": ""}

    ans_lower = answer.lower()

    # Detect exclusions: "no me interesan los empleos de X", "excluye Y"
    _excl_patterns = [
        re.compile(r"(?:no me interesa|no quiero|excluye?|elimina?)\s+(?:los?\s+)?(?:empleos?\s+(?:de\s+)?)?(.{4,40})", re.I),
        re.compile(r"(?:ya no busques?|para de buscar)\s+(?:empleos?\s+(?:de\s+)?)?(.{4,40})", re.I),
    ]
    for pat in _excl_patterns:
        m = pat.search(answer)
        if m:
            excl = m.group(1).strip().rstrip(".,;")
            if excl and excl not in filters["excluded_roles"]:
                filters["excluded_roles"].append(excl)

    # Save updated filters
    try:
        _JOB_FILTERS_FILE.write_text(json.dumps(filters, ensure_ascii=False, indent=2), encoding="utf-8")
        _log.info("job_filters_updated: %s", filters["excluded_roles"])
    except Exception as e:
        _log.warning("job_filters_save failed: %s", e)


# ── Question generator ────────────────────────────────────────────────────────

def _generate_and_save_question() -> dict:
    """
    Generate the next strategic question using:
    - Document scan of /Volumes/ZLab_Documents/Zlab_Document/Documents
    - Recent dossiers (research context)
    - Previous Q&A (avoid repetition)
    - Current directives
    """
    # Load context
    doc_summary = scan_documents_folder()

    kb_text = ""
    for _kb_path in [
        Path.home() / "Documents" / "knowledge_base_backup.md",
        BASE_DIR / "data" / "knowledge_base.md",
    ]:
        try:
            _c = _kb_path.read_text(encoding="utf-8", errors="ignore")
            if len(_c) > len(kb_text):
                kb_text = _c
        except Exception:
            pass
    kb_text = kb_text[:2000]

    directives = ""
    try:
        _dp = BASE_DIR / "data" / "directives.md"
        if _dp.exists():
            directives = _dp.read_text(encoding="utf-8")[:800]
    except Exception:
        pass

    # Load recent dossiers (top 6 by fit score)
    dossier_dir = BASE_DIR / "data" / "dossiers"
    dossier_lines: list[str] = []
    try:
        dossiers = []
        for df in sorted(dossier_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:12]:
            try:
                d = json.loads(df.read_text(encoding="utf-8"))
                fit = (d.get("fit_assessment") or {}).get("fit_score", 0) or 0
                dossiers.append((fit, d))
            except Exception:
                pass
        dossiers.sort(key=lambda x: -x[0])
        for fit, d in dossiers[:6]:
            na = (d.get("next_actions") or [])[:1]
            dossier_lines.append(
                f"- {d.get('name','?')} ({d.get('type','?')}, fit={fit}): "
                f"{(na[0] if na else d.get('description',''))[:120]}"
            )
    except Exception:
        pass
    dossier_text = "\n".join(dossier_lines) if dossier_lines else "Sin dossiers aún."

    # Avoid repeating recent questions
    recent_qa = load_profile_qa(10)
    answered_q = "\n".join(f"- {e['question'][:80]}" for e in recent_qa) if recent_qa else "Ninguna."

    # Question type rotation: cycle through types to keep variety
    _last_type = "strategic"
    try:
        if _ACTIVE_Q_FILE.exists():
            _last = json.loads(_ACTIVE_Q_FILE.read_text(encoding="utf-8"))
            _last_type = _last.get("type", "strategic")
    except Exception:
        pass
    _type_sequence = ["strategic", "priority", "profile", "jobs", "strategic", "strategic"]
    try:
        _next_idx = (_type_sequence.index(_last_type) + 1) % len(_type_sequence)
    except ValueError:
        _next_idx = 0
    next_type = _type_sequence[_next_idx]

    type_instruction = {
        "strategic": "Una pregunta de DECISIÓN ESTRATÉGICA urgente sobre sus oportunidades de inversión, grants o partnerships investigados. Referencia nombres específicos de empresas/fondos/grants del dossier. Debe ayudar a David a tomar una decisión esta semana.",
        "priority":  "Una pregunta para que David AJUSTE LAS PRIORIDADES de investigación de Phi. ¿Qué tipos de oportunidades deben subir/bajar de prioridad? ¿Hay sectores que ya no le interesan?",
        "profile":   "Una pregunta sobre CONTEXTO PROFESIONAL de David que Phi no puede inferir de sus documentos. Algo que cambia la forma en que Phi analiza las oportunidades (estado de una negociación, capacidad de equipo, decisión reciente).",
        "jobs":      "Una pregunta sobre PREFERENCIAS DE EMPLEO: tipos de roles que sí/no le interesan, ubicaciones, industrias, condiciones que no acepta. La respuesta actualizará el filtro de búsqueda de empleos de Phi.",
    }.get(next_type, "Una pregunta estratégica relevante.")

    prompt = f"""Eres Phi, el agente profesional de David Lagarejo. Genera UNA sola pregunta estratégica basada en el contexto real de David.

PERFIL DE DAVID:
{kb_text or "CEO/Founder, Zircular (IIoT sensor), ZION ING (consultoría), NYC, pre-seed."}

DIRECTIVAS ACTUALES:
{directives or "Sin directivas específicas."}

DOCUMENTOS ENCONTRADOS EN SU COMPUTADOR:
{doc_summary[:1500] if doc_summary else "No se pudo acceder a los documentos."}

OPORTUNIDADES INVESTIGADAS (dossiers con mayor fit):
{dossier_text}

PREGUNTAS YA RESPONDIDAS (no repetir):
{answered_q}

TIPO DE PREGUNTA REQUERIDA: {next_type.upper()}
INSTRUCCIÓN: {type_instruction}

REGLAS:
- La pregunta debe ser ESPECÍFICA: referencia datos reales de los documentos o dossiers
- Máximo 2 oraciones. Directa. Sin preámbulo.
- Incluye CONTEXTO: en 1 oración, por qué haces esta pregunta ahora
- Si el tipo es "priority" o "jobs", la pregunta debe llevar a David a darte instrucciones concretas sobre qué hacer diferente

Devuelve JSON únicamente:
{{"question": "...", "context": "...", "type": "{next_type}", "urgency": "high|medium|low", "related_entities": ["nombre1"]}}"""

    raw = call_phi_sync([
        {"role": "system", "content": "Output JSON only. No markdown. No explanation."},
        {"role": "user", "content": prompt},
    ], num_ctx=5000)

    # Parse response
    question_data: dict = {}
    try:
        s = raw.find("{")
        e = raw.rfind("}") + 1
        if s >= 0 and e > s:
            question_data = json.loads(raw[s:e])
    except Exception as exc:
        _log.warning("question_parse_failed: %s", exc)

    # Fallback if parse failed
    if not question_data.get("question"):
        question_data = {
            "question": "¿Cuál es la próxima acción más importante para avanzar con Zircular esta semana?",
            "context": "Phi necesita actualizar su modelo de prioridades.",
            "type": "strategic",
            "urgency": "high",
            "related_entities": [],
        }

    # Build full question object
    q_id = "q_" + hashlib.md5((question_data["question"] + str(time.time())).encode()).hexdigest()[:10]
    active_q = {
        "question_id":    q_id,
        "question":       question_data.get("question", ""),
        "context":        question_data.get("context", ""),
        "type":           question_data.get("type", next_type),
        "urgency":        question_data.get("urgency", "medium"),
        "related_entities": question_data.get("related_entities", []),
        "generated_at":   datetime.now(timezone.utc).isoformat()[:19] + "Z",
        "generated_at_ts": time.time(),
        "answered":       False,
        "answer":         None,
        "answered_at":    None,
    }

    try:
        _ACTIVE_Q_FILE.write_text(json.dumps(active_q, ensure_ascii=False, indent=2), encoding="utf-8")
        _log.info("active_question_saved type=%s urgency=%s", active_q["type"], active_q["urgency"])
    except Exception as e:
        _log.warning("active_question_save_failed: %s", e)

    return active_q
