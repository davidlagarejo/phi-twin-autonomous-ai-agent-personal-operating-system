"""
services/translator.py — Context-aware English→Spanish translator.

Strategy (in order of preference):
  1. argostranslate — local OPUS-MT model, no API key, context-aware within sentence
  2. Phi fallback   — uses the local Ollama model with a minimal translation prompt
     (slower but zero extra dependencies)

Usage:
    from services.translator import translate_to_spanish, is_english

    spanish = translate_to_spanish("The company raised $5M in pre-seed funding.")
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

_log = logging.getLogger("phi.translator")

# ── State ──────────────────────────────────────────────────────────────────────
_argos_ready      = False   # True once OPUS-MT en→es package is loaded
_argos_lock       = threading.Lock()
_argos_checked    = False   # avoid re-checking on every call
_argos_translation = None  # cached argostranslate Translation object (thread-safe reuse)

# Download flag file — written after first successful model install
_FLAG = Path(__file__).parent.parent / "data" / ".argos_en_es_ready"


# ── Language detection ────────────────────────────────────────────────────────

def is_english(text: str) -> bool:
    """
    Return True if the text is primarily English.
    Falls back to True (translate anyway) if langdetect is unavailable.
    """
    if not text or len(text.strip()) < 20:
        return False
    try:
        from langdetect import detect, LangDetectException
        lang = detect(text)
        return lang == "en"
    except Exception:
        # If langdetect not installed or fails, assume English for Phi responses
        return True


# ── argostranslate backend ────────────────────────────────────────────────────

def _ensure_argos() -> bool:
    """Return True if argostranslate en→es is ready to use."""
    global _argos_ready, _argos_checked, _argos_translation
    if _argos_ready:
        return True
    if _argos_checked:
        return False

    with _argos_lock:
        if _argos_ready:
            return True
        _argos_checked = True
        try:
            import argostranslate.package
            import argostranslate.translate

            # Check if en→es package is already installed
            installed = argostranslate.translate.get_installed_languages()
            en_lang = next((l for l in installed if l.code == "en"), None)
            if en_lang:
                es_lang = next((l for l in installed if l.code == "es"), None)
                if es_lang:
                    translation = en_lang.get_translation(es_lang)
                    if translation:
                        _argos_translation = translation  # cache object for reuse
                        _argos_ready = True
                        _log.info("argostranslate en→es ready (already installed)")
                        return True

            # Try to install if flag file exists (previous successful install)
            if _FLAG.exists():
                _argos_ready = False  # flag exists but package not found — stale
                return False

            # Download and install en→es package (runs once, ~150 MB)
            _log.info("argostranslate: downloading en→es package...")
            argostranslate.package.update_package_index()
            available = argostranslate.package.get_available_packages()
            pkg = next(
                (p for p in available if p.from_code == "en" and p.to_code == "es"),
                None,
            )
            if not pkg:
                _log.warning("argostranslate: en→es package not found in index")
                return False

            argostranslate.package.install_from_path(pkg.download())
            _FLAG.parent.mkdir(parents=True, exist_ok=True)
            _FLAG.touch()
            # Re-load and cache translation object
            installed = argostranslate.translate.get_installed_languages()
            en_lang = next((l for l in installed if l.code == "en"), None)
            es_lang = next((l for l in installed if l.code == "es"), None)
            if en_lang and es_lang:
                _argos_translation = en_lang.get_translation(es_lang)
            _argos_ready = True
            _log.info("argostranslate en→es installed successfully")
            return True

        except Exception as e:
            _log.warning("argostranslate unavailable: %s", e)
            return False


def _translate_argos(text: str) -> str:
    """Translate English text to Spanish using the cached argostranslate object."""
    translation = _argos_translation
    if not translation:
        raise RuntimeError("argostranslate translation object not cached")

    # Translate paragraph by paragraph to preserve markdown structure
    paragraphs = text.split("\n")
    translated = []
    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            translated.append(para)
            continue
        # Keep markdown headers, bullets, code blocks intact
        prefix_match = re.match(r"^(#{1,4}\s+|\-\s+|\*\s+|\d+\.\s+|>\s*|```.*)", para)
        if prefix_match:
            prefix = prefix_match.group(0)
            rest = para[len(prefix):]
            translated.append(prefix + (translation.translate(rest) if rest.strip() else ""))
        else:
            translated.append(translation.translate(para))
    return "\n".join(translated)


# ── Phi fallback backend ──────────────────────────────────────────────────────

def _translate_phi(text: str) -> str:
    """
    Translate using the local Phi model — context-aware but slower.
    Only used when argostranslate is not available.
    """
    from llm.client import call_phi as _call_phi
    import asyncio

    prompt = (
        "Translate the following text from English to natural, professional Spanish. "
        "Preserve all markdown formatting, emojis, bullet points, and proper nouns exactly. "
        "Output ONLY the translated text — no explanations, no prefix, no suffix.\n\n"
        f"TEXT:\n{text}"
    )
    msgs = [
        {"role": "system", "content": "You are a professional English-to-Spanish translator."},
        {"role": "user",   "content": prompt},
    ]
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an async context — run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(asyncio.run, _call_phi(msgs, num_ctx=2048, temperature=0.05))
                return future.result(timeout=45)
        else:
            return loop.run_until_complete(_call_phi(msgs, num_ctx=2048, temperature=0.05))
    except Exception as e:
        _log.warning("phi_translate failed: %s — returning original", e)
        return text


# ── Public API ────────────────────────────────────────────────────────────────

def translate_to_spanish(text: str) -> str:
    """
    Translate English text to Spanish, preserving markdown and context.
    Returns the original text unchanged if translation fails or is not needed.
    """
    if not text or not text.strip():
        return text

    if not is_english(text):
        return text  # already Spanish or mixed — don't touch

    # Try argostranslate first (fast, local)
    if _ensure_argos():
        try:
            result = _translate_argos(text)
            if result and len(result) > 5:
                return result
        except Exception as e:
            _log.warning("argos_translate failed: %s — falling back to phi", e)

    # Fallback: use Phi model (slower but always available)
    return _translate_phi(text)


def warm_up() -> None:
    """
    Call this at server startup (in background thread) to pre-load the
    translation model. Avoids cold-start latency on first chat message.
    """
    def _bg():
        _log.info("translator warm-up starting...")
        _ensure_argos()
        _log.info("translator warm-up complete (argos_ready=%s)", _argos_ready)

    t = threading.Thread(target=_bg, daemon=True, name="translator-warmup")
    t.start()
