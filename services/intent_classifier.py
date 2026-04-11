"""
services/intent_classifier.py — Script-based intent classifier for chat messages.

Pure Python — NO LLM calls, NO prompts. Pattern matching only.
Safety rationale: LLM-based classifiers can be fooled by prompt injection in
research results. This script runs deterministically on user messages only.

Detects three intents:
  INVESTIGATE_NOW  — "investiga X", "busca X", "quiero saber de X"
  CANCEL_QUEUE     — "cancela el research de X", "para de investigar X"
  DIRECTIVE_CHANGE — "enfócate en X", "ya no me interesa Y", "prioriza Z"
  NONE             — everything else

Returns an Intent dataclass with entity/topic extracted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# ── Intent types ──────────────────────────────────────────────────────────────

IntentType = Literal["INVESTIGATE_NOW", "CANCEL_QUEUE", "DIRECTIVE_CHANGE", "MAIL_CHECK", "NONE"]


@dataclass
class Intent:
    type: IntentType
    entity: str = ""           # primary entity name extracted
    topic: str = ""            # topic/area extracted
    raw: str = ""              # original message
    confidence: float = 0.0   # 0.0–1.0

    @property
    def actionable(self) -> bool:
        return self.type != "NONE" and (bool(self.entity) or bool(self.topic))


# ── Normalization ─────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation at ends."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# ── INVESTIGATE_NOW patterns ──────────────────────────────────────────────────
# Triggers when user asks Phi to research a specific entity NOW.

_INVESTIGATE_VERBS = [
    "investiga", "investigate", "investiga a", "investiga la", "investiga el",
    "busca información", "busca info", "busca más sobre", "busca sobre",
    "quiero saber de", "quiero saber sobre", "necesito saber de",
    "dime todo sobre", "dime sobre", "hazme un dossier de",
    "research", "look up", "find out about", "dig into",
    "quiero que investigues", "necesito un dossier",
    "analiza a", "analiza la", "analiza el", "analiza las",
    "estudia a", "estudia la", "estudia el",
]

# After the verb, capture the entity name (up to 60 chars, stopping at comma/period/question)
_INVESTIGATE_RE = re.compile(
    r"(?:"
    + "|".join(re.escape(v) for v in sorted(_INVESTIGATE_VERBS, key=len, reverse=True))
    + r")\s+([^\.,\?!]{3,60})",
    re.IGNORECASE,
)


def _extract_investigate(text: str) -> tuple[str, float]:
    """Return (entity_name, confidence) or ('', 0.0)."""
    m = _INVESTIGATE_RE.search(text)
    if m:
        raw_entity = m.group(1).strip().rstrip(".,;:!?")
        # Strip leading articles and prepositions
        raw_entity = re.sub(r"^(a |an |the |la |el |los |las |un |una |sobre |acerca de |de )", "", raw_entity, flags=re.I)
        if len(raw_entity) >= 3:
            return raw_entity, 0.9
    return "", 0.0


# ── CANCEL_QUEUE patterns ─────────────────────────────────────────────────────
# Triggers when user wants to stop investigating something.

_CANCEL_PATTERNS = [
    (re.compile(r"(?:cancela|cancel|para|stop|deja de|detén|detente de|no sigas)\s+(?:el\s+)?(?:research|investigación|búsqueda|buscar)\s+(?:de\s+|sobre\s+|en\s+)?(.{3,60})", re.I), 0.95),
    (re.compile(r"(?:elimina|borra|quita|remove)\s+(?:las?\s+)?(?:tareas?|tasks?)\s+(?:de\s+)?(.{3,60})", re.I), 0.9),
    (re.compile(r"(?:ya no investigues|ya no busques|no investigues más|no busques más)\s+(?:sobre\s+|a\s+|la\s+|el\s+)?(.{3,60})", re.I), 0.95),
    (re.compile(r"(?:pausa|pause)\s+(?:todo\s+)?(?:lo de|el research de|las tareas de)\s+(.{3,60})", re.I), 0.85),
]


def _extract_cancel(text: str) -> tuple[str, float]:
    """Return (topic_to_cancel, confidence) or ('', 0.0)."""
    for pattern, conf in _CANCEL_PATTERNS:
        m = pattern.search(text)
        if m:
            topic = m.group(1).strip().rstrip(".,;:!?")
            if len(topic) >= 3:
                return topic, conf
    return "", 0.0


# ── DIRECTIVE_CHANGE patterns ─────────────────────────────────────────────────
# Triggers when user changes overall strategy/focus.

_DIRECTIVE_SIGNALS_STRONG = [
    r"(?:cambia|change)\s+(?:el\s+)?(?:foco|focus|estrategia|strategy|rumbo|dirección)\s+(?:a|to|hacia)?\s*(.{5,80})",
    r"(?:enfócate|enfocar|enfoca)\s+(?:en|en la|en el|on)\s+(.{5,80})",
    r"(?:prioriza|prioritize|priorizar)\s+(.{5,80})",
    r"(?:ya no me interesa|no me interesa|no me interesan)\s+(.{5,80})",
    r"(?:nuevo objetivo|new goal|nueva estrategia|nueva meta)\s*[:—]?\s*(.{5,80})",
    r"(?:a partir de ahora|from now on|de ahora en adelante)\s+(?:quiero que)?\s*(.{5,80})",
    r"(?:pivota|pivot)\s+(?:a|hacia|to)\s+(.{5,80})",
]

_DIRECTIVE_RE_LIST = [(re.compile(p, re.I), 0.9) for p in _DIRECTIVE_SIGNALS_STRONG]

# Softer signals — lower confidence, still directive
_DIRECTIVE_SOFT_KEYWORDS = [
    "ya no", "deja de investigar", "para de buscar", "olvida el",
    "quiero enfocarme en", "quiero concentrarme en", "cambia el rumbo",
    "no quiero más", "no me hagas más research de",
]


def _extract_directive(text: str) -> tuple[str, float]:
    """Return (topic_description, confidence) or ('', 0.0)."""
    for pattern, conf in _DIRECTIVE_RE_LIST:
        m = pattern.search(text)
        if m:
            topic = m.group(1).strip().rstrip(".,;:!?")
            if len(topic) >= 5:
                return topic, conf

    norm = _normalize(text)
    for kw in _DIRECTIVE_SOFT_KEYWORDS:
        if kw in norm:
            # Extract what comes after the keyword
            idx = norm.find(kw)
            after = text[idx + len(kw):].strip()[:80].rstrip(".,;:!?")
            if len(after) >= 5:
                return after, 0.7
    return "", 0.0


# ── MAIL_CHECK patterns ───────────────────────────────────────────────────────
# Triggers when user wants to scan inbox for relevant emails.

_MAIL_CHECK_RE = re.compile(
    r"(?:revisa|escanea|chequea|busca en|mira|check|scan|look at|any|hay algo en)\s+"
    r"(?:mi\s+)?(?:correo|correos|email|emails|inbox|bandeja|buzón|mail|mensajes)"
    r"|(?:qué hay en mi correo|qué hay en mi inbox|nuevos correos|nuevos emails"
    r"|correos nuevos|emails nuevos|mail new|new emails?|new mail"
    r"|algún correo|algún email|tienes correos|tienes emails"
    r"|correos? relevantes|emails? relevantes|correo de inversores"
    r"|correo de grants|email.*investor|email.*grant|email.*job)",
    re.I,
)


def _is_mail_check(text: str) -> bool:
    return bool(_MAIL_CHECK_RE.search(text))


# ── Minimum message length guard ─────────────────────────────────────────────
# Very short messages are unlikely to contain actionable intents.
_MIN_LEN = 8


# ── Public API ────────────────────────────────────────────────────────────────

def classify(message: str) -> Intent:
    """
    Classify a user message into an actionable intent.
    Pure script — no I/O, no LLM, deterministic.

    Priority: CANCEL_QUEUE > MAIL_CHECK > INVESTIGATE_NOW > DIRECTIVE_CHANGE > NONE
    """
    if not message or len(message.strip()) < _MIN_LEN:
        return Intent(type="NONE", raw=message)

    text = message.strip()

    # 1. Cancel first
    cancel_topic, cancel_conf = _extract_cancel(text)
    if cancel_conf >= 0.8:
        return Intent(type="CANCEL_QUEUE", topic=cancel_topic, raw=text, confidence=cancel_conf)

    # 2. Mail check
    if _is_mail_check(text):
        return Intent(type="MAIL_CHECK", raw=text, confidence=0.95)

    # 3. Investigate NOW
    entity, inv_conf = _extract_investigate(text)
    if inv_conf >= 0.85:
        return Intent(type="INVESTIGATE_NOW", entity=entity, raw=text, confidence=inv_conf)

    # 4. Directive change
    directive_topic, dir_conf = _extract_directive(text)
    if dir_conf >= 0.7:
        return Intent(type="DIRECTIVE_CHANGE", topic=directive_topic, raw=text, confidence=dir_conf)

    return Intent(type="NONE", raw=text)


def classify_batch(messages: list[str]) -> list[Intent]:
    """Classify multiple messages. Returns list of Intents."""
    return [classify(m) for m in messages]
