"""
memories/emotion_classifier.py
Helpers for extracting emotion and directive fields from the interpreter LLM response.

No extra API call is made. The interpreter prompt outputs:
    "emotion":          str  — categorical label (frustration, excitement, neutral, etc.)
    "emotion_strength": str  — "low" | "medium" | "high"
    "directive":        bool — True if input specifies a rule/preference to remember permanently
"""
from __future__ import annotations

from core.helpers import log_it

_ENTITY = "emotion_classifier"

# Long-term keyword triggers (structural, checked in tandem with directive)
_PERMANENT_KEYWORDS = {"always", "never", "permanently", "from now on", "every time", "prefer"}
_LT_EXPLICIT_KEYWORDS = {"remember forever", "remember this permanently", "never forget", "remember that", "remember"}


def _normalize_emotion_strength(val: any) -> str:
    """Normalize raw emotion strength to 'low', 'medium', or 'high'."""
    if isinstance(val, str):
        val_lower = val.strip().lower()
        if val_lower in ("low", "medium", "high"):
            return val_lower
        if val_lower in ("none", "neutral"):
            return "low"
    elif isinstance(val, (int, float)):
        if val >= 0.75:
            return "high"
        if val >= 0.35:
            return "medium"
        return "low"
    return "low"


def parse_classifier_fields(
    response_dict: dict,
) -> tuple[str, str, bool]:
    """
    Safely extract (emotion, emotion_strength, directive) from an
    interpreter response dict, returning safe defaults if fields are absent.
    """
    emotion = str(response_dict.get("emotion", "neutral")).strip().lower()
    raw_strength = response_dict.get("emotion_strength", "low")
    emotion_strength = _normalize_emotion_strength(raw_strength)

    # Directive boolean extraction (supports legacy intent_strength fallback)
    directive = False
    if "directive" in response_dict:
        directive = bool(response_dict.get("directive"))
    elif "intent_strength" in response_dict:
        try:
            directive = float(response_dict["intent_strength"]) > 0.85
        except (TypeError, ValueError):
            directive = False

    log_it(
        f"Classifier: emotion={emotion!r} strength={emotion_strength} directive={directive}",
        _ENTITY,
    )
    return emotion, emotion_strength, directive


def should_promote_short_term(emotion_strength: str | float, directive: bool | float = False) -> bool:
    """
    Return True if an entry qualifies for short-term promotion based on
    emotion_strength == 'high' or directive == True.
    """
    is_high = (
        emotion_strength == "high"
        or (isinstance(emotion_strength, (int, float)) and emotion_strength >= 0.75)
    )
    is_dir = (
        bool(directive)
        or (isinstance(directive, (int, float)) and directive > 0.85)
    )

    result = is_high or is_dir
    if result:
        log_it(
            f"Short-term promotion triggered (emo_high={is_high}, directive={is_dir}).",
            _ENTITY,
        )
    return result


def should_promote_long_term(directive: bool | float, content: str) -> tuple[bool, str]:
    """
    Return (promote, ltype) where ltype is 'behaviour' or 'fact'.
    Promotion fires when directive is True AND content contains persistent keywords.
    """
    is_dir = (
        bool(directive)
        or (isinstance(directive, (int, float)) and directive > 0.85)
    )
    if not is_dir:
        return False, ""

    content_lower = content.lower()

    if any(kw in content_lower for kw in _PERMANENT_KEYWORDS):
        log_it(
            "Long-term (behaviour) promotion triggered via directive + keyword.",
            _ENTITY,
        )
        return True, "behaviour"

    if any(kw in content_lower for kw in _LT_EXPLICIT_KEYWORDS):
        log_it(
            "Long-term (fact) promotion triggered via directive + keyword.",
            _ENTITY,
        )
        return True, "fact"

    return False, ""
