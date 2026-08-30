"""
memories/emotion_classifier.py
Helpers for extracting emotion / intent fields from the interpreter LLM response.

No extra API call is made.  The interpreter prompt is extended (see
prompts/prompt_templates.py) to include three new fields in its JSON:

    "emotion":          str   — categorical label (frustration, excitement, …)
    "emotion_strength": float — 0–1 intensity of that emotion
    "intent_strength":  float — 0–1 strength of an explicit memory directive

This module only parses those fields and applies the promotion thresholds.
"""

from core.config import cfg
from core.helpers import log_it

_ENTITY = "emotion_classifier"

# Long-term keyword triggers (not tuneable at runtime — structural, not thresholds)
_PERMANENT_KEYWORDS = {"always", "never", "permanently", "from now on", "every time"}
_LT_EXPLICIT_KEYWORDS = {"remember forever", "remember this permanently", "never forget"}


def _emotion_threshold() -> float:
    return cfg.memory.get("emotion_strength_threshold", 0.80)

def _intent_threshold() -> float:
    return cfg.memory.get("intent_strength_threshold", 0.85)

def _lt_intent_threshold() -> float:
    return cfg.memory.get("lt_intent_threshold", 0.90)


def parse_classifier_fields(
    response_dict: dict,
) -> tuple[str, float, float]:
    """
    Safely extract (emotion, emotion_strength, intent_strength) from an
    interpreter response dict, returning safe defaults if fields are absent.
    """
    emotion = response_dict.get("emotion", "neutral")
    try:
        emotion_strength = float(response_dict.get("emotion_strength", 0.0))
    except (TypeError, ValueError):
        emotion_strength = 0.0
    try:
        intent_strength = float(response_dict.get("intent_strength", 0.0))
    except (TypeError, ValueError):
        intent_strength = 0.0

    log_it(
        f"Classifier: emotion={emotion!r} strength={emotion_strength:.2f} "
        f"intent={intent_strength:.2f}",
        _ENTITY,
    )
    return emotion, emotion_strength, intent_strength


def should_promote_short_term(emotion_strength: float, intent_strength: float) -> bool:
    """
    Return True if an entry qualifies for short-term promotion based on the
    classifier scores alone (triggers 1 & 2 from FUTURE.md).

    Trigger 3 (repetition) and trigger 4 (tool failure) are checked by the
    short-term worker itself using working-memory embeddings and failure flags.
    """
    result = emotion_strength > _emotion_threshold() or intent_strength > _intent_threshold()
    if result:
        log_it(
            f"Short-term promotion triggered (emo={emotion_strength:.2f}, "
            f"intent={intent_strength:.2f}, thresholds: emo>{_emotion_threshold()}, intent>{_intent_threshold()}).",
            _ENTITY,
        )
    return result


def should_promote_long_term(intent_strength: float, content: str) -> tuple[bool, str]:
    """
    Return (promote, ltype) where ltype is 'behaviour' or 'fact'.

    Promotion fires when intent_strength > 0.90 AND the content contains
    keywords signalling a permanent-behaviour or explicit-long-term request.
    """
    if intent_strength <= _lt_intent_threshold():
        return False, ""

    content_lower = content.lower()

    if any(kw in content_lower for kw in _PERMANENT_KEYWORDS):
        log_it(
            f"Long-term (behaviour) promotion triggered (intent={intent_strength:.2f}).",
            _ENTITY,
        )
        return True, "behaviour"

    if any(kw in content_lower for kw in _LT_EXPLICIT_KEYWORDS):
        log_it(
            f"Long-term (fact) promotion triggered (intent={intent_strength:.2f}).",
            _ENTITY,
        )
        return True, "fact"

    return False, ""
