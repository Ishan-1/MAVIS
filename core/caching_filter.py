"""
core/caching_filter.py
Fast deterministic structural pre-filter for MAVIS Semantic Pipeline Cache.

Inverted structural design:
- Bypasses any bare pronouns (it, this, that, them, these, those, him, her)
- Bypasses continuation openers (also, now, then, same, again, keep, continue, another, and, still, so)
- Bypasses comparative/anaphoric determiners (the same, the other, the second, the previous, the next, etc.)
- Bypasses conversational feedback, rejections, agreements, and transitions
- Bypasses discourse pointers and relative temporal references
- Bypasses ultra-short or elliptic directives
- Preserves self-contained queries and inline payloads (e.g., "Explain this code: def foo()...")
"""
from __future__ import annotations

import re

# Workspace prefix pattern injected by benchmark / isolated workspace harnesses
_WORKSPACE_PREFIX_RE = re.compile(r"^\[Workspace:\s*[^\]]+\]\s*", re.IGNORECASE)

# Continuation openers when starting a query
_CONTINUATION_OPENERS_RE = re.compile(
    r"^(also|now|then|same|again|keep|continue|another|and|still|so|furthermore|moreover)\b",
    re.IGNORECASE,
)

# Bare pronouns & demonstratives that refer back to earlier conversation entities
_ANAPHORIC_PRONOUNS_RE = re.compile(
    r"\b(it|this|that|them|these|those|him|her)\b",
    re.IGNORECASE,
)

# Comparative or anaphoric phrases
_COMPARATIVE_PHRASES_RE = re.compile(
    r"\b(the\s+same|the\s+other|the\s+second|the\s+third|the\s+previous|the\s+next|the\s+above|the\s+former|the\s+latter|another\s+one|same\s+as|instead|as\s+well)\b",
    re.IGNORECASE,
)

# Conversational feedback, rejections, agreements, and transitions
_FEEDBACK_TRANSITIONS_RE = re.compile(
    r"^(no|nope|not\s+quite|wrong|incorrect)\b|"
    r"^(that|it|this)\s+(is|was|looks|seems|feels)?\s*(a\s+)?(not\s+good|wrong|incorrect|bad|broken|terrible|awesome|great|cool|fine|ok|okay|good|solid|start)\b|"
    r"^that('s|s|\s+is)\s+(a\s+)?(not|wrong|incorrect|invalid|broken|good|great|fine|a\s+bit|solid)\b|"
    r"^(that|it)\s+(worked|failed|passed|didn't\s+work|did\s+not\s+work)\b|"
    r"^(sounds|looks)\s+like\s+a\s+(good|solid|great)\s+plan\b|"
    r"^(thanks|thank\s+you)\b|"
    r"^(ok|okay)\b[\s,.]+(then|now|please|go\s+ahead|do\s+it|'|\"|what|how|why|can)?",
    re.IGNORECASE,
)

# Discourse pointers & relative temporal references
_DISCOURSE_RE = re.compile(
    r"\b(previous|prior|earlier|last|above|mentioned)\s+(turn|step|output|error|response|message|command|result|file|version|code)\b|"
    r"\bwhat\s+(did\s+)?you\s+(just\s+)?(say|do|write|suggest|show|create|said|did|wrote|suggested|showed|created)\b|"
    r"\b(still|now)\s+(failing|broken|passing|working)\b",
    re.IGNORECASE,
)

# Short conversational continuations and flow commands
_ELLIPTIC_COMMANDS = {
    "why", "why?", "how", "how?", "what", "what?", "where", "where?",
    "and?", "and then?", "what next?", "what next",
    "continue", "proceed", "go ahead", "next", "more", "details",
    "expand", "retry", "try again", "once more", "one more time", "again",
    "undo", "undo that", "revert", "revert that", "stop", "cancel",
    "yes", "yeah", "yep", "sure", "no", "nope",
}


def is_context_dependent_query(query: str) -> tuple[bool, str]:
    """
    Evaluate whether a query depends on previous conversational context.

    Returns:
        (is_dependent: bool, reason: str)
        - True if the query references prior turns, demonstratives, feedback, or ellipsis.
        - False if the query is self-contained and independently executable.
    """
    if not query:
        return True, "empty_query"

    cleaned = _WORKSPACE_PREFIX_RE.sub("", query).strip()
    if not cleaned:
        return True, "empty_query"

    lower = cleaned.lower()
    if lower in _ELLIPTIC_COMMANDS:
        return True, "elliptic_continuation"

    words = cleaned.split()
    if len(words) <= 1 and not (":" in cleaned):
        return True, "ultra_short"

    # Inline payload exception (e.g. "Explain this code: def foo()...")
    has_inline_payload = False
    prefix = ""
    if ":" in cleaned:
        prefix, suffix = cleaned.split(":", 1)
        if len(suffix.strip()) > 8:
            has_inline_payload = True

    if _CONTINUATION_OPENERS_RE.search(cleaned):
        return True, "continuation_opener"

    if _COMPARATIVE_PHRASES_RE.search(cleaned):
        return True, "comparative_phrase"

    if _FEEDBACK_TRANSITIONS_RE.search(cleaned):
        return True, "feedback_transition"

    if _DISCOURSE_RE.search(cleaned):
        return True, "discourse_reference"

    m = _ANAPHORIC_PRONOUNS_RE.search(cleaned)
    if m:
        if has_inline_payload:
            prefix_lower = prefix.lower()
            if any(w in prefix_lower for w in ("code", "error", "file", "sentence", "function", "input", "command", "text", "query", "script", "url")):
                pass
            else:
                return True, f"anaphoric_pronoun:{m.group(0)}"
        else:
            return True, f"anaphoric_pronoun:{m.group(0)}"

    return False, "standalone"


def is_cacheable_query(query: str) -> tuple[bool, str]:
    """
    Determine if a query is eligible for pipeline caching.

    Returns:
        (is_cacheable: bool, reason: str)
    """
    if not query or not query.strip():
        return False, "empty_query"

    is_dep, reason = is_context_dependent_query(query)
    if is_dep:
        return False, f"context_dependent:{reason}"

    return True, "eligible"
