"""
core/caching_filter.py
Fast deterministic linguistic pre-filter for MAVIS Semantic Pipeline Cache.

Prevents context-dependent queries (such as "Fix this", "No, that is not good",
"Run it again", conversational feedback, anaphoric references, and elliptic follow-ups)
from being cached or checked against the semantic cache.
"""
from __future__ import annotations

import re

# Workspace prefix pattern injected by benchmark / isolated workspace harnesses
_WORKSPACE_PREFIX_RE = re.compile(r"^\[Workspace:\s*[^\]]+\]\s*", re.IGNORECASE)

# Short conversational continuations and flow commands (< 4 words)
_ELLIPTIC_COMMANDS = {
    "why", "why?", "how", "how?", "what", "what?", "where", "where?",
    "and?", "and then?", "what next?", "what next",
    "continue", "proceed", "go ahead", "next", "more", "details",
    "expand", "retry", "try again", "once more", "one more time", "again",
    "undo", "undo that", "revert", "revert that", "stop", "cancel",
    "yes", "yeah", "yep", "sure", "ok", "okay", "no", "nope",
}

# Regex patterns matching anaphora, demonstratives, and conversational feedback
_ANAPHORA_PATTERNS = [
    # Verbs directly acting on bare demonstratives / pronouns (e.g., "Fix this", "Explain that", "Show it")
    re.compile(
        r"^(can you\s+|please\s+)?(fix|change|update|modify|edit|correct|patch|repair|redo|re-run|rerun|retry|run|test|execute|try|explain|clarify|elaborate on|expand on|show|display|undo|revert)\s+(this|that|it|these|those)(\s+again)?\b",
        re.IGNORECASE,
    ),
    # Demonstratives pointing to discourse entities without explicit payload (e.g. "this error", "that bug")
    re.compile(
        r"\b(this|that|these|those)\s+(error|bug|issue|exception|failure|crash|traceback|warning|typo|problem|mistake|output|response|result|draft|reply|plan|suggestion|step|command|test|file|script|code|function)\b",
        re.IGNORECASE,
    ),
    # Questions referencing this/that/it or trailing about that/it
    re.compile(r"^(why|what|how)\s+(is|was|did|does|do|about)\s+(this|that|it)\b", re.IGNORECASE),
    re.compile(r"\b(about|on|with|for|regarding)\s+(this|that|it|these|those)\s*[?.!]*$", re.IGNORECASE),
    # Conversational rejections & feedback
    re.compile(r"^(no|nope|not quite|wrong|incorrect)\b", re.IGNORECASE),
    # Judgments of previous response (e.g. "That's a good start", "That is wrong", "That worked", "That's a solid plan")
    re.compile(
        r"^(that|it|this)\s+(is|was|looks|seems|feels)?\s*(a\s+)?(not\s+good|wrong|incorrect|bad|broken|terrible|awesome|great|cool|fine|ok|okay|good|solid|start)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"^that('s|s| is)\s+(a\s+)?(not|wrong|incorrect|invalid|broken|good|great|fine|a bit|solid)\b",
        re.IGNORECASE,
    ),
    re.compile(r"^(that|it)\s+(worked|failed|passed|didn't work|did not work)\b", re.IGNORECASE),
    # Conversational agreement / plan acceptance / conversational transitions
    re.compile(r"^(sounds|looks)\s+like\s+a\s+(good|solid|great)\s+plan\b", re.IGNORECASE),
    re.compile(r"^(thanks|thank you)\b", re.IGNORECASE),
    re.compile(r"^(ok|okay)\s*[,.]?\s*(then|now|please|go ahead|do it|'|\")?", re.IGNORECASE),
    # Temporal / relative discourse references
    re.compile(
        r"\b(previous|prior|earlier|last|above|mentioned)\s+(turn|step|output|error|response|message|command|result|file|version|code)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat\s+(did\s+)?you\s+(just\s+)?(say|do|write|suggest|show|create|said|did|wrote|suggested|showed|created)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(still|now)\s+(failing|broken|passing|working)\b", re.IGNORECASE),
]


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

    # Inline payload exception:
    # "Explain this code: def foo()..." or "Fix this error: ZeroDivisionError..."
    has_inline_payload = False
    prefix = ""
    if ":" in cleaned:
        prefix, suffix = cleaned.split(":", 1)
        if len(suffix.strip()) > 8:
            has_inline_payload = True

    for pat in _ANAPHORA_PATTERNS:
        m = pat.search(cleaned)
        if m:
            if has_inline_payload:
                prefix_lower = prefix.lower()
                if any(w in prefix_lower for w in ("code", "error", "file", "sentence", "function", "input", "command", "text", "query")):
                    continue
            return True, f"pattern:{pat.pattern[:40]}"

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
