"""
tasks/short_term_worker.py
Background worker: Working memory → Short-term promotion.

Runs every 15 minutes when registered with TaskRunner.
Active-session guard: skips if last user input was > 30 minutes ago.

Promotion triggers (from FUTURE.md):
  1. emotion_strength > 0.75  OR  intent_strength > 0.85
  2. (same threshold — handled by the classifier, already set on the turn)
  3. Repetition: cosine similarity ≥ 0.85 in last 50 turns, ≥ 3rd occurrence
  4. tool_failure flag is True on the entry

The MemoryStore singleton must be injected via `set_store()` before the
TaskRunner calls `run()`.
"""

import time
import uuid

from core.config import cfg
from core.helpers import log_it
from memories.embedding import cosine_similarity
from memories.emotion_classifier import should_promote_short_term

_ENTITY = "short_term_worker"

# Module-level store reference — injected by main.py
_store = None


def set_store(store):
    """Inject the MemoryStore singleton.  Must be called before run()."""
    global _store
    _store = store


def run():
    """
    Zero-argument callable registered with TaskRunner (every 15 minutes).
    Scans new working-memory entries and promotes qualifying ones to
    short-term memory.
    """
    if _store is None:
        log_it("short_term_worker: MemoryStore not injected — skipping.", _ENTITY)
        return

    # Read tunable params from central config on each run (supports runtime /config set)
    session_timeout_s = cfg.memory.get("session_timeout_minutes", 30) * 60
    rep_window       = cfg.memory.get("repetition_window", 50)
    rep_threshold    = cfg.memory.get("repetition_similarity_threshold", 0.85)
    rep_min_count    = cfg.memory.get("repetition_min_count", 3)

    # ── Active-session guard ────────────────────────────────────────────────────────────
    idle_s = time.time() - _store.last_user_input_ts
    if idle_s > session_timeout_s:
        log_it(
            f"short_term_worker: session idle for {idle_s / 60:.1f} min — skipping.",
            _ENTITY,
        )
        return

    # ── Delta processing ───────────────────────────────────────────────────────────────
    cursor = _store.read_cursor(_store.st_cursor_path)
    new_turns = _store.get_working_memory_since(cursor)

    if not new_turns:
        log_it("short_term_worker: no new turns since last run.", _ENTITY)
        return

    log_it(f"short_term_worker: processing {len(new_turns)} new turn(s).", _ENTITY)

    # For repetition detection we need the full recent window
    all_recent = _store.get_working_memory()[-rep_window:]

    promoted = 0
    latest_ts = cursor

    for turn in new_turns:
        reason = _promotion_reason(turn, all_recent, rep_threshold, rep_min_count)
        if reason:
            entry = {
                "id": turn.get("id", str(uuid.uuid4())),
                "role": turn["role"],
                "content": turn["content"],
                "timestamp": turn["timestamp"],
                "emotion": turn.get("emotion", "neutral"),
                "emotion_strength": turn.get("emotion_strength", 0.0),
                "intent_strength": turn.get("intent_strength", 0.0),
                "promotion_reason": reason,
                "embedding": turn.get("embedding"),  # pass cached vector
            }
            _store.write_short_term(entry)
            promoted += 1
            log_it(
                f"short_term_worker: promoted turn id={entry['id']!r} reason={reason!r}",
                _ENTITY,
            )

        latest_ts = max(latest_ts, turn["timestamp"])

    # Advance cursor to the timestamp of the last processed turn
    _store.write_cursor(_store.st_cursor_path, latest_ts)
    log_it(
        f"short_term_worker: promoted {promoted}/{len(new_turns)} turns. "
        f"Cursor → {latest_ts}.",
        _ENTITY,
    )


# ── Internal helpers ─────────────────────────────────────────────────────────

def _promotion_reason(
    turn: dict,
    all_recent: list[dict],
    rep_threshold: float,
    rep_min_count: int,
) -> str:
    """
    Return a non-empty reason string if the turn qualifies for promotion,
    or an empty string otherwise.
    """
    emo = turn.get("emotion_strength", 0.0)
    intent = turn.get("intent_strength", 0.0)

    # Trigger 1 & 2: emotion / intent classifier thresholds
    if should_promote_short_term(emo, intent):
        return f"classifier(emo={emo:.2f},intent={intent:.2f})"

    # Trigger 3: repetition
    rep_count = _repetition_count(turn, all_recent, rep_threshold)
    if rep_count >= rep_min_count:
        return f"repetition(count={rep_count})"

    # Trigger 4: tool failure
    if turn.get("tool_failure"):
        return "tool_failure"

    return ""


def _repetition_count(turn: dict, window: list[dict], threshold: float) -> int:
    """
    Count how many turns in *window* have cosine similarity ≥ threshold
    with *turn* (excluding *turn* itself).
    """
    vec = turn.get("embedding")
    if not vec:
        return 0

    count = 0
    for other in window:
        if other is turn:
            continue
        other_vec = other.get("embedding")
        if not other_vec:
            continue
        if cosine_similarity(vec, other_vec) >= threshold:
            count += 1
    return count
