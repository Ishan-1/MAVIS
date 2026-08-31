"""
tasks/long_term_worker.py
Background worker: Short-term → Long-term promotion + short-term TTL pruning.

Runs every 8 hours regardless of session state.

Promotion triggers (from FUTURE.md):
  1. Permanent behaviour change: intent_strength > 0.90 AND keywords like
     "always", "never", "permanently", "from now on", "every time"
     → ltype = "behaviour"
  2. Explicit long-term request: intent_strength > 0.90 AND keywords like
     "remember forever", "never forget"
     → ltype = "fact"

Short-term pruning:
  - JSON files older than 7 days are deleted.
  - ChromaDB entries with date < cutoff are deleted from the collection.

The MemoryStore singleton must be injected via `set_store()` before the
TaskRunner calls `run()`.
"""

import json
import os
import uuid
from datetime import datetime, timedelta, timezone

from core.config import cfg
from core.helpers import log_it
from memories.emotion_classifier import should_promote_long_term

_ENTITY = "long_term_worker"

# Module-level store reference — injected by main.py
_store = None


def set_store(store):
    """Inject the MemoryStore singleton.  Must be called before run()."""
    global _store
    _store = store


def run():
    """
    Zero-argument callable registered with TaskRunner (every 480 minutes).
    Scans short-term JSON entries added since the last cursor position,
    promotes qualifying entries to long-term, and prunes stale short-term data.
    """
    if _store is None:
        log_it("long_term_worker: MemoryStore not injected — skipping.", _ENTITY)
        return

    log_it("long_term_worker: starting run.", _ENTITY)

    # ── Delta scan of short-term JSON files ─────────────────────────────────
    cursor_ts = _store.read_cursor(_store.lt_cursor_path)
    promoted = 0
    latest_ts = cursor_ts

    st_json_dir = _store.st_json_dir
    try:
        json_files = sorted(
            f for f in os.listdir(st_json_dir) if f.endswith(".json")
        )
    except FileNotFoundError:
        json_files = []

    for fname in json_files:
        path = os.path.join(st_json_dir, fname)
        try:
            with open(path) as f:
                entries = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        for entry in entries:
            ts = entry.get("timestamp", 0.0)
            if ts <= cursor_ts:
                continue  # already processed

            latest_ts = max(latest_ts, ts)
            directive = entry.get("directive", entry.get("intent_strength", 0.0) > 0.85)

            promote, ltype = should_promote_long_term(directive, content)
            if promote:
                lt_entry = {
                    "id": entry.get("id", str(uuid.uuid4())),
                    "content": content,
                    "timestamp": ts,
                    "date": entry.get("date", fname.replace(".json", "")),
                    "emotion": entry.get("emotion", "neutral"),
                    "emotion_strength": entry.get("emotion_strength", "low"),
                    "directive": directive,
                    "source_file": fname,
                }
                _store.write_long_term(lt_entry, ltype)
                promoted += 1
                log_it(
                    f"long_term_worker: promoted id={lt_entry['id']!r} "
                    f"type={ltype!r} from {fname}",
                    _ENTITY,
                )

    # Advance cursor
    _store.write_cursor(_store.lt_cursor_path, latest_ts)
    log_it(
        f"long_term_worker: promoted {promoted} entries. Cursor → {latest_ts}.",
        _ENTITY,
    )

    # ── Prune short-term entries older than 7 days ───────────────────────────
    _prune_short_term()

    log_it("long_term_worker: run complete.", _ENTITY)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _prune_short_term():
    """
    Delete short-term JSON files older than ST_TTL_DAYS and remove the
    corresponding ChromaDB entries.
    """
    if _store is None:
        return

    cutoff_date = (
        datetime.now(tz=timezone.utc) - timedelta(days=cfg.memory.get("short_term_ttl_days", 7))
    ).strftime("%Y-%m-%d")

    st_json_dir = _store.st_json_dir
    try:
        json_files = [f for f in os.listdir(st_json_dir) if f.endswith(".json")]
    except FileNotFoundError:
        return

    deleted_files = 0
    for fname in json_files:
        date_str = fname.replace(".json", "")
        if date_str < cutoff_date:
            path = os.path.join(st_json_dir, fname)
            try:
                os.remove(path)
                deleted_files += 1
                log_it(f"long_term_worker: pruned {fname}.", _ENTITY)
            except OSError as exc:
                log_it(f"long_term_worker: could not delete {fname}: {exc}", _ENTITY)

    # Prune ChromaDB short-term entries older than cutoff
    try:
        # ChromaDB supports `where` with metadata filtering on delete
        _store._st_col.delete(where={"date": {"$lt": cutoff_date}})
        log_it(
            f"long_term_worker: pruned ChromaDB entries with date < {cutoff_date}.",
            _ENTITY,
        )
    except Exception as exc:
        log_it(f"long_term_worker: ChromaDB prune error: {exc}", _ENTITY)

    log_it(
        f"long_term_worker: pruned {deleted_files} JSON file(s) older than {cutoff_date}.",
        _ENTITY,
    )
