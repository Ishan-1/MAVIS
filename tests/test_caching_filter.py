"""
tests/test_caching_filter.py
Unit tests for semantic cache pre-filtering and context-dependency detection.
"""
import sqlite3
import hashlib
import pytest

from core.caching_filter import is_context_dependent_query, is_cacheable_query
from core.caching import CacheManager
from core.config import cfg
from core.metrics import get_metrics_summary, MetricEmitter


def test_table_driven_battery_phrasings():
    """
    Comprehensive table-driven battery testing real user phrasings,
    ensuring all context-dependent commands bypass the cache while
    standalone queries remain eligible.
    """
    cases = [
        # ── Bare-pronoun commands (must bypass) ──────────────────────────────
        ("Summarize that", True),
        ("Make it shorter", True),
        ("Do it again", True),
        ("Translate that to Hindi", True),
        ("Rewrite it", True),
        ("Delete it", True),
        ("Send it to her", True),
        ("Shorten that", True),
        ("Is this good?", True),
        ("Do that", True),
        ("Why did that happen?", True),
        ("Fix this", True),
        ("Run it again", True),
        ("Explain it", True),
        # ── Continuations and transitions (must bypass) ───────────────────────
        ("Do the same for the other file", True),
        ("Same as before", True),
        ("Also add logging", True),
        ("Now do the tests", True),
        ("Keep going", True),
        ("And the other file?", True),
        ("What about the second one?", True),
        ("No, that is not good", True),
        ("No, that is wrong", True),
        ("That worked", True),
        ("That failed", True),
        ("Sounds like a solid plan. Go ahead with those steps.", True),
        ("ok then prepare a proper commit message and do the commit", True),
        ("Thanks for the draft to Sarah. Could you please save that reply...", True),
        # ── Elliptic commands (must bypass) ──────────────────────────────────
        ("Why?", True),
        ("How?", True),
        ("Next", True),
        ("Continue", True),
        ("undo that", True),
        # ── Standalone queries (MUST NOT bypass / must be cacheable) ──────────
        ("Okta SSO setup steps", False),
        ("What is the capital of France?", False),
        ("Current date and time?", False),
        ("Run pytest tests/test_metrics.py", False),
        ("How do I fix a ZeroDivisionError in Python?", False),
        ("List all files in /tmp", False),
        ("can you go through the uncommited changes and tell me what did they add/fix?", False),
        # ── Inline payload exemptions (MUST NOT bypass) ──────────────────────
        ("Explain this code: def foo(): return 42", False),
        ("What is the meaning of this error: NameError: name x is not defined", False),
        ("Can you fix this error: SyntaxError: invalid syntax in math_lib.py?", False),
    ]

    for query, expected_dep in cases:
        is_dep, reason = is_context_dependent_query(query)
        assert is_dep is expected_dep, f"Query: {query!r} -> got is_dep={is_dep} (reason={reason}), expected {expected_dep}"


def test_workspace_prefix_handling():
    q_dep = "[Workspace: /home/user/project]\nThanks for the update. Now fix the bug."
    is_dep, _ = is_context_dependent_query(q_dep)
    assert is_dep is True

    q_standalone = "[Workspace: /home/user/project]\nCurrent date and time?"
    is_dep, _ = is_context_dependent_query(q_standalone)
    assert is_dep is False


def test_cache_manager_check_cache_bypasses(tmp_path, monkeypatch):
    db_file = tmp_path / "test_cache.db"
    monkeypatch.setitem(cfg.caching, "enabled", True)
    monkeypatch.setitem(cfg.caching, "filter_context_dependent", True)

    class DummyClient:
        def generate(self, *args, **kwargs):
            raise AssertionError("LLM should not be called on context-dependent check")

    cm = CacheManager(client=DummyClient(), db_path=str(db_file))

    # Context-dependent query: check_cache returns None without calling embed or DB search
    hit = cm.check_cache("Summarize that", turn_id="t_dep")
    assert hit is None


def test_cache_manager_save_cache_prefilters_and_stable_id(tmp_path, monkeypatch):
    db_file = tmp_path / "test_cache.db"
    monkeypatch.setitem(cfg.caching, "enabled", True)
    monkeypatch.setitem(cfg.caching, "filter_context_dependent", True)

    embed_calls = []

    def fake_embed(text, client=None):
        embed_calls.append(text)
        return [0.1] * 10

    monkeypatch.setattr("core.caching.embed", fake_embed)

    cm = CacheManager(client=None, db_path=str(db_file))

    # 1. Try saving context-dependent query -> should be skipped
    cm.save_cache("Make it shorter", [{"id": "n1"}], {"status": 0})
    assert len(embed_calls) == 0

    with sqlite3.connect(str(db_file)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM pipeline_cache").fetchone()[0]
        assert count == 0

    # 2. Try saving with cacheable=False -> should be skipped
    cm.save_cache("Current date and time?", [{"id": "n1"}], {"status": 0}, cacheable=False)
    assert len(embed_calls) == 0

    # 3. Save standalone query with cacheable=True -> should succeed with stable sha256 id
    cm.save_cache("Current date and time?", [{"id": "n1"}], {"status": 0}, cacheable=True)
    assert len(embed_calls) == 1

    expected_id = hashlib.sha256("current date and time?".encode("utf-8")).hexdigest()[:16]
    with sqlite3.connect(str(db_file)) as conn:
        row = conn.execute("SELECT id, query FROM pipeline_cache").fetchone()
        assert row is not None
        assert row[0] == expected_id
        assert row[1] == "Current date and time?"


def test_cache_manager_purge_context_dependent(tmp_path, monkeypatch):
    db_file = tmp_path / "test_cache.db"
    monkeypatch.setitem(cfg.caching, "enabled", True)
    monkeypatch.setitem(cfg.caching, "filter_context_dependent", False)  # disable auto-purge on init

    cm = CacheManager(client=None, db_path=str(db_file))

    # Insert mixed entries into sqlite
    entries = [
        ("id1", "Current date and time?", "[]", "null", "[0.1]", 9999999999, "generalized", 100),
        ("id2", "Summarize that", "[]", "null", "[0.1]", 9999999999, "specialized", 100),
        ("id3", "Do the same for the other file", "[]", "null", "[0.1]", 9999999999, "specialized", 100),
        ("id4", "Sounds like a solid plan. Go ahead with those steps.", "[]", "null", "[0.1]", 9999999999, "specialized", 100),
        ("id5", "Run pytest tests/test_metrics.py", "[]", "null", "[0.1]", 9999999999, "generalized", 100),
    ]

    with sqlite3.connect(str(db_file)) as conn:
        conn.executemany("""
            INSERT INTO pipeline_cache (id, query, pipeline, result, embedding, ttl_timestamp, generalizability, inserted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, entries)
        conn.commit()

    # Verify initial count
    with sqlite3.connect(str(db_file)) as conn:
        count = conn.execute("SELECT COUNT(*) FROM pipeline_cache").fetchone()[0]
        assert count == 5

    # Run purge
    purged_count = cm.purge_context_dependent_entries()
    assert purged_count == 3

    # Verify only the 2 standalone entries remain
    with sqlite3.connect(str(db_file)) as conn:
        remaining = [r[0] for r in conn.execute("SELECT query FROM pipeline_cache").fetchall()]
        assert "Current date and time?" in remaining
        assert "Run pytest tests/test_metrics.py" in remaining
        assert len(remaining) == 2


def test_metrics_eligible_hit_rate_not_diluted(tmp_path, monkeypatch):
    """Verify that bypassed/pre-filtered checks do not dilute the cache hit rate."""
    monkeypatch.setattr("core.metrics.METRICS_DIR", str(tmp_path))
    emitter = MetricEmitter("caching")

    # 1. One instant hit
    emitter.log({
        "turn_id": "t1",
        "cache_status": "hit",
        "hit_tier": "instant",
        "latency_ms": 1.0,
    })
    # 2. One miss
    emitter.log({
        "turn_id": "t2",
        "cache_status": "miss",
        "hit_tier": "miss",
        "latency_ms": 50.0,
    })
    # 3. Two bypassed (context-dependent) checks
    emitter.log({
        "turn_id": "t3",
        "cache_status": "bypassed",
        "hit_tier": "prefiltered_context",
        "latency_ms": 0.0,
    })
    emitter.log({
        "turn_id": "t4",
        "cache_status": "bypassed",
        "hit_tier": "prefiltered_context",
        "latency_ms": 0.0,
    })

    summary = get_metrics_summary()
    c = summary["caching"]

    assert c["total_checks"] == 4
    assert c["bypassed"] == 2
    assert c["eligible_checks"] == 2
    assert c["hits"] == 1
    # Hit rate must be 1 / 2 = 50.0%, NOT 1 / 4 = 25.0%
    assert c["hit_rate_pct"] == 50.0
    assert c["bypass_rate_pct"] == 50.0
