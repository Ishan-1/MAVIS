"""
tasks/reflection_worker.py
Background worker: Autonomous Reflection Engine execution.

Runs periodically (default: once daily, every 1440 minutes).
Executes a compute-bounded reflection cycle:
  Phase 1: Initiation & Topic Selection (math-gated exploration vs pivot)
  Phase 2: Triage Gate (low-temperature utility filter)
  Phase 3: Adversarial Exploration Loop (Generator vs Critic + sandboxed tools)
  Phase 4: Semantic Delta Evaluation & Dedicated Reflection Storage

The MemoryStore singleton must be injected via `set_store()` before `run()`.
"""
from __future__ import annotations

import time

from core.config import cfg
from core.helpers import log_it
from core.llm import get_llm_client
from core.reflection.engine import ReflectionEngine

_ENTITY = "reflection_worker"

# Module-level store reference — injected by worker_process.py / main.py
_store = None


def set_store(store):
    """Inject the MemoryStore singleton. Must be called before run()."""
    global _store
    _store = store


def run():
    """
    Zero-argument callable registered with TaskRunner or invoked by worker_process.py.
    Executes one autonomous reflection cycle with strict ONI sandboxing.
    """
    if _store is None:
        log_it("reflection_worker: MemoryStore not injected — skipping.", _ENTITY)
        return

    # Check if reflection is enabled in configuration
    if not cfg.get("reflection", "enabled", default=True):
        log_it("reflection_worker: reflection is disabled in config — skipping.", _ENTITY)
        return

    # Enforce strict ONI sandbox isolation for background execution
    try:
        from oni import set_context_trust
        set_context_trust("whitelist_only")
    except Exception:
        pass

    log_it("reflection_worker: starting scheduled autonomous reflection cycle.", _ENTITY)
    try:
        client = get_llm_client()
        engine = ReflectionEngine(client=client, store=_store)
        result = engine.run_cycle()

        log_it(
            f"reflection_worker finished cycle {result.id}: status={result.status}, "
            f"topic='{result.selected_topic}', delta={result.semantic_delta:.3f}, "
            f"committed={result.committed}, duration={result.duration_seconds:.2f}s",
            _ENTITY,
        )
    except Exception as exc:
        log_it(f"reflection_worker failed with unhandled error: {exc}", _ENTITY)
