"""
core/reflection/__init__.py
Autonomous reflection engine package for MAVIS.
"""
from core.reflection.engine import (
    ReflectionEngine,
    ReflectionResult,
    load_seed_pool,
    add_seed_to_pool,
    remove_seed_from_pool,
    load_reflection_journal,
    verify_and_promote_reflection,
    delete_reflection_entry,
)

__all__ = [
    "ReflectionEngine",
    "ReflectionResult",
    "load_seed_pool",
    "add_seed_to_pool",
    "remove_seed_from_pool",
    "load_reflection_journal",
    "verify_and_promote_reflection",
    "delete_reflection_entry",
]
