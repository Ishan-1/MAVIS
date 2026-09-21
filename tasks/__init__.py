"""
tasks/
Package for autonomous background tasks and daemon worker processes.

Houses background memory consolidation workers:
  - short_term_worker.py: Promotes high-salience working memories to short-term storage (every 15 min).
  - long_term_worker.py: Promotes durable knowledge and rules to long-term storage (every 8 hrs).
  - worker_process.py: Standalone decoupled daemon process managing scheduled worker cycles.
"""
from __future__ import annotations

