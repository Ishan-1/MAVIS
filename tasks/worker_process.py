"""
worker_process.py
Standalone worker process for MAVIS background memory tasks.
Spawned as independent subprocesses by main.py.

Usage:
    python worker_process.py <short_term|long_term> <interval_minutes>

Monitors data/.mavis_heartbeat and self-terminates if heartbeat becomes stale (>120s).
"""
from __future__ import annotations

import os
import signal
import sys
import time

from dotenv import load_dotenv

# Ensure root dir is on sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

load_dotenv()

from core.helpers import log_it
from core.config import cfg

_HEARTBEAT_PATH = os.path.join(_ROOT, "data", ".mavis_heartbeat")
_HEARTBEAT_MAX_AGE_SECONDS = 180
_RUNNING = True
_STARTUP_TIME = time.time()


def _sig_handler(signum, frame):
    global _RUNNING
    _RUNNING = False


signal.signal(signal.SIGTERM, _sig_handler)
signal.signal(signal.SIGINT, _sig_handler)


def is_heartbeat_alive() -> bool:
    """Return False if heartbeat file is missing (>30s after startup) or has not been touched in >180s."""
    if not os.path.exists(_HEARTBEAT_PATH):
        return (time.time() - _STARTUP_TIME) <= 30
    try:
        with open(_HEARTBEAT_PATH, "r") as f:
            ts = float(f.read().strip())
        return (time.time() - ts) <= _HEARTBEAT_MAX_AGE_SECONDS
    except Exception:
        try:
            return (time.time() - os.path.getmtime(_HEARTBEAT_PATH)) <= _HEARTBEAT_MAX_AGE_SECONDS
        except Exception:
            return False


def sleep_with_heartbeat(duration_seconds: float) -> bool:
    """Sleep for duration_seconds in small slices, checking heartbeat and termination signals.
    Returns True if completed normally, False if interrupted or heartbeat expired."""
    elapsed = 0.0
    slice_sec = 5.0
    while elapsed < duration_seconds and _RUNNING:
        if not is_heartbeat_alive():
            return False
        to_sleep = min(slice_sec, duration_seconds - elapsed)
        time.sleep(to_sleep)
        elapsed += to_sleep
    return _RUNNING


def main():
    if len(sys.argv) < 3:
        print("Usage: worker_process.py <worker_name> <interval_minutes>")
        sys.exit(1)

    worker_name = sys.argv[1]
    interval_min = int(sys.argv[2])
    interval_seconds = interval_min * 60

    entity = f"{worker_name}_worker_proc"
    log_it(f"Worker process started for {worker_name} (interval={interval_min}m).", entity)

    # Initialize genai client and MemoryStore
    from google import genai
    from memories.memory_store import MemoryStore

    client = genai.Client(vertexai=True, api_key=os.getenv("VERTEX_API_KEY"))
    store = MemoryStore(client)

    if worker_name == "short_term":
        import tasks.short_term_worker as task_mod
    elif worker_name == "long_term":
        import tasks.long_term_worker as task_mod
    else:
        log_it(f"Unknown worker name: {worker_name}", entity)
        sys.exit(1)

    task_mod.set_store(store)

    # Startup grace period: wait the interval before running for the first time
    # so we don't redundantly re-process memory on restarts or crashes
    log_it(f"Entering startup grace period ({interval_min}m)...", entity)
    if not sleep_with_heartbeat(interval_seconds):
        log_it("Heartbeat lost or stopped during grace period. Exiting.", entity)
        sys.exit(0)

    while _RUNNING:
        try:
            if not is_heartbeat_alive():
                log_it("Heartbeat expired (main process appears stopped). Worker exiting.", entity)
                break

            store.reload_working_memory()
            task_mod.run()
        except Exception as e:
            log_it(f"Exception during worker run: {e}", entity)

        if not sleep_with_heartbeat(interval_seconds):
            break

    log_it("Worker process exiting cleanly.", entity)


if __name__ == "__main__":
    main()
