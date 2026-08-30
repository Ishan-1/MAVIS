import threading
import schedule
import time
from core.helpers import log_it


class TaskRunner:
    """
    A generic background scheduler that runs registered periodic jobs
    in a daemon thread alongside the main process.

    Usage:
        runner = TaskRunner()
        runner.register(my_fn, interval_minutes=15, task_name="my_task")
        runner.start()
        # ... main loop ...
        runner.stop()
    """

    def __init__(self, entity_name: str = "scheduler", tick_seconds: int = 30):
        """
        Args:
            entity_name: Name used for log files (logs/<entity_name>.log).
            tick_seconds: How often the background thread checks for pending jobs.
        """
        self.entity_name = entity_name
        self.tick_seconds = tick_seconds
        self._scheduler = schedule.Scheduler()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._registry: list[dict] = []  # [{name, interval_minutes, trust_level, last_run_ts}]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        task_fn,
        interval_minutes: int,
        task_name: str,
        trust_level: str = "whitelist_only",
    ):
        """
        Register a zero-argument callable to be run every `interval_minutes`.

        The callable is wrapped so that any exception it raises is caught,
        logged, and does NOT crash the scheduler thread.

        Args:
            task_fn:          A zero-argument callable (the task to run).
            interval_minutes: How often to run the task, in minutes.
            task_name:        Human-readable name used in logs and registry.
            trust_level:      ONI trust level for this task's thread.
                              Defaults to "whitelist_only" — background tasks
                              never inherit the session trust level, even in
                              YOLO mode.
        """
        def _safe_run():
            # Isolate this background thread's ONI trust from the session level.
            try:
                from oni import set_context_trust
                set_context_trust(trust_level)
            except Exception:
                pass  # ONI not yet loaded — proceed without trust isolation

            log_it(
                f"Running task '{task_name}' (trust={trust_level}).",
                self.entity_name,
            )
            try:
                task_fn()
                log_it(f"Task '{task_name}' completed successfully.", self.entity_name)
            except Exception as e:
                log_it(f"Task '{task_name}' raised an exception: {e}", self.entity_name)
            finally:
                # Record completion time for /status reporting
                for entry in self._registry:
                    if entry["name"] == task_name:
                        entry["last_run_ts"] = time.time()
                        break

        self._scheduler.every(interval_minutes).minutes.do(_safe_run)
        self._registry.append({
            "name": task_name,
            "interval_minutes": interval_minutes,
            "trust_level": trust_level,
            "last_run_ts": None,  # None until first execution
        })
        log_it(
            f"Registered task '{task_name}' (every {interval_minutes} min, trust={trust_level}).",
            self.entity_name,
        )

    def list_tasks(self) -> list[dict]:
        """
        Return a list of registered task metadata dicts.
        Each dict has keys: 'name', 'interval_minutes', 'trust_level', 'last_run_ts'.
        Useful for the '/status' command in the main loop.
        """
        return list(self._registry)

    def get_last_run(self, task_name: str) -> float | None:
        """Return the timestamp of the last completed run for *task_name*, or None."""
        for entry in self._registry:
            if entry["name"] == task_name:
                return entry["last_run_ts"]
        return None

    def start(self):
        """
        Start the background daemon thread. Safe to call even if no tasks
        are registered yet — tasks can be registered after start() and the
        scheduler will pick them up on the next tick.
        """
        if self._thread and self._thread.is_alive():
            log_it("TaskRunner already running.", self.entity_name)
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="MAV-Scheduler")
        self._thread.start()
        log_it(
            f"Scheduler started (tick={self.tick_seconds}s, "
            f"{len(self._registry)} task(s) registered).",
            self.entity_name,
        )
        print(f"[Scheduler] Started with {len(self._registry)} task(s).")

    def stop(self):
        """
        Signal the background thread to stop and wait for it to exit.
        """
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=self.tick_seconds + 5)
            log_it("Scheduler stopped.", self.entity_name)
            print("[Scheduler] Stopped.")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_loop(self):
        """Main loop executed in the daemon thread."""
        while not self._stop_event.is_set():
            self._scheduler.run_pending()
            time.sleep(self.tick_seconds)
