"""
oni/gate.py
ConfirmationGate — synchronous greylist approval for the main thread;
automatic denial for background threads (no user present).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time


class ConfirmationGate:
    """
    Routes greylist approval requests to the user or auto-denies them.

    Design:
      - Main thread calls: block on input(), wait for yes/no.
      - Tool subprocess calls: delegate via IPC to the parent MAVIS process.
      - Background thread calls: auto-deny immediately and log the reason.
        Background tasks should never interrupt the user or block indefinitely.
    """

    def __init__(self) -> None:
        self._main_thread_id: int = threading.main_thread().ident  # type: ignore[assignment]

    # ── Public API ────────────────────────────────────────────────────────────

    def request_approval(self, description: str) -> bool:
        """
        Request user approval for a greylisted operation.

        Args:
            description: Human-readable description of what needs approval.

        Returns:
            True if the user approved, False if denied or called from a
            background thread (auto-deny).
        """
        # If running inside a tool runner child subprocess, forward request to parent MAVIS
        if os.environ.get("MAVIS_TOOL_SUBPROCESS") == "1":
            try:
                payload = json.dumps({
                    "__oni_ipc__": True,
                    "type": "approval_request",
                    "description": description,
                })
                sys.__stdout__.write(payload + "\n")
                sys.__stdout__.flush()
                line = sys.__stdin__.readline()
                if not line:
                    return False
                resp = json.loads(line.strip())
                return bool(resp.get("approved", False))
            except Exception:
                return False

        if not self._is_main_thread():
            print(
                f"\n[ONI] ⚠ Background task requested approval for: {description}\n"
                f"      Auto-denied (no user present in background thread)."
            )
            return False

        t0 = time.perf_counter()
        approved = False
        try:
            from core.output import oni_gate_panel, interactive_select_yes_no
            oni_gate_panel(description)
            approved = interactive_select_yes_no("Allow operation?", default=False)
        except Exception:
            # Fallback if output layer encounters any terminal issues
            print(f"\n[ONI] ⚠ Approval required:\n      {description}")
            try:
                response = input("  Allow? (yes/no, default no): ").strip().lower()
                approved = response in ("yes", "y")
            except (EOFError, KeyboardInterrupt):
                print("\n[ONI] Approval cancelled — defaulting to deny.")
                approved = False

        dwell_time_ms = round((time.perf_counter() - t0) * 1000, 2)
        try:
            from core.metrics import MetricEmitter
            MetricEmitter("oni").log({
                "target_command_or_path": description[:100],
                "phase": "runtime",
                "trust_level": "ask",
                "oni_decision": "greylist_prompted",
                "user_decision": "allowed" if approved else "denied",
                "dwell_time_ms": dwell_time_ms,
            })
        except Exception:
            pass

        return approved

    # ── Internal ──────────────────────────────────────────────────────────────

    def _is_main_thread(self) -> bool:
        return threading.current_thread().ident == self._main_thread_id

