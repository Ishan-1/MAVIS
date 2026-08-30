"""
oni/gate.py
ConfirmationGate — synchronous greylist approval for the main thread;
automatic denial for background threads (no user present).
"""
from __future__ import annotations

import threading


class ConfirmationGate:
    """
    Routes greylist approval requests to the user or auto-denies them.

    Design:
      - Main thread calls: block on input(), wait for yes/no.
      - Background thread calls: auto-deny immediately and log the reason.
        Background tasks should never interrupt the user or block indefinitely.

    This is intentionally simple — no async queue needed because the
    pre-flight scan serialises all greylist prompts before any execution begins.
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
        if not self._is_main_thread():
            print(
                f"\n[ONI] ⚠ Background task requested approval for: {description}\n"
                f"      Auto-denied (no user present in background thread)."
            )
            return False

        try:
            from core.output import oni_gate_panel, interactive_select_yes_no
            oni_gate_panel(description)
            return interactive_select_yes_no("Allow operation?", default=False)
        except Exception:
            # Fallback if output layer encounters any terminal issues
            print(f"\n[ONI] ⚠ Approval required:\n      {description}")
            try:
                response = input("  Allow? (yes/no, default no): ").strip().lower()
                return response in ("yes", "y")
            except (EOFError, KeyboardInterrupt):
                print("\n[ONI] Approval cancelled — defaulting to deny.")
                return False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _is_main_thread(self) -> bool:
        return threading.current_thread().ident == self._main_thread_id
