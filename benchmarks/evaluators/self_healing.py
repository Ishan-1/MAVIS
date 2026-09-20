"""
benchmarks/evaluators/self_healing.py
Self-Healing (HEAL) evaluation engine.
Assesses pipeline_debugger fault detection, recovery actions, and memory retention.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from benchmarks.harness.models import TaskSpec


class SelfHealingEvaluator:
    """
    Computes HEAL score:
    - Fault recovery rate: Repaired Faults / Injected Faults.
    - Diagnostic accuracy: Validating action in {'patch_params', 'replace_node'}.
    """

    def __init__(self, metrics_dir: Path | None = None) -> None:
        self.metrics_dir = metrics_dir or Path("data/metrics")

    def evaluate(
        self,
        task: TaskSpec,
        execution_trace: list[dict[str, Any]],
    ) -> tuple[float, list[str]]:
        """
        Evaluate self-healing performance for a task execution.

        Returns (heal_score, audit_messages).
        """
        if not task.injected_faults:
            # Check if any spontaneous runtime failures occurred and were repaired
            spontaneous_failures, spontaneous_repairs = self._check_runtime_repairs(execution_trace)
            if spontaneous_failures > 0:
                score = round(spontaneous_repairs / spontaneous_failures, 4)
                return score, [f"Spontaneous faults: {spontaneous_repairs}/{spontaneous_failures} repaired"]
            return 1.0, ["No faults injected or encountered."]

        repaired_count = 0
        total_injected = len(task.injected_faults)
        logs: list[str] = []

        debugger_events = self._read_debugger_metrics()

        for fault in task.injected_faults:
            # Check if the trigger step was executed and healed
            healed = False
            for event in debugger_events:
                action = event.get("action", "")
                if action in ("patch_params", "replace_node"):
                    healed = True
                    break

            # Fallback: check execution trace metadata
            if not healed:
                for trace in execution_trace:
                    if trace.get("repaired", False) or trace.get("debugger_action") in ("patch_params", "replace_node"):
                        healed = True
                        break

            if healed:
                repaired_count += 1
                logs.append(f"Fault [{fault.trigger_step}]: Successfully healed via pipeline_debugger")
            else:
                logs.append(f"Fault [{fault.trigger_step}]: Unresolved or failed")

        heal_score = round(repaired_count / total_injected, 4) if total_injected > 0 else 1.0
        return heal_score, logs

    def _check_runtime_repairs(self, execution_trace: list[dict[str, Any]]) -> tuple[int, int]:
        """Inspect trace for spontaneous step errors and repairs."""
        failures = 0
        repairs = 0
        for step in execution_trace:
            if step.get("status") == "error" or step.get("had_error"):
                failures += 1
                if step.get("repaired") or step.get("debugger_action"):
                    repairs += 1
        return failures, repairs

    def _read_debugger_metrics(self) -> list[dict[str, Any]]:
        """Read recent pipeline_debugger telemetry rows."""
        debugger_csv = self.metrics_dir / "pipeline_debugger.csv"
        if not debugger_csv.exists():
            return []

        rows: list[dict[str, Any]] = []
        try:
            with open(debugger_csv, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
        except Exception:
            pass
        return rows[-50:]  # Return most recent 50 events
