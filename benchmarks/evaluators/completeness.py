"""
benchmarks/evaluators/completeness.py
Completeness (COMP) evaluation engine.
Executes programmatic verification scripts and verifies artifact assertions.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.harness.models import TaskSpec, VerificationArtifact


class CompletenessEvaluator:
    """
    Computes COMP score based on deterministic checkpoints:
    1. Static artifact assertions (existence, keywords, line counts).
    2. Execution of task-specific verify.py script.
    """

    def evaluate(
        self,
        task: TaskSpec,
        workspace_dir: Path,
    ) -> tuple[float, bool, str]:
        """
        Evaluate completeness for a task.

        Returns (comp_score, passed, output_log).
        """
        checkpoints_passed = 0
        total_checkpoints = 0
        log_lines: list[str] = []

        # 1. Evaluate static artifact checkpoints
        for artifact in task.verification.artifacts:
            total_checkpoints += 1
            passed, msg = self._verify_artifact(artifact, workspace_dir)
            log_lines.append(msg)
            if passed:
                checkpoints_passed += 1

        # 2. Evaluate verification script if present
        script_path = None
        if task.task_dir:
            candidate = task.task_dir / task.verification.script
            if candidate.exists():
                script_path = candidate

        if script_path:
            total_checkpoints += 1
            passed, script_output = self._run_verification_script(script_path, workspace_dir)
            log_lines.append(f"Script [{script_path.name}]: {'PASSED' if passed else 'FAILED'}")
            if script_output:
                log_lines.append(f"  {script_output.strip()}")
            if passed:
                checkpoints_passed += 1

        if total_checkpoints == 0:
            # No checkpoints specified; default to 1.0 if completed
            return 1.0, True, "No verification checkpoints declared."

        comp_score = round(checkpoints_passed / total_checkpoints, 4)
        all_passed = checkpoints_passed == total_checkpoints
        return comp_score, all_passed, "\n".join(log_lines)

    def _verify_artifact(
        self,
        artifact: VerificationArtifact,
        workspace_dir: Path,
    ) -> tuple[bool, str]:
        """Check artifact file existence and textual constraints."""
        target_file = workspace_dir / artifact.path
        if not target_file.exists():
            return False, f"Artifact [{artifact.path}]: MISSING"

        try:
            content = target_file.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return False, f"Artifact [{artifact.path}]: Read error ({exc})"

        if artifact.min_lines is not None:
            line_count = len(content.splitlines())
            if line_count < artifact.min_lines:
                return (
                    False,
                    f"Artifact [{artifact.path}]: Line count {line_count} < min {artifact.min_lines}",
                )

        for required_str in artifact.must_contain:
            if required_str.lower() not in content.lower():
                return (
                    False,
                    f"Artifact [{artifact.path}]: Missing required text '{required_str}'",
                )

        return True, f"Artifact [{artifact.path}]: PASSED"

    def _run_verification_script(
        self,
        script_path: Path,
        workspace_dir: Path,
    ) -> tuple[bool, str]:
        """Execute the task verify.py script against the workspace directory."""
        # Check if script defines a verify(workspace_path) function
        try:
            spec = importlib.util.spec_from_file_location("task_verifier", script_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "verify") and callable(module.verify):
                    result = module.verify(str(workspace_dir))
                    if isinstance(result, bool):
                        return result, "verify() returned boolean"
                    if isinstance(result, tuple) and len(result) == 2:
                        return bool(result[0]), str(result[1])
        except Exception as exc:
            # Fall back to subprocess execution if import fails
            pass

        # Subprocess fallback
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(workspace_dir.parent)
            res = subprocess.run(
                [sys.executable, str(script_path), str(workspace_dir)],
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            passed = res.returncode == 0
            output = res.stdout if passed else (res.stderr or res.stdout)
            return passed, output
        except subprocess.TimeoutExpired:
            return False, "Verification script timed out after 30s"
        except Exception as exc:
            return False, f"Execution failed: {exc}"
