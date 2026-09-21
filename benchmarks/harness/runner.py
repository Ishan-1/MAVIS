"""
benchmarks/harness/runner.py
Benchmark Runner for MAVIS-Bench.
Orchestrates task loading, workspace isolation, simulated user interactions,
headless MAVIS goal execution, and multi-axis evaluation scoring.
"""
from __future__ import annotations

import asyncio
import contextlib
import datetime
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import yaml

from benchmarks.evaluators.completeness import CompletenessEvaluator
from benchmarks.evaluators.proactivity import ProactivityEvaluator
from benchmarks.evaluators.security import SecurityEvaluator
from benchmarks.evaluators.self_healing import SelfHealingEvaluator
from benchmarks.harness.mock_channel import MockChannel
from benchmarks.harness.models import (
    EpisodeResult,
    EpisodeSpec,
    PersonaProfile,
    TaskResult,
    TaskSpec,
)
from benchmarks.harness.simulated_user import SimulatedUser


@contextlib.contextmanager
def task_asset_context(ws: Path):
    """
    Temporarily exposes task-specific simulated resources (such as data/emails.json)
    at root paths expected by legacy MAVIS tools, cleaning them up upon completion.
    """
    root_data_emails = Path("data/emails.json")
    created_emails = False
    backup_content = None

    emails_file = ws / "emails.json"
    if not emails_file.exists() and (ws / "data" / "emails.json").exists():
        emails_file = ws / "data" / "emails.json"
    elif not emails_file.exists() and (ws / "inbox").exists():
        emails_list = []
        for f in sorted((ws / "inbox").glob("*.json")):
            try:
                emails_list.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        if emails_list:
            emails_file.write_text(json.dumps(emails_list, indent=2), encoding="utf-8")

    if emails_file.exists():
        if root_data_emails.exists():
            backup_content = root_data_emails.read_text(encoding="utf-8")
        root_data_emails.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(emails_file, root_data_emails)
        created_emails = True

    root_workspace = Path("workspace")
    created_workspace_symlink = False
    temp_backup_ws = None

    if root_workspace.is_symlink():
        try:
            root_workspace.unlink()
        except OSError:
            pass
    elif root_workspace.exists() and root_workspace.is_dir():
        temp_backup_ws = Path("workspace_backup_benchmark")
        if temp_backup_ws.exists():
            shutil.rmtree(temp_backup_ws, ignore_errors=True)
        root_workspace.rename(temp_backup_ws)

    ws_abs = str(ws.resolve())
    prev_ws = os.environ.get("MAVIS_ACTIVE_WORKSPACE")
    os.environ["MAVIS_ACTIVE_WORKSPACE"] = ws_abs

    try:
        root_workspace.symlink_to(ws.resolve(), target_is_directory=True)
        created_workspace_symlink = True
    except Exception:
        pass

    try:
        yield
    finally:
        if prev_ws is not None:
            os.environ["MAVIS_ACTIVE_WORKSPACE"] = prev_ws
        elif "MAVIS_ACTIVE_WORKSPACE" in os.environ:
            del os.environ["MAVIS_ACTIVE_WORKSPACE"]

        if created_workspace_symlink and (root_workspace.is_symlink() or root_workspace.exists()):
            try:
                root_workspace.unlink()
            except OSError:
                pass
        if temp_backup_ws is not None and temp_backup_ws.exists():
            try:
                temp_backup_ws.rename(root_workspace)
            except Exception:
                pass
        if created_emails:
            if backup_content is not None:
                root_data_emails.write_text(backup_content, encoding="utf-8")
            elif root_data_emails.exists():
                try:
                    root_data_emails.unlink()
                except OSError:
                    pass


class BenchmarkRunner:
    """
    Main orchestrator for running MAVIS-Bench episodes and tasks.
    """

    def __init__(
        self,
        bench_root: Path | None = None,
        output_dir: Path | None = None,
        max_turns: int = 30,
        turn_timeout: float = 60.0,
        llm_client: Any | None = None,
        mavis_runner_fn: Callable[[str, Path], dict[str, Any]] | None = None,
    ) -> None:
        self.bench_root = bench_root or Path("benchmarks")
        self.data_dir = self.bench_root / "data"
        self.output_dir = output_dir or (self.bench_root / "outputs")
        self.max_turns = max_turns
        self.turn_timeout = turn_timeout
        self.llm_client = llm_client
        self.mavis_runner_fn = mavis_runner_fn

        # Evaluator components
        self.proc_evaluator = ProactivityEvaluator(llm_client=llm_client)
        self.comp_evaluator = CompletenessEvaluator()
        self.heal_evaluator = SelfHealingEvaluator()
        self.safe_evaluator = SecurityEvaluator()

    def load_task(self, persona: str, task_id: str) -> TaskSpec:
        """Load a single task.yaml spec from benchmarks/data/<persona>/tasks/<task_id>/."""
        task_dir = self.data_dir / persona / "tasks" / task_id
        task_yaml = task_dir / "task.yaml"
        if not task_yaml.exists():
            raise FileNotFoundError(f"Task definition not found: {task_yaml}")

        with open(task_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return TaskSpec.from_dict(data, task_dir=task_dir)

    def load_episode(self, persona: str) -> tuple[EpisodeSpec, PersonaProfile]:
        """Load episode.yaml and profile.yaml for a persona."""
        persona_dir = self.data_dir / persona
        episode_yaml = persona_dir / "episode.yaml"
        profile_yaml = persona_dir / "profile.yaml"

        if not episode_yaml.exists():
            raise FileNotFoundError(f"Episode file missing: {episode_yaml}")

        with open(episode_yaml, "r", encoding="utf-8") as f:
            ep_data = yaml.safe_load(f)
        episode = EpisodeSpec.from_dict(ep_data)

        profile = None
        if profile_yaml.exists():
            with open(profile_yaml, "r", encoding="utf-8") as f:
                pr_data = yaml.safe_load(f)
            profile = PersonaProfile.from_dict(pr_data)
        else:
            profile = PersonaProfile(
                persona_id=persona,
                name=persona.replace("_", " ").title(),
                description=f"Persona {persona}",
                default_workspace="workspace",
            )

        return episode, profile

    def prepare_workspace(self, persona: str, task: TaskSpec) -> Path:
        """Set up a clean, isolated workspace for the task with dynamic date substitution."""
        workspace_dir = self.bench_root / "workspaces" / persona / task.task_id
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir, ignore_errors=True)
        workspace_dir.mkdir(parents=True, exist_ok=True)

        today_dt = datetime.date.today()
        now_dt = datetime.datetime.now()
        week_start_dt = today_dt - datetime.timedelta(days=today_dt.weekday())

        replacements = {
            "{{today}}": today_dt.isoformat(),
            "{{yesterday}}": (today_dt - datetime.timedelta(days=1)).isoformat(),
            "{{tomorrow}}": (today_dt + datetime.timedelta(days=1)).isoformat(),
            "{{this_week_start}}": week_start_dt.isoformat(),
            "{{now}}": now_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "{{current_year}}": str(today_dt.year),
            "{{current_month}}": f"{today_dt.month:02d}",
            "{{current_day}}": f"{today_dt.day:02d}",
        }

        # Copy any initial task assets if present with dynamic templating
        if task.task_dir:
            assets_dir = task.task_dir / "assets"
            if assets_dir.exists():
                for item in assets_dir.rglob("*"):
                    if item.is_file():
                        rel = item.relative_to(assets_dir)
                        dest = workspace_dir / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if item.suffix in (".json", ".yaml", ".yml", ".md", ".txt", ".py", ".csv"):
                            try:
                                content = item.read_text(encoding="utf-8")
                                for placeholder, val in replacements.items():
                                    content = content.replace(placeholder, val)
                                dest.write_text(content, encoding="utf-8")
                            except Exception:
                                shutil.copy2(item, dest)
                        else:
                            shutil.copy2(item, dest)

        # Pre-compile inbox into emails.json if inbox folder exists
        inbox_dir = workspace_dir / "inbox"
        if inbox_dir.exists():
            emails_list = []
            for f in sorted(inbox_dir.glob("*.json")):
                try:
                    emails_list.append(json.loads(f.read_text(encoding="utf-8")))
                except Exception:
                    pass
            if emails_list:
                (workspace_dir / "emails.json").write_text(json.dumps(emails_list, indent=2), encoding="utf-8")
                data_sub = workspace_dir / "data"
                data_sub.mkdir(parents=True, exist_ok=True)
                (data_sub / "emails.json").write_text(json.dumps(emails_list, indent=2), encoding="utf-8")

        return workspace_dir

    async def run_task(
        self,
        task: TaskSpec,
        profile: PersonaProfile,
        workspace_dir: Path | None = None,
    ) -> TaskResult:
        """
        Execute an end-to-end task run between the simulated user and MAVIS.
        """
        ws = workspace_dir or self.prepare_workspace(task.persona, task)
        channel = MockChannel(default_timeout=self.turn_timeout)
        simulated_user = SimulatedUser(task, profile, llm_client=self.llm_client)

        channel.set_runtime_identity(sender_id=task.persona, chat_id=task.task_id)
        task_start = time.perf_counter()

        turns_taken = 0
        waves_taken = 1
        status = "IN_PROGRESS"
        execution_trace: list[dict[str, Any]] = []
        command_logs: list[dict[str, Any]] = []

        with task_asset_context(ws):
            # Clear MAVIS working memory so each task starts in clean isolation
            from memories.memory_store import clear_all_working_memories
            clear_all_working_memories()

            # Send initial message
            initial_prompt = simulated_user.get_initial_message()
            await channel.send_user_message(initial_prompt)

            while turns_taken < self.max_turns:
                turns_taken += 1
                user_msg = await channel.receive_user_message(timeout=self.turn_timeout)

                # Delegate execution to MAVIS runner function or simulate response
                if self.mavis_runner_fn is not None:
                    mavis_out = self.mavis_runner_fn(user_msg, ws)
                    reply_text = str(mavis_out.get("reply", "Task completed."))
                    waves_taken = int(mavis_out.get("waves", waves_taken))
                    execution_trace.extend(mavis_out.get("trace", []))
                    command_logs.extend(mavis_out.get("commands", []))
                else:
                    # Dry-run placeholder when no live runner attached
                    reply_text = f"Received: {user_msg}. Processing completed."

                await channel.send_agent_reply(reply_text)

                # 1. Early-exit check: If all task artifacts & verification checks already passed, complete!
                _, verify_passed_early, _ = self.comp_evaluator.evaluate(task, ws)
                if verify_passed_early:
                    status = "SUCCESS"
                    print(f"\n[MAVIS-Bench] Task {task.task_id} verification PASSED on turn {turns_taken}. Completing task.")
                    break

                # 2. Check next action from simulated user
                action = await simulated_user.next_action(reply_text)
                if action.action_type == "terminate" or not action.message.strip():
                    status = "SUCCESS"
                    break

                await channel.send_user_message(action.message)

        if status == "IN_PROGRESS":
            status = "MAX_TURNS"

        duration = round(time.perf_counter() - task_start, 2)

        # Run multi-axis evaluations
        print(f"\n[MAVIS-Bench] Evaluating multi-axis metrics for {task.task_id}...")
        channel_history = channel.get_history()
        print("  → Computing Proactivity (PROC)...")
        proc_score, satisfied_intents = self.proc_evaluator.evaluate(task, channel_history)
        print("  → Computing Task Completeness (COMP)...")
        comp_score, verify_passed, verify_output = self.comp_evaluator.evaluate(task, ws)
        print("  → Computing Self-Healing (HEAL)...")
        heal_score, heal_logs = self.heal_evaluator.evaluate(task, execution_trace)
        print("  → Computing Safety (SAFE)...")
        safe_score, safe_logs = self.safe_evaluator.evaluate(
            task, command_logs, execution_trace=execution_trace
        )
        print("  ✓ Evaluations complete.")
        task_status = "SUCCESS" if verify_passed else ("MAX_TURNS" if status == "MAX_TURNS" else "FAILED")

        result = TaskResult(
            task_id=task.task_id,
            persona=task.persona,
            status=task_status,
            turns_taken=turns_taken,
            waves_taken=waves_taken,
            proc_score=proc_score,
            comp_score=comp_score,
            heal_score=heal_score,
            safe_score=safe_score,
            satisfied_intents=satisfied_intents,
            verification_passed=verify_passed,
            verification_output=verify_output,
            messages=channel_history,
            telemetry={
                "heal_logs": heal_logs,
                "safe_logs": safe_logs,
            },
            duration_seconds=duration,
        )

        self._save_task_result(result)
        return result

    async def run_episode(self, persona: str) -> EpisodeResult:
        """Execute all tasks in an episode for a given persona."""
        episode_spec, profile = self.load_episode(persona)
        ep_start = time.perf_counter()
        task_results: list[TaskResult] = []

        for task_meta in episode_spec.tasks:
            t_id = task_meta["task_id"]
            task_spec = self.load_task(persona, t_id)
            res = await self.run_task(task_spec, profile)
            task_results.append(res)

        ep_result = EpisodeResult(
            episode_id=episode_spec.episode_id,
            persona=persona,
            task_results=task_results,
            duration_seconds=round(time.perf_counter() - ep_start, 2),
        )
        ep_result.calculate_averages()
        self._save_episode_result(ep_result)
        return ep_result

    def _save_task_result(self, result: TaskResult) -> None:
        """Write task scorecard JSON."""
        dest_dir = self.output_dir / result.persona / "tasks"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"{result.task_id}.json"
        with open(dest_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

    def _save_episode_result(self, result: EpisodeResult) -> None:
        """Write episode scorecard JSON."""
        dest_dir = self.output_dir / result.persona
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / f"episode_summary.json"
        with open(dest_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
