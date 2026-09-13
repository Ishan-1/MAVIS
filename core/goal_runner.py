"""
core/goal_runner.py

Autonomous Goal Runner for MAVIS (/goal).
Executes high-level autonomous goals through iterative waves of DAG pipelines.
Each wave is planned dynamically based on previous execution outcomes,
guaranteeing mathematical acyclicity (no cyclic back-edges) while supporting
multi-iteration loops and gate-based conditional execution up to max_iterations.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from typing import Any, Callable

from core.helpers import log_it
from core.llm import BaseLLMClient, get_llm_client
from core.metrics import MetricEmitter
from memories.memory_store import MemoryStore
from oni import oni as _oni
from core.output import mavis_error, mavis_ok, mavis_print, mavis_status

_ENTITY = "goal_runner"
_EMITTER = MetricEmitter("goal_runner")

GOAL_PLANNER_SYSTEM_PROMPT = """You are the MAVIS Autonomous Goal Planner.
Your job is to achieve a high-level user goal by planning and executing iterative waves of Directed Acyclic Graph (DAG) pipelines.
In each iteration, analyze what has already been accomplished and plan the NEXT wave of actions.

Each wave is an acyclic list of steps. Steps can be:
1. "tool": standard tool invocation with "function_name" and "params".
2. "subagent": delegate complex sub-tasks to specialized subagents.
3. "gate": conditional check with:
   - "type": "gate"
   - "mode": "deterministic" or "nlp"
   - "condition": e.g. "$step1.status == 0" or "len($step1.output) > 0"
   - "if_true": "next_step_id"
   - "if_false": "alternative_step_id" or null

If the goal is fully achieved, set "status": "completed" and "pipeline": [].
If further steps are required, set "status": "in_progress" and provide the "pipeline".
If unrecoverable blockers prevent finishing, set "status": "blocked" and "pipeline": [].

Output schema:
{
  "status": "in_progress" | "completed" | "blocked",
  "reasoning": "What has been achieved so far and why this wave is planned.",
  "pipeline": [ ... list of DAG nodes ... ],
  "final_summary": "If status is completed or blocked, provide the final report."
}

Respond ONLY with valid JSON.
"""


class GoalRunner:
    """
    Orchestrates autonomous multi-wave goal execution with task-scoped ONI lease.
    """

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        execute_pipeline_fn: Callable[..., Any] | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.llm = llm or get_llm_client()
        self.execute_pipeline = execute_pipeline_fn
        self.memory_store = memory_store or MemoryStore(self.llm, namespace="interpreter")

    def plan_wave(
        self,
        goal: str,
        iteration: int,
        max_iterations: int,
        history: list[dict[str, Any]],
        commands_list: dict[str, Any] | None = None,
        agents_list: list[str] | None = None,
    ) -> dict[str, Any]:
        """Ask LLM to plan the next execution wave based on progress so far."""
        history_summary = ""
        for h in history:
            history_summary += f"\n--- Wave {h.get('wave')} ---\n"
            history_summary += f"Reasoning: {h.get('reasoning')}\n"
            history_summary += f"Outcome: {h.get('outcome')}\n"

        user_content = (
            f"GOAL: {goal}\n"
            f"CURRENT ITERATION: {iteration} of {max_iterations}\n\n"
            f"HISTORY OF PREVIOUS WAVES:\n{history_summary or 'None (initial wave)'}\n\n"
            f"AVAILABLE COMMANDS:\n{json.dumps(commands_list or {}, indent=2)[:2000]}\n\n"
            f"AVAILABLE AGENTS:\n{json.dumps(agents_list or [], indent=2)}\n\n"
            "Plan the next wave DAG or declare the goal completed/blocked."
        )

        try:
            raw_response = self.llm.chat(
                messages=[
                    {"role": "system", "content": GOAL_PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            # Parse JSON
            m = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if m:
                return json.loads(m.group(0))
        except Exception as e:
            log_it(_ENTITY, f"Goal wave planning failed: {e}", level="WARN")

        return {
            "status": "blocked",
            "reasoning": "Failed to generate valid plan for next wave.",
            "pipeline": [],
            "final_summary": f"Planning failed on iteration {iteration}.",
        }

    def run_goal(
        self,
        goal: str,
        max_iterations: int = 8,
        turn_id: str | None = None,
        commands_list: dict[str, Any] | None = None,
        agents_list: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Execute an autonomous goal to completion or until max_iterations.
        """
        goal_id = turn_id or f"goal_{int(time.time())}"
        mavis_print(f"\n[bold cyan]═══ AUTONOMOUS GOAL RUNNER ═══[/bold cyan]")
        mavis_print(f"[bold]Goal:[/bold] {goal}")
        mavis_print(f"[dim]Max Waves:[/dim] {max_iterations}  [dim]Task Lease:[/dim] ONI yolo lease active\n")

        # Acquire task-scoped lease
        prev_lease = _oni.acquire_task_lease(f"goal_{goal_id}", lease_trust="yolo")

        history: list[dict[str, Any]] = []
        status = "in_progress"
        final_summary = ""
        total_steps_executed = 0
        start_time = datetime.now()

        try:
            for iteration in range(1, max_iterations + 1):
                mavis_status(f"[Wave {iteration}/{max_iterations}] Planning next actions...")

                wave_plan = self.plan_wave(
                    goal=goal,
                    iteration=iteration,
                    max_iterations=max_iterations,
                    history=history,
                    commands_list=commands_list,
                    agents_list=agents_list,
                )

                status = wave_plan.get("status", "in_progress")
                reasoning = wave_plan.get("reasoning", "")
                pipeline = wave_plan.get("pipeline", [])
                plan_summary = wave_plan.get("final_summary", "")

                mavis_status(f"[Wave {iteration}] Plan: {reasoning}")

                if status == "completed":
                    mavis_ok(f"Goal achieved on Wave {iteration}!")
                    final_summary = plan_summary or reasoning
                    history.append({
                        "wave": iteration,
                        "reasoning": reasoning,
                        "steps_count": 0,
                        "outcome": "Goal completed.",
                        "status": "completed",
                    })
                    break

                if status == "blocked" and not pipeline:
                    mavis_error(f"Goal blocked on Wave {iteration}: {plan_summary or reasoning}")
                    final_summary = plan_summary or reasoning
                    break

                if not pipeline:
                    mavis_ok("Wave plan produced no further pipeline steps. Concluding goal.")
                    status = "completed"
                    final_summary = plan_summary or reasoning
                    break

                # Execute wave pipeline
                wave_turn_id = f"{goal_id}_wave{iteration}"
                wave_results = None
                if self.execute_pipeline:
                    wave_results = self.execute_pipeline(
                        pipeline=pipeline,
                        query=goal,
                        turn_id=wave_turn_id,
                    )

                step_count = len(pipeline)
                total_steps_executed += step_count
                wave_outcome = "success" if wave_results is not None else "failed"

                # Compact outcome for history
                outcome_snippet = ""
                if wave_results:
                    for nid, res in wave_results.items():
                        snippet = str(res)[:120].replace("\n", " ")
                        outcome_snippet += f"[{nid}: {snippet}] "
                else:
                    outcome_snippet = "Pipeline execution encountered unrecoverable step failure."

                history.append({
                    "wave": iteration,
                    "reasoning": reasoning,
                    "steps_count": step_count,
                    "outcome": outcome_snippet,
                    "status": wave_outcome,
                })

                mavis_ok(f"Wave {iteration} completed ({step_count} steps executed).")

            if status != "completed" and iteration >= max_iterations:
                mavis_status(f"Reached max iterations limit ({max_iterations}). Finalizing.")
                status = "max_iterations_reached"
                final_summary = final_summary or f"Completed {max_iterations} execution waves."

        finally:
            # Release task lease
            _oni.release_task_lease(f"goal_{goal_id}", prev_lease)

        # Record to memory store
        try:
            self.memory_store.add_to_working_memory(
                role="goal_runner",
                content=f"Goal '{goal}' concluded with status '{status}'. {final_summary}",
                turn_id=goal_id,
            )
        except Exception:
            pass

        _EMITTER.log({
            "goal_id": goal_id,
            "goal": goal,
            "status": status,
            "waves_executed": len(history),
            "total_steps": total_steps_executed,
            "duration_s": (datetime.now() - start_time).total_seconds(),
        })

        mavis_print(f"\n[bold green]═══ GOAL RUNNER COMPLETED ═══[/bold green]")
        mavis_print(f"[bold]Status:[/bold] {status}")
        mavis_print(f"[bold]Summary:[/bold] {final_summary}\n")

        return {
            "goal_id": goal_id,
            "status": status,
            "waves": len(history),
            "total_steps": total_steps_executed,
            "summary": final_summary,
            "history": history,
        }
