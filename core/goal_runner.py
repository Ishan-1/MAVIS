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
from pathlib import Path
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
In each iteration, analyze what has already been accomplished from previous waves and plan the NEXT wave of actions.

Wave Planning Rules:
1. Each wave is an acyclic DAG of steps. Steps execute in topological dependency order.
2. Step Types & Formats:
   - "tool": Deterministic environment actions (filesystem, shell, APIs).
     Format: {"id": "step_id", "type": "tool", "function_name": "<name from AVAILABLE COMMANDS or missing_commands>", "params": {...}}
   - "cognitive": 1-shot in-memory semantic processing (summarization, extraction, parsing, code quality analysis).
     Format: {"id": "step_id", "type": "cognitive", "function_name": "semantic_transform", "params": {"content": <data or $dep.output>, "instruction": "<what to analyze/extract>"}}
     CRITICAL: Always use "function_name": "semantic_transform" (or a registered agent from AVAILABLE AGENTS or missing_agents). Never leave function_name blank and never use arbitrary undeclared names.
   - "subagent": Iterative multi-turn ReAct tool-calling loops.
     Format: {"id": "step_id", "type": "subagent", "function_name": "<agent from AVAILABLE AGENTS or missing_agents>", "params": {...}, "max_turns": 4}
   - "gate": Conditional branch check:
     Format: {"id": "step_id", "type": "gate", "mode": "deterministic" | "nlp", "condition": "<e.g. $s1.status == 0>", "if_true": "step_ok", "if_false": "step_alt"}
   - "control": Verification post-condition check to confirm goal milestone:
     Format: {"id": "step_id", "type": "control", "mode": "deterministic" | "nlp", "condition": "<condition>", "expected_outcome": "<expected>", "on_failure": "trigger_debugger" | "report_failure"}

3. CRITICAL Dependency Rule:
   - Step parameter references (e.g. "$step1.output") can ONLY reference step IDs defined within the CURRENT wave's "pipeline".
   - NEVER reference step IDs from previous waves. If you need data from a previous wave, read the outcome from HISTORY OF PREVIOUS WAVES, pass the literal values in params, or re-read the relevant files.

4. In-Place Tool/Agent Evolution (Anti-Proliferation):
   - Check AVAILABLE COMMANDS and AVAILABLE AGENTS first. If an existing tool or agent can be enhanced with an added parameter, edge-case handling, or refined prompt, PREFER UPDATING IT in-place rather than creating a duplicate new tool.
   - To update an existing tool in-place:
     "update_commands": [
       {"tool_name": "read_file_contents", "requested_changes": "Add optional create_if_missing boolean parameter"}
     ]
   - To update an existing agent in-place:
     "update_agents": [
       {"agent_name": "...", "requested_changes": "..."}
     ]
   - Any new parameters will be added with backwards-compatible defaults, preserving previous functionality while adding the new capability.

5. Synthesizing Missing Capabilities:
   - If achieving the goal requires an entirely new capability not present in AVAILABLE COMMANDS, declare it in "missing_commands":
     "missing_commands": [
       {"signature": "func_name(param1: type) -> tuple[int, return_type]", "description": "What it does."}
     ]
     The system will automatically generate, test, and register the tool before executing the wave pipeline, so you can immediately call it in your "pipeline".
   - If achieving the goal requires a specialized subagent not in AVAILABLE AGENTS, declare it in "missing_agents":
     "missing_agents": [
       {"name": "agent_name", "type": "cognitive" | "subagent", "description": "...", "input_schema": {...}}
     ]

6. Goal Scratchpad (Blackboard):
   - You have a persistent scratchpad available across all waves shown in CURRENT GOAL SCRATCHPAD.
   - It contains:
     * iteration_state: persistent iteration state (e.g. current_index, processed_items, remaining_backlog).
     * artifacts: verified paths to files generated in previous waves or offloaded to disk. NEVER hallucinate file paths! Always check artifacts or HISTORY OF PREVIOUS WAVES for actual file paths.
     * notes_and_findings: key discoveries and notes across waves.
   - You can update the scratchpad in your response via "scratchpad_updates":
     "scratchpad_updates": {
       "iteration_state": { "current_index": 2, "backlog": [...] },
       "artifacts": { "checkpoint": {"path": "data/scratch/...", "description": "..."} },
       "notes_and_findings": ["Discovered syntax error in line 45"]
     }

If the goal is fully achieved, set "status": "completed" and "pipeline": [].
If further steps are required, set "status": "in_progress" and provide the "pipeline".
If unrecoverable blockers prevent finishing, set "status": "blocked" and "pipeline": [].

Output schema:
{
  "status": "in_progress" | "completed" | "blocked",
  "reasoning": "What has been achieved so far and why this wave is planned.",
  "scratchpad_updates": { ... optional state updates ... },
  "update_commands": [ ... optional list of tools to evolve in-place ... ],
  "update_agents": [ ... optional list of agents to evolve in-place ... ],
  "missing_commands": [ ... optional list of new tools to build ... ],
  "missing_agents": [ ... optional list of new agents to build ... ],
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
        tool_builder: Any | None = None,
        agent_builder: Any | None = None,
        tool_retriever: Any | None = None,
    ) -> None:
        self.llm = llm or get_llm_client()
        self.execute_pipeline = execute_pipeline_fn
        self.memory_store = memory_store or MemoryStore(self.llm, namespace="interpreter")
        self.tool_builder = tool_builder
        self.agent_builder = agent_builder
        self.tool_retriever = tool_retriever

    # ── Scratchpad & Blackboard Management ────────────────────────────────────

    def _get_scratchpad_path(self, goal_id: str) -> Path:
        p = Path("data/scratch/goals") / goal_id / "scratchpad.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _init_scratchpad(self, goal_id: str, goal: str) -> dict[str, Any]:
        """Initialize or load the persistent cross-wave goal scratchpad."""
        sp_path = self._get_scratchpad_path(goal_id)
        if sp_path.exists():
            try:
                with open(sp_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        sp = {
            "goal_id": goal_id,
            "goal": goal,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "iteration_state": {},
            "artifacts": {},
            "notes_and_findings": [],
        }
        self._save_scratchpad(goal_id, sp)
        return sp

    def _save_scratchpad(self, goal_id: str, scratchpad: dict[str, Any]) -> None:
        """Persist scratchpad to disk."""
        sp_path = self._get_scratchpad_path(goal_id)
        try:
            with open(sp_path, "w", encoding="utf-8") as f:
                json.dump(scratchpad, f, indent=2, default=str)
        except Exception as e:
            log_it(f"Failed to save scratchpad for {goal_id}: {e}", _ENTITY)

    def _update_scratchpad(
        self,
        goal_id: str,
        scratchpad: dict[str, Any],
        updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge wave updates into persistent scratchpad."""
        if not isinstance(updates, dict):
            return scratchpad

        if "iteration_state" in updates and isinstance(updates["iteration_state"], dict):
            scratchpad.setdefault("iteration_state", {}).update(updates["iteration_state"])

        if "artifacts" in updates and isinstance(updates["artifacts"], dict):
            scratchpad.setdefault("artifacts", {}).update(updates["artifacts"])

        if "notes_and_findings" in updates:
            new_notes = updates["notes_and_findings"]
            if isinstance(new_notes, list):
                scratchpad.setdefault("notes_and_findings", []).extend(new_notes)
            elif isinstance(new_notes, str):
                scratchpad.setdefault("notes_and_findings", []).append(new_notes)

        for k, v in updates.items():
            if k not in ("iteration_state", "artifacts", "notes_and_findings"):
                scratchpad[k] = v

        scratchpad["updated_at"] = datetime.now().isoformat()
        self._save_scratchpad(goal_id, scratchpad)
        return scratchpad

    def _register_artifacts_from_results(
        self,
        goal_id: str,
        scratchpad: dict[str, Any],
        wave: int,
        wave_results: dict[str, Any] | None,
    ) -> None:
        """Scan wave_results for any spilled scratchpad files and auto-register them."""
        if not wave_results or not isinstance(wave_results, dict):
            return

        artifacts = scratchpad.setdefault("artifacts", {})
        updated = False

        for nid, res in wave_results.items():
            res_str = str(res)
            matches = re.findall(r"data/scratch/[a-zA-Z0-9_\-\./]+", res_str)
            for m in set(matches):
                clean_path = m.rstrip(".,;)]}\"'")
                p = Path(clean_path)
                if p.exists() and p.is_file():
                    key = f"wave_{wave}_{nid}"
                    artifacts[key] = {
                        "path": str(p),
                        "wave": wave,
                        "node_id": nid,
                        "size_bytes": p.stat().st_size,
                        "description": f"Output artifact from step '{nid}' in wave {wave}",
                    }
                    updated = True

        if updated:
            scratchpad["updated_at"] = datetime.now().isoformat()
            self._save_scratchpad(goal_id, scratchpad)

    # ── Wave Planning ─────────────────────────────────────────────────────────

    def plan_wave(
        self,
        goal: str,
        iteration: int,
        max_iterations: int,
        history: list[dict[str, Any]],
        scratchpad: dict[str, Any] | None = None,
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
            f"CURRENT GOAL SCRATCHPAD (BLACKBOARD):\n{json.dumps(scratchpad or {}, indent=2)}\n\n"
            f"HISTORY OF PREVIOUS WAVES:\n{history_summary or 'None (initial wave)'}\n\n"
            f"AVAILABLE COMMANDS:\n{json.dumps(commands_list or {}, indent=2)[:2000]}\n\n"
            f"AVAILABLE AGENTS:\n{json.dumps(agents_list or [], indent=2)}\n\n"
            "Plan the next wave DAG, specify scratchpad updates, or declare the goal completed/blocked."
        )

        try:
            raw_response = self.llm.generate(
                prompt=user_content,
                json_mode=True,
                system_instruction=GOAL_PLANNER_SYSTEM_PROMPT,
                temperature=0.2,
            )

            # Parse JSON
            try:
                return json.loads(raw_response)
            except Exception:
                m = re.search(r"\{.*\}", raw_response, re.DOTALL)
                if m:
                    return json.loads(m.group(0))
        except Exception as e:
            log_it(f"Goal wave planning failed: {e}", _ENTITY)


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
        lease_trust: str | None = None,
        commands_list: dict[str, Any] | None = None,
        agents_list: list[str] | None = None,
        auto_confirm_lease: bool = False,
    ) -> dict[str, Any]:
        """
        Execute an autonomous goal to completion or until max_iterations.
        Defaults to the session's active trust level. If 'yolo' is explicitly
        requested, requires an upfront user confirmation gate before wave 1.
        """
        goal_id = turn_id or f"goal_{int(time.time())}"
        effective_lease = lease_trust or _oni._effective_trust()

        mavis_print(f"\n[bold cyan]═══ AUTONOMOUS GOAL RUNNER ═══[/bold cyan]")
        mavis_print(f"[bold]Goal:[/bold] {goal}")
        mavis_print(f"[dim]Max Waves:[/dim] {max_iterations}  [dim]Execution Trust:[/dim] [bold]{effective_lease}[/bold]")

        # If elevated unattended access ('yolo') is requested, prompt for explicit upfront consent
        if effective_lease == "yolo" and not auto_confirm_lease:
            desc = (
                f"AUTONOMOUS GOAL EXECUTION:\n"
                f"Goal: '{goal}'\n"
                f"This goal requests unattended 'yolo' permissions (no per-step confirmation) for up to {max_iterations} waves."
            )
            mavis_print(f"[bold yellow]⚠️  ONI Security Gate: Elevated 'yolo' lease requested for autonomous goal.[/bold yellow]")
            approved = _oni.gate.request_approval(desc)
            if not approved:
                mavis_error("Goal execution aborted: Elevated 'yolo' task lease was declined by user.\n")
                return {
                    "goal_id": goal_id,
                    "status": "cancelled",
                    "waves": 0,
                    "total_steps": 0,
                    "summary": "Cancelled: 'yolo' task lease declined by user.",
                    "history": [],
                }
            mavis_ok("Elevated 'yolo' task lease approved by user for this goal run.\n")
        else:
            mavis_print(f"[dim]Task Lease: Running under '{effective_lease}' trust level.[/dim]\n")

        # Acquire task-scoped lease
        prev_lease = _oni.acquire_task_lease(f"goal_{goal_id}", lease_trust=effective_lease)

        scratchpad = self._init_scratchpad(goal_id=goal_id, goal=goal)
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
                    scratchpad=scratchpad,
                    commands_list=commands_list,
                    agents_list=agents_list,
                )

                status = wave_plan.get("status", "in_progress")
                reasoning = wave_plan.get("reasoning", "")
                pipeline = wave_plan.get("pipeline", [])
                plan_summary = wave_plan.get("final_summary", "")

                # Update scratchpad if updates provided
                sp_updates = wave_plan.get("scratchpad_updates")
                if sp_updates and isinstance(sp_updates, dict):
                    scratchpad = self._update_scratchpad(goal_id, scratchpad, sp_updates)

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

                # 1. Evolve existing tools in-place if requested
                update_tools = wave_plan.get("update_commands", [])
                if update_tools and self.tool_builder:
                    mavis_status(f"[Wave {iteration}] Evolving {len(update_tools)} existing tool(s)...")
                    for u_tool in update_tools:
                        t_name = u_tool.get("tool_name", "")
                        req_changes = u_tool.get("requested_changes", "")
                        if not t_name:
                            continue
                        mavis_status(f"Updating tool '{t_name}': {req_changes[:80]}...")
                        try:
                            gen_score = self.tool_builder.update_tool(t_name, req_changes) or "repurposable"
                            mavis_ok(f"Updated tool '{t_name}' ({gen_score}).")
                        except Exception as ute:
                            mavis_error(f"Couldn't update tool '{t_name}': {ute}. See logs/tool_builder.log.")

                # 2. Evolve existing agents in-place if requested
                update_agents = wave_plan.get("update_agents", [])
                if update_agents and self.agent_builder:
                    mavis_status(f"[Wave {iteration}] Evolving {len(update_agents)} existing agent(s)...")
                    for u_agent in update_agents:
                        ag_name = u_agent.get("agent_name", "").strip().lower()
                        req_changes = u_agent.get("requested_changes", "")
                        if not ag_name:
                            continue
                        mavis_status(f"Updating agent '{ag_name}'...")
                        try:
                            self.agent_builder.update_agent(
                                agent_name=ag_name,
                                requested_changes=req_changes,
                                available_tools=commands_list or {},
                            )
                            mavis_ok(f"Updated agent '{ag_name}'.")
                        except Exception as uae:
                            mavis_error(f"Couldn't update agent '{ag_name}': {uae}. See logs/agent_builder.log.")

                # 3. Synthesize missing tools if requested
                missing_tools = wave_plan.get("missing_commands", [])
                if missing_tools and self.tool_builder:
                    mavis_status(f"[Wave {iteration}] Synthesizing {len(missing_tools)} missing tool(s)...")
                    for new_tool in missing_tools:
                        signature = new_tool.get("signature", "")
                        description = new_tool.get("description", "")
                        if not signature:
                            continue
                        func_name = signature.split("(")[0].strip()
                        mavis_status(f"Building tool '{func_name}': {signature}")
                        try:
                            gen_score = self.tool_builder.build_tool(signature, description) or "repurposable"
                            if commands_list is not None:
                                commands_list[func_name] = {
                                    "description": description,
                                    "generalizability": gen_score,
                                }
                            if self.tool_retriever:
                                self.tool_retriever.index_tool(signature, description, generalizability=gen_score)
                            mavis_ok(f"Built '{func_name}'.")
                        except Exception as te:
                            mavis_error(f"Couldn't build '{func_name}': {te}. See logs/tool_builder.log.")

                # 4. Synthesize missing agents if requested
                missing_agents = wave_plan.get("missing_agents", [])
                if missing_agents and self.agent_builder:
                    mavis_status(f"[Wave {iteration}] Synthesizing {len(missing_agents)} missing agent(s)...")
                    for new_ag in missing_agents:
                        ag_name = new_ag.get("name", "").strip().lower()
                        ag_type = new_ag.get("type", "cognitive")
                        ag_desc = new_ag.get("description", "")
                        ag_schema = new_ag.get("input_schema", {})
                        ag_turns = new_ag.get("default_max_turns", 4)
                        ag_allowed = new_ag.get("allowed_tools", None)
                        if not ag_name:
                            continue
                        mavis_status(f"Building agent '{ag_name}' ({ag_type})...")
                        try:
                            self.agent_builder.build_agent(
                                agent_name=ag_name,
                                agent_description=ag_desc,
                                input_schema=ag_schema,
                                agent_type=ag_type,
                                default_max_turns=ag_turns,
                                allowed_tools=ag_allowed,
                                available_tools=commands_list or {},
                            )
                            if agents_list is not None and ag_name not in agents_list:
                                agents_list.append(ag_name)
                            mavis_ok(f"Built agent '{ag_name}'.")
                        except Exception as ae:
                            mavis_error(f"Couldn't build agent '{ag_name}': {ae}. See logs/agent_builder.log.")

                # Execute wave pipeline
                wave_turn_id = f"{goal_id}_wave{iteration}"
                wave_results = None
                if self.execute_pipeline:
                    try:
                        wave_results = self.execute_pipeline(
                            pipeline=pipeline,
                            query=goal,
                            turn_id=wave_turn_id,
                            is_goal_wave=True,
                        )
                    except TypeError:
                        wave_results = self.execute_pipeline(
                            pipeline=pipeline,
                            query=goal,
                            turn_id=wave_turn_id,
                        )



                # Auto-register any artifacts spilled or produced in this wave
                self._register_artifacts_from_results(goal_id, scratchpad, iteration, wave_results)

                step_count = len(pipeline)
                total_steps_executed += step_count
                wave_outcome = "success" if wave_results is not None else "failed"

                # Check if wave had a control node
                control_passed = None
                if wave_results:
                    for nid, res in wave_results.items():
                        if isinstance(res, dict) and res.get("node_type") == "control":
                            control_passed = bool(res.get("success"))
                            if control_passed:
                                mavis_ok(f"[Wave {iteration}] Post-condition verified: {res.get('reason')}")
                            else:
                                mavis_status(f"[Wave {iteration}] Post-condition unmet: {res.get('reason')}")

                # Compact outcome for history
                outcome_snippet = ""
                if wave_results:
                    for nid, res in wave_results.items():
                        snippet = str(res)[:300].replace("\n", " ")
                        outcome_snippet += f"[{nid}: {snippet}] "
                else:
                    outcome_snippet = "Pipeline execution encountered unrecoverable step failure."


                history.append({
                    "wave": iteration,
                    "reasoning": reasoning,
                    "steps_count": step_count,
                    "outcome": outcome_snippet,
                    "control_verified": control_passed,
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
            "scratchpad": scratchpad,
        }

