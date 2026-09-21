"""
agent_builder/agent_builder.py
AgentBuilder: Symmetrically synthesizes, tests, and debugs reusable CognitiveNode and Subagent modules for MAVIS.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any
from core.config import cfg
from core.helpers import log_it
from core.llm.base import BaseLLMClient
from core.agents import load_agent, load_agent_with_error
from agent_builder.tester import AgentTester
from agent_builder.debugger import AgentDebugger
from prompts.agent_prompt_templates import (
    cognitive_builder_prompt,
    subagent_builder_prompt,
    cognitive_updater_prompt,
    subagent_updater_prompt,
)
from core.metrics import MetricEmitter

_ENTITY = "agent_builder"
_EMITTER = MetricEmitter("builders")


class AgentBuildError(Exception):
    """Raised when an agent fails build or verification after all retries."""
    pass


class AgentBuilder:
    """
    Lifecycle manager for cognitive nodes and ReAct subagents in MAVIS:
    generate → write to agents/ → LLM-as-a-Judge test → [debug loop] → register in agents_list.json.
    """

    @property
    def MAX_RETRIES(self) -> int:
        return cfg.get("agentbuilder", "max_retries", default=3)

    def __init__(self, client: BaseLLMClient):
        self.client = client
        self.tester = AgentTester(client)
        self.debugger = AgentDebugger(client)
        from memories.memory_store import MemoryStore
        self.memory = MemoryStore(self.client, namespace="agent_debugger")

    def _to_pascal_case(self, name: str) -> str:
        """Convert snake_case to PascalCase (e.g. 'summarize_news' -> 'SummarizeNewsAgent')."""
        clean = re.sub(r"[^a-zA-Z0-9_]", "", name)
        words = clean.split("_")
        pascal = "".join(w.capitalize() for w in words if w)
        if not pascal.endswith("Agent"):
            pascal += "Agent"
        return pascal

    def _write_agent_file(self, agent_name: str, code: str):
        """Write agent module through ONI's call_fs."""
        from oni import oni as _oni
        file_path = f"agents/{agent_name}.py"
        status, result = _oni.call_fs("write", file_path, code)
        if status != 0:
            raise IOError(f"ONI denied write to '{file_path}': {result}")
        log_it(f"Agent '{agent_name}' written to {file_path}", _ENTITY)

    def _register_agent(
        self,
        agent_name: str,
        description: str,
        input_schema: dict,
        agent_type: str = "cognitive",
        default_max_turns: int = 4,
        allowed_tools: list[str] | None = None,
        generalizability: str = "specialized",
    ):
        """Add successfully verified agent to data/agents_list.json."""
        catalog_path = "data/agents_list.json"
        try:
            with open(catalog_path, "r") as f:
                catalog = json.load(f)
        except Exception:
            catalog = {}

        # Construct signature string: agent_name(param1: type, ...) -> tuple[int, Any]
        params_str = ", ".join(f"{k}: Any" for k in input_schema.keys())
        sig = f"{agent_name}({params_str}) -> tuple[int, Any]"

        entry: dict[str, Any] = {
            "type": agent_type,
            "description": description,
            "generalizability": generalizability,
        }
        if agent_type == "subagent":
            entry["default_max_turns"] = default_max_turns
            entry["allowed_tools"] = allowed_tools

        catalog[sig] = entry

        with open(catalog_path, "w") as f:
            json.dump(catalog, f, indent=4)
        log_it(f"Agent '{sig}' ({agent_type}) registered in {catalog_path}", _ENTITY)

    def _mark_needs_manual_fix(self, agent_name: str, last_error: str):
        """Prepend a warning comment to an agent file that exhausted retries."""
        file_path = f"agents/{agent_name}.py"
        try:
            with open(file_path, "r") as f:
                existing = f.read()
            header = (
                "# NEEDS MANUAL FIX\n"
                f"# Automated prompt debugging exhausted {self.MAX_RETRIES} retries.\n"
                f"# Last error: {last_error}\n"
                "# Review and adjust system instruction/constraints manually.\n\n"
            )
            with open(file_path, "w") as f:
                f.write(header + existing)
        except FileNotFoundError:
            pass

    def build_agent(
        self,
        agent_name: str,
        agent_description: str = "",
        input_schema: dict | None = None,
        output_schema: dict | None = None,
        agent_type: str = "cognitive",
        default_max_turns: int = 4,
        allowed_tools: list[str] | None = None,
        generalizability: str = "specialized",
        executor: Any | None = None,
        tool_retriever: Any | None = None,
        available_tools: dict[str, Any] | None = None,
        description: str = "",
    ) -> str:
        """
        Full lifecycle: generate → write → test → debug loop → register.
        Supports both 1-shot CognitiveNode and bounded ReAct Subagent.
        """
        if not agent_name or not isinstance(agent_name, str):
            raise AgentBuildError(f"agent_name must be a non-empty string, got {agent_name!r}")
        clean_name = agent_name.strip().lower()
        agent_description = agent_description or description or f"Agent {clean_name}"
        class_name = self._to_pascal_case(clean_name)
        input_schema = input_schema or {}

        is_subagent = str(agent_type).lower() == "subagent"
        clean_type = "subagent" if is_subagent else "cognitive"

        t0 = time.perf_counter()
        # ── 1. Query past agent debugger memories for priors ─────────────────────
        ref_context = ""
        try:
            mem_context = self.memory.retrieve_context(
                f"{clean_name} {clean_type} {agent_description}",
                top_k=2,
            )
            if mem_context:
                ref_context = f"RELEVANT DEBUGGED PROMPT FIXES:\n{mem_context}\n"
        except Exception as me:
            log_it(f"Memory retrieval in AgentBuilder failed: {me}", _ENTITY)

        # ── 2. Synthesize Agent Code ─────────────────────────────────────────────
        if is_subagent:
            prompt = subagent_builder_prompt.format(
                reference_context=ref_context,
                agent_name=clean_name,
                agent_description=agent_description,
                input_schema=json.dumps(input_schema, indent=2),
                output_schema=json.dumps(output_schema, indent=2) if output_schema else "None (Unstructured text)",
                default_max_turns=default_max_turns,
                allowed_tools=json.dumps(allowed_tools) if allowed_tools else "None",
                class_name=class_name,
            )
        else:
            prompt = cognitive_builder_prompt.format(
                reference_context=ref_context,
                agent_name=clean_name,
                agent_description=agent_description,
                input_schema=json.dumps(input_schema, indent=2),
                output_schema=json.dumps(output_schema, indent=2) if output_schema else "None (Unstructured text)",
                class_name=class_name,
            )

        raw_response = self.client.generate(prompt, json_mode=True)
        response_dict = json.loads(raw_response)
        agent_code = response_dict.get("code", "")

        self._write_agent_file(clean_name, agent_code)

        # ── 3. Test → Debug Loop ────────────────────────────────────────────────
        test_cases = self.tester.generate_test_cases(
            agent_name=clean_name,
            agent_description=agent_description,
            input_schema=input_schema,
            output_schema=output_schema,
            agent_type=clean_type,
        )

        attempt = 0
        last_failure_reason = "Unknown failure"
        while attempt <= self.MAX_RETRIES:
            agent_instance, load_err = load_agent_with_error(clean_name, self.client)
            if not agent_instance:
                last_failure_reason = f"Module 'agents/{clean_name}.py' failed to load: {load_err or 'No valid agent class found'}"
                log_it(
                    f"Agent '{clean_name}' ({clean_type}) load failure (attempt {attempt}/{self.MAX_RETRIES}): {last_failure_reason}",
                    _ENTITY,
                )
                if attempt == self.MAX_RETRIES:
                    break
                current_code = open(f"agents/{clean_name}.py").read()
                fixed_code, fix_summary = self.debugger.debug_agent(
                    agent_name=clean_name,
                    agent_description=agent_description,
                    broken_code=current_code,
                    failed_case={"inputs": {}},
                    actual_output="",
                    failure_reason=last_failure_reason,
                    agent_type=clean_type,
                )
                self._write_agent_file(clean_name, fixed_code)
                attempt += 1
                continue

            status, test_result = self.tester.test_agent(
                agent_instance,
                test_cases=test_cases,
                executor=executor,
                tool_retriever=tool_retriever,
                available_tools=available_tools,
            )

            if status == 0:
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                log_it(f"Agent '{clean_name}' ({clean_type}) passed verification on attempt {attempt}.", _ENTITY)
                self._register_agent(
                    agent_name=clean_name,
                    description=agent_description,
                    input_schema=input_schema,
                    agent_type=clean_type,
                    default_max_turns=default_max_turns,
                    allowed_tools=allowed_tools,
                    generalizability=generalizability,
                )

                _EMITTER.log({
                    "target_name": clean_name,
                    "builder_type": clean_type,
                    "latency_ms": latency_ms,
                    "status": "passed",
                    "attempt_count": attempt,
                    "failure_reason": "none",
                    "debugger_prior_used": ref_context != "",
                    "input_tokens": len(prompt) // 4,
                    "output_tokens": len(agent_code) // 4,
                })

                # Record successful repair in agent_debugger memory if fixed after retry
                if attempt > 0:
                    try:
                        self.memory.write_agent_fix(
                            clean_name,
                            last_failure_reason,
                            f"Repaired and verified after {attempt} retry attempts for {clean_type} agent.",
                        )
                    except Exception as e:
                        log_it(f"Failed to record agent memory: {e}", _ENTITY)

                return generalizability

            # Verification failed
            failed_case = test_result.get("failed_case", {})
            actual_output = test_result.get("actual_output", "")
            last_failure_reason = test_result.get("reason", "Verdict: failed")

            log_it(
                f"Agent '{clean_name}' ({clean_type}) failed verification (attempt {attempt}/{self.MAX_RETRIES}): {last_failure_reason}",
                _ENTITY,
            )

            if attempt == self.MAX_RETRIES:
                break

            current_code = open(f"agents/{clean_name}.py").read()
            fixed_code, fix_summary = self.debugger.debug_agent(
                agent_name=clean_name,
                agent_description=agent_description,
                broken_code=current_code,
                failed_case=failed_case,
                actual_output=actual_output,
                failure_reason=last_failure_reason,
                agent_type=clean_type,
            )
            self._write_agent_file(clean_name, fixed_code)
            attempt += 1

        # ── 4. All Retries Exhausted ────────────────────────────────────────────
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        _EMITTER.log({
            "target_name": clean_name,
            "builder_type": clean_type,
            "latency_ms": latency_ms,
            "status": "failed",
            "attempt_count": attempt,
            "failure_reason": str(last_failure_reason)[:100],
            "debugger_prior_used": ref_context != "",
            "input_tokens": len(prompt) // 4,
            "output_tokens": 0,
        })
        self._mark_needs_manual_fix(clean_name, last_failure_reason)
        raise AgentBuildError(
            f"Agent '{clean_name}' ({clean_type}) failed all {self.MAX_RETRIES} verification attempts. "
            f"Last reason: {last_failure_reason}"
        )

    def update_agent(
        self,
        agent_name: str,
        requested_changes: str,
        executor: Any | None = None,
        tool_retriever: Any | None = None,
        available_tools: dict[str, Any] | None = None,
    ) -> str:
        """
        Evolve an existing cognitive node or subagent in-place based on requested changes.
        Enforces immutability on PROTECTED_BASELINE_AGENTS ('semantic_transform'),
        regenerates code with backwards-compatible input schema, tests with AgentTester,
        and updates data/agents_list.json in-place.

        Raises:
            AgentBuildError: if agent is protected, not found, or fails all verification retries.
        """
        from core.manager import PROTECTED_BASELINE_AGENTS
        clean_name = agent_name.strip().lower()
        if clean_name in PROTECTED_BASELINE_AGENTS:
            raise AgentBuildError(f"Cannot update protected baseline agent '{clean_name}'.")

        agent_file = f"agents/{clean_name}.py"
        if not os.path.exists(agent_file):
            raise AgentBuildError(f"Agent '{clean_name}' does not exist at {agent_file}.")

        with open(agent_file, "r", encoding="utf-8") as f:
            current_code = f.read()

        catalog_path = "data/agents_list.json"
        agent_type = "cognitive"
        current_desc = ""
        current_input_schema: dict[str, Any] = {}
        current_output_schema: dict[str, Any] | None = None
        current_max_turns = 4
        current_allowed_tools: list[str] | None = None
        current_gen = "specialized"
        old_sig = None

        if os.path.exists(catalog_path):
            try:
                with open(catalog_path, "r", encoding="utf-8") as f:
                    catalog = json.load(f)
                for sig, meta in catalog.items():
                    sig_name = sig.split("(")[0].strip().lower()
                    if sig_name == clean_name:
                        old_sig = sig
                        agent_type = meta.get("type", "cognitive")
                        current_desc = meta.get("description", "")
                        current_gen = meta.get("generalizability", "specialized")
                        current_max_turns = meta.get("default_max_turns", 4)
                        current_allowed_tools = meta.get("allowed_tools", None)
                        break
            except Exception as e:
                log_it(f"Error reading agents_list.json in update_agent: {e}", _ENTITY)

        is_subagent = agent_type == "subagent"
        clean_type = "subagent" if is_subagent else "cognitive"
        class_name = self._to_pascal_case(clean_name)
        t0 = time.perf_counter()

        ref_context = ""
        try:
            mem_context = self.memory.retrieve_context(
                f"{clean_name} {clean_type} {requested_changes}",
                top_k=2,
            )
            if mem_context:
                ref_context = f"RELEVANT DEBUGGED PROMPT FIXES:\n{mem_context}\n"
        except Exception as me:
            log_it(f"Memory retrieval in AgentBuilder.update_agent failed: {me}", _ENTITY)

        # 1. Synthesize Updated Agent Code
        if is_subagent:
            prompt = subagent_updater_prompt.format(
                class_name=class_name,
                agent_name=clean_name,
                current_description=current_desc,
                current_input_schema=json.dumps(current_input_schema),
                current_output_schema=json.dumps(current_output_schema) if current_output_schema else "None",
                current_allowed_tools=json.dumps(current_allowed_tools) if current_allowed_tools else "None",
                current_max_turns=current_max_turns,
                default_max_turns=current_max_turns,
                allowed_tools=json.dumps(current_allowed_tools) if current_allowed_tools else "None",
                current_code=current_code,
                requested_changes=f"{requested_changes}\n\n{ref_context}",
            )
        else:
            prompt = cognitive_updater_prompt.format(
                class_name=class_name,
                agent_name=clean_name,
                current_description=current_desc,
                current_input_schema=json.dumps(current_input_schema),
                current_output_schema=json.dumps(current_output_schema) if current_output_schema else "None",
                current_code=current_code,
                requested_changes=f"{requested_changes}\n\n{ref_context}",
            )

        raw_response = self.client.generate(prompt, json_mode=True)
        response_dict = json.loads(raw_response)
        agent_code = response_dict.get("code", "")
        updated_desc = response_dict.get("updated_description", current_desc)
        updated_input_schema = response_dict.get("updated_input_schema", current_input_schema)
        updated_output_schema = response_dict.get("updated_output_schema", current_output_schema)
        raw_gen = str(response_dict.get("generalizability", current_gen)).strip().lower()
        generalizability = raw_gen if raw_gen in ("specialized", "repurposable", "generalizable") else "specialized"
        if is_subagent:
            current_max_turns = response_dict.get("default_max_turns", current_max_turns)
            current_allowed_tools = response_dict.get("allowed_tools", current_allowed_tools)

        self._write_agent_file(clean_name, agent_code)

        # 2. Test → Debug Loop
        test_cases = self.tester.generate_test_cases(
            agent_name=clean_name,
            agent_description=updated_desc,
            input_schema=updated_input_schema,
            output_schema=updated_output_schema,
            agent_type=clean_type,
        )

        attempt = 0
        last_failure_reason = "Unknown failure"
        while attempt <= self.MAX_RETRIES:
            agent_instance, load_err = load_agent_with_error(clean_name, self.client)
            if not agent_instance:
                last_failure_reason = f"Module 'agents/{clean_name}.py' failed to load: {load_err or 'No valid agent class found'}"
                log_it(
                    f"Agent '{clean_name}' ({clean_type}) load failure on update (attempt {attempt}/{self.MAX_RETRIES}): {last_failure_reason}",
                    _ENTITY,
                )
                if attempt == self.MAX_RETRIES:
                    break
                current_code_on_disk = open(f"agents/{clean_name}.py", "r", encoding="utf-8").read()
                fixed_code, fix_summary = self.debugger.debug_agent(
                    agent_name=clean_name,
                    agent_description=updated_desc,
                    broken_code=current_code_on_disk,
                    failed_case={"inputs": {}},
                    actual_output="",
                    failure_reason=last_failure_reason,
                    agent_type=clean_type,
                )
                self._write_agent_file(clean_name, fixed_code)
                attempt += 1
                continue

            status, test_result = self.tester.test_agent(
                agent_instance,
                test_cases=test_cases,
                executor=executor,
                tool_retriever=tool_retriever,
                available_tools=available_tools,
            )

            if status == 0:
                latency_ms = round((time.perf_counter() - t0) * 1000, 2)
                log_it(f"Updated agent '{clean_name}' ({clean_type}) passed verification on attempt {attempt}.", _ENTITY)

                if os.path.exists(catalog_path):
                    with open(catalog_path, "r", encoding="utf-8") as f:
                        catalog = json.load(f)
                    if old_sig and old_sig in catalog:
                        del catalog[old_sig]
                    with open(catalog_path, "w", encoding="utf-8") as f:
                        json.dump(catalog, f, indent=4)

                self._register_agent(
                    agent_name=clean_name,
                    description=updated_desc,
                    input_schema=updated_input_schema,
                    agent_type=clean_type,
                    default_max_turns=current_max_turns,
                    allowed_tools=current_allowed_tools,
                    generalizability=generalizability,
                )

                _EMITTER.log({
                    "target_name": clean_name,
                    "builder_type": f"{clean_type}_update",
                    "latency_ms": latency_ms,
                    "status": "passed",
                    "attempt_count": attempt,
                    "failure_reason": "none",
                    "debugger_prior_used": ref_context != "",
                    "input_tokens": len(prompt) // 4,
                    "output_tokens": len(agent_code) // 4,
                })

                if attempt > 0:
                    try:
                        self.memory.write_agent_fix(
                            clean_name,
                            last_failure_reason,
                            f"Repaired and verified after {attempt} retry attempts for updated {clean_type} agent.",
                        )
                    except Exception as e:
                        log_it(f"Failed to record agent memory: {e}", _ENTITY)

                return generalizability

            failed_case = test_result.get("failed_case", {})
            actual_output = test_result.get("actual_output", "")
            last_failure_reason = test_result.get("reason", "Verdict: failed")

            log_it(
                f"Updated agent '{clean_name}' ({clean_type}) failed verification (attempt {attempt}/{self.MAX_RETRIES}): {last_failure_reason}",
                _ENTITY,
            )
            if attempt == self.MAX_RETRIES:
                break

            current_code_on_disk = open(f"agents/{clean_name}.py", "r", encoding="utf-8").read()
            fixed_code, fix_summary = self.debugger.debug_agent(
                agent_name=clean_name,
                agent_description=updated_desc,
                broken_code=current_code_on_disk,
                failed_case=failed_case,
                actual_output=actual_output,
                failure_reason=last_failure_reason,
                agent_type=clean_type,
            )
            self._write_agent_file(clean_name, fixed_code)
            attempt += 1

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        _EMITTER.log({
            "target_name": clean_name,
            "builder_type": f"{clean_type}_update",
            "latency_ms": latency_ms,
            "status": "failed",
            "attempt_count": attempt,
            "failure_reason": str(last_failure_reason)[:100],
            "debugger_prior_used": ref_context != "",
            "input_tokens": len(prompt) // 4,
            "output_tokens": 0,
        })
        self._mark_needs_manual_fix(clean_name, last_failure_reason)
        raise AgentBuildError(
            f"Updated agent '{clean_name}' ({clean_type}) failed all {self.MAX_RETRIES} verification attempts. "
            f"Last reason: {last_failure_reason}"
        )

