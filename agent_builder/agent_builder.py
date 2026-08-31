"""
agent_builder/agent_builder.py
AgentBuilder: Synthesizes, tests, and debugs reusable BaseAgent modules for MAVIS.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any
from core.config import cfg
from core.helpers import log_it
from core.llm.base import BaseLLMClient
from core.agents import load_agent
from agent_builder.tester import AgentTester
from agent_builder.debugger import AgentDebugger
from prompts.agent_prompt_templates import agent_builder_prompt

_ENTITY = "agent_builder"


class AgentBuildError(Exception):
    """Raised when an agent fails build or verification after all retries."""
    pass


class AgentBuilder:
    """
    Lifecycle manager for cognitive sub-agents in MAVIS:
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

        catalog[sig] = {
            "description": description,
            "generalizability": generalizability,
        }

        with open(catalog_path, "w") as f:
            json.dump(catalog, f, indent=4)
        log_it(f"Agent '{sig}' registered in {catalog_path}", _ENTITY)

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
        agent_description: str,
        input_schema: dict,
        output_schema: dict | None = None,
        generalizability: str = "specialized",
    ) -> str:
        """
        Full lifecycle: generate → write → test → debug loop → register.
        """
        if not agent_name or not isinstance(agent_name, str):
            raise AgentBuildError(f"agent_name must be a non-empty string, got {agent_name!r}")
        clean_name = agent_name.strip().lower()
        class_name = self._to_pascal_case(clean_name)

        # ── 1. Query past agent debugger memories for priors ─────────────────────
        ref_context = ""
        try:
            mem_context = self.memory.retrieve_context(
                f"{clean_name} {agent_description}",
                top_k=2,
            )
            if mem_context:
                ref_context = f"RELEVANT DEBUGGED PROMPT FIXES:\n{mem_context}\n"
        except Exception as me:
            log_it(f"Memory retrieval in AgentBuilder failed: {me}", _ENTITY)

        # ── 2. Synthesize Agent Code ─────────────────────────────────────────────
        prompt = agent_builder_prompt.format(
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
        )

        attempt = 0
        last_failure_reason = "Unknown failure"
        while attempt <= self.MAX_RETRIES:
            agent_instance = load_agent(clean_name, self.client)
            if not agent_instance:
                raise AgentBuildError(f"Could not load generated agent module 'agents/{clean_name}.py'")

            status, test_result = self.tester.test_agent(agent_instance, test_cases=test_cases)

            if status == 0:
                log_it(f"Agent '{clean_name}' passed verification on attempt {attempt}.", _ENTITY)
                self._register_agent(clean_name, agent_description, input_schema, generalizability)

                # Record successful repair in agent_debugger memory if fixed after retry
                if attempt > 0:
                    try:
                        self.memory.write_agent_fix(
                            clean_name,
                            last_failure_reason,
                            f"Repaired and verified after {attempt} retry attempts.",
                        )
                    except Exception as e:
                        log_it(f"Failed to record agent memory: {e}", _ENTITY)

                return generalizability

            # Verification failed
            failed_case = test_result.get("failed_case", {})
            actual_output = test_result.get("actual_output", "")
            last_failure_reason = test_result.get("reason", "Verdict: failed")

            log_it(
                f"Agent '{clean_name}' failed verification (attempt {attempt}/{self.MAX_RETRIES}): {last_failure_reason}",
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
            )
            self._write_agent_file(clean_name, fixed_code)
            attempt += 1

        # ── 4. All Retries Exhausted ────────────────────────────────────────────
        self._mark_needs_manual_fix(clean_name, last_failure_reason)
        raise AgentBuildError(
            f"Agent '{clean_name}' failed all {self.MAX_RETRIES} verification attempts. "
            f"Last reason: {last_failure_reason}"
        )
