"""
agent_builder/debugger.py
AgentDebugger: Diagnoses prompt and schema failure modes from LLM-as-a-Judge diagnostics
and synthesizes refined system instructions, constraints, and few-shots.
"""
from __future__ import annotations

import json
from typing import Any
from core.helpers import log_it
from core.llm.base import BaseLLMClient
from prompts.agent_prompt_templates import agent_debugger_prompt

_ENTITY = "agent_debugger"


class AgentDebugger:
    """
    Automated debugging loop for candidate cognitive agents.
    Tuning prompts, negative constraints, and output validation upon Judge failure.
    """

    def __init__(self, client: BaseLLMClient):
        self.client = client

    def debug_agent(
        self,
        agent_name: str,
        agent_description: str,
        broken_code: str,
        failed_case: dict,
        actual_output: Any,
        failure_reason: str,
    ) -> tuple[str, str]:
        """
        Synthesize a corrected agent implementation based on failure feedback.

        Returns:
            (fixed_code, fix_summary)
        """
        failing_inputs = failed_case.get("inputs", {})
        prompt = agent_debugger_prompt.format(
            agent_name=agent_name,
            agent_description=agent_description,
            broken_code=broken_code,
            failing_input=json.dumps(failing_inputs, indent=2, default=str),
            actual_output=json.dumps(actual_output, indent=2, default=str) if not isinstance(actual_output, str) else actual_output,
            judge_reason=failure_reason,
        )

        try:
            raw = self.client.generate(prompt, json_mode=True)
            data = json.loads(raw)
            fixed_code = data.get("code", broken_code)
            fix_summary = data.get("fix_summary", "Refined prompt and negative constraints.")
            log_it(f"AgentDebugger produced fix for '{agent_name}': {fix_summary}", _ENTITY)
            return fixed_code, fix_summary
        except Exception as e:
            log_it(f"AgentDebugger failed to repair agent '{agent_name}': {e}", _ENTITY)
            return broken_code, f"Debug failed: {e}"
