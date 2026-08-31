"""
agent_builder/tester.py
AgentTester harness: Evaluates candidate cognitive sub-agents using LLM-as-a-Judge.
Produces discrete binary 'passed' or 'failed' verdicts.
"""
from __future__ import annotations

import json
from typing import Any
from core.agents.base import BaseAgent
from core.helpers import log_it
from core.llm.base import BaseLLMClient
from prompts.agent_prompt_templates import agent_tester_input_prompt, agent_judge_prompt

_ENTITY = "agent_tester"


class AgentTester:
    """
    Automated test generator and LLM-as-a-Judge evaluator for MAVIS sub-agents.
    """

    def __init__(self, client: BaseLLMClient):
        self.client = client

    def generate_test_cases(
        self,
        agent_name: str,
        agent_description: str,
        input_schema: dict,
        output_schema: dict | None,
    ) -> list[dict]:
        """Generate 2-3 synthetic test inputs matching the agent's schema."""
        prompt = agent_tester_input_prompt.format(
            agent_name=agent_name,
            agent_description=agent_description,
            input_schema=json.dumps(input_schema, indent=2),
            output_schema=json.dumps(output_schema, indent=2) if output_schema else "None (Unstructured text)",
        )
        try:
            raw = self.client.generate(prompt, json_mode=True)
            data = json.loads(raw)
            cases = data.get("test_cases", [])
            if isinstance(cases, list) and cases:
                return cases
        except Exception as e:
            log_it(f"Failed to generate synthetic test cases: {e}", _ENTITY)

        # Fallback minimal test case
        return [{"id": "fallback_tc", "description": "Default invocation", "inputs": {k: "test" for k in input_schema.keys()}}]

    def evaluate_output(
        self,
        agent_name: str,
        agent_description: str,
        system_instruction: str,
        output_schema: dict | None,
        test_input: dict,
        actual_output: Any,
    ) -> tuple[str, str]:
        """
        Evaluate agent output with LLM-as-a-Judge.
        Returns:
            ("passed" | "failed", reason)
        """
        judge_prompt = agent_judge_prompt.format(
            agent_name=agent_name,
            agent_description=agent_description,
            system_instruction=system_instruction,
            output_schema=json.dumps(output_schema, indent=2) if output_schema else "None (Unstructured text)",
            test_input=json.dumps(test_input, indent=2, default=str),
            actual_output=json.dumps(actual_output, indent=2, default=str) if not isinstance(actual_output, str) else actual_output,
        )

        try:
            raw = self.client.generate(judge_prompt, json_mode=True)
            result = json.loads(raw)
            verdict = str(result.get("verdict", "failed")).strip().lower()
            reason = str(result.get("reason", "No reason provided."))
            clean_verdict = "passed" if verdict == "passed" else "failed"
            return clean_verdict, reason
        except Exception as e:
            log_it(f"Judge evaluation failed: {e}", _ENTITY)
            return "failed", f"Judge evaluation error: {e}"

    def test_agent(
        self,
        agent_instance: BaseAgent,
        test_cases: list[dict] | None = None,
    ) -> tuple[int, dict]:
        """
        Run test suite on candidate agent instance.
        Returns:
            (0, {"summary": "..."}) if all test cases passed.
            (-1, {"failed_case": ..., "actual_output": ..., "reason": ...}) if any failed.
        """
        cases = test_cases or self.generate_test_cases(
            agent_name=agent_instance.name,
            agent_description=agent_instance.description,
            input_schema=agent_instance.input_schema,
            output_schema=agent_instance.output_schema,
        )

        log_it(f"Running {len(cases)} test case(s) for agent '{agent_instance.name}'", _ENTITY)

        for case in cases:
            case_id = case.get("id", "unknown")
            inputs = case.get("inputs", {})

            # 1. Dynamic run
            status, result = agent_instance.run(**inputs)
            if status != 0:
                fail_reason = f"Execution error in test case '{case_id}': {result}"
                log_it(fail_reason, _ENTITY)
                return -1, {
                    "failed_case": case,
                    "actual_output": result,
                    "reason": fail_reason,
                }

            # 2. Judge evaluation
            verdict, reason = self.evaluate_output(
                agent_name=agent_instance.name,
                agent_description=agent_instance.description,
                system_instruction=agent_instance.system_instruction,
                output_schema=agent_instance.output_schema,
                test_input=inputs,
                actual_output=result,
            )

            if verdict != "passed":
                log_it(f"Agent '{agent_instance.name}' FAILED test case '{case_id}': {reason}", _ENTITY)
                return -1, {
                    "failed_case": case,
                    "actual_output": result,
                    "reason": reason,
                }

            log_it(f"Agent '{agent_instance.name}' PASSED test case '{case_id}'", _ENTITY)

        return 0, {"summary": f"All {len(cases)} test cases passed with verdict: passed."}
