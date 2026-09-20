"""
core/pipeline_debugger.py

Runtime execution debugger for MAVIS DAG pipelines.
Diagnoses step failures, patches parameters, replaces incompatible nodes,
and provides closed-loop self-correction during pipeline execution.
"""
from __future__ import annotations

import json
import time
from typing import Any

from core.helpers import log_it
from core.llm import BaseLLMClient, get_llm_client
from core.metrics import MetricEmitter

_ENTITY = "pipeline_debugger"
_EMITTER = MetricEmitter("pipeline_debugger")

DEBUGGER_SYSTEM_PROMPT = """You are the MAVIS Pipeline Debugger.
A step in an automated Directed Acyclic Graph (DAG) pipeline failed during execution.
Your job is to analyze the root cause and generate a structured JSON repair plan so the pipeline can self-heal and proceed without crashing.

You will be given:
1. The original USER REQUEST
2. The PREVIOUS STEP RESULTS (data produced by steps that already succeeded)
3. The FAILED NODE specification (id, type, function_name, declared params)
4. The RESOLVED PARAMETERS that were actually passed into the failed step
5. The EXACT ERROR MESSAGE / FAILURE STATUS returned by the step
6. A summary of AVAILABLE COMMANDS and AGENTS

Analyze the root cause:
- Did a path lack a necessary directory prefix (e.g. 'docs/' was requested, but 'file.md' was passed)?
- Was a parameter passed as an unparsed string when a list/dict was expected, or vice versa?
- Did a tool fail because an alternative tool or cognitive agent (e.g. 'semantic_transform') is better suited?
- For a verification/control step (`type: "control"`):
  Did the verification condition fail because it was over-constrained, syntactically malformed, or looking for a pattern that differs slightly from what successful upstream steps actually output?
  CORE PHILOSOPHY: User experience and forward progress take precedence over rigid verification checks. If the previous steps substantially fulfilled the user's intent, PREFER patching/relaxing the control condition (`action: "patch_params"` with a relaxed or corrected `condition`) so the pipeline does not fail the user's task.
- Is this an external, unfixable error (e.g. invalid credentials, resource permanently missing, network down)?

Output a JSON object with this exact schema:
{
  "action": "patch_params" | "replace_node" | "unrecoverable",
  "diagnosis": "Brief, clear explanation of why the step failed and what is being corrected.",
  "patched_node": {
    "id": "node_id",
    "type": "tool" | "cognitive" | "subagent" | "control",
    "function_name": "function_or_agent_name",
    "condition": "updated condition string if type is control",
    "params": { ... corrected parameters ... }
  }
}

Rules:
1. If "action" is "patch_params": keep the same "id" and "type". If it is a tool/subagent, keep "function_name" and supply corrected "params". If it is a control node, supply a corrected or relaxed "condition" (in top-level "condition" or inside "params").
2. If "action" is "replace_node", provide an alternative "function_name" from the available tools or agents that achieves the same goal.
3. If "action" is "unrecoverable", set "patched_node" to null.
4. Respond ONLY with valid JSON.
"""


class PipelineDebugger:
    """
    Closed-loop runtime debugger for live DAG pipelines.
    Integrates with MemoryStore and Neo4j Knowledge Graph to retrieve past
    runtime fixes and persist newly discovered repairs.
    """

    def __init__(self, client: BaseLLMClient | None = None, memory_store: Any | None = None):
        self._client = client
        self.memory_store = memory_store

    def _get_client(self) -> BaseLLMClient:
        if self._client is None:
            self._client = get_llm_client()
        return self._client

    def diagnose_and_repair(
        self,
        failed_node: dict[str, Any],
        resolved_params: dict[str, Any],
        error_message: str,
        node_results: dict[str, Any],
        user_query: str,
        commands_list: dict[str, Any] | None = None,
        agents_list: dict[str, Any] | None = None,
        turn_id: str = "",
    ) -> dict[str, Any]:
        """
        Analyze a failed node and produce a repair plan.

        Returns a dictionary with:
          - action: 'patch_params' | 'replace_node' | 'unrecoverable'
          - diagnosis: str
          - patched_node: dict | None
        """
        t0 = time.perf_counter()
        node_id = failed_node.get("id", "unknown")
        func_name = failed_node.get("function_name", "unknown")

        log_it(
            f"PipelineDebugger invoked for step '{node_id}' ({func_name}): {error_message[:120]}",
            _ENTITY,
        )

        # Retrieve relevant memory priors on this tool and failure mode
        memory_priors = ""
        if self.memory_store is not None:
            try:
                memory_priors = self.memory_store.retrieve_context(
                    f"{func_name} {error_message}",
                    extra_namespaces=["debugger", "toolbuilder"],
                )
            except Exception as mem_err:
                log_it(f"PipelineDebugger memory retrieval failed: {mem_err}", _ENTITY)

        prompt_payload = {
            "user_request": user_query,
            "relevant_past_memory": memory_priors,
            "previous_step_results": node_results,
            "failed_node": failed_node,
            "resolved_parameters": resolved_params,
            "error_message": error_message,
            "available_commands": list((commands_list or {}).keys())[:20],
            "available_agents": list((agents_list or {}).keys())[:10],
        }

        user_content = json.dumps(prompt_payload, indent=2, default=str)

        try:
            client = self._get_client()
            raw_response = client.generate(
                user_content,
                json_mode=True,
                system_instruction=DEBUGGER_SYSTEM_PROMPT,
            )

            # Strip markdown fence if present
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()

            plan = json.loads(cleaned)
            action = plan.get("action", "unrecoverable")
            diagnosis = plan.get("diagnosis", "Automated diagnosis completed.")
            patched_node = plan.get("patched_node")

            if action in ("patch_params", "replace_node") and not isinstance(patched_node, dict):
                action = "unrecoverable"
                diagnosis = "Debugger generated invalid patched_node structure."
                patched_node = None

            if failed_node.get("type") == "control" and isinstance(patched_node, dict):
                patched_node.setdefault("type", "control")
                if "condition" in patched_node.get("params", {}):
                    patched_node["condition"] = patched_node["params"]["condition"]

            # If repair succeeded, persist fix to memory_store and Neo4j
            if action in ("patch_params", "replace_node") and patched_node and self.memory_store is not None:
                try:
                    fix_entry = {
                        "command": func_name,
                        "error_signature": error_message[:200],
                        "diagnosis": diagnosis,
                        "action": action,
                        "content": f"Pipeline repair for {func_name}: on error '{error_message[:100]}', {diagnosis}",
                        "patched_params": patched_node.get("params", {}),
                    }
                    self.memory_store.write_long_term(fix_entry, ltype="fix")
                except Exception as e:
                    log_it(f"PipelineDebugger failed to write long_term memory: {e}", _ENTITY)

                kg = getattr(self.memory_store, "kg", None)
                if kg and hasattr(kg, "add_pipeline_fix"):
                    try:
                        kg.add_pipeline_fix(
                            command_name=func_name,
                            error_snippet=error_message[:100],
                            fix_summary=diagnosis[:200],
                            client=self._client,
                        )
                    except Exception as e:
                        log_it(f"PipelineDebugger failed to write to Neo4j: {e}", _ENTITY)

            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            _EMITTER.log({
                "turn_id": turn_id,
                "node_id": node_id,
                "action": action,
                "latency_ms": latency_ms,
                "success": action in ("patch_params", "replace_node"),
            })

            log_it(
                f"PipelineDebugger result for '{node_id}': action={action} diagnosis={diagnosis!r}",
                _ENTITY,
            )
            return {
                "action": action,
                "diagnosis": diagnosis,
                "patched_node": patched_node,
            }

        except Exception as exc:
            latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            log_it(f"PipelineDebugger error during repair: {exc}", _ENTITY)
            _EMITTER.log({
                "turn_id": turn_id,
                "node_id": node_id,
                "action": "error",
                "latency_ms": latency_ms,
                "success": False,
            })
            return {
                "action": "unrecoverable",
                "diagnosis": f"PipelineDebugger encountered an exception: {exc}",
                "patched_node": None,
            }
