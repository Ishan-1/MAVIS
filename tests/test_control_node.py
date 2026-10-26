"""
Unit tests for Terminal Control / Success Verification Nodes (core/gate_evaluator.py, core/dag.py, core/pipeline_debugger.py, core/answerer.py).
"""
import pytest
from unittest.mock import MagicMock
from core.gate_evaluator import (
    evaluate_control_node,
    evaluate_deterministic_with_reason,
    evaluate_nlp_with_reason,
)
from core.dag import validate_and_sort_dag, extract_node_dependencies
from core.pipeline_debugger import PipelineDebugger
from core.answerer import Answerer


def test_evaluate_control_node_deterministic_success():
    control_node = {
        "id": "n_verify",
        "type": "control",
        "mode": "deterministic",
        "condition": "$n1.status == 0 and len($n2.output) > 0",
        "expected_outcome": "Tests pass and output is non-empty",
    }
    node_results = {
        "n1": {"status": 0, "output": "3 passed"},
        "n2": {"status": 0, "output": "Report generated."},
    }

    res = evaluate_control_node(control_node, node_results)
    assert res["node_id"] == "n_verify"
    assert res["node_type"] == "control"
    assert res["success"] is True
    assert res["status"] == 0
    assert "passed" in res["output"]


def test_evaluate_control_node_deterministic_failure():
    control_node = {
        "id": "n_verify",
        "type": "control",
        "mode": "deterministic",
        "condition": "$n1.status == 0 and 'PASSED' in $n1.output",
    }
    node_results = {
        "n1": {"status": 1, "output": "FAILED: 2 tests failed"},
    }

    res = evaluate_control_node(control_node, node_results)
    assert res["success"] is False
    assert res["status"] == 1
    assert "failed" in res["output"]


def test_evaluate_control_node_malformed_condition():
    control_node = {
        "id": "n_verify",
        "type": "control",
        "mode": "deterministic",
        "condition": "$n1.status == == 0",  # Syntax error
    }
    node_results = {"n1": {"status": 0, "output": "ok"}}

    res = evaluate_control_node(control_node, node_results)
    assert res["success"] is False
    assert res["status"] == 1
    assert "Evaluation error" in res["reason"]


def test_evaluate_control_node_nlp():
    mock_llm = MagicMock()
    mock_llm.chat.return_value = '{"verdict": true, "reason": "All tasks verified complete"}'

    control_node = {
        "id": "n_verify",
        "type": "control",
        "mode": "nlp",
        "condition": "Did the tool successfully generate a valid summary?",
    }
    node_results = {"n1": {"output": "Executive summary: Sales increased 15%."}}

    res = evaluate_control_node(control_node, node_results, client=mock_llm)
    assert res["success"] is True
    assert res["status"] == 0
    assert "All tasks verified complete" in res["reason"]


def test_dag_control_node_topological_sort():
    # Pipeline where control node has no explicit deps, should run last
    pipeline = [
        {"id": "n_verify", "type": "control", "condition": "$n2.status == 0"},
        {"id": "n1", "type": "tool", "function_name": "step1"},
        {"id": "n2", "type": "tool", "function_name": "step2", "params": {"prev": "$n1.output"}},
    ]

    sorted_pipeline, err = validate_and_sort_dag(pipeline)
    assert err is None
    ids = [n["id"] for n in sorted_pipeline]
    assert ids == ["n1", "n2", "n_verify"]


def test_dag_control_node_implicit_dependencies():
    # Even without $node references, control node should implicitly run last
    pipeline = [
        {"id": "n_verify", "type": "control", "mode": "nlp", "condition": "Everything looks good"},
        {"id": "n1", "type": "tool", "function_name": "step1"},
        {"id": "n2", "type": "tool", "function_name": "step2"},
    ]

    sorted_pipeline, err = validate_and_sort_dag(pipeline)
    assert err is None
    assert sorted_pipeline[-1]["id"] == "n_verify"


def test_pipeline_debugger_control_node_relaxation():
    mock_llm = MagicMock()
    # LLM debugger relaxes the condition because user intent was substantially fulfilled
    mock_llm.generate.return_value = '''
    {
      "action": "patch_params",
      "diagnosis": "Condition expected exact match 'SUCCESS' but upstream returned 'OK'. Relaxing condition to accept status == 0.",
      "patched_node": {
        "id": "n_verify",
        "type": "control",
        "condition": "$n1.status == 0",
        "params": {}
      }
    }
    '''
    debugger = PipelineDebugger(client=mock_llm)
    failed_node = {
        "id": "n_verify",
        "type": "control",
        "condition": "$n1.output == 'SUCCESS'",
    }
    node_results = {"n1": {"status": 0, "output": "OK"}}

    plan = debugger.diagnose_and_repair(
        failed_node=failed_node,
        resolved_params=failed_node,
        error_message="Condition evaluated to False: $n1.output == 'SUCCESS'",
        node_results=node_results,
        user_query="Run task and verify",
    )

    assert plan["action"] == "patch_params"
    assert plan["patched_node"]["type"] == "control"
    assert plan["patched_node"]["condition"] == "$n1.status == 0"


def test_answerer_warns_on_control_failure():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "The task had an issue with verification."
    answerer = Answerer(client=mock_llm)

    pipeline_results = {
        "n1": {"status": 0, "output": "Files generated"},
        "n_verify": {
            "node_type": "control",
            "success": False,
            "status": 1,
            "reason": "Test suite had 1 failing test",
        },
    }

    answer = answerer.synthesize(
        query="Generate files and verify tests pass",
        pipeline_results=pipeline_results,
    )

    # Verify that the generated prompt to the LLM included the critical post-condition warning
    call_prompt = mock_llm.generate.call_args[0][0]
    assert "CRITICAL POST-CONDITION VERIFICATION WARNING" in call_prompt
    assert "Test suite had 1 failing test" in call_prompt


def test_answerer_confirms_on_control_success():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Git push completed: Everything is up-to-date."
    answerer = Answerer(client=mock_llm)

    pipeline_results = {
        "n1": (0, "Everything up-to-date\n"),
        "n_verify": {
            "node_type": "control",
            "success": True,
            "status": 0,
            "reason": "Condition satisfied: '$n1.status == 0'",
        },
    }

    answer = answerer.synthesize(
        query="do a git push",
        pipeline_results=pipeline_results,
    )

    # Verify that the generated prompt to the LLM confirms successful completion
    call_prompt = mock_llm.generate.call_args[0][0]
    assert "POST-CONDITION VERIFICATION" in call_prompt
    assert "Condition satisfied: '$n1.status == 0'" in call_prompt
    assert "All verification checks passed successfully" in call_prompt
    assert answer == "Git push completed: Everything is up-to-date."

