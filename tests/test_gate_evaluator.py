"""
Unit tests for core/gate_evaluator.py
"""
import pytest
from unittest.mock import MagicMock
from core.gate_evaluator import (
    evaluate_deterministic,
    evaluate_nlp,
    evaluate_gate,
    NodeResultProxy,
)


def test_node_result_proxy_attributes():
    # Success dict
    proxy = NodeResultProxy({"status": 0, "output": "build successful", "items": [1, 2, 3]})
    assert proxy.status == 0
    assert proxy.output == "build successful"
    assert "build" in proxy.output
    assert proxy["status"] == 0
    assert proxy.get("items") == [1, 2, 3]

    # Failed string
    proxy_fail = NodeResultProxy("FAILED: KeyError: 'user_id'")
    assert proxy_fail.status == 1
    assert "KeyError" in proxy_fail.output


def test_deterministic_evaluator_comparisons():
    node_results = {
        "step_test": {"status": 0, "output": "3 passed, 0 failed"},
        "step_lint": {"status": 1, "output": "2 errors found"},
        "step_count": {"status": 0, "output": 10},
    }

    # Status checks
    assert evaluate_deterministic("$step_test.status == 0", node_results) is True
    assert evaluate_deterministic("$step_lint.status == 0", node_results) is False

    # String containment
    assert evaluate_deterministic("'0 failed' in $step_test.output", node_results) is True
    assert evaluate_deterministic("'errors' in $step_lint.output", node_results) is True

    # Methods
    assert evaluate_deterministic("$step_test.output.startswith('3 passed')", node_results) is True

    # Numeric comparisons
    assert evaluate_deterministic("$step_count.output >= 10", node_results) is True
    assert evaluate_deterministic("$step_count.output < 5", node_results) is False

    # Boolean logic
    cond = "$step_test.status == 0 and 'errors' in $step_lint.output"
    assert evaluate_deterministic(cond, node_results) is True

    cond_neg = "not ($step_lint.status == 0)"
    assert evaluate_deterministic(cond_neg, node_results) is True


def test_deterministic_evaluator_safety():
    # Attempting to call unsafe functions or arbitrary code should return False safely
    node_results = {"n1": "test"}
    assert evaluate_deterministic("__import__('os').system('ls')", node_results) is False
    assert evaluate_deterministic("open('/etc/passwd').read()", node_results) is False


def test_nlp_evaluator_mock():
    mock_llm = MagicMock()
    mock_llm.chat.return_value = '{"verdict": true, "reason": "The output confirms success"}'

    verdict = evaluate_nlp(
        condition_str="Does the output show all tests passed cleanly?",
        context_payload={"output": "All 48 tests passed in 1.2s"},
        client=mock_llm,
    )
    assert verdict is True

    mock_llm.chat.return_value = '{"verdict": false, "reason": "Syntax errors found"}'
    verdict_false = evaluate_nlp(
        condition_str="Is the code clean?",
        context_payload={"output": "SyntaxError on line 5"},
        client=mock_llm,
    )
    assert verdict_false is False


def test_evaluate_gate_deterministic_branching():
    gate_node = {
        "id": "gate_check",
        "type": "gate",
        "mode": "deterministic",
        "condition": "$n1.status == 0",
        "if_true": "commit_node",
        "if_false": "fix_node",
    }

    # Case 1: True
    verdict, active, inactive = evaluate_gate(gate_node, {"n1": {"status": 0, "output": "ok"}})
    assert verdict is True
    assert active == ["commit_node"]
    assert inactive == ["fix_node"]

    # Case 2: False
    verdict, active, inactive = evaluate_gate(gate_node, {"n1": {"status": 1, "output": "err"}})
    assert verdict is False
    assert active == ["fix_node"]
    assert inactive == ["commit_node"]
