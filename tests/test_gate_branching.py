"""
Unit tests for gate nodes in DAG ordering and dynamic branch pruning.
"""
import pytest
from core.dag import (
    validate_and_sort_dag,
    compute_pruned_nodes,
    extract_node_dependencies,
)


def test_dag_with_gate_topological_sort():
    pipeline = [
        {
            "id": "step_tests",
            "type": "tool",
            "function_name": "pytest",
            "params": {},
        },
        {
            "id": "gate_test_pass",
            "type": "gate",
            "condition": "$step_tests.status == 0",
            "if_true": "step_commit",
            "if_false": "step_diagnose",
        },
        {
            "id": "step_commit",
            "type": "tool",
            "function_name": "git_commit",
            "params": {},
        },
        {
            "id": "step_diagnose",
            "type": "tool",
            "function_name": "diagnose_errors",
            "params": {},
        },
    ]

    sorted_pipeline, err = validate_and_sort_dag(pipeline)
    assert err is None
    ids = [n["id"] for n in sorted_pipeline]
    # step_tests must come before gate_test_pass
    assert ids.index("step_tests") < ids.index("gate_test_pass")
    # gate_test_pass must come before both branches
    assert ids.index("gate_test_pass") < ids.index("step_commit")
    assert ids.index("gate_test_pass") < ids.index("step_diagnose")


def test_compute_pruned_nodes_true_branch():
    gate = {
        "id": "gate_1",
        "type": "gate",
        "if_true": ["step_true_1", "step_true_2"],
        "if_false": ["step_false_1", "step_false_2"],
    }
    nodes_by_id = {
        "gate_1": gate,
        "step_true_1": {"id": "step_true_1"},
        "step_true_2": {"id": "step_true_2"},
        "step_false_1": {"id": "step_false_1"},
        "step_false_2": {"id": "step_false_2"},
    }
    node_deps = {
        "gate_1": set(),
        "step_true_1": {"gate_1"},
        "step_true_2": {"step_true_1"},
        "step_false_1": {"gate_1"},
        "step_false_2": {"step_false_1"},
    }

    # When verdict is True, false branch is pruned
    pruned = compute_pruned_nodes(gate, verdict=True, nodes_by_id=nodes_by_id, node_deps=node_deps)
    assert "step_false_1" in pruned
    assert "step_false_2" in pruned
    assert "step_true_1" not in pruned
    assert "step_true_2" not in pruned

    # When verdict is False, true branch is pruned
    pruned_false = compute_pruned_nodes(gate, verdict=False, nodes_by_id=nodes_by_id, node_deps=node_deps)
    assert "step_true_1" in pruned_false
    assert "step_true_2" in pruned_false
    assert "step_false_1" not in pruned_false
    assert "step_false_2" not in pruned_false
