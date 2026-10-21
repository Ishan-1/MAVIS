"""
tests/test_dag.py

Pytest test suite for DAG validation, topological sorting, and dependency resolution in MAVIS.
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.dag import (
    extract_dependencies_from_value,
    extract_node_dependencies,
    validate_and_sort_dag,
    resolve_params,
)


def test_extract_dependencies_scalar_and_nested():
    """Verify dependencies are extracted from scalars, lists, and nested dictionaries."""
    assert extract_dependencies_from_value("$n1.output") == {"n1"}
    assert extract_dependencies_from_value("$n2") == {"n2"}
    assert extract_dependencies_from_value("plain string") == set()

    # List of dependencies
    list_val = ["$n1.output", "static", "$n2.field"]
    assert extract_dependencies_from_value(list_val) == {"n1", "n2"}

    # Nested dictionary
    nested_val = {
        "outer": {
            "inner": "$n3.output",
            "items": ["$n4.status", "$n5"],
        },
        "other": "val"
    }
    assert extract_dependencies_from_value(nested_val) == {"n3", "n4", "n5"}


def test_extract_node_dependencies():
    """Verify extracting dependencies directly from a node dict."""
    node = {
        "id": "step_b",
        "function_name": "process_data",
        "params": {
            "source_files": ["$step_a.output"],
            "options": {"extra": "$step_0.config"}
        }
    }
    assert extract_node_dependencies(node) == {"step_a", "step_0"}

    # Node with empty or non-dict params
    assert extract_node_dependencies({"id": "step_c", "params": None}) == set()
    assert extract_node_dependencies({"id": "step_d"}) == set()


def test_linear_dag_topological_sort():
    """Verify out-of-order linear pipeline is sorted into correct execution sequence."""
    pipeline = [
        {"id": "n3", "function_name": "final_step", "params": {"data": "$n2.output"}},
        {"id": "n1", "function_name": "first_step", "params": {"query": "hello"}},
        {"id": "n2", "function_name": "second_step", "params": {"data": "$n1.output"}},
    ]

    sorted_pipeline, err = validate_and_sort_dag(pipeline)
    assert err is None
    assert sorted_pipeline is not None
    order = [node["id"] for node in sorted_pipeline]
    assert order == ["n1", "n2", "n3"]


def test_diamond_dag_topological_sort():
    """Verify a diamond DAG: n1 -> (n2, n3) -> n4 is sorted so dependencies execute first."""
    pipeline = [
        {"id": "n4", "function_name": "aggregate", "params": {"b": "$n2.output", "c": "$n3.output"}},
        {"id": "n2", "function_name": "branch_a", "params": {"src": "$n1.output"}},
        {"id": "n3", "function_name": "branch_b", "params": {"src": "$n1.output"}},
        {"id": "n1", "function_name": "source", "params": {}},
    ]

    sorted_pipeline, err = validate_and_sort_dag(pipeline)
    assert err is None
    assert sorted_pipeline is not None
    order = [node["id"] for node in sorted_pipeline]

    # n1 must be first, n4 must be last; n2 and n3 must both be between n1 and n4
    assert order[0] == "n1"
    assert order[-1] == "n4"
    assert order.index("n2") < order.index("n4")
    assert order.index("n3") < order.index("n4")


def test_cycle_detection_direct():
    """Verify 2-node circular dependency is detected and aborted with an error."""
    pipeline = [
        {"id": "n1", "function_name": "step_a", "params": {"input": "$n2.output"}},
        {"id": "n2", "function_name": "step_b", "params": {"input": "$n1.output"}},
    ]

    sorted_pipeline, err = validate_and_sort_dag(pipeline)
    assert sorted_pipeline is None
    assert err is not None
    assert "cycle" in err.lower()


def test_cycle_detection_self():
    """Verify self-dependency is detected immediately."""
    pipeline = [
        {"id": "n1", "function_name": "step_a", "params": {"input": "$n1.output"}},
    ]

    sorted_pipeline, err = validate_and_sort_dag(pipeline)
    assert sorted_pipeline is None
    assert err is not None
    assert "cycle" in err.lower() or "depends on itself" in err.lower()


def test_cycle_detection_multi_node():
    """Verify multi-node loop: n1 -> n2 -> n3 -> n1 is detected."""
    pipeline = [
        {"id": "n1", "function_name": "step_a", "params": {"input": "$n3.output"}},
        {"id": "n2", "function_name": "step_b", "params": {"input": "$n1.output"}},
        {"id": "n3", "function_name": "step_c", "params": {"input": "$n2.output"}},
    ]

    sorted_pipeline, err = validate_and_sort_dag(pipeline)
    assert sorted_pipeline is None
    assert err is not None
    assert "cycle" in err.lower()


def test_dangling_dependency_detection():
    """Verify referencing an undeclared step triggers a clean validation error."""
    pipeline = [
        {"id": "n1", "function_name": "step_a", "params": {"data": "$missing_step.output"}},
    ]

    sorted_pipeline, err = validate_and_sort_dag(pipeline)
    assert sorted_pipeline is None
    assert err is not None
    assert "unknown step" in err.lower()
    assert "missing_step" in err


def test_duplicate_node_id_detection():
    """Verify duplicate node IDs are rejected."""
    pipeline = [
        {"id": "step_1", "function_name": "action_a", "params": {}},
        {"id": "step_1", "function_name": "action_b", "params": {}},
    ]

    sorted_pipeline, err = validate_and_sort_dag(pipeline)
    assert sorted_pipeline is None
    assert err is not None
    assert "duplicate" in err.lower()


def test_resolve_params():
    """Verify resolving parameters against completed node outputs."""
    node_results = {
        "n1": "raw_string_result",
        "n2": {"status_code": 200, "data": [1, 2, 3]},
    }

    # Direct output resolution
    res, err = resolve_params("$n1.output", node_results)
    assert err is None
    assert res == "raw_string_result"

    # Status resolution
    res, err = resolve_params("$n1.status", node_results)
    assert err is None
    assert res == 0

    # Dict field resolution
    res, err = resolve_params("$n2.data", node_results)
    assert err is None
    assert res == [1, 2, 3]

    # Nested dictionary and list resolution
    params = {
        "summary": "$n1.output",
        "items": ["$n2.data", "literal"],
        "nested": {"code": "$n2.status_code"}
    }
    resolved, err = resolve_params(params, node_results)
    assert err is None
    assert resolved == {
        "summary": "raw_string_result",
        "items": [[1, 2, 3], "literal"],
        "nested": {"code": 200}
    }

    # Missing dependency result
    missing_res, missing_err = resolve_params("$n3.output", node_results)
    assert missing_res is None
    assert missing_err is not None
    assert "n3" in missing_err


def test_resolve_params_embedded_string():
    """Verify embedded $node_id and $node_id.output references within strings are interpolated."""
    node_results = {
        "n1": "feat: initialize repository",
        "n2": {"code": 0, "msg": "all tests passed"},
        "n3": ["file1.py", "file2.py"],
    }

    # Shell command with quoted $node.output
    cmd = 'git commit -m "$n1.output"'
    res, err = resolve_params(cmd, node_results)
    assert err is None
    assert res == 'git commit -m "feat: initialize repository"'

    # Embedded $node without .output
    cmd2 = 'git commit -a -m "$n1"'
    res2, err = resolve_params(cmd2, node_results)
    assert err is None
    assert res2 == 'git commit -a -m "feat: initialize repository"'

    # Multiple embedded references and dict/status field
    msg = 'Status: $n2.code, result: $n2.msg'
    res3, err = resolve_params(msg, node_results)
    assert err is None
    assert res3 == 'Status: 0, result: all tests passed'

    # Embedded list serialization
    log = 'Files: $n3.output'
    res4, err = resolve_params(log, node_results)
    assert err is None
    assert res4 == 'Files: ["file1.py", "file2.py"]'

    # Embedded reference to missing dependency fails cleanly
    missing_cmd = 'git commit -m "$missing_node.output"'
    res_miss, err_miss = resolve_params(missing_cmd, node_results)
    assert res_miss is None
    assert err_miss is not None
    assert "missing_node" in err_miss


def test_execute_pipeline_topological_execution():
    """Integration test verifying execute_pipeline sorts out-of-order nodes and executes in order."""
    from main import execute_pipeline

    executed_calls = []

    def mock_call_command(cmd, params):
        executed_calls.append((cmd, params))
        if cmd == "fetch_urls":
            return 0, ["https://example.com/1", "https://example.com/2"]
        elif cmd == "scrape_content":
            return 0, "Scraped webpage text"
        return 0, "ok"

    pipeline = [
        {
            "id": "step2",
            "function_name": "scrape_content",
            "params": {"urls": "$step1.output"}
        },
        {
            "id": "step1",
            "function_name": "fetch_urls",
            "params": {"topic": "AI"}
        }
    ]

    with patch("main._oni.preflight_scan", return_value=(True, [])), \
         patch("main.call_command", side_effect=mock_call_command), \
         patch("main.answerer.synthesize", return_value="Final Answer"):

        execute_pipeline(pipeline, query="Test Query")

    # Ensure step1 (fetch_urls) executed first, followed by step2 (scrape_content) with step1's output
    assert len(executed_calls) == 2
    assert executed_calls[0][0] == "fetch_urls"
    assert executed_calls[0][1] == {"topic": "AI"}

    assert executed_calls[1][0] == "scrape_content"
    assert executed_calls[1][1] == {"urls": ["https://example.com/1", "https://example.com/2"]}


def test_execute_pipeline_handles_cycle_gracefully():
    """Verify that execute_pipeline aborts without executing any tools when a cycle is detected."""
    from main import execute_pipeline

    executed_calls = []

    def mock_call_command(cmd, params):
        executed_calls.append(cmd)
        return 0, "ok"

    cyclic_pipeline = [
        {"id": "n1", "function_name": "cmd_a", "params": {"x": "$n2.output"}},
        {"id": "n2", "function_name": "cmd_b", "params": {"x": "$n1.output"}},
    ]

    with patch("main.call_command", side_effect=mock_call_command), \
         patch("main.mavis_error") as mock_err:

        execute_pipeline(cyclic_pipeline, query="Cycle Query")

    # Nothing should have executed
    assert len(executed_calls) == 0
    mock_err.assert_called_once()
    assert "cycle" in mock_err.call_args[0][0].lower()


def test_execute_pipeline_handles_dangling_dep_gracefully():
    """Verify that execute_pipeline aborts without executing any tools when a dependency is missing."""
    from main import execute_pipeline

    executed_calls = []

    def mock_call_command(cmd, params):
        executed_calls.append(cmd)
        return 0, "ok"

    broken_pipeline = [
        {"id": "n1", "function_name": "cmd_a", "params": {"x": "$nonexistent.output"}},
    ]

    with patch("main.call_command", side_effect=mock_call_command), \
         patch("main.mavis_error") as mock_err:

        execute_pipeline(broken_pipeline, query="Broken Query")

    # Nothing should have executed
    assert len(executed_calls) == 0
    mock_err.assert_called_once()
    assert "unknown step" in mock_err.call_args[0][0].lower()


def test_tool_receives_untruncated_output_and_scratchpad_dereferencing(tmp_path):
    """Verify downstream tool steps receive full untruncated output and dereference scratchpad notices."""
    from main import execute_pipeline
    from core.dag import dereference_scratchpad_value
    from pathlib import Path

    # 1. Verify dereference_scratchpad_value
    scratch_file = Path("data/scratch/test_deref_dummy.txt")
    scratch_file.parent.mkdir(parents=True, exist_ok=True)
    full_text = "COMPLETE UNTRUNCATED CODE REPORT CONTENT " * 100
    scratch_file.write_text(full_text, encoding="utf-8")

    try:
        digest_str = f"Head\n... [50 lines / 4500 bytes offloaded to file: {scratch_file}] ...\nTail"
        derefed = dereference_scratchpad_value(digest_str)
        assert derefed == full_text

        # 2. Integration test: step 1 produces 10KB, step 2 (tool) receives full 10KB
        executed_params = []

        def mock_call(cmd, params):
            if cmd == "producer":
                return 0, full_text
            if cmd == "consumer":
                executed_params.append(params.get("data"))
                return 0, "written"
            return 0, "ok"

        pipeline = [
            {"id": "s1", "type": "tool", "function_name": "producer", "params": {}},
            {"id": "s2", "type": "tool", "function_name": "consumer", "params": {"data": "$s1.output"}},
        ]

        with patch("main._oni.preflight_scan", return_value=(True, [])), \
             patch("main.call_command", side_effect=mock_call):
            results = execute_pipeline(pipeline, query="Test Query", turn_id="test_untruncated_turn")

        assert len(executed_params) == 1
        assert executed_params[0] == full_text
        assert "offloaded to file" not in executed_params[0]

    finally:
        if scratch_file.exists():
            scratch_file.unlink()


def test_execute_pipeline_is_goal_wave_suppresses_answerer():
    """Verify execute_pipeline does not call answerer.synthesize when is_goal_wave=True."""
    from main import execute_pipeline

    pipeline = [
        {"id": "s1", "type": "tool", "function_name": "dummy_cmd", "params": {}},
    ]

    with patch("main._oni.preflight_scan", return_value=(True, [])), \
         patch("main.call_command", return_value=(0, "step done")), \
         patch("main.answerer.synthesize") as mock_synth:

        mock_synth.return_value = "Synthesized CLI answer"
        execute_pipeline(pipeline, query="Goal Query", turn_id="wave_turn", is_goal_wave=True)
        mock_synth.assert_not_called()

        execute_pipeline(pipeline, query="Single CLI Query", turn_id="cli_turn", is_goal_wave=False)
        mock_synth.assert_called_once()


