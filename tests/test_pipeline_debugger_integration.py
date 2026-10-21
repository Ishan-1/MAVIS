"""
tests/test_pipeline_debugger_integration.py

Integration tests for live DAG execution recovery with PipelineDebugger.
"""
from unittest.mock import MagicMock, patch
import pytest

from main import execute_pipeline


def test_execute_pipeline_recovers_via_pipeline_debugger():
    """Verify that a failing step is caught, diagnosed, repaired, and successfully completed."""
    call_counts = {"read_files": 0}

    def mock_call_command(cmd, params):
        if cmd == "list_dir":
            return 0, "bugs.md\nCaching.md"
        elif cmd == "read_files":
            call_counts["read_files"] += 1
            # Fails on first call with relative path, succeeds after debugger patches with docs/ prefix
            filenames = params.get("filenames")
            if filenames == "bugs.md\nCaching.md":
                return -1, "Failed to read file 'bugs.md': [Errno 2] No such file or directory"
            elif filenames == ["docs/bugs.md", "docs/Caching.md"]:
                return 0, "All file contents read successfully."
        return 0, "ok"

    pipeline = [
        {"id": "n1", "type": "tool", "function_name": "list_dir", "params": {"path": "docs"}},
        {"id": "n2", "type": "tool", "function_name": "read_files", "params": {"filenames": "$n1.output"}},
    ]

    mock_repair_plan = {
        "action": "patch_params",
        "diagnosis": "Missing directory prefix 'docs/'. Prepending 'docs/'.",
        "patched_node": {
            "id": "n2",
            "type": "tool",
            "function_name": "read_files",
            "params": {"filenames": ["docs/bugs.md", "docs/Caching.md"]},
        },
    }

    with patch("main._oni.preflight_scan", return_value=(True, [])), \
         patch("main.call_command", side_effect=mock_call_command), \
         patch("main.pipeline_debugger.diagnose_and_repair", return_value=mock_repair_plan) as mock_diag, \
         patch("main.answerer.synthesize", return_value="Here is the summary of docs files.") as mock_synth:

        res = execute_pipeline(pipeline, query="Summarize docs/")

        assert res is not None
        assert res["n1"] == "bugs.md\nCaching.md"
        assert res["n2"] == "All file contents read successfully."
        assert call_counts["read_files"] == 2
        mock_diag.assert_called_once()
        mock_synth.assert_called_once()


def test_execute_pipeline_handles_unrecoverable_failure_and_hands_off_to_answerer():
    """Verify that an unrecoverable failure passes partial results and error details to Answerer."""
    def mock_call_command(cmd, params):
        if cmd == "fetch_data":
            return 0, "Raw data payload"
        elif cmd == "broken_tool":
            return -1, "Permanent network failure (503 Service Unavailable)"
        return 0, "ok"

    pipeline = [
        {"id": "n1", "type": "tool", "function_name": "fetch_data", "params": {}},
        {"id": "n2", "type": "tool", "function_name": "broken_tool", "params": {"input": "$n1.output"}},
    ]

    mock_unrecoverable_plan = {
        "action": "unrecoverable",
        "diagnosis": "Remote service 503 is offline.",
        "patched_node": None,
    }

    with patch("main._oni.preflight_scan", return_value=(True, [])), \
         patch("main.call_command", side_effect=mock_call_command), \
         patch("main.pipeline_debugger.diagnose_and_repair", return_value=mock_unrecoverable_plan), \
         patch("main.answerer.synthesize", return_value="I fetched the data, but the processing tool is down.") as mock_synth:

        res = execute_pipeline(pipeline, query="Process external data")

        # Returns None to indicate execution failure
        assert res is None
        # But answerer.synthesize MUST be called with partial results including the failure info
        mock_synth.assert_called_once()
        synth_args = mock_synth.call_args[1]
        pipeline_results = synth_args["pipeline_results"]
        assert "n1" in pipeline_results
        assert pipeline_results["n1"] == "Raw data payload"
        assert "n2" in pipeline_results
        assert "FAILED" in pipeline_results["n2"]
