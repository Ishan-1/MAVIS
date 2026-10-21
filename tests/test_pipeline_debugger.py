"""
tests/test_pipeline_debugger.py

Unit tests for PipelineDebugger (diagnose_and_repair, error handling, telemetry).
"""
import json
import pytest
from unittest.mock import MagicMock

from core.pipeline_debugger import PipelineDebugger


def test_pipeline_debugger_patch_params():
    """Verify PipelineDebugger parses a patch_params repair plan."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = json.dumps({
        "action": "patch_params",
        "diagnosis": "Missing directory prefix 'docs/'.",
        "patched_node": {
            "id": "n2",
            "type": "tool",
            "function_name": "read_and_concatenate_files",
            "params": {"filenames": ["docs/bugs.md", "docs/Caching.md"]}
        }
    })

    debugger = PipelineDebugger(mock_llm)
    plan = debugger.diagnose_and_repair(
        failed_node={"id": "n2", "type": "tool", "function_name": "read_and_concatenate_files", "params": {"filenames": "bugs.md"}},
        resolved_params={"filenames": "bugs.md"},
        error_message="Failed to read file 'bugs.md': [Errno 2] No such file or directory",
        node_results={"n1": "bugs.md\nCaching.md"},
        user_query="Summarize files in docs/",
        turn_id="test1",
    )

    assert plan["action"] == "patch_params"
    assert "docs/" in plan["diagnosis"]
    assert plan["patched_node"]["params"]["filenames"] == ["docs/bugs.md", "docs/Caching.md"]


def test_pipeline_debugger_replace_node():
    """Verify PipelineDebugger supports replacing a failed node with an alternative tool/agent."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = json.dumps({
        "action": "replace_node",
        "diagnosis": "Tool search_academic_papers failed; switching to general search_news.",
        "patched_node": {
            "id": "n1",
            "type": "tool",
            "function_name": "search_news",
            "params": {"query": "AI advances"}
        }
    })

    debugger = PipelineDebugger(mock_llm)
    plan = debugger.diagnose_and_repair(
        failed_node={"id": "n1", "type": "tool", "function_name": "search_academic_papers", "params": {"query": "AI"}},
        resolved_params={"query": "AI"},
        error_message="API Rate limit exceeded (429)",
        node_results={},
        user_query="Find recent AI papers",
        turn_id="test2",
    )

    assert plan["action"] == "replace_node"
    assert plan["patched_node"]["function_name"] == "search_news"


def test_pipeline_debugger_unrecoverable():
    """Verify PipelineDebugger handles unrecoverable errors gracefully."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = json.dumps({
        "action": "unrecoverable",
        "diagnosis": "Invalid API credentials for remote service.",
        "patched_node": None
    })

    debugger = PipelineDebugger(mock_llm)
    plan = debugger.diagnose_and_repair(
        failed_node={"id": "n1", "type": "tool", "function_name": "custom_api", "params": {}},
        resolved_params={},
        error_message="401 Unauthorized",
        node_results={},
        user_query="Fetch private repo stats",
        turn_id="test3",
    )

    assert plan["action"] == "unrecoverable"
    assert plan["patched_node"] is None


def test_pipeline_debugger_handles_llm_exception():
    """Verify PipelineDebugger gracefully returns unrecoverable if LLM raises an exception."""
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = RuntimeError("LLM connection timed out")

    debugger = PipelineDebugger(mock_llm)
    plan = debugger.diagnose_and_repair(
        failed_node={"id": "n1", "type": "tool", "function_name": "some_tool", "params": {}},
        resolved_params={},
        error_message="Some error",
        node_results={},
        user_query="Test query",
        turn_id="test4",
    )

    assert plan["action"] == "unrecoverable"
    assert "exception" in plan["diagnosis"].lower()
    assert plan["patched_node"] is None


def test_pipeline_debugger_memory_interaction():
    """Verify PipelineDebugger retrieves past priors and records successful fixes to MemoryStore & Neo4j."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = json.dumps({
        "action": "patch_params",
        "diagnosis": "Missing docs prefix.",
        "patched_node": {
            "id": "n2",
            "type": "tool",
            "function_name": "read_files",
            "params": {"filenames": ["docs/bugs.md"]}
        }
    })

    mock_mem = MagicMock()
    mock_mem.retrieve_context.return_value = "Past fix: prefix files with docs/"
    mock_kg = MagicMock()
    mock_kg.is_available.return_value = True
    mock_mem.kg = mock_kg

    debugger = PipelineDebugger(mock_llm, memory_store=mock_mem)
    plan = debugger.diagnose_and_repair(
        failed_node={"id": "n2", "type": "tool", "function_name": "read_files", "params": {"filenames": "bugs.md"}},
        resolved_params={"filenames": "bugs.md"},
        error_message="File not found: bugs.md",
        node_results={},
        user_query="Read docs files",
        turn_id="test5",
    )

    # 1. Verify memory priors were retrieved
    mock_mem.retrieve_context.assert_called_once()
    assert "read_files" in mock_mem.retrieve_context.call_args[0][0]

    # 2. Verify diagnosis payload sent to LLM included memory context
    prompt_sent = mock_llm.generate.call_args[0][0]
    assert "Past fix: prefix files with docs/" in prompt_sent

    # 3. Verify fix was written to MemoryStore long-term storage
    mock_mem.write_long_term.assert_called_once()
    fix_entry = mock_mem.write_long_term.call_args[0][0]
    assert fix_entry["command"] == "read_files"
    assert "Missing docs prefix" in fix_entry["diagnosis"]

    # 4. Verify fix was written to Neo4j graph
    mock_kg.add_pipeline_fix.assert_called_once_with(
        command_name="read_files",
        error_snippet="File not found: bugs.md",
        fix_summary="Missing docs prefix.",
        client=mock_llm,
    )

