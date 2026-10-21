"""
Unit tests for symmetrical AgentBuilder, AgentTester, and AgentDebugger (cognitive nodes and subagents).
"""
import json
import pytest
from unittest.mock import MagicMock, patch, mock_open

from core.agents.base import CognitiveNode
from core.agents.subagent import Subagent
from agent_builder.agent_builder import AgentBuilder
from agent_builder.tester import AgentTester
from agent_builder.debugger import AgentDebugger


def test_agent_tester_cognitive_node():
    mock_llm = MagicMock()
    # LLM test input generator
    mock_llm.generate.side_effect = [
        # 1. generate_test_cases
        json.dumps({
            "test_cases": [
                {"id": "tc_1", "description": "Happy path", "inputs": {"text": "hello world"}}
            ]
        }),
        # 2. agent run output (CognitiveNode.run)
        "Summary: hello world",
        # 3. LLM judge
        json.dumps({"verdict": "passed", "reason": "Accurate summary"}),
    ]

    class MockCognitiveAgent(CognitiveNode):
        name = "mock_cognitive"
        description = "Summarizes text"
        system_instruction = "You summarize text"
        input_schema = {"text": "str"}

    tester = AgentTester(mock_llm)
    agent_instance = MockCognitiveAgent(mock_llm)
    status, result = tester.test_agent(agent_instance)
    assert status == 0
    assert "passed" in result["summary"]


def test_agent_tester_subagent_with_executor():
    mock_llm = MagicMock()
    # LLM test input generator + ReAct loop + LLM judge
    mock_llm.generate.side_effect = [
        # 1. generate_test_cases
        json.dumps({
            "test_cases": [
                {"id": "tc_1", "description": "Happy path", "inputs": {"query": "find file"}}
            ]
        }),
        # 2. Subagent turn 1: Call tool
        'Action: {"tool": "find_file", "params": {"pattern": "*.py"}}',
        # 3. Subagent turn 2: Final answer
        'Final Answer: Found 3 files.',
        # 4. LLM judge
        json.dumps({"verdict": "passed", "reason": "Successfully used tools to find files"}),
    ]

    class MockSubagent(Subagent):
        name = "mock_subagent"
        description = "Searches for files"
        system_instruction = "Search and report"
        default_max_turns = 4
        allowed_tools = ["find_file"]
        input_schema = {"query": "str"}

    mock_executor = MagicMock(return_value=(0, "main.py, test.py, core.py"))
    tester = AgentTester(mock_llm)
    subagent_instance = MockSubagent(mock_llm)

    status, result = tester.test_agent(
        subagent_instance,
        executor=mock_executor,
        available_tools={"find_file": "Find files matching pattern"},
    )

    assert status == 0
    assert "passed" in result["summary"]
    mock_executor.assert_called_once_with("find_file", {"pattern": "*.py"})


def test_agent_debugger_dispatches_by_type():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = json.dumps({
        "code": "class FixedAgent: pass",
        "fix_summary": "Tuned negative constraints"
    })

    debugger = AgentDebugger(mock_llm)

    # 1. Cognitive debug
    code, summary = debugger.debug_agent(
        agent_name="my_cog",
        agent_description="1-shot task",
        broken_code="class BrokenCog: pass",
        failed_case={"inputs": {"x": 1}},
        actual_output="bad output",
        failure_reason="Schema violated",
        agent_type="cognitive",
    )
    assert code == "class FixedAgent: pass"
    prompt_used = mock_llm.generate.call_args[0][0]
    assert "cognitive" in prompt_used.lower()

    # 2. Subagent debug
    code, summary = debugger.debug_agent(
        agent_name="my_sub",
        agent_description="ReAct loop",
        broken_code="class BrokenSub: pass",
        failed_case={"inputs": {"x": 1}},
        actual_output="exceeded turns",
        failure_reason="Max turns reached",
        agent_type="subagent",
    )
    assert code == "class FixedAgent: pass"
    prompt_used = mock_llm.generate.call_args[0][0]
    assert "react" in prompt_used.lower() or "subagent" in prompt_used.lower()


def test_agent_builder_build_cognitive(tmp_path):
    mock_llm = MagicMock()
    # Code generation response
    cog_code = '''from core.agents.base import CognitiveNode

class TestCogAgent(CognitiveNode):
    name = "test_cog"
    description = "Test cognitive unit"
    system_instruction = "Analyze data"
    input_schema = {"data": "str"}
    output_schema = None
'''
    mock_llm.generate.side_effect = [
        # build_agent code synthesis
        json.dumps({"code": cog_code}),
        # test_agent test case generation
        json.dumps({"test_cases": [{"id": "tc1", "inputs": {"data": "test"}}]}),
        # judge
        json.dumps({"verdict": "passed", "reason": "Correct"}),
    ]

    builder = AgentBuilder(mock_llm)

    with patch.object(builder, "_write_agent_file") as mock_write, \
         patch.object(builder, "_register_agent") as mock_reg, \
         patch("agent_builder.agent_builder.load_agent_with_error") as mock_load:

        mock_instance = MagicMock(spec=CognitiveNode)
        mock_instance.name = "test_cog"
        mock_instance.description = "Test cognitive unit"
        mock_instance.system_instruction = "Analyze data"
        mock_instance.input_schema = {"data": "str"}
        mock_instance.output_schema = None
        mock_instance.run.return_value = (0, "Analysis result")
        mock_load.return_value = (mock_instance, None)

        gen = builder.build_agent(
            agent_name="test_cog",
            agent_description="Test cognitive unit",
            input_schema={"data": "str"},
            agent_type="cognitive",
        )

        assert gen == "specialized"
        mock_write.assert_called_once()
        mock_reg.assert_called_once_with(
            agent_name="test_cog",
            description="Test cognitive unit",
            input_schema={"data": "str"},
            agent_type="cognitive",
            default_max_turns=4,
            allowed_tools=None,
            generalizability="specialized",
        )


def test_agent_builder_build_subagent(tmp_path):
    mock_llm = MagicMock()
    sub_code = '''from core.agents.subagent import Subagent

class TestSubAgent(Subagent):
    name = "test_sub"
    description = "Test subagent"
    system_instruction = "Think Act Observe"
    default_max_turns = 5
    allowed_tools = ["tool_a"]
    input_schema = {"data": "str"}
    output_schema = None
'''
    mock_llm.generate.side_effect = [
        # build_agent code synthesis
        json.dumps({"code": sub_code}),
        # test_agent test case generation
        json.dumps({"test_cases": [{"id": "tc1", "inputs": {"data": "test"}}]}),
        # judge
        json.dumps({"verdict": "passed", "reason": "Correct"}),
    ]

    builder = AgentBuilder(mock_llm)

    with patch.object(builder, "_write_agent_file") as mock_write, \
         patch.object(builder, "_register_agent") as mock_reg, \
         patch("agent_builder.agent_builder.load_agent_with_error") as mock_load:

        mock_instance = MagicMock(spec=Subagent)
        mock_instance.name = "test_sub"
        mock_instance.description = "Test subagent"
        mock_instance.system_instruction = "Think Act Observe"
        mock_instance.input_schema = {"data": "str"}
        mock_instance.output_schema = None
        mock_instance.run.return_value = (0, "Final Answer: Done")
        mock_load.return_value = (mock_instance, None)

        gen = builder.build_agent(
            agent_name="test_sub",
            agent_description="Test subagent",
            input_schema={"data": "str"},
            agent_type="subagent",
            default_max_turns=5,
            allowed_tools=["tool_a"],
        )

        assert gen == "specialized"
        mock_write.assert_called_once()
        mock_reg.assert_called_once_with(
            agent_name="test_sub",
            description="Test subagent",
            input_schema={"data": "str"},
            agent_type="subagent",
            default_max_turns=5,
            allowed_tools=["tool_a"],
            generalizability="specialized",
        )


def test_agent_builder_debugger_console_output():
    """Verify AgentDebugger outputs console status and ok messages when repairing an agent."""
    mock_llm = MagicMock()
    code_initial = "class BrokenAgent: pass"
    code_repaired = "class RepairedAgent: pass"

    mock_llm.generate.side_effect = [
        # 1. Initial build code
        json.dumps({"code": code_initial}),
        # 2. Test cases
        json.dumps({"test_cases": [{"id": "tc1", "inputs": {"x": 1}}]}),
        # 3. Judge verdict attempt 0 (fails)
        json.dumps({"verdict": "failed", "reason": "Expected output format missing"}),
        # 4. Debugger fix
        json.dumps({"code": code_repaired, "fix_summary": "Added proper output format."}),
        # 5. Judge verdict attempt 1 (passes)
        json.dumps({"verdict": "passed", "reason": "Output correctly formatted"}),
    ]

    builder = AgentBuilder(mock_llm)

    with patch.object(builder, "_write_agent_file"), \
         patch.object(builder, "_register_agent"), \
         patch("agent_builder.agent_builder.load_agent_with_error") as mock_load, \
         patch("agent_builder.agent_builder.mavis_status") as mock_status, \
         patch("agent_builder.agent_builder.mavis_ok") as mock_ok, \
         patch("builtins.open", mock_open(read_data="class BrokenAgent: pass")):

        mock_instance = MagicMock(spec=CognitiveNode)
        mock_instance.name = "retry_cog"
        mock_instance.description = "Agent requiring debug retry"
        mock_instance.input_schema = {"x": "int"}
        mock_instance.output_schema = None
        mock_instance.run.return_value = (0, "Some output")
        mock_load.return_value = (mock_instance, None)

        gen = builder.build_agent(
            agent_name="retry_cog",
            agent_description="Agent requiring debug retry",
            agent_type="cognitive",
        )

        assert gen == "specialized"
        # Verify [AgentDebugger] status and ok were broadcast to user terminal
        status_calls = [c.args[0] for c in mock_status.call_args_list]
        ok_calls = [c.args[0] for c in mock_ok.call_args_list]

        assert any("[AgentDebugger]" in msg and "failed verification" in msg for msg in status_calls)
        assert any("[AgentDebugger] Repair action:" in msg for msg in status_calls)
        assert any("[AgentDebugger]" in msg and "repaired and verified" in msg for msg in ok_calls)
