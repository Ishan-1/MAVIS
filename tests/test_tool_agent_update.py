"""
Unit tests for ToolBuilder.update_tool and AgentBuilder.update_agent.
"""
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from tool_builder.toolbuilder import ToolBuilder, ToolBuildError
from agent_builder.agent_builder import AgentBuilder, AgentBuildError


def test_update_tool_protected_baseline():
    mock_client = MagicMock()
    tb = ToolBuilder(client=mock_client)

    with pytest.raises(ToolBuildError, match="Cannot update protected baseline tool"):
        tb.update_tool("run_shell_command", "Add a new flag")


def test_update_tool_nonexistent():
    mock_client = MagicMock()
    tb = ToolBuilder(client=mock_client)

    with pytest.raises(ToolBuildError, match="does not exist"):
        tb.update_tool("completely_nonexistent_tool_xyz123", "Add a flag")


def test_update_tool_success(tmp_path):
    mock_client = MagicMock()
    tb = ToolBuilder(client=mock_client)

    tool_name = "test_custom_math_tool"
    tool_file = Path(f"tools/{tool_name}.py")
    tool_file.parent.mkdir(parents=True, exist_ok=True)
    tool_file.write_text("def test_custom_math_tool(a: int) -> tuple[int, int]:\n    return 0, a * 2\n", encoding="utf-8")

    # Mock commands_list
    commands_path = "data/commands_list.json"
    existing_commands = {}
    if os.path.exists(commands_path):
        with open(commands_path, "r") as f:
            existing_commands = json.load(f)

    try:
        updated_sig = f"{tool_name}(a: int, multiplier: int = 2) -> tuple[int, int]"
        mock_response = {
            "requirements": [],
            "env": [],
            "generalizability": "repurposable",
            "code": f"def {tool_name}(a: int, multiplier: int = 2) -> tuple[int, int]:\n    return 0, a * multiplier\n",
            "updated_signature": updated_sig,
            "updated_description": "Multiplies a by multiplier",
        }
        mock_test_response = {
            "code": f"def test_{tool_name}():\n    from tools.{tool_name} import {tool_name}\n    status, res = {tool_name}(3)\n    assert status == 0 and res == 6\n"
        }

        tb._llm = MagicMock(side_effect=[
            json.dumps(mock_response),
            json.dumps(mock_test_response),
        ])

        with patch("tool_builder.tester.ToolTester.test_tool", return_value=(0, "All tests passed")), \
             patch.object(tb, "_write_code_to_file", side_effect=lambda path, code: Path(path).write_text(code, encoding="utf-8")), \
             patch("core.tool_retriever.ToolRetriever.index_tool"):
            gen = tb.update_tool(tool_name, "Add optional multiplier argument with default 2")
            assert gen == "repurposable"

        with open(commands_path, "r") as f:
            cmd_data = json.load(f)
        assert updated_sig in cmd_data
        assert cmd_data[updated_sig]["description"] == "Multiplies a by multiplier"

    finally:
        if tool_file.exists():
            tool_file.unlink()
        test_file = Path(f"tests/test_{tool_name}.py")
        if test_file.exists():
            test_file.unlink()
        with open(commands_path, "w") as f:
            json.dump(existing_commands, f, indent=4)


def test_update_agent_protected_baseline():
    mock_client = MagicMock()
    ab = AgentBuilder(client=mock_client)

    with pytest.raises(AgentBuildError, match="Cannot update protected baseline agent"):
        ab.update_agent("semantic_transform", "Add a new prompt instruction")


def test_update_agent_nonexistent():
    mock_client = MagicMock()
    ab = AgentBuilder(client=mock_client)

    with pytest.raises(AgentBuildError, match="does not exist"):
        ab.update_agent("completely_nonexistent_agent_xyz123", "Add instructions")


def test_update_agent_success():
    mock_client = MagicMock()
    ab = AgentBuilder(client=mock_client)

    agent_name = "test_custom_eval_node"
    agent_file = Path(f"agents/{agent_name}.py")
    agent_file.parent.mkdir(parents=True, exist_ok=True)
    agent_file.write_text(
        "from core.agents.base import CognitiveNode\n\n"
        "class TestCustomEvalNodeAgent(CognitiveNode):\n"
        "    name = 'test_custom_eval_node'\n"
        "    description = 'Evaluates text'\n"
        "    def run(self, turn_id='', **inputs):\n"
        "        return 0, 'ok'\n",
        encoding="utf-8"
    )

    catalog_path = "data/agents_list.json"
    existing_agents = {}
    if os.path.exists(catalog_path):
        with open(catalog_path, "r") as f:
            existing_agents = json.load(f)

    try:
        mock_response = {
            "code": (
                "from core.agents.base import CognitiveNode\n\n"
                "class TestCustomEvalNodeAgent(CognitiveNode):\n"
                "    name = 'test_custom_eval_node'\n"
                "    description = 'Evaluates text with metric score'\n"
                "    def run(self, turn_id='', **inputs):\n"
                "        return 0, {'score': 100}\n"
            ),
            "updated_description": "Evaluates text with metric score",
            "updated_input_schema": {"text": "str", "scale": "int"},
            "updated_output_schema": {"score": "int"},
            "generalizability": "repurposable",
        }
        mock_client.generate.return_value = json.dumps(mock_response)

        with patch.object(ab, "_write_agent_file", side_effect=lambda name, code: agent_file.write_text(code, encoding="utf-8")), \
             patch.object(ab.tester, "generate_test_cases", return_value=[]), \
             patch.object(ab.tester, "test_agent", return_value=(0, {"verdict": "passed"})):
            gen = ab.update_agent(agent_name, "Add scale parameter to input schema and return score object")
            assert gen == "repurposable"

        with open(catalog_path, "r") as f:
            ag_data = json.load(f)
        matching = [k for k in ag_data if k.startswith(agent_name)]
        assert len(matching) > 0
        assert ag_data[matching[0]]["description"] == "Evaluates text with metric score"


    finally:
        if agent_file.exists():
            agent_file.unlink()
        with open(catalog_path, "w") as f:
            json.dump(existing_agents, f, indent=4)


def test_tool_tester_tmp_path_fixture_injection(tmp_path):
    """Verify ToolTester automatically injects tmp_path and registers temporary paths in ONI."""
    from tool_builder.tester import ToolTester
    import pathlib

    tool_name = "dummy_tmp_tool"
    tool_file = Path(f"tools/{tool_name}.py")
    test_file = Path(f"tests/test_{tool_name}.py")

    tool_code = f"""
def {tool_name}(path: str, content: str) -> tuple[int, bool]:
    with open(path, "w") as f:
        f.write(content)
    return 0, True
"""

    test_code = f"""
def test_{tool_name}(tmp_path):
    from tools.{tool_name} import {tool_name}
    p = tmp_path / "out.txt"
    status, res = {tool_name}(str(p), "hello tmp_path")
    assert status == 0
    assert p.read_text() == "hello tmp_path"
"""

    try:
        tool_file.write_text(tool_code, encoding="utf-8")
        test_file.write_text(test_code, encoding="utf-8")

        tester = ToolTester(client=None)
        status, result = tester.test_tool(tool_name, check_imports=False)
        assert status == 0, f"Expected status 0, got {status}: {result}"
    finally:
        if tool_file.exists():
            tool_file.unlink()
        if test_file.exists():
            test_file.unlink()

