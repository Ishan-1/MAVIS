"""
tests/test_catalog_manager.py
Tests for core.manager: Tool and agent lifecycles (list, archive/deregister, restore, delete)
and MCP management.
"""
import json
import os
import sys
from pathlib import Path
import pytest

from core.manager import (
    PROTECTED_BASELINE_TOOLS,
    PROTECTED_BASELINE_AGENTS,
    list_all_tools,
    deregister_tool,
    restore_tool,
    delete_tool_permanently,
    list_all_agents,
    deregister_agent,
    restore_agent,
    delete_agent_permanently,
    save_mcp_server,
    toggle_mcp_server,
    delete_mcp_server,
    verify_mcp_server,
    TOOLS_DIR,
    ARCHIVE_TOOLS_DIR,
    AGENTS_DIR,
    ARCHIVE_AGENTS_DIR,
    COMMANDS_LIST_PATH,
    AGENTS_LIST_PATH,
)
from core.tool_retriever import ToolRetriever


@pytest.fixture(autouse=True)
def cleanup_temp_artifacts():
    """Ensure temporary test tools/agents are cleaned up before and after each test."""
    dummy_tools = ["temp_test_tool_abc"]
    dummy_agents = ["temp_test_agent_abc"]

    def _cleanup():
        for t in dummy_tools:
            delete_tool_permanently(t)
        for a in dummy_agents:
            delete_agent_permanently(a)
        delete_mcp_server("dummy_test_mcp_server")

    _cleanup()
    yield
    _cleanup()


def test_protected_baseline_primitive_lock():
    """Verify run_shell_command and semantic_transform cannot be deregistered or permanently deleted."""
    assert "run_shell_command" in PROTECTED_BASELINE_TOOLS
    assert "semantic_transform" in PROTECTED_BASELINE_AGENTS

    ok_dreg, msg_dreg = deregister_tool("run_shell_command")
    assert ok_dreg is False
    assert "protected" in msg_dreg.lower()

    ok_del, msg_del = delete_tool_permanently("run_shell_command")
    assert ok_del is False
    assert "protected" in msg_del.lower()

    ok_adreg, msg_adreg = deregister_agent("semantic_transform")
    assert ok_adreg is False
    assert "protected" in msg_adreg.lower()

    ok_adel, msg_adel = delete_agent_permanently("semantic_transform")
    assert ok_adel is False
    assert "protected" in msg_adel.lower()


def test_tool_deregister_restore_and_delete_lifecycle():
    """Test the full tool lifecycle: create -> deregister/archive -> restore -> permanent delete."""
    tool_name = "temp_test_tool_abc"
    tool_file = TOOLS_DIR / f"{tool_name}.py"
    tool_sig = f"{tool_name}(arg1: str) -> tuple[int, str]"

    code = f'def {tool_name}(arg1: str) -> tuple[int, str]:\n    """Test dummy tool."""\n    return 0, arg1\n'
    tool_file.write_text(code, encoding="utf-8")

    # Register in commands_list.json
    with open(COMMANDS_LIST_PATH, "r") as f:
        cmds = json.load(f)
    cmds[tool_sig] = {"description": "Test dummy tool.", "generalizability": "repurposable"}
    with open(COMMANDS_LIST_PATH, "w") as f:
        json.dump(cmds, f, indent=4)

    # Index in SQLite
    retriever = ToolRetriever()
    retriever.index_tool(tool_sig, "Test dummy tool.", "repurposable")

    # 1. Verify it shows up in list_all_tools() as active
    tools = list_all_tools()
    found = next((t for t in tools if t["name"] == tool_name), None)
    assert found is not None
    assert found["status"] == "active"
    assert found["is_protected"] is False

    # 2. Deregister (Archive)
    ok, msg = deregister_tool(tool_name)
    assert ok is True
    assert not tool_file.exists()
    assert (ARCHIVE_TOOLS_DIR / f"{tool_name}.py").exists()

    # Verify removed from commands_list
    with open(COMMANDS_LIST_PATH, "r") as f:
        cmds_after = json.load(f)
    assert tool_sig not in cmds_after

    # Verify status in list_all_tools is now archived
    tools_after_dreg = list_all_tools()
    found_archived = next((t for t in tools_after_dreg if t["name"] == tool_name), None)
    assert found_archived is not None
    assert found_archived["status"] == "archived"

    # 3. Restore
    ok_res, msg_res = restore_tool(tool_name)
    assert ok_res is True
    assert tool_file.exists()
    assert not (ARCHIVE_TOOLS_DIR / f"{tool_name}.py").exists()

    # Verify restored to commands_list
    with open(COMMANDS_LIST_PATH, "r") as f:
        cmds_restored = json.load(f)
    assert tool_sig in cmds_restored

    # 4. Permanently Delete
    ok_del, msg_del = delete_tool_permanently(tool_name)
    assert ok_del is True
    assert not tool_file.exists()
    assert not (ARCHIVE_TOOLS_DIR / f"{tool_name}.py").exists()

    with open(COMMANDS_LIST_PATH, "r") as f:
        cmds_final = json.load(f)
    assert tool_sig not in cmds_final


def test_agent_deregister_restore_and_delete_lifecycle():
    """Test full subagent lifecycle: create -> deregister/archive -> restore -> permanent delete."""
    agent_name = "temp_test_agent_abc"
    agent_file = AGENTS_DIR / f"{agent_name}.py"
    agent_sig = f"{agent_name}(query: str) -> tuple[int, Any]"

    code = f'# Dummy agent code\nclass {agent_name}:\n    """Dummy agent class."""\n    pass\n'
    agent_file.write_text(code, encoding="utf-8")

    # Register in agents_list.json
    with open(AGENTS_LIST_PATH, "r") as f:
        agents = json.load(f)
    agents[agent_sig] = {"type": "cognitive", "description": "Dummy agent."}
    with open(AGENTS_LIST_PATH, "w") as f:
        json.dump(agents, f, indent=4)

    # 1. Verify it shows up in list_all_agents() as active
    agents_list = list_all_agents()
    found = next((a for a in agents_list if a["name"] == agent_name), None)
    assert found is not None
    assert found["status"] == "active"

    # 2. Deregister (Archive)
    ok_dreg, msg_dreg = deregister_agent(agent_name)
    assert ok_dreg is True
    assert not agent_file.exists()
    assert (ARCHIVE_AGENTS_DIR / f"{agent_name}.py").exists()

    with open(AGENTS_LIST_PATH, "r") as f:
        agents_after = json.load(f)
    assert agent_sig not in agents_after

    # 3. Restore
    ok_res, msg_res = restore_agent(agent_name)
    assert ok_res is True
    assert agent_file.exists()
    assert not (ARCHIVE_AGENTS_DIR / f"{agent_name}.py").exists()

    with open(AGENTS_LIST_PATH, "r") as f:
        agents_restored = json.load(f)
    assert agent_sig in agents_restored

    # 4. Permanently Delete
    ok_del, msg_del = delete_agent_permanently(agent_name)
    assert ok_del is True
    assert not agent_file.exists()
    assert not (ARCHIVE_AGENTS_DIR / f"{agent_name}.py").exists()


def test_mcp_server_management():
    """Test adding, toggling, testing, and deleting an MCP server configuration."""
    mock_server_path = str(Path(__file__).parent / "mock_mcp_server.py")
    server_name = "dummy_test_mcp_server"

    server_conf = {
        "enabled": True,
        "command": sys.executable,
        "args": [mock_server_path],
        "env": {},
    }

    # 1. Save server
    ok_save, msg_save = save_mcp_server(server_name, server_conf)
    assert ok_save is True

    # 2. Toggle server
    ok_tog, msg_tog = toggle_mcp_server(server_name, False)
    assert ok_tog is True

    # 3. Test server handshake with mock server
    ok_test, tools, test_msg = verify_mcp_server(server_name, server_conf)
    assert ok_test is True
    assert "mock_search" in tools

    # 4. Delete server
    ok_del, msg_del = delete_mcp_server(server_name)
    assert ok_del is True
