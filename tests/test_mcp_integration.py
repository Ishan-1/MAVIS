"""
tests/test_mcp_integration.py
Integration tests for MCP client with mock MCP server.
"""
import os
import sys
import pytest

from core.mcp_client import MCPServerProcess, MCPManager
from oni.config import ONIConfig
from oni.permissions import classify_command


def test_mock_mcp_server_lifecycle():
    mock_server_path = os.path.join(os.path.dirname(__file__), "mock_mcp_server.py")
    server = MCPServerProcess(
        name="test_mock",
        command=sys.executable,
        args=[mock_server_path],
        timeout=5.0,
    )

    try:
        # 1. Start and handshake
        connected = server.start()
        assert connected is True
        assert server.is_connected is True
        assert server.server_info.get("name") == "mock-mcp-server"

        # 2. List tools
        tools, err = server.list_tools()
        assert err is None
        assert "mock_search" in tools
        assert "mock_fail" in tools

        # 3. Call tool successfully
        status, res = server.call_tool("mock_search", {"query": "pytest", "limit": 2})
        assert status == 0
        assert isinstance(res, list)
        assert len(res) == 2
        assert res[0]["title"] == "Result for pytest"

        # 4. Call tool with failure
        fail_status, fail_res = server.call_tool("mock_fail", {})
        assert fail_status == -1
        assert "Mock simulated tool failure" in fail_res

    finally:
        server.stop()
        assert server.is_connected is False


def test_mcp_manager_discovery_and_routing():
    mock_server_path = os.path.join(os.path.dirname(__file__), "mock_mcp_server.py")
    manager = MCPManager()
    server = MCPServerProcess(
        name="mock_svc",
        command=sys.executable,
        args=[mock_server_path],
        timeout=5.0,
    )
    manager.servers["mock_svc"] = server

    try:
        discovered = manager.discover_all_tools()
        assert len(discovered) == 2

        # Check signature generation
        search_sigs = [k for k in discovered.keys() if "mock_search" in k]
        assert len(search_sigs) == 1
        assert "mock_search(query: str, limit: int = 5) -> tuple[int, Any]" in search_sigs[0]

        # Verify tool routing
        assert manager.is_mcp_tool("mock_search") is True
        status, res = manager.call_tool("mock_search", {"query": "AI", "limit": 3})
        assert status == 0
        assert len(res) == 3

    finally:
        manager.shutdown()
        assert len(manager.servers) == 0
        assert len(manager.tool_map) == 0


def test_mcp_tool_oni_classification():
    """Verify that external MCP tools respect ONI's trust level rules."""
    class MockConfig:
        whitelist = {"get_current_datetime"}
        greylist = {"mock_search"}
        blacklist = {"dangerous_tool"}

    config = MockConfig()

    # Greylisted tool prompts confirmation under 'ask' mode
    assert classify_command("mock_search", config, "ask") == "greylist"

    # Whitelisted under yolo mode
    assert classify_command("mock_search", config, "yolo") == "allow"

    # Blacklist blocks under 'ask' and 'whitelist_only' mode
    assert classify_command("dangerous_tool", config, "ask") == "deny"
    assert classify_command("dangerous_tool", config, "whitelist_only") == "deny"
