"""
tests/test_mcp_client.py
Unit tests for MCP client components: schema converter, signature generator, and return contracts.
"""
import pytest
from core.mcp_client import (
    json_schema_type_to_python,
    format_mcp_tool_signature,
    MCPServerProcess,
)


def test_json_schema_type_to_python():
    assert json_schema_type_to_python({"type": "string"}) == "str"
    assert json_schema_type_to_python({"type": "integer"}) == "int"
    assert json_schema_type_to_python({"type": "number"}) == "float"
    assert json_schema_type_to_python({"type": "boolean"}) == "bool"
    assert json_schema_type_to_python({"type": "array"}) == "list"
    assert json_schema_type_to_python({"type": "object"}) == "dict"
    assert json_schema_type_to_python({"type": ["string", "null"]}) == "str"
    assert json_schema_type_to_python({}) == "Any"


def test_format_mcp_tool_signature_simple():
    name = "search_web"
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer", "default": 10},
        },
        "required": ["query"],
    }
    sig = format_mcp_tool_signature(name, schema)
    assert sig == "search_web(query: str, count: int = 10) -> tuple[int, Any]"


def test_format_mcp_tool_signature_no_required():
    name = "get_status"
    schema = {
        "type": "object",
        "properties": {
            "verbose": {"type": "boolean", "default": False},
        },
    }
    sig = format_mcp_tool_signature(name, schema)
    assert sig == "get_status(verbose: bool = False) -> tuple[int, Any]"


def test_format_mcp_tool_signature_empty_schema():
    name = "ping"
    sig = format_mcp_tool_signature(name, {})
    assert sig == "ping() -> tuple[int, Any]"
