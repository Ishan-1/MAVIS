"""
tests/mock_mcp_server.py
A minimal mock MCP server communicating over stdio JSON-RPC 2.0.
"""
import json
import sys

def main():
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except Exception:
            continue

        method = req.get("method")
        req_id = req.get("id")

        if method == "initialize":
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mock-mcp-server", "version": "1.0.0"},
                },
            }
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()

        elif method == "notifications/initialized":
            # Notification: no response
            pass

        elif method == "tools/list":
            res = {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "mock_search",
                            "description": "Mock search tool for testing.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "limit": {"type": "integer", "default": 5},
                                },
                                "required": ["query"],
                            },
                        },
                        {
                            "name": "mock_fail",
                            "description": "Mock tool that returns an error.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {},
                            },
                        },
                    ]
                },
            }
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()

        elif method == "tools/call":
            params = req.get("params", {})
            t_name = params.get("name")
            arguments = params.get("arguments", {})

            if t_name == "mock_search":
                q = arguments.get("query", "")
                limit = arguments.get("limit", 5)
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps([{"title": f"Result for {q}", "rank": i} for i in range(limit)]),
                            }
                        ],
                        "isError": False,
                    },
                }
            elif t_name == "mock_fail":
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": "Mock simulated tool failure"}
                        ],
                        "isError": True,
                    },
                }
            else:
                res = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method/tool {t_name} not found"},
                }

            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()
