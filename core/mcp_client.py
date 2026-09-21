"""
core/mcp_client.py
Model Context Protocol (MCP) client adapter for MAVIS.

Enables MAVIS to connect to external MCP servers (stdio transport), discover tools,
normalize schemas into MAVIS signature contracts, and execute tools within DAG pipelines.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from core.helpers import log_it

_ENTITY = "mcp_client"


def json_schema_type_to_python(prop: dict) -> str:
    """Map a JSON schema property dictionary to a Python type annotation string."""
    prop_type = prop.get("type", "Any")
    if isinstance(prop_type, list):
        # Handle union types like ["string", "null"]
        types = [t for t in prop_type if t != "null"]
        if types:
            prop_type = types[0]
        else:
            return "Any"

    mapping = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }
    return mapping.get(str(prop_type).lower(), "Any")


def format_mcp_tool_signature(name: str, input_schema: dict) -> str:
    """
    Convert an MCP tool definition (name + JSON Schema) into a MAVIS Python signature string.

    Example:
        name = "brave_web_search"
        input_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "count": {"type": "integer", "default": 5}
            },
            "required": ["query"]
        }
        Returns: "brave_web_search(query: str, count: int = 5) -> tuple[int, Any]"
    """
    properties = input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
    required = set(input_schema.get("required", [])) if isinstance(input_schema, dict) else set()

    args_parts = []
    # Separate required and optional to ensure valid Python parameter ordering
    req_args = []
    opt_args = []

    for prop_name, prop_def in properties.items():
        if not isinstance(prop_def, dict):
            prop_def = {}
        py_type = json_schema_type_to_python(prop_def)

        if prop_name in required:
            req_args.append(f"{prop_name}: {py_type}")
        else:
            if "default" in prop_def:
                default_val = prop_def["default"]
                if isinstance(default_val, str):
                    default_repr = f'"{default_val}"'
                else:
                    default_repr = str(default_val)
                opt_args.append(f"{prop_name}: {py_type} = {default_repr}")
            else:
                opt_args.append(f"{prop_name}: {py_type} = None")

    all_args = req_args + opt_args
    args_str = ", ".join(all_args)
    return f"{name}({args_str}) -> tuple[int, Any]"


class MCPServerProcess:
    """
    Manages a single long-running MCP server subprocess over stdio JSON-RPC 2.0.
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.command = command
        self.args = args or []
        self.custom_env = env or {}
        self.cwd = cwd
        self.timeout = timeout

        self.proc: Optional[subprocess.Popen] = None
        self._req_id: int = 0
        self._lock = threading.Lock()
        self._pending_requests: Dict[int, Tuple[threading.Event, Dict[str, Any]]] = {}
        self._reader_thread: Optional[threading.Thread] = None
        self.is_connected = False
        self.server_info: Dict[str, Any] = {}
        self.discovered_tools: Dict[str, dict] = {}

    def start(self) -> bool:
        """Launch server process and perform initialization handshake."""
        with self._lock:
            if self.is_connected and self.proc and self.proc.poll() is None:
                return True

            merged_env = os.environ.copy()
            # Substitute environment variables in config env
            for k, v in self.custom_env.items():
                if isinstance(v, str):
                    # Resolve ${VAR} or $VAR
                    expanded = os.path.expandvars(v)
                    merged_env[k] = expanded
                else:
                    merged_env[k] = str(v)

            cmd_list = [self.command] + self.args
            try:
                self.proc = subprocess.Popen(
                    cmd_list,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=merged_env,
                    cwd=self.cwd,
                )
            except Exception as e:
                log_it(f"Failed to spawn MCP server '{self.name}': {e}", _ENTITY, level="ERROR")
                return False

            self._reader_thread = threading.Thread(
                target=self._read_loop,
                name=f"mcp-reader-{self.name}",
                daemon=True,
            )
            self._reader_thread.start()

        # Perform initialize handshake outside the creation lock
        init_res, err = self._initialize()
        if err:
            log_it(f"MCP '{self.name}' initialize failed: {err}", _ENTITY, level="ERROR")
            self.stop()
            return False

        self.server_info = init_res.get("serverInfo", {})
        self.is_connected = True
        log_it(f"MCP server '{self.name}' connected successfully ({self.server_info}).", _ENTITY)
        return True

    def _read_loop(self) -> None:
        """Read stdout lines from the child process and dispatch JSON-RPC responses."""
        if not self.proc or not self.proc.stdout:
            return

        try:
            for line in self.proc.stdout:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    msg = json.loads(line_str)
                except json.JSONDecodeError:
                    # Ignore non-JSON lines or log them
                    continue

                # Handle response
                if "id" in msg:
                    req_id = msg["id"]
                    with self._lock:
                        if req_id in self._pending_requests:
                            event, container = self._pending_requests[req_id]
                            container["response"] = msg
                            event.set()
                # Handle server-initiated requests or notifications (ignoring roots/sampling for now)
        except Exception as exc:
            log_it(f"MCP '{self.name}' reader loop terminated: {exc}", _ENTITY, level="DEBUG")
        finally:
            self.is_connected = False

    def send_request(self, method: str, params: Optional[dict] = None) -> Tuple[Optional[dict], Optional[str]]:
        """Send a JSON-RPC 2.0 request and wait for the response synchronously."""
        if not self.proc or self.proc.poll() is not None:
            return None, f"MCP server '{self.name}' is not running."

        event = threading.Event()
        container: Dict[str, Any] = {}

        with self._lock:
            self._req_id += 1
            req_id = self._req_id
            self._pending_requests[req_id] = (event, container)

        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        try:
            req_str = json.dumps(payload) + "\n"
            if not self.proc.stdin:
                return None, "Process stdin is closed."
            self.proc.stdin.write(req_str)
            self.proc.stdin.flush()
        except Exception as e:
            with self._lock:
                self._pending_requests.pop(req_id, None)
            return None, f"Failed to write to MCP server '{self.name}': {e}"

        # Wait for response
        finished = event.wait(timeout=self.timeout)
        with self._lock:
            self._pending_requests.pop(req_id, None)

        if not finished:
            return None, f"MCP request '{method}' to '{self.name}' timed out after {self.timeout}s."

        response = container.get("response", {})
        if "error" in response:
            err_obj = response["error"]
            err_msg = err_obj.get("message", str(err_obj))
            return None, f"MCP error ({err_obj.get('code', 'unknown')}): {err_msg}"

        return response.get("result", {}), None

    def send_notification(self, method: str, params: Optional[dict] = None) -> None:
        """Send a JSON-RPC 2.0 notification without waiting for a reply."""
        if not self.proc or self.proc.poll() is not None or not self.proc.stdin:
            return

        payload = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        try:
            req_str = json.dumps(payload) + "\n"
            self.proc.stdin.write(req_str)
            self.proc.stdin.flush()
        except Exception:
            pass

    def _initialize(self) -> Tuple[Optional[dict], Optional[str]]:
        """Perform MCP initialize handshake."""
        res, err = self.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": False},
                    "sampling": {},
                },
                "clientInfo": {
                    "name": "MAVIS",
                    "version": "1.3",
                },
            },
        )
        if err:
            return None, err

        # Acknowledge with notifications/initialized
        self.send_notification("notifications/initialized")
        return res, None

    def list_tools(self) -> Tuple[Dict[str, dict], Optional[str]]:
        """Query tools/list from the server and store discovered tools."""
        res, err = self.send_request("tools/list", {})
        if err:
            return {}, err

        tools = res.get("tools", [])
        discovered = {}
        for t in tools:
            t_name = t.get("name")
            if t_name:
                discovered[t_name] = t

        self.discovered_tools = discovered
        return discovered, None

    def call_tool(self, tool_name: str, arguments: dict) -> Tuple[int, Any]:
        """
        Execute tools/call on the server.
        Converts the MCP response to MAVIS execution contract: (status_code: int, result: Any).
        """
        res, err = self.send_request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
        )
        if err:
            return -1, err

        is_error = res.get("isError", False)
        content_items = res.get("content", [])

        # Format output payload
        extracted = []
        for item in content_items:
            if isinstance(item, dict):
                c_type = item.get("type")
                if c_type == "text":
                    extracted.append(item.get("text", ""))
                elif c_type == "image":
                    extracted.append(f"[Image: mimeType={item.get('mimeType', 'unknown')}]")
                elif c_type == "resource":
                    extracted.append(f"[Resource: {item.get('resource', {})}]")
                else:
                    extracted.append(str(item))
            else:
                extracted.append(str(item))

        output_str = "\n".join(extracted) if extracted else str(res)

        # If it looks like JSON, try to parse it so downstream DAG steps get structured data
        parsed_result = output_str
        if output_str.strip().startswith(("{", "[")):
            try:
                parsed_result = json.loads(output_str)
            except Exception:
                parsed_result = output_str

        status_code = -1 if is_error else 0
        return status_code, parsed_result

    def stop(self) -> None:
        """Terminate server process cleanly."""
        with self._lock:
            self.is_connected = False
            if self.proc:
                try:
                    self.proc.terminate()
                    self.proc.wait(timeout=2.0)
                except Exception:
                    try:
                        self.proc.kill()
                    except Exception:
                        pass
                self.proc = None


class MCPManager:
    """
    Central process-wide manager for MCP servers and tools in MAVIS.
    """

    def __init__(self) -> None:
        self.servers: Dict[str, MCPServerProcess] = {}
        # tool_name -> (server_instance, original_mcp_tool_dict)
        self.tool_map: Dict[str, Tuple[MCPServerProcess, dict]] = {}
        # Normalized MAVIS commands dictionary: signature -> metadata
        self.mavis_commands: Dict[str, dict] = {}
        # Quick lookup: func_name -> signature
        self.func_to_sig: Dict[str, str] = {}
        self._lock = threading.Lock()

    def initialize_from_config(self) -> None:
        """Load configured MCP servers from MAVIS config and discover tools."""
        from core.config import cfg

        mcp_cfg = cfg.get("mcp", default={})
        if not isinstance(mcp_cfg, dict) or not mcp_cfg.get("enabled", True):
            log_it("MCP integration is disabled in configuration.", _ENTITY)
            return

        timeout = float(mcp_cfg.get("timeout_seconds", 30.0))
        configured_servers = mcp_cfg.get("servers", {})

        if not isinstance(configured_servers, dict):
            return

        for s_name, s_conf in configured_servers.items():
            if not isinstance(s_conf, dict):
                continue
            # Check enabled flag per server (default True)
            if not s_conf.get("enabled", True):
                continue

            command = s_conf.get("command")
            if not command:
                continue

            args = s_conf.get("args", [])
            env = s_conf.get("env", {})
            cwd = s_conf.get("cwd")

            server_proc = MCPServerProcess(
                name=s_name,
                command=command,
                args=args,
                env=env,
                cwd=cwd,
                timeout=timeout,
            )
            self.servers[s_name] = server_proc

        self.discover_all_tools()

    def discover_all_tools(self) -> Dict[str, dict]:
        """Connect to all servers and refresh available tools."""
        with self._lock:
            self.tool_map.clear()
            self.mavis_commands.clear()
            self.func_to_sig.clear()

            for s_name, server in self.servers.items():
                ok = server.start()
                if not ok:
                    continue

                tools, err = server.list_tools()
                if err:
                    log_it(f"Error listing tools from '{s_name}': {err}", _ENTITY, level="ERROR")
                    continue

                for t_name, t_def in tools.items():
                    # Handle name conflicts by namespacing if already taken
                    target_func_name = t_name
                    if target_func_name in self.func_to_sig:
                        target_func_name = f"{s_name}_{t_name}"

                    sig = format_mcp_tool_signature(
                        target_func_name,
                        t_def.get("inputSchema", {}),
                    )
                    desc = t_def.get("description", f"MCP tool from {s_name}")
                    prefix_desc = f"[MCP: {s_name}] {desc}"

                    self.tool_map[target_func_name] = (server, t_def)
                    self.func_to_sig[target_func_name] = sig
                    self.mavis_commands[sig] = {
                        "description": prefix_desc,
                        "generalizability": "repurposable",
                        "is_mcp": True,
                        "mcp_server": s_name,
                        "mcp_original_name": t_name,
                    }

            log_it(
                f"Discovered {len(self.tool_map)} MCP tool(s) across {len(self.servers)} server(s).",
                _ENTITY,
            )
            return dict(self.mavis_commands)

    def is_mcp_tool(self, func_name: str) -> bool:
        """Check if a function name is backed by an active MCP server."""
        return func_name in self.tool_map

    def call_tool(self, func_name: str, params: dict) -> Tuple[int, Any]:
        """Route tool invocation to the corresponding MCP server."""
        if func_name not in self.tool_map:
            return -1, f"Unknown MCP tool: '{func_name}'"

        server, t_def = self.tool_map[func_name]
        orig_name = t_def.get("name", func_name)
        return server.call_tool(orig_name, params)

    def get_status_summary(self) -> List[Dict[str, Any]]:
        """Return human-readable status for all configured MCP servers."""
        summaries = []
        for s_name, server in self.servers.items():
            tool_count = sum(1 for (s, _) in self.tool_map.values() if s.name == s_name)
            summaries.append({
                "server": s_name,
                "command": f"{server.command} {' '.join(server.args)}",
                "connected": server.is_connected,
                "tool_count": tool_count,
                "info": server.server_info,
            })
        return summaries

    def shutdown(self) -> None:
        """Cleanly stop all running MCP servers."""
        with self._lock:
            for server in self.servers.values():
                server.stop()
            self.servers.clear()
            self.tool_map.clear()
            self.mavis_commands.clear()
            self.func_to_sig.clear()


# Process-wide singleton
mcp_manager = MCPManager()
