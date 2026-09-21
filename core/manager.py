"""
core/manager.py
Central catalog and lifecycle management engine for MAVIS.
Provides unified CRUD, archiving, restoration, and deletion across:
- Tools (filesystem, commands_list.json, sys.modules, SQLite tools_registry, Neo4j)
- Subagents & Cognitive Nodes (filesystem, agents_list.json, sys.modules, memories)
- MCP Servers & Discovered Tools (data/mavis_config.json, MCPServerProcess)
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.config import cfg
from core.helpers import log_it
from core.tool_retriever import ToolRetriever

_ENTITY = "catalog_manager"
_REPO_ROOT = Path(__file__).resolve().parent.parent

TOOLS_DIR = _REPO_ROOT / "tools"
AGENTS_DIR = _REPO_ROOT / "agents"
ARCHIVE_TOOLS_DIR = _REPO_ROOT / "data" / "archived_tools"
ARCHIVE_AGENTS_DIR = _REPO_ROOT / "data" / "archived_agents"
ARCHIVE_REGISTRY_PATH = _REPO_ROOT / "data" / "archive_registry.json"
COMMANDS_LIST_PATH = _REPO_ROOT / "data" / "commands_list.json"
AGENTS_LIST_PATH = _REPO_ROOT / "data" / "agents_list.json"
CONFIG_PATH = _REPO_ROOT / "data" / "mavis_config.json"

PROTECTED_BASELINE_TOOLS = {"run_shell_command"}
PROTECTED_BASELINE_AGENTS = {"semantic_transform"}


# ── Archive Registry Helpers ───────────────────────────────────────────────────

def _load_archive_registry() -> dict:
    """Load the JSON archive catalog or return empty default structure."""
    if not ARCHIVE_REGISTRY_PATH.exists():
        return {"tools": {}, "agents": {}}
    try:
        with open(ARCHIVE_REGISTRY_PATH, "r") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {"tools": {}, "agents": {}}
            data.setdefault("tools", {})
            data.setdefault("agents", {})
            return data
    except Exception as exc:
        log_it(f"Failed to read archive registry: {exc}", _ENTITY)
        return {"tools": {}, "agents": {}}


def _save_archive_registry(registry: dict) -> None:
    """Persist the archive catalog to disk."""
    ARCHIVE_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ARCHIVE_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)


def _load_commands_list() -> dict:
    """Load data/commands_list.json."""
    if not COMMANDS_LIST_PATH.exists():
        return {}
    try:
        with open(COMMANDS_LIST_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_commands_list(commands: dict) -> None:
    """Save to data/commands_list.json."""
    COMMANDS_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COMMANDS_LIST_PATH, "w") as f:
        json.dump(commands, f, indent=4)


def _load_agents_list() -> dict:
    """Load data/agents_list.json."""
    if not AGENTS_LIST_PATH.exists():
        return {}
    try:
        with open(AGENTS_LIST_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_agents_list(agents: dict) -> None:
    """Save to data/agents_list.json."""
    AGENTS_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AGENTS_LIST_PATH, "w") as f:
        json.dump(agents, f, indent=4)


def _extract_func_name_from_sig(sig: str) -> str:
    """Extract clean function name from signature string like 'foo(bar: str)'."""
    return sig.split("(")[0].strip()


def _extract_docstring_and_sig_from_code(code: str, default_name: str) -> Tuple[str, str]:
    """Inspect Python code AST to extract docstring and construct basic signature."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == default_name:
                doc = ast.get_docstring(node) or ""
                params = [arg.arg for arg in node.args.args]
                sig = f"{node.name}({', '.join(params)})"
                return doc, sig
            elif isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node) or ""
                return doc, f"{node.name}()"
    except Exception:
        pass
    return "", f"{default_name}()"


# ── Tools Management API ───────────────────────────────────────────────────────

def list_all_tools() -> List[Dict[str, Any]]:
    """
    List all tools across active filesystem, archived storage, and MCP servers.
    """
    tools: List[Dict[str, Any]] = []
    seen_names = set()

    commands = _load_commands_list()
    archive = _load_archive_registry().get("tools", {})

    # 1. Scan active tools in tools/
    if TOOLS_DIR.exists():
        for py_file in sorted(TOOLS_DIR.glob("*.py")):
            if py_file.name.startswith("__"):
                continue
            name = py_file.stem
            seen_names.add(name)

            # Match signature from commands_list.json
            sig = None
            meta = {}
            for s, m in commands.items():
                if _extract_func_name_from_sig(s) == name:
                    sig = s
                    meta = m if isinstance(m, dict) else {"description": str(m)}
                    break

            try:
                code = py_file.read_text(encoding="utf-8")
            except Exception:
                code = ""

            docstring, fallback_sig = _extract_docstring_and_sig_from_code(code, name)

            tools.append({
                "name": name,
                "signature": sig or fallback_sig,
                "description": meta.get("description") or docstring or "No description provided.",
                "generalizability": meta.get("generalizability", "generalizable" if name in PROTECTED_BASELINE_TOOLS else "repurposable"),
                "status": "active",
                "is_protected": (name in PROTECTED_BASELINE_TOOLS),
                "file_path": str(py_file),
                "source_code": code,
                "is_mcp": False,
                "mcp_server": None,
            })

    # 2. Scan archived tools in data/archived_tools/
    if ARCHIVE_TOOLS_DIR.exists():
        for py_file in sorted(ARCHIVE_TOOLS_DIR.glob("*.py")):
            if py_file.name.startswith("__"):
                continue
            name = py_file.stem
            if name in seen_names:
                continue
            seen_names.add(name)

            archived_meta = archive.get(name, {})
            sig = archived_meta.get("signature")

            try:
                code = py_file.read_text(encoding="utf-8")
            except Exception:
                code = ""

            docstring, fallback_sig = _extract_docstring_and_sig_from_code(code, name)

            tools.append({
                "name": name,
                "signature": sig or fallback_sig,
                "description": archived_meta.get("description") or docstring or "Archived tool.",
                "generalizability": archived_meta.get("generalizability", "repurposable"),
                "status": "archived",
                "is_protected": False,
                "file_path": str(py_file),
                "source_code": code,
                "is_mcp": False,
                "mcp_server": None,
            })

    # 3. Add MCP tools
    try:
        from core.mcp_client import mcp_manager
        for sig, meta in mcp_manager.mavis_commands.items():
            tool_name = _extract_func_name_from_sig(sig)
            if tool_name in seen_names:
                continue
            server_name = meta.get("server") or "mcp"
            tools.append({
                "name": tool_name,
                "signature": sig,
                "description": meta.get("description", "Discovered from MCP server."),
                "generalizability": meta.get("generalizability", "repurposable"),
                "status": "mcp",
                "is_protected": False,
                "file_path": f"mcp://{server_name}/{tool_name}",
                "source_code": f"# MCP Tool supplied by server: {server_name}\n# Signature: {sig}",
                "is_mcp": True,
                "mcp_server": server_name,
            })
    except Exception:
        pass

    return tools


def deregister_tool(tool_name: str) -> Tuple[bool, str]:
    """
    Deregister (archive) an active tool:
    - Moves tools/<tool_name>.py to data/archived_tools/<tool_name>.py
    - Records metadata in data/archive_registry.json
    - Removes from data/commands_list.json
    - Evicts from sys.modules
    - Purges from SQLite tools_registry.db
    """
    if tool_name in PROTECTED_BASELINE_TOOLS:
        return False, f"Cannot deregister protected baseline primitive '{tool_name}'."

    py_path = TOOLS_DIR / f"{tool_name}.py"
    if not py_path.exists():
        return False, f"Active tool file '{py_path}' does not exist."

    ARCHIVE_TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = ARCHIVE_TOOLS_DIR / f"{tool_name}.py"

    # Find tool signature and metadata in commands_list.json
    commands = _load_commands_list()
    found_sig = None
    meta = {}
    for s, m in list(commands.items()):
        if _extract_func_name_from_sig(s) == tool_name:
            found_sig = s
            meta = m if isinstance(m, dict) else {"description": str(m)}
            del commands[s]
            break

    # If signature wasn't in commands_list, inspect code
    if not found_sig:
        try:
            code = py_path.read_text(encoding="utf-8")
            doc, found_sig = _extract_docstring_and_sig_from_code(code, tool_name)
            meta = {"description": doc, "generalizability": "repurposable"}
        except Exception:
            found_sig = f"{tool_name}()"
            meta = {"description": "User tool", "generalizability": "repurposable"}

    # Move file to archive
    try:
        shutil.move(str(py_path), str(target_path))
    except Exception as exc:
        return False, f"Failed to move file to archive: {exc}"

    # Save to archive registry
    archive = _load_archive_registry()
    archive["tools"][tool_name] = {
        "signature": found_sig,
        "description": meta.get("description", ""),
        "generalizability": meta.get("generalizability", "repurposable"),
    }
    _save_archive_registry(archive)

    # Save updated commands_list
    _save_commands_list(commands)

    # Evict from sys.modules
    sys.modules.pop(f"tools.{tool_name}", None)
    sys.modules.pop(tool_name, None)

    # Clean pycache
    pycache_dir = TOOLS_DIR / "__pycache__"
    if pycache_dir.exists():
        for f in pycache_dir.glob(f"{tool_name}.*.pyc"):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass

    # Evict from SQLite registry
    try:
        retriever = ToolRetriever()
        retriever.delete_tool(tool_name)
    except Exception as exc:
        log_it(f"Failed to evict tool from SQLite during deregister: {exc}", _ENTITY)

    log_it(f"Deregistered and archived tool '{tool_name}'", _ENTITY)
    return True, f"Tool '{tool_name}' successfully deregistered and moved to archive."


def restore_tool(tool_name: str) -> Tuple[bool, str]:
    """
    Restore an archived tool:
    - Moves data/archived_tools/<tool_name>.py back to tools/<tool_name>.py
    - Restores entry in data/commands_list.json
    - Re-indexes in SQLite tools_registry.db
    - Cleans record from data/archive_registry.json
    """
    archived_path = ARCHIVE_TOOLS_DIR / f"{tool_name}.py"
    if not archived_path.exists():
        return False, f"Archived tool file '{archived_path}' does not exist."

    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = TOOLS_DIR / f"{tool_name}.py"

    archive = _load_archive_registry()
    meta = archive.get("tools", {}).pop(tool_name, {})

    sig = meta.get("signature")
    if not sig:
        try:
            code = archived_path.read_text(encoding="utf-8")
            doc, sig = _extract_docstring_and_sig_from_code(code, tool_name)
            meta["description"] = doc
            meta["generalizability"] = "repurposable"
        except Exception:
            sig = f"{tool_name}() -> tuple[int, Any]"

    # Move back to active directory
    try:
        shutil.move(str(archived_path), str(target_path))
    except Exception as exc:
        return False, f"Failed to restore file to tools/: {exc}"

    # Re-add to commands_list.json
    commands = _load_commands_list()
    commands[sig] = {
        "description": meta.get("description", ""),
        "generalizability": meta.get("generalizability", "repurposable"),
    }
    _save_commands_list(commands)
    _save_archive_registry(archive)

    # Re-index in SQLite
    try:
        retriever = ToolRetriever()
        retriever.index_tool(sig, meta.get("description", ""), generalizability=meta.get("generalizability", "repurposable"))
    except Exception as exc:
        log_it(f"Failed to re-index restored tool in SQLite: {exc}", _ENTITY)

    log_it(f"Restored tool '{tool_name}' to active tools", _ENTITY)
    return True, f"Tool '{tool_name}' successfully restored."


def delete_tool_permanently(tool_name: str) -> Tuple[bool, str]:
    """
    Permanently delete a tool:
    - Destroys file from tools/ or data/archived_tools/
    - Purges from data/commands_list.json and data/archive_registry.json
    - Evicts from sys.modules
    - Purges from SQLite tools_registry.db
    - Detaches from Neo4j Knowledge Graph (if online)
    """
    if tool_name in PROTECTED_BASELINE_TOOLS:
        return False, f"Cannot permanently delete protected baseline primitive '{tool_name}'."

    deleted_any = False

    # Check active
    py_path = TOOLS_DIR / f"{tool_name}.py"
    if py_path.exists():
        try:
            py_path.unlink()
            deleted_any = True
        except Exception as exc:
            return False, f"Failed to delete active file: {exc}"

    # Check archive
    arch_path = ARCHIVE_TOOLS_DIR / f"{tool_name}.py"
    if arch_path.exists():
        try:
            arch_path.unlink()
            deleted_any = True
        except Exception as exc:
            return False, f"Failed to delete archived file: {exc}"

    # Purge pycache
    for parent_dir in (TOOLS_DIR, ARCHIVE_TOOLS_DIR):
        pycache = parent_dir / "__pycache__"
        if pycache.exists():
            for f in pycache.glob(f"{tool_name}.*.pyc"):
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass

    # Purge from commands_list
    commands = _load_commands_list()
    to_del = [s for s in commands if _extract_func_name_from_sig(s) == tool_name]
    for s in to_del:
        del commands[s]
    if to_del:
        _save_commands_list(commands)
        deleted_any = True

    # Purge from archive registry
    archive = _load_archive_registry()
    if tool_name in archive.get("tools", {}):
        del archive["tools"][tool_name]
        _save_archive_registry(archive)
        deleted_any = True

    # Evict from sys.modules
    sys.modules.pop(f"tools.{tool_name}", None)
    sys.modules.pop(tool_name, None)

    # Purge from SQLite
    try:
        retriever = ToolRetriever()
        retriever.delete_tool(tool_name)
    except Exception as exc:
        log_it(f"SQLite purge error for '{tool_name}': {exc}", _ENTITY)

    # Purge from Neo4j (if available)
    try:
        from memories.memory_store import MemoryStore
        from core.llm import get_llm_client
        store = MemoryStore(get_llm_client(), namespace="interpreter")
        if store.kg and getattr(store.kg, "_driver", None):
            with store.kg._driver.session() as session:
                session.run("MATCH (t:Tool {name: $name}) DETACH DELETE t", name=tool_name)
                log_it(f"Detached tool '{tool_name}' from Neo4j Knowledge Graph", _ENTITY)
    except Exception:
        pass

    if not deleted_any:
        return False, f"Tool '{tool_name}' was not found in active or archived catalogs."

    log_it(f"Permanently deleted tool '{tool_name}'", _ENTITY)
    return True, f"Tool '{tool_name}' permanently deleted from filesystem and registries."


# ── Subagents Management API ──────────────────────────────────────────────────

def list_all_agents() -> List[Dict[str, Any]]:
    """
    List all subagents across active filesystem and archived storage.
    """
    agents: List[Dict[str, Any]] = []
    seen_names = set()

    agents_list = _load_agents_list()
    archive = _load_archive_registry().get("agents", {})

    # 1. Scan active agents in agents/
    if AGENTS_DIR.exists():
        for py_file in sorted(AGENTS_DIR.glob("*.py")):
            if py_file.name.startswith("__"):
                continue
            name = py_file.stem
            seen_names.add(name)

            sig = None
            meta = {}
            for s, m in agents_list.items():
                if _extract_func_name_from_sig(s) == name:
                    sig = s
                    meta = m if isinstance(m, dict) else {}
                    break

            try:
                code = py_file.read_text(encoding="utf-8")
            except Exception:
                code = ""

            docstring, fallback_sig = _extract_docstring_and_sig_from_code(code, name)
            is_prot = (name in PROTECTED_BASELINE_AGENTS)

            agents.append({
                "name": name,
                "signature": sig or fallback_sig,
                "type": meta.get("type", "cognitive"),
                "description": meta.get("description") or docstring or "Subagent module.",
                "generalizability": meta.get("generalizability", "generalizable" if is_prot else "specialized"),
                "allowed_tools": meta.get("allowed_tools", []),
                "default_max_turns": meta.get("default_max_turns", 4),
                "status": "active",
                "is_protected": is_prot,
                "file_path": str(py_file),
                "source_code": code,
            })

    # 2. Scan archived agents in data/archived_agents/
    if ARCHIVE_AGENTS_DIR.exists():
        for py_file in sorted(ARCHIVE_AGENTS_DIR.glob("*.py")):
            if py_file.name.startswith("__"):
                continue
            name = py_file.stem
            if name in seen_names:
                continue
            seen_names.add(name)

            archived_meta = archive.get(name, {})
            sig = archived_meta.get("signature")

            try:
                code = py_file.read_text(encoding="utf-8")
            except Exception:
                code = ""

            docstring, fallback_sig = _extract_docstring_and_sig_from_code(code, name)

            agents.append({
                "name": name,
                "signature": sig or fallback_sig,
                "type": archived_meta.get("type", "cognitive"),
                "description": archived_meta.get("description") or docstring or "Archived agent.",
                "generalizability": archived_meta.get("generalizability", "specialized"),
                "allowed_tools": archived_meta.get("allowed_tools", []),
                "default_max_turns": archived_meta.get("default_max_turns", 4),
                "status": "archived",
                "is_protected": False,
                "file_path": str(py_file),
                "source_code": code,
            })

    return agents


def deregister_agent(agent_name: str) -> Tuple[bool, str]:
    """
    Deregister (archive) an active agent:
    - Moves agents/<agent_name>.py to data/archived_agents/<agent_name>.py
    - Records snapshot in data/archive_registry.json
    - Removes from data/agents_list.json
    - Evicts from sys.modules
    """
    if agent_name in PROTECTED_BASELINE_AGENTS:
        return False, f"Cannot deregister protected baseline cognitive primitive '{agent_name}'."

    py_path = AGENTS_DIR / f"{agent_name}.py"
    if not py_path.exists():
        return False, f"Active agent file '{py_path}' does not exist."

    ARCHIVE_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = ARCHIVE_AGENTS_DIR / f"{agent_name}.py"

    agents_list = _load_agents_list()
    found_sig = None
    meta = {}
    for s, m in list(agents_list.items()):
        if _extract_func_name_from_sig(s) == agent_name:
            found_sig = s
            meta = m if isinstance(m, dict) else {}
            del agents_list[s]
            break

    if not found_sig:
        found_sig = f"{agent_name}()"
        meta = {"type": "cognitive", "description": "Subagent module"}

    try:
        shutil.move(str(py_path), str(target_path))
    except Exception as exc:
        return False, f"Failed to move agent file to archive: {exc}"

    archive = _load_archive_registry()
    archive["agents"][agent_name] = {
        "signature": found_sig,
        "type": meta.get("type", "cognitive"),
        "description": meta.get("description", ""),
        "generalizability": meta.get("generalizability", "specialized"),
        "allowed_tools": meta.get("allowed_tools", []),
        "default_max_turns": meta.get("default_max_turns", 4),
    }
    _save_archive_registry(archive)
    _save_agents_list(agents_list)

    sys.modules.pop(f"agents.{agent_name}", None)
    sys.modules.pop(agent_name, None)

    pycache_dir = AGENTS_DIR / "__pycache__"
    if pycache_dir.exists():
        for f in pycache_dir.glob(f"{agent_name}.*.pyc"):
            try:
                f.unlink(missing_ok=True)
            except Exception:
                pass

    log_it(f"Deregistered and archived agent '{agent_name}'", _ENTITY)
    return True, f"Agent '{agent_name}' successfully deregistered and moved to archive."


def restore_agent(agent_name: str) -> Tuple[bool, str]:
    """
    Restore an archived agent:
    - Moves data/archived_agents/<agent_name>.py back to agents/<agent_name>.py
    - Restores entry in data/agents_list.json
    - Cleans record from data/archive_registry.json
    """
    archived_path = ARCHIVE_AGENTS_DIR / f"{agent_name}.py"
    if not archived_path.exists():
        return False, f"Archived agent file '{archived_path}' does not exist."

    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = AGENTS_DIR / f"{agent_name}.py"

    archive = _load_archive_registry()
    meta = archive.get("agents", {}).pop(agent_name, {})

    sig = meta.get("signature") or f"{agent_name}()"

    try:
        shutil.move(str(archived_path), str(target_path))
    except Exception as exc:
        return False, f"Failed to restore file to agents/: {exc}"

    agents_list = _load_agents_list()
    entry = {
        "type": meta.get("type", "cognitive"),
        "description": meta.get("description", ""),
        "generalizability": meta.get("generalizability", "specialized"),
    }
    if meta.get("type") == "subagent":
        entry["allowed_tools"] = meta.get("allowed_tools", [])
        entry["default_max_turns"] = meta.get("default_max_turns", 4)

    agents_list[sig] = entry
    _save_agents_list(agents_list)
    _save_archive_registry(archive)

    log_it(f"Restored agent '{agent_name}' to active agents", _ENTITY)
    return True, f"Agent '{agent_name}' successfully restored."


def delete_agent_permanently(agent_name: str) -> Tuple[bool, str]:
    """
    Permanently delete an agent:
    - Destroys file from agents/ or data/archived_agents/
    - Purges from data/agents_list.json and data/archive_registry.json
    - Evicts from sys.modules
    """
    if agent_name in PROTECTED_BASELINE_AGENTS:
        return False, f"Cannot permanently delete protected baseline cognitive primitive '{agent_name}'."

    deleted_any = False

    py_path = AGENTS_DIR / f"{agent_name}.py"
    if py_path.exists():
        try:
            py_path.unlink()
            deleted_any = True
        except Exception as exc:
            return False, f"Failed to delete active agent file: {exc}"

    arch_path = ARCHIVE_AGENTS_DIR / f"{agent_name}.py"
    if arch_path.exists():
        try:
            arch_path.unlink()
            deleted_any = True
        except Exception as exc:
            return False, f"Failed to delete archived agent file: {exc}"

    for parent_dir in (AGENTS_DIR, ARCHIVE_AGENTS_DIR):
        pycache = parent_dir / "__pycache__"
        if pycache.exists():
            for f in pycache.glob(f"{agent_name}.*.pyc"):
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass

    agents_list = _load_agents_list()
    to_del = [s for s in agents_list if _extract_func_name_from_sig(s) == agent_name]
    for s in to_del:
        del agents_list[s]
    if to_del:
        _save_agents_list(agents_list)
        deleted_any = True

    archive = _load_archive_registry()
    if agent_name in archive.get("agents", {}):
        del archive["agents"][agent_name]
        _save_archive_registry(archive)
        deleted_any = True

    sys.modules.pop(f"agents.{agent_name}", None)
    sys.modules.pop(agent_name, None)

    if not deleted_any:
        return False, f"Agent '{agent_name}' was not found in active or archived catalogs."

    log_it(f"Permanently deleted agent '{agent_name}'", _ENTITY)
    return True, f"Agent '{agent_name}' permanently deleted."


# ── MCP Management API ─────────────────────────────────────────────────────────

def get_mcp_overview() -> Dict[str, Any]:
    """
    Retrieve configured MCP servers, active process statuses, and discovered tools.
    """
    mcp_cfg = cfg.get("mcp", default={})
    servers_cfg = mcp_cfg.get("servers", {}) if isinstance(mcp_cfg, dict) else {}

    overview: Dict[str, Any] = {
        "enabled": mcp_cfg.get("enabled", True),
        "timeout_seconds": mcp_cfg.get("timeout_seconds", 30),
        "servers": {},
        "discovered_tools": [],
    }

    from core.mcp_client import mcp_manager

    for s_name, s_conf in servers_cfg.items():
        if not isinstance(s_conf, dict):
            continue

        running = False
        pid = None
        server_instance = mcp_manager.servers.get(s_name)
        if server_instance and server_instance.is_connected:
            running = True
            pid = getattr(server_instance, "_proc", None)
            pid = pid.pid if pid else None

        tool_count = 0
        server_tools = []
        for sig, meta in mcp_manager.mavis_commands.items():
            if meta.get("server") == s_name:
                tool_count += 1
                server_tools.append({
                    "signature": sig,
                    "name": _extract_func_name_from_sig(sig),
                    "description": meta.get("description", ""),
                    "input_schema": meta.get("input_schema", {}),
                })

        overview["servers"][s_name] = {
            "name": s_name,
            "enabled": s_conf.get("enabled", True),
            "command": s_conf.get("command", ""),
            "args": s_conf.get("args", []),
            "env": s_conf.get("env", {}),
            "cwd": s_conf.get("cwd", None),
            "is_running": running,
            "pid": pid,
            "tool_count": tool_count,
            "tools": server_tools,
        }

    overview["discovered_tools"] = list(mcp_manager.mavis_commands.keys())
    return overview


def save_mcp_server(name: str, server_conf: dict) -> Tuple[bool, str]:
    """Save or update an MCP server configuration in data/mavis_config.json."""
    clean_name = name.strip()
    if not clean_name:
        return False, "Server name cannot be empty."

    cmd = server_conf.get("command")
    if not cmd:
        return False, "Server command is required (e.g. 'npx', 'python')."

    try:
        mcp_cfg = cfg.get("mcp", default={})
        servers = mcp_cfg.setdefault("servers", {})
        servers[clean_name] = {
            "enabled": server_conf.get("enabled", True),
            "command": cmd,
            "args": server_conf.get("args", []),
            "env": server_conf.get("env", {}),
        }
        if server_conf.get("cwd"):
            servers[clean_name]["cwd"] = server_conf["cwd"]

        cfg.set("mcp", "servers", servers)
        cfg.save()
        log_it(f"Saved MCP server configuration for '{clean_name}'", _ENTITY)
        return True, f"MCP server '{clean_name}' configuration saved."
    except Exception as exc:
        return False, f"Failed to save MCP server: {exc}"


def toggle_mcp_server(name: str, enabled: bool) -> Tuple[bool, str]:
    """Enable or disable an MCP server in configuration."""
    mcp_cfg = cfg.get("mcp", default={})
    servers = mcp_cfg.get("servers", {})
    if name not in servers:
        return False, f"Server '{name}' does not exist in configuration."

    servers[name]["enabled"] = enabled
    cfg.set("mcp", "servers", servers)
    cfg.save()
    status_str = "enabled" if enabled else "disabled"
    return True, f"MCP server '{name}' {status_str}."


def delete_mcp_server(name: str) -> Tuple[bool, str]:
    """Delete an MCP server from configuration and terminate if running."""
    from core.mcp_client import mcp_manager
    mcp_cfg = cfg.get("mcp", default={})
    servers = mcp_cfg.get("servers", {})
    if name not in servers:
        return False, f"Server '{name}' not found in configuration."

    del servers[name]
    cfg.set("mcp", "servers", servers)
    cfg.save()

    # Stop server if active
    server_instance = mcp_manager.servers.pop(name, None)
    if server_instance:
        server_instance.stop()

    return True, f"MCP server '{name}' removed from configuration."


def verify_mcp_server(name: str, server_conf: dict) -> Tuple[bool, List[str], str]:
    """
    Test handshake and discover tools on an MCP server process without modifying global state.
    """
    from core.mcp_client import MCPServerProcess
    cmd = server_conf.get("command")
    if not cmd:
        return False, [], "Missing command."

    test_server = MCPServerProcess(
        name=f"test_{name}",
        command=cmd,
        args=server_conf.get("args", []),
        env=server_conf.get("env", {}),
        cwd=server_conf.get("cwd"),
        timeout=8.0,
    )

    try:
        ok = test_server.start()
        if not ok:
            return False, [], "Failed to start or handshake with server process."

        tools, err = test_server.list_tools()
        if err:
            return False, [], f"Handshake succeeded but tools/list failed: {err}"

        tool_names = list(tools.keys())
        return True, tool_names, f"Successfully connected! Discovered {len(tool_names)} tool(s)."
    except Exception as exc:
        return False, [], f"Test failed with error: {exc}"
    finally:
        try:
            test_server.stop()
        except Exception:
            pass

verify_mcp_server.__test__ = False
test_mcp_server = verify_mcp_server
test_mcp_server.__test__ = False



def reload_mcp_system() -> Tuple[bool, str]:
    """
    Shutdown, reload from config, and refresh all MCP tools into MAVIS commands registry.
    """
    try:
        from core.mcp_client import mcp_manager
        mcp_manager.shutdown()
        mcp_manager.initialize_from_config()

        commands = _load_commands_list()
        # Remove old MCP tools
        for k in list(commands.keys()):
            if isinstance(commands[k], dict) and commands[k].get("is_mcp"):
                del commands[k]

        # Add new tools
        new_tools = mcp_manager.mavis_commands
        commands.update(new_tools)
        _save_commands_list(commands)

        try:
            retriever = ToolRetriever()
            retriever.sync_tools(new_tools)
        except Exception:
            pass

        return True, f"MCP reloaded: {len(new_tools)} tool(s) discovered across {len(mcp_manager.servers)} server(s)."
    except Exception as exc:
        return False, f"Failed to reload MCP: {exc}"
