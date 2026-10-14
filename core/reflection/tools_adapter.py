"""
core/reflection/tools_adapter.py
Sandboxed tools adapter for the Autonomous Reflection Engine.
Supports:
1. Sandboxed web search (via tools/search_web.py)
2. Safe read-only local inspection (logs, metrics, core code, docs)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Tuple

from core.helpers import log_it

_ENTITY = "reflection_tools"
_MAV_ROOT = Path(__file__).resolve().parent.parent.parent

# Allowed directories for local reflection reading
_SAFE_LOCAL_PREFIXES = (
    "logs",
    "data/metrics",
    "core",
    "docs",
    "memories",
    "tasks",
)

# Explicitly forbidden files
_FORBIDDEN_FILES = {
    ".env",
    "id_rsa",
    "id_ed25519",
}


def search_web_sandboxed(query: str, max_results: int = 3) -> Tuple[int, str]:
    """
    Execute tools/search_web.py in an isolated subprocess via core/run_tool.py.
    Falls back to direct tool invocation if run_tool fails.
    """
    query = (query or "").strip()
    if not query:
        return 0, "No query provided."

    params = {"query": query, "max_results": max_results}

    # Attempt 1: Call via main.call_command if loaded
    try:
        if "main" in sys.modules:
            from main import call_command
            status, result = call_command("search_web", params)
            return int(status), str(result)
    except Exception as exc:
        log_it(f"search_web via call_command failed: {exc}", _ENTITY)

    # Attempt 2: Subprocess via core/run_tool.py
    try:
        run_tool_path = _MAV_ROOT / "core" / "run_tool.py"
        if run_tool_path.exists():
            env = os.environ.copy()
            env["MAVIS_TOOL_SUBPROCESS"] = "1"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(run_tool_path),
                    "search_web",
                    json.dumps(params),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(_MAV_ROOT),
                env=env,
            )
            stdout = proc.stdout.strip()
            if stdout:
                try:
                    payload = json.loads(stdout)
                    if isinstance(payload, list) and len(payload) >= 2:
                        return int(payload[0]), str(payload[1])
                except json.JSONDecodeError:
                    pass
            if proc.returncode == 0 and stdout:
                return 0, stdout
    except Exception as exc:
        log_it(f"Subprocess run_tool search_web failed: {exc}", _ENTITY)

    # Attempt 3: Direct baseline import
    try:
        from tools.search_web import search_web
        status, res = search_web(query, max_results=max_results)
        return int(status), str(res)
    except Exception as exc:
        log_it(f"Direct search_web import failed: {exc}", _ENTITY)
        return -1, f"Search failed: {exc}"


def read_local_context(relative_path: str, max_chars: int = 4000) -> Tuple[int, str]:
    """
    Safely read local files for self-reflection (logs, metrics, docs, core logic).
    Enforces directory whitelist and prevents path traversal.
    """
    clean_path = os.path.normpath(relative_path.strip().lstrip("/"))
    
    # Path traversal check
    if clean_path.startswith("..") or os.path.isabs(clean_path):
        return -1, "Access denied: Path traversal or absolute paths are forbidden."

    # Check forbidden filenames
    base_name = os.path.basename(clean_path)
    if base_name in _FORBIDDEN_FILES or base_name.endswith(".env"):
        return -1, "Access denied: Sensitive file access is forbidden."

    # Check safe directory prefixes
    if not any(clean_path == prefix or clean_path.startswith(prefix + "/") for prefix in _SAFE_LOCAL_PREFIXES):
        return -1, f"Access denied: '{clean_path}' is outside approved reflection inspection paths ({', '.join(_SAFE_LOCAL_PREFIXES)})."

    full_path = _MAV_ROOT / clean_path
    if not full_path.exists():
        return -1, f"File not found: {clean_path}"

    if not full_path.is_file():
        return -1, f"Target is not a regular file: {clean_path}"

    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars + 1)
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... [truncated for reflection]"
        return 0, content
    except Exception as exc:
        log_it(f"read_local_context failed for '{clean_path}': {exc}", _ENTITY)
        return -1, f"Failed to read file: {exc}"
