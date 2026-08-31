"""
oni/oni.py
ONI — Operating System & Network Interface.

The single gateway for all system, network, and filesystem I/O in MAVIS.
No tool, worker, or internal component may bypass this module.

Three primitive calls are exposed:
  call_system_command(command, params) — OS-level commands
  call_network(url, ...)              — outbound HTTP requests
  call_fs(operation, path, ...)       — filesystem operations

Additionally:
  preflight_scan(pipeline)            — validates a full pipeline before execution
  set_trust_level(level)             — updates global session trust
  set_context_trust(level)           — sets thread-local trust (for background tasks)
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
from typing import Any

import requests as _requests

from core.helpers import log_it
from oni.audit import record
from oni.config import ONIConfig, TrustLevel
from oni.gate import ConfirmationGate
from oni.permissions import classify_command, classify_url, classify_fs_write

_ENTITY = "oni"
_thread_local = threading.local()


class ONI:
    """
    Operating System & Network Interface — the MAVIS security harness.

    Instantiated once at process start (see oni/__init__.py).
    Thread-safe: each background thread can carry its own trust level
    via set_context_trust() without affecting the session trust level.
    """

    def __init__(self) -> None:
        self.config = ONIConfig()
        self.gate = ConfirmationGate()
        self.session_allowances: set[str] = set()

        env_allowances = os.environ.get("MAVIS_SESSION_ALLOWANCES")
        if env_allowances:
            try:
                self.session_allowances.update(json.loads(env_allowances))
            except Exception:
                pass

        log_it(
            f"ONI initialised. Trust level: {self.config.trust_level}",
            _ENTITY,
        )
        record({
            "type": "lifecycle",
            "event": "oni_init",
            "trust_level": self.config.trust_level,
        })

    # ── Trust level management ────────────────────────────────────────────────

    def set_trust_level(self, level: TrustLevel) -> None:
        """Set the global session trust level (affects main thread and unlabelled calls)."""
        self.config.trust_level = level
        log_it(f"Global trust level set to: {level}", _ENTITY)
        record({"type": "trust_change", "new_level": level})

    def set_context_trust(self, level: TrustLevel) -> None:
        """
        Set a thread-local trust level override.
        Used by TaskRunner so background workers run at a fixed, isolated level
        regardless of what the user-facing session trust is set to.
        """
        _thread_local.trust_level = level

    def _effective_trust(self) -> TrustLevel:
        """Return thread-local override if present, else global trust."""
        return getattr(_thread_local, "trust_level", None) or self.config.trust_level

    # ── Pre-flight scan ───────────────────────────────────────────────────────

    def preflight_scan(self, pipeline: list[dict]) -> tuple[bool, list[str]]:
        """
        Scan all nodes in *pipeline* before execution begins.

        Phase 1 — blacklist check: if any command is blacklisted, abort
                   immediately (before any node has run).
        Phase 2 — greylist check: collect all greylisted commands and ask
                   the user once, upfront, in a single batch prompt.

        Returns:
            (ok, issues):
              ok=True  → pipeline may proceed.
              ok=False → pipeline must be aborted; *issues* describes why.
        """
        trust = self._effective_trust()
        denied: list[str] = []
        greylisted: list[str] = []

        for node in pipeline:
            command = node.get("function_name", "")
            params = node.get("params", {})
            decision = "allow" if command in self.session_allowances else classify_command(command, self.config, trust)

            if decision == "deny":
                denied.append(command)
                record({
                    "type": "preflight",
                    "command": command,
                    "decision": "denied",
                    "reason": "blacklisted",
                    "trust_level": trust,
                })
            elif decision == "greylist":
                greylisted.append(command)

            # Deep parameter scan for known tools
            if command == "run_shell_command" and isinstance(params.get("command"), str):
                shell_cmd = params["command"]
                for segment in shell_cmd.split("|"):
                    try:
                        parts = shlex.split(segment.strip())
                        sub_exe = parts[0] if parts else ""
                    except ValueError:
                        sub_exe = segment.strip().split()[0] if segment.strip() else ""
                    if sub_exe:
                        sub_dec = "allow" if sub_exe in self.session_allowances else classify_command(sub_exe, self.config, trust)
                        if sub_dec == "deny":
                            denied.append(f"{sub_exe} (in shell command)")
                        elif sub_dec == "greylist":
                            greylisted.append(sub_exe)
            elif command == "call_system_command" and isinstance(params.get("command"), str):
                sub_cmd = params["command"]
                sub_dec = "allow" if sub_cmd in self.session_allowances else classify_command(sub_cmd, self.config, trust)
                if sub_dec == "deny":
                    denied.append(sub_cmd)
                elif sub_dec == "greylist":
                    greylisted.append(sub_cmd)

        # Phase 1 — any blacklisted command kills the pipeline
        if denied:
            msg = f"Blacklisted command(s): {', '.join(denied)}"
            print(f"\n[ONI] ✗ Pipeline BLOCKED — {msg}")
            return False, [f"Blacklisted: {c}" for c in denied]

        # Phase 2 — present greylisted commands to user as a single batch
        if greylisted:
            approved = self._batch_greylist_prompt(greylisted, trust)
            if not approved:
                return False, [f"User denied greylisted command(s): {', '.join(greylisted)}"]

        return True, []

    def _batch_greylist_prompt(self, commands: list[str], trust: str) -> bool:
        """Present all greylisted pipeline commands to the user for batch approval."""
        unique = list(dict.fromkeys(commands))  # preserve order, deduplicate
        print("\n[ONI] ⚠ The following pipeline commands require approval:")
        for i, cmd in enumerate(unique, 1):
            print(f"  {i}. {cmd}")
        approved = self.gate.request_approval(
            f"Run pipeline with greylist command(s): {', '.join(unique)}"
        )
        if approved:
            self.session_allowances.update(unique)
        record({
            "type": "preflight_greylist",
            "commands": unique,
            "decision": "allowed" if approved else "denied",
            "approved_by": "user" if approved else "auto_deny",
            "trust_level": trust,
        })
        return approved

    # ── call_system_command ───────────────────────────────────────────────────

    def call_system_command(self, command: str, params: dict) -> tuple[int, Any]:
        """
        Execute an OS-level command through ONI's permission layer.

        Args:
            command: Command name (e.g. "restart_process", "echo").
            params:  Key-value parameters forwarded to the command.

        Returns:
            (0, result) on success, (-1, error_message) on denial or failure.
        """
        trust = self._effective_trust()
        decision = "allow" if command in self.session_allowances else classify_command(command, self.config, trust)

        audit_entry: dict = {
            "type": "system",
            "command": command,
            "params": params,
            "trust_level": trust,
        }

        if decision == "deny":
            audit_entry.update({"decision": "denied", "reason": "blacklisted"})
            record(audit_entry)
            log_it(f"DENIED system command: {command}", _ENTITY)
            return -1, f"Command '{command}' is blacklisted by ONI."

        if decision == "greylist":
            approved = self.gate.request_approval(
                f"System command: '{command}' with params {params}"
            )
            audit_entry["decision"] = "allowed" if approved else "denied"
            audit_entry["approved_by"] = "user" if approved else "auto_deny"
            record(audit_entry)
            if not approved:
                log_it(f"DENIED (greylist) system command: {command}", _ENTITY)
                return -1, f"Command '{command}' denied (greylist)."
            self.session_allowances.add(command)
        else:
            # decision == "allow"
            audit_entry.update({"decision": "allowed", "approved_by": "whitelist" if command in self.config.whitelist else "session_allowance"})
            record(audit_entry)

        log_it(f"Executing system command: {command}", _ENTITY)
        return self._execute_system(command, params)

    def _execute_system(self, command: str, params: dict) -> tuple[int, Any]:
        """Run an approved system command."""
        if command == "restart_process":
            log_it("ONI executing restart_process.", _ENTITY)
            python = sys.executable
            os.execv(python, [python] + sys.argv)
            return 0, "restarting"  # unreachable

        # Generic shell command
        cmd_parts = [command] + [str(v) for v in params.values()]
        try:
            result = subprocess.run(
                cmd_parts,
                capture_output=True,
                text=True,
                timeout=self.config.tool_execution_timeout,
            )
            if result.returncode == 0:
                return 0, result.stdout.strip()
            return -1, result.stderr.strip()
        except subprocess.TimeoutExpired:
            return -1, f"Command '{command}' timed out."
        except FileNotFoundError:
            return -1, f"Command '{command}' not found on PATH."
        except Exception as e:
            return -1, str(e)

    # ── call_shell ────────────────────────────────────────────────────────────

    def call_shell(self, shell_string: str) -> tuple[int, str]:
        """
        Execute a full shell command string, including pipes.

        Args:
            shell_string: A complete shell command, e.g. "ls -la | grep foo | sed 's/x/y/'"

        Trust behaviour:
            - Each executable in the pipe is classified individually.
            - If any executable is blacklisted → abort immediately.
            - If any executable is greylisted → single batch prompt
              (e.g. "Allow executing: ls, grep, sed?").
            - If all are whitelisted → auto-approve (no prompt).

        Returns:
            (0, stdout) on success, (-1, error_message) on denial or failure.
        """
        trust = self._effective_trust()

        # Parse pipe segments and extract executables
        executables = []
        for segment in shell_string.split("|"):
            try:
                parts = shlex.split(segment.strip())
                if parts:
                    executables.append(parts[0])
            except ValueError:
                executables.append(segment.strip().split()[0])

        denied = []
        greylisted = []

        for exe in executables:
            decision = "allow" if exe in self.session_allowances else classify_command(exe, self.config, trust)
            if decision == "deny":
                denied.append(exe)
            elif decision == "greylist":
                greylisted.append(exe)

        audit_entry: dict = {
            "type": "shell",
            "shell_string": shell_string,
            "executables": executables,
            "trust_level": trust,
            "shell": True,
        }

        # Phase 1 — any blacklisted executable kills the call
        if denied:
            msg = f"Blacklisted executable(s) in pipe: {', '.join(denied)}"
            audit_entry.update({"decision": "denied", "reason": "blacklisted", "denied": denied})
            record(audit_entry)
            log_it(f"DENIED call_shell (blacklisted): {shell_string}", _ENTITY)
            print(f"\n[ONI] ✗ Shell command BLOCKED — {msg}")
            return -1, msg

        # Phase 2 — batch-prompt any greylisted executables
        if greylisted:
            unique_grey = list(dict.fromkeys(greylisted))
            approved = self.gate.request_approval(
                f"Allow executing: {', '.join(unique_grey)}?"
            )
            audit_entry["decision"] = "allowed" if approved else "denied"
            audit_entry["approved_by"] = "user" if approved else "auto_deny"
            audit_entry["greylisted"] = unique_grey
            record(audit_entry)
            if not approved:
                log_it(f"DENIED (greylist) call_shell: {shell_string}", _ENTITY)
                return -1, f"Shell command denied (greylist: {', '.join(unique_grey)})."
            self.session_allowances.update(unique_grey)
        else:
            # All whitelisted or in session allowances — auto-approve
            audit_entry.update({"decision": "allowed", "approved_by": "whitelist" if all(e in self.config.whitelist for e in executables) else "session_allowance"})
            record(audit_entry)

        log_it(f"Executing shell: {shell_string}", _ENTITY)
        try:
            result = subprocess.run(
                shell_string,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.config.tool_execution_timeout,
            )
            if result.returncode == 0:
                return 0, result.stdout.strip()
            return -1, result.stderr.strip() or f"Shell command failed (exit {result.returncode})."
        except subprocess.TimeoutExpired:
            return -1, f"Shell command timed out."
        except Exception as e:
            return -1, str(e)

    # ── call_network ─────────────────────────────────────────────────────────

    def call_network(
        self,
        url: str,
        data: Any = None,
        method: str = "GET",
        params: dict | None = None,
        headers: dict | None = None,
        timeout: int = 10,
    ) -> tuple[int, Any]:
        """
        Make an outbound HTTP request through ONI's permission layer.

        Args:
            url:     Target URL.
            data:    Request body (JSON-serialisable). Sent for POST/PUT/PATCH.
            method:  HTTP verb (default "GET").
            params:  Query-string parameters.
            headers: Additional request headers.
            timeout: Request timeout in seconds.

        Returns:
            (0, response_data) on success, (-1, error_message) on failure/denial.
        """
        trust = self._effective_trust()
        decision = classify_url(url, self.config, trust)

        audit_entry: dict = {
            "type": "network",
            "url": url,
            "method": method.upper(),
            "trust_level": trust,
        }

        if decision == "deny":
            audit_entry.update({"decision": "denied", "reason": "url_blacklisted"})
            record(audit_entry)
            log_it(f"DENIED network request to: {url}", _ENTITY)
            return -1, f"URL '{url}' is blacklisted by ONI."

        audit_entry["decision"] = "allowed"
        record(audit_entry)
        log_it(f"Network {method.upper()} → {url}", _ENTITY)

        try:
            response = _requests.request(
                method=method.upper(),
                url=url,
                json=data if data and method.upper() in ("POST", "PUT", "PATCH") else None,
                params=params,
                headers=headers or {},
                timeout=timeout,
            )
            response.raise_for_status()
            try:
                return 0, response.json()
            except ValueError:
                return 0, response.text

        except _requests.exceptions.HTTPError as e:
            return -1, f"HTTP error: {e}"
        except _requests.exceptions.ConnectionError as e:
            return -1, f"Connection error: {e}"
        except _requests.exceptions.Timeout:
            return -1, f"Request to '{url}' timed out."
        except Exception as e:
            return -1, str(e)

    # ── call_fs ───────────────────────────────────────────────────────────────

    def call_fs(
        self,
        operation: str,
        path: str,
        data: Any = None,
        params: dict | None = None,
    ) -> tuple[int, Any]:
        """
        Perform a filesystem operation through ONI's permission layer.

        Operations:
          "read"            — Read file contents (always allowed).
          "write"           — Write data to path (path must be in approved list).
          "delete"          — Delete a file (path must be in approved list).
          "install_package" — pip install *path* (always greylisted).

        Args:
            operation: One of "read", "write", "delete", "install_package".
            path:      File path (or package name for install_package).
            data:      Content to write (for "write" operation).
            params:    Extra options (e.g. {"mode": "a"} for append).

        Returns:
            (0, result) on success, (-1, error_message) on denial or failure.
        """
        trust = self._effective_trust()
        params = params or {}

        audit_entry: dict = {
            "type": "fs",
            "operation": operation,
            "path": path,
            "trust_level": trust,
        }

        # --- install_package: always greylisted ---
        if operation == "install_package":
            approved = self.gate.request_approval(f"pip install {path}")
            audit_entry["decision"] = "allowed" if approved else "denied"
            audit_entry["approved_by"] = "user" if approved else "auto_deny"
            record(audit_entry)
            if not approved:
                log_it(f"DENIED install_package: {path}", _ENTITY)
                return -1, f"Package install '{path}' denied."
            log_it(f"Installing package: {path}", _ENTITY)
            return self._do_install(path)

        # --- write / delete: check approved paths ---
        if operation in ("write", "delete"):
            decision = classify_fs_write(path, self.config, trust)

            if decision == "deny":
                audit_entry.update({"decision": "denied", "reason": "path_not_approved"})
                record(audit_entry)
                log_it(f"DENIED fs {operation} to: {path}", _ENTITY)
                return -1, f"Write to '{path}' denied: not in approved paths."

            if decision == "greylist":
                approved = self.gate.request_approval(
                    f"Filesystem {operation}: '{path}'"
                )
                audit_entry["decision"] = "allowed" if approved else "denied"
                audit_entry["approved_by"] = "user" if approved else "auto_deny"
                record(audit_entry)
                if not approved:
                    log_it(f"DENIED (greylist) fs {operation}: {path}", _ENTITY)
                    return -1, f"Filesystem {operation} to '{path}' denied."
            else:
                # decision == "allow"
                audit_entry.update({"decision": "allowed", "approved_by": "approved_path"})
                record(audit_entry)
        else:
            # "read" — always permitted
            audit_entry["decision"] = "allowed"
            record(audit_entry)

        log_it(f"fs {operation}: {path}", _ENTITY)
        return self._do_fs(operation, path, data, params)

    # ── Internal execution helpers ────────────────────────────────────────────

    def _do_fs(
        self, operation: str, path: str, data: Any, params: dict
    ) -> tuple[int, Any]:
        """Execute an approved filesystem operation."""
        try:
            if operation == "read":
                with open(path, "r") as f:
                    return 0, f.read()

            elif operation == "write":
                parent = os.path.dirname(path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                mode = params.get("mode", "w")
                with open(path, mode) as f:
                    f.write(data or "")
                return 0, f"Written to {path}"

            elif operation == "delete":
                os.remove(path)
                return 0, f"Deleted {path}"

            else:
                return -1, f"Unknown fs operation: '{operation}'"

        except FileNotFoundError as e:
            return -1, str(e)
        except PermissionError as e:
            return -1, f"Permission denied: {e}"
        except Exception as e:
            return -1, str(e)

    def _do_install(self, package: str) -> tuple[int, Any]:
        """Execute an approved pip install."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                log_it(f"Package '{package}' installed successfully.", _ENTITY)
                return 0, f"Package '{package}' installed."
            return -1, result.stderr.strip()
        except subprocess.TimeoutExpired:
            return -1, f"pip install '{package}' timed out."
        except Exception as e:
            return -1, str(e)
