"""
tester.py
Tool test runner with AST-based forbidden-import scanner.

For LLM-generated tools (check_imports=True):
  1. Static AST scan for forbidden imports/calls — instant fail if found.
  2. Dynamic test execution — run the auto-generated test function.

For existing hand-written tools (check_imports=False, default):
  Only step 2 runs (grandfathering existing tools).
"""
import ast
import os
import traceback as tb_module
from core.config import cfg
from core.helpers import log_it

# Default fallbacks (used if config is unavailable at import time)
_DEFAULT_FORBIDDEN_IMPORTS = {
    "subprocess", "socket", "ftplib", "paramiko", "pexpect", "httplib2",
}
_DEFAULT_FORBIDDEN_OS_ATTRS = {
    "system", "popen", "fork", "execv", "execl", "execle",
    "execlp", "execlpe", "execvp", "execvpe", "spawn", "spawnl",
}


class ToolTester:
    def __init__(self, client, entity_name: str = "tool_tester"):
        self.client = client
        self.entity_name = entity_name

    # ── Public API ────────────────────────────────────────────────────────────

    def test_tool(self, func_name: str, check_imports: bool = False):
        """
        Test a tool module.

        Args:
            func_name:     Name of the tool (matches tools/<func_name>.py).
            check_imports: If True, run the AST scanner before executing the
                           test. Set to True for all LLM-generated tools.

        Returns:
            (0, result)  — test passed.
            (-1, message) — test failed (includes AST violations if found).
        """
        tool_file = f"tools/{func_name}.py"

        # Step 1: Static AST scan (only for generated tools)
        if check_imports:
            violations = self._check_forbidden_imports(tool_file)
            if violations:
                msg = (
                    f"ONI AST scan FAILED for '{func_name}'. "
                    f"Forbidden usage detected: {', '.join(violations)}. "
                    f"Use `from oni import call_network, call_system_command, call_fs` instead."
                )
                log_it(msg, self.entity_name)
                return -1, msg

        # Step 2: Dynamic test execution
        test_file_path = f"tests/test_{func_name}.py"
        if not os.path.exists(test_file_path):
            log_it(f"Test file for {func_name} does not exist.", self.entity_name)
            return -1, f"Test file for {func_name} does not exist."

        try:
            import importlib.util
            import sys

            # Evict cached modules from sys.modules so debugged files are reloaded from disk
            for mod in (f"tools.{func_name}", func_name, "test_module"):
                if mod in sys.modules:
                    del sys.modules[mod]

            spec = importlib.util.spec_from_file_location("test_module", test_file_path)
            test_module = importlib.util.module_from_spec(spec)
            sys.modules["test_module"] = test_module
            spec.loader.exec_module(test_module)

            test_function = getattr(test_module, f"test_{func_name}", None)
            if not test_function:
                log_it(
                    f"Test function for {func_name} not found in {test_file_path}.",
                    self.entity_name,
                )
                return -1, f"Test function for {func_name} not found."

            result = test_function()
            log_it(f"Test for {func_name} executed successfully.", self.entity_name)
            return 0, result

        except Exception as e:
            full_tb = tb_module.format_exc()
            log_it(f"Error while testing {func_name}:\n{full_tb}", self.entity_name)
            return -1, full_tb

    # ── AST scanner ───────────────────────────────────────────────────────────

    def _check_forbidden_imports(self, file_path: str) -> list[str]:
        """
        Parse the tool file and walk its AST looking for:
          - Direct imports of forbidden modules (subprocess, socket, etc.)
          - Calls to forbidden os.* methods (os.system, os.popen, etc.)

        Returns a list of violation strings, or an empty list if clean.
        """
        try:
            with open(file_path, "r") as f:
                source = f.read()
            tree = ast.parse(source, filename=file_path)
        except FileNotFoundError:
            return [f"file_not_found:{file_path}"]
        except SyntaxError as e:
            return [f"syntax_error:{e}"]

        forbidden_imports = set(cfg.get("toolbuilder", "forbidden_imports", default=list(_DEFAULT_FORBIDDEN_IMPORTS)))
        forbidden_os_attrs = set(cfg.get("toolbuilder", "forbidden_os_attrs", default=list(_DEFAULT_FORBIDDEN_OS_ATTRS)))

        violations: list[str] = []

        for node in ast.walk(tree):
            # import subprocess  /  import socket
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in forbidden_imports:
                        violations.append(f"import {root}")

            # from subprocess import ...  /  from socket import ...
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root in forbidden_imports:
                        violations.append(f"from {root} import ...")

            # os.system(...) / os.popen(...) etc.
            elif isinstance(node, ast.Attribute):
                if (
                    isinstance(node.value, ast.Name)
                    and node.value.id == "os"
                    and node.attr in forbidden_os_attrs
                ):
                    violations.append(f"os.{node.attr}()")

        return list(dict.fromkeys(violations))  # deduplicate, preserve order