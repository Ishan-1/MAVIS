import os
import traceback
from google import genai
from prompts.prompt_templates import builder_prompt, tester_prompt, debug_prompt
import json
from core.config import cfg
from core.helpers import log_it
from core.output import mavis_status, mavis_warn


class ToolBuildError(Exception):
    """Raised when a tool cannot be built or debugged after all retries."""
    pass


class ToolBuilder:
    """
    Builds, tests, and (if necessary) debugs Python tool files on-the-fly
    using an LLM.

    Full lifecycle for each new tool:
        build → write to disk (via ONI call_fs)
          └─ generate & write test
               └─ AST import scan (ONI enforcement)
                    └─ run test
                         ├─ PASS → register in commands_list → done
                         └─ FAIL → debug_tool(broken_code, traceback)
                               └─ overwrite tool file with fixed code
                                    └─ run test again
                                         ├─ PASS → register → done
                                         └─ FAIL (repeat up to MAX_RETRIES)
                                              └─ exhausted → notify user, do NOT register

    All file writes and package installs go through ONI's call_fs() so they
    are subject to path checks, greylist prompts, and audit logging.
    """

    @property
    def MAX_RETRIES(self) -> int:  # type: ignore[override]
        return cfg.get("toolbuilder", "max_retries", default=3)

    def __init__(self, client, entity_name: str = "tool_builder"):
        self.client = client
        self.builder_prompt = builder_prompt
        self.entity_name = entity_name
        from memories.memory_store import MemoryStore
        self.memory = MemoryStore(self.client, namespace="toolbuilder")

    # ------------------------------------------------------------------
    # Dependency / environment helpers
    # ------------------------------------------------------------------

    def update_requirements(self, requirements: list[str]):
        """Add missing pip packages to requirements.txt and install them via ONI."""
        if not requirements:
            return

        from oni import oni as _oni

        # Read existing requirements through ONI
        status, existing = _oni.call_fs("read", "requirements.txt")
        if status != 0:
            existing = ""

        # Filter out internal repository modules and empty entries
        internal_modules = {
            "oni", "tools", "memories", "agent_builder",
            "tool_builder", "pipeline", "prompts", "core", "data", "tests"
        }
        existing_pkgs = {line.strip().lower() for line in existing.splitlines() if line.strip()}
        new_packages = [
            req.strip() for req in requirements
            if req.strip()
            and req.strip().lower() not in existing_pkgs
            and req.strip().lower() not in internal_modules
        ]
        if not new_packages:
            log_it(f"All requirements already present or ignored: {requirements}", self.entity_name)
            return

        # Write updated requirements.txt (approved path)
        updated = existing.rstrip("\n") + "\n" + "\n".join(new_packages) + "\n"
        _oni.call_fs("write", "requirements.txt", updated)

        # Install each new package — goes through greylist prompt
        for req in new_packages:
            status, result = _oni.call_fs("install_package", req)
            if status != 0:
                log_it(f"Failed to install {req}: {result}", self.entity_name)
            else:
                log_it(f"Installed package: {req}", self.entity_name)

    def update_env(self, env: list[str]):
        """
        Add missing environment variable stubs to .env via ONI.

        .env is NOT in the approved write paths, so ONI will prompt the user
        for confirmation (ask mode) or deny outright (whitelist_only mode).
        This is intentional — .env holds API keys and deserves scrutiny.
        """
        if not env:
            return

        from oni import oni as _oni

        status, existing = _oni.call_fs("read", ".env")
        if status != 0:
            existing = ""

        additions = "".join(
            f'{var}=""\n' for var in env if var not in existing
        )
        if additions:
            status, result = _oni.call_fs("write", ".env", existing + additions)
            if status != 0:
                log_it(f"ONI denied .env write: {result}", self.entity_name)
            else:
                log_it(f"Environment variable stubs added: {env}", self.entity_name)

    # ------------------------------------------------------------------
    # File writers
    # ------------------------------------------------------------------

    def _write_code_to_file(self, file_path: str, code: str):
        """Write code to file_path through ONI's call_fs."""
        from oni import oni as _oni
        status, result = _oni.call_fs("write", file_path, code)
        if status != 0:
            raise IOError(f"ONI denied write to '{file_path}': {result}")

    def write_tool(self, requirements: list[str], env: list[str], code: str) -> int:
        """Write tool code to tools/<func_name>.py and handle side-effects."""
        func_name = code.split("(")[0].split()[-1]
        file_path = f"tools/{func_name}.py"
        self._write_code_to_file(file_path, code)
        log_it(f"Tool '{func_name}' written to {file_path}.", self.entity_name)
        self.update_requirements(requirements)
        self.update_env(env)
        return 0

    def write_tool_tests(self, func_name: str, code: str):
        """Write test code to tests/test_<func_name>.py."""
        file_path = f"tests/test_{func_name}.py"
        self._write_code_to_file(file_path, code)
        log_it(f"Test for '{func_name}' written to {file_path}.", self.entity_name)

    # ------------------------------------------------------------------
    # LLM calls
    # ------------------------------------------------------------------

    def _llm(self, prompt: str) -> str:
        """Call the LLM and return the stripped text response."""
        if hasattr(self.client, "generate"):
            return self.client.generate(prompt, json_mode=True)
        raw = self.client.models.generate_content(
            model="gemini-2.5-flash", contents=prompt
        ).text
        return raw.removeprefix("```json").removesuffix("```").strip()

    def _find_reference_tools(self, func_sig: str, func_desc: str) -> str:
        """
        Identify existing tools that might share state, conventions, or files,
        and return a formatted reference string containing their source code.
        """
        try:
            if not os.path.exists("data/commands_list.json"):
                return ""
            with open("data/commands_list.json", "r") as f:
                commands = json.load(f)
            if not commands:
                return ""

            # Quick LLM selection of relevant reference tools
            commands_summary = "\n".join([
                f"- {sig}: {desc.get('description', '') if isinstance(desc, dict) else desc}"
                for sig, desc in commands.items()
            ])
            ref_prompt = (
                f"You are helping build a new Python tool with signature: `{func_sig}`\n"
                f"Description: `{func_desc}`\n\n"
                f"Existing tools in the system:\n{commands_summary}\n\n"
                f"Identify up to 3 existing tools from above that are directly relevant (e.g. they share data files, storage locations, formats, or domain conventions) with the new tool.\n"
                f"Return ONLY a JSON list of function names (e.g. [\"tool_a\", \"tool_b\"]), or an empty list [] if none are relevant.\n"
            )
            raw = self._llm(ref_prompt)
            chosen_tools = json.loads(raw)
            if not isinstance(chosen_tools, list) or not chosen_tools:
                return ""

            sections = []
            for tool in chosen_tools[:3]:
                clean_name = str(tool).split("(")[0].strip()
                tool_path = f"tools/{clean_name}.py"
                if os.path.exists(tool_path):
                    with open(tool_path, "r") as tf:
                        code_content = tf.read()
                    sections.append(f"--- Existing Tool Reference: {clean_name}.py ---\n{code_content}\n")

            if sections:
                return "REFERENCE TOOLS (Use these for conventions, data file paths, or patterns if relevant):\n" + "\n".join(sections) + "\n"
            return ""
        except Exception as e:
            log_it(f"Error finding reference tools: {e}", self.entity_name)
            return ""

    def build_tool(self, func_sig: str, func_desc: str):
        """
        Full lifecycle: generate → AST scan → test → [debug loop] → register.

        Raises:
            Exception: propagated LLM or JSON parse errors from the build step.
        """
        func_name = func_sig.split("(")[0].strip()

        # ── 1. Generate tool code ──────────────────────────────────────
        ref_context = self._find_reference_tools(func_sig, func_desc)

        # Cross-namespace retrieval: query toolbuilder conventions and debugger fixes
        try:
            mem_context = self.memory.retrieve_context(
                f"{func_sig} {func_desc}",
                extra_namespaces=["debugger"],
                top_k=2,
            )
            if mem_context:
                ref_context = f"{ref_context}\n\n{mem_context}\n"
        except Exception as me:
            log_it(f"Memory retrieval in ToolBuilder failed: {me}", self.entity_name)

        build_prompt = self.builder_prompt.format(
            reference_tools=ref_context,
            function_signature=func_sig,
            function_description=func_desc,
        )
        response_raw = self._llm(build_prompt)
        log_it(f"Build LLM response: {response_raw}", self.entity_name)
        response = json.loads(response_raw)
        tool_code = response["code"]
        raw_gen = str(response.get("generalizability", "repurposable")).strip().lower()
        if raw_gen in ("generalizable", "repurposable", "specialized"):
            generalizability = raw_gen
        else:
            try:
                f_val = float(raw_gen)
                generalizability = (
                    "generalizable" if f_val >= 0.75 else ("repurposable" if f_val >= 0.4 else "specialized")
                )
            except (ValueError, TypeError):
                generalizability = "repurposable"

        self.write_tool(response["requirements"], response["env"], tool_code)

        # ── 2. Generate test ───────────────────────────────────────────
        self.build_tool_tests(func_sig, func_desc)

        # ── 3. Test → debug loop ───────────────────────────────────────
        from tool_builder.tester import ToolTester
        tester = ToolTester(self.client)

        attempt = 0
        while attempt <= self.MAX_RETRIES:
            # check_imports=True: enforce ONI compliance on all generated tools
            status, result = tester.test_tool(func_name, check_imports=True)

            if status == 0:
                log_it(
                    f"Tool '{func_name}' passed tests on attempt {attempt}.",
                    self.entity_name,
                )
                self._register_tool(func_sig, func_desc, generalizability=generalizability)

                # Specialized Memory: Record successful pattern or debug fix
                try:
                    if attempt == 0:
                        self.memory.write_pattern(func_name, func_sig, func_desc)
                    else:
                        from memories.memory_store import MemoryStore
                        debugger_mem = MemoryStore(self.client, namespace="debugger")
                        debugger_mem.write_fix(func_name, "Build retry needed", f"Fixed and verified after {attempt} retries.")
                except Exception as e:
                    log_it(f"Failed to record memory in ToolBuilder: {e}", self.entity_name)

                return generalizability

            # Test failed
            tb = str(result)
            log_it(
                f"Tool '{func_name}' failed test (attempt {attempt}/{self.MAX_RETRIES}): {tb}",
                self.entity_name,
            )

            if attempt == self.MAX_RETRIES:
                break  # exhausted — fall through to failure handling

            mavis_status(
                f"Test failed (attempt {attempt}/{self.MAX_RETRIES}). Retrying with LLM fix..."
            )
            current_code = open(f"tools/{func_name}.py").read()
            tool_code = self.debug_tool(func_sig, func_desc, current_code, tb)
            attempt += 1

        # ── 4. All retries exhausted ───────────────────────────────────
        self._mark_needs_manual_fix(func_name)
        mavis_warn(
            f"Tool '{func_name}' failed all {self.MAX_RETRIES} debug attempts.\n"
            f"  File marked with '# NEEDS MANUAL FIX'. Last error: {tb}"
        )
        log_it(
            f"Tool '{func_name}' exhausted retries. Needs manual fix.", self.entity_name
        )
        raise ToolBuildError(
            f"Tool '{func_name}' failed all {self.MAX_RETRIES} debug attempts."
        )

    def debug_tool(
        self,
        func_sig: str,
        func_desc: str,
        broken_code: str,
        error_traceback: str,
    ) -> str:
        """
        Ask the LLM to fix a broken tool implementation.

        Overwrites the tool file (via ONI) with the corrected code and
        returns the corrected code string.
        """
        prompt = debug_prompt.format(
            function_signature=func_sig,
            function_description=func_desc,
            broken_code=broken_code,
            error_traceback=error_traceback,
        )
        response_raw = self._llm(prompt)
        log_it(f"Debug LLM response: {response_raw}", self.entity_name)
        fixed_code = json.loads(response_raw)["code"]

        func_name = func_sig.split("(")[0].strip()
        self._write_code_to_file(f"tools/{func_name}.py", fixed_code)
        log_it(f"Overwrote tools/{func_name}.py with debugged code.", self.entity_name)
        return fixed_code

    def build_tool_tests(self, func_sig: str, func_desc: str):
        """Generate and write a test file for the given function."""
        func_name = func_sig.split("(")[0].strip()
        prompt = tester_prompt.format(
            function_signature=func_sig,
            function_description=func_desc,
            func_name=func_name,
        )
        response_raw = self._llm(prompt)
        log_it(f"Tester LLM response: {response_raw}", self.entity_name)
        test_code = json.loads(response_raw)["code"]
        self.write_tool_tests(func_name, test_code)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_tool(self, func_sig: str, func_desc: str, generalizability: str = "repurposable"):
        """Add a successfully tested tool to data/commands_list.json."""
        with open("data/commands_list.json", "r") as f:
            commands = json.load(f)
        commands[func_sig] = {
            "description": func_desc,
            "generalizability": generalizability,
        }
        with open("data/commands_list.json", "w") as f:
            json.dump(commands, f, indent=4)
        log_it(f"Tool '{func_sig}' registered in commands_list.", self.entity_name)

    def _mark_needs_manual_fix(self, func_name: str):
        """Prepend a warning comment to a tool file that failed all debug retries."""
        file_path = f"tools/{func_name}.py"
        try:
            with open(file_path, "r") as f:
                existing = f.read()
            header = (
                "# NEEDS MANUAL FIX\n"
                f"# Automated debugging exhausted {self.MAX_RETRIES} retries.\n"
                "# Review the error in logs/tool_builder.log and fix this file manually.\n\n"
            )
            with open(file_path, "w") as f:
                f.write(header + existing)
        except FileNotFoundError:
            pass
