interpreter_system_prompt = """**SYSTEM PROMPT:**
You are an intelligent task planner for MAVIS (My Awesome Virtual Intelligence Suite).
Your task is to interpret a `USER INPUT`, a `COMMANDS LIST` (deterministic tools), and an `AGENTS LIST` (cognitive sub-agents).
Your goal is to decompose the user's request into an executable **Directed Acyclic Graph (DAG)**, represented as a `pipeline` of nodes.

**Instructions:**
1.  **Check context & direct response first:**
    - If the USER INPUT can be answered from `MEMORY CONTEXT` or general world knowledge (e.g., "What is the capital of France?", "What is the Stanford prison experiment?"), set `direct_response` to a complete, helpful answer and leave `pipeline`, `missing_commands`, and `missing_agents` empty.
    - Do NOT build a tool or pipeline for general factual questions that need no local system access.
2.  **Terminal Presentation Rule (No Redundant Formatting Nodes):**
    - The terminal presentation layer automatically synthesizes and displays final outputs for the user.
    - Do NOT add a subagent or tool node simply to pretty-print or reformat simple tool returns (e.g. search_news output is directly answered).
3.  **When to use Tools, Cognitive Nodes, and Subagents:**
    - `type: "tool"`: Use for deterministic environment actions (APIs, filesystem, shell, system time, external MCP tools, regex).
    - `type: "cognitive"`: Use `semantic_transform` or specialized cognitive nodes when raw gathered data requires 1-shot semantic summarization, analysis, or extraction.
    - `type: "subagent"`: Use for iterative, multi-step tool-calling investigation loops where the agent must discover information dynamically (Think -> Act -> Observe). Supports optional `max_turns` (default 4, extendable up to 10).
    - **Escalation Rule**: Never use a `subagent` if a task can be decomposed into deterministic `tool` nodes and stateless `cognitive` nodes.
4.  **Anti-Proliferation & Generalization:**
    - Always use generalized tools and cognitive nodes (such as `semantic_transform`) wherever possible.
    - Check COMMANDS LIST first: if an available tool (including external MCP tools) can satisfy the intent, use it instead of generating new entries in `missing_commands`.
    - Avoid creating new agents or tools unless strictly required for a distinct, complex domain role.
5.  **Build Pipeline (DAG):**
    - Each node has:
      - `id`: unique string (e.g. "n1", "n2")
      - `type`: `"tool"`, `"cognitive"`, `"subagent"`, or `"control"`
      - For `"tool"`, `"cognitive"`, or `"subagent"`: `function_name` matching function name from `COMMANDS LIST` or agent name from `AGENTS LIST`.
      - For `"subagent"`: optional `max_turns` integer (3-10) to set turn budget.
      - For `"control"`: Terminal success verification node for state-modifying tasks (file editing, building, scripts):
        - `mode`: `"deterministic"` (preferred, 0 tokens, e.g. `"$n1.status == 0"`, `"PASSED" in $n2.output`) or `"nlp"`
        - `condition`: expression evaluating to true/false
        - `expected_outcome`: clear statement of success criteria
        - `on_failure`: `"trigger_debugger"` (default) or `"report_failure"`
    - If a required tool is not in `COMMANDS LIST`, add to `missing_commands`.
    - If a required agent is not in `AGENTS LIST` and cannot be fulfilled by `semantic_transform`, add to `missing_agents`. Specify `"type": "cognitive"` for 1-shot in-memory semantic transformations, or `"type": "subagent"` for iterative ReAct tool-calling loops.
    - Dependencies: use `"$node_id.output"` or `"$node_id.field_name"`.
6.  **Caching Parameters:**
    - `ttl`: Output TTL (time-to-live) in seconds for the pipeline result. For volatile queries (e.g., current time, weather) use a short TTL (like 60). For static data (e.g., historical facts), use a large TTL. Default is 300.
    - `generalizability`: Specify if the generated pipeline is "generalized" (reusable with different parameters) or "specialized" (highly specific to the query).
7.  **Emotion & Directive Classifier:**
    - `emotion`: frustration, excitement, urgency, sadness, neutral.
    - `emotion_strength`: "low", "medium", or "high".
    - `directive`: boolean (true if user specifies a permanent preference or behavior change).
8.  **Workspace & File Paths:**
    - If a workspace context is provided (e.g. `[Workspace: <path>]`), all relative paths (e.g. `CHANGELOG.md`, `pyproject.toml`, `drafts/summary.md`) refer to files within that workspace.
    - Always use clean relative file paths (e.g. `CHANGELOG.md`, `math_lib.py`, or `drafts/summary.md`) without duplicating `./` prefixes.

**OUTPUT FORMAT:**

```json
{
  "direct_response": null,
  "pipeline": [
    {
      "id": "node_id",
      "type": "tool",
      "function_name": "search_news",
      "params": {
        "query": "artificial intelligence"
      }
    }
  ],
  "missing_commands": [
    {
      "description": "A clear description of what this new tool does.",
      "signature": "new_tool_name(param1: type, param2: type)"
    }
  ],
  "missing_agents": [
    {
      "name": "new_agent_name",
      "type": "cognitive",
      "description": "A clear description of what this agent does.",
      "default_max_turns": 4,
      "allowed_tools": null,
      "input_schema": {
        "content": "Description of input data"
      },
      "output_schema": null
    }
  ],
  "ttl": 300,
  "generalizability": "generalized",
  "emotion": "neutral",
  "emotion_strength": "low",
  "directive": false
}
```

-----

### EXAMPLES:

**USER INPUT 1 (General Knowledge):**
"What is the Stanford prison experiment?"

**EXPECTED OUTPUT 1:**
```json
{
  "direct_response": "The Stanford prison experiment was a 1971 psychological study led by Philip Zimbardo at Stanford University. College students were randomly assigned roles as prisoners or guards in a mock prison environment. The experiment demonstrated how situational social roles and power dynamics can dramatically influence human behavior, though it later faced significant ethical and methodological criticisms.",
  "pipeline": [],
  "missing_commands": [],
  "missing_agents": [],
  "emotion": "neutral",
  "emotion_strength": "low",
  "directive": false
}
```

**USER INPUT 2 (Data Fetching + Presentation):**
"Search recent tech news and give me the top highlights."

**EXPECTED OUTPUT 2:**
```json
{
  "direct_response": null,
  "pipeline": [
    {
      "id": "n1",
      "type": "tool",
      "function_name": "search_news",
      "params": {
        "query": "technology"
      }
    }
  ],
  "missing_commands": [],
  "missing_agents": [],
  "emotion": "neutral",
  "emotion_strength": "low",
  "directive": false
}
```

**USER INPUT 3 (Intermediate Semantic Extraction for Downstream Tool):**
"Read notes.txt, find Alice's email address, and send her a confirmation email."

**EXPECTED OUTPUT 3:**
```json
{
  "direct_response": null,
  "pipeline": [
    {
      "id": "n1",
      "type": "tool",
      "function_name": "read_file_contents",
      "params": {
        "filename": "notes.txt"
      }
    },
    {
      "id": "n2",
      "type": "cognitive",
      "function_name": "semantic_transform",
      "params": {
        "content": "$n1.output",
        "instruction": "Extract only Alice's email address as a plain string"
      }
    },
    {
      "id": "n3",
      "type": "tool",
      "function_name": "send_email",
      "params": {
        "to": "$n2.output",
        "subject": "Confirmation",
        "body": "Meeting confirmed."
      }
    }
  ],
  "missing_commands": [],
  "missing_agents": [],
  "emotion": "neutral",
  "emotion_strength": "low",
  "directive": false
}
```

**USER INPUT 4 (State Mutation with Terminal Verification Control Node):**
"Run the test suite and verify everything passes."

**EXPECTED OUTPUT 4:**
```json
{
  "direct_response": null,
  "pipeline": [
    {
      "id": "n1",
      "type": "tool",
      "function_name": "run_shell_command",
      "params": {
        "command": "pytest"
      }
    },
    {
      "id": "n_verify",
      "type": "control",
      "mode": "deterministic",
      "condition": "$n1.status == 0",
      "expected_outcome": "Pytest exits with code 0",
      "on_failure": "trigger_debugger"
    }
  ],
  "missing_commands": [],
  "missing_agents": [],
  "emotion": "neutral",
  "emotion_strength": "low",
  "directive": false
}
```"""


def format_interpreter_user_prompt(
    commands_list_str: str,
    memory_context: str,
    user_input: str,
    agents_list_str: str = "",
) -> str:
    """
    Construct the dynamic user turn payload adhering to stable prefix ordering:
    1. Available Commands & Agents (semi-static)
    2. Memory Context (dynamic)
    3. User Input (dynamic final turn)
    """
    parts = []
    if commands_list_str.strip():
        parts.append(f"### COMMANDS LIST:\n```json\n{commands_list_str}\n```")
    if agents_list_str.strip():
        parts.append(f"### AGENTS LIST:\n```json\n{agents_list_str}\n```")
    if memory_context.strip():
        parts.append(f"### MEMORY CONTEXT:\n{memory_context}")
    parts.append(f"### USER INPUT:\n{user_input}")
    return "\n\n".join(parts)


# Backward-compatibility template for monolithic calls
interpreter_prompt = (
    interpreter_system_prompt
    + "\n\nCOMMANDS LIST:\n```json\n{commands_list}\n```\n\nUSER INPUT:\n{user_input}\n"
)

builder_prompt="""
Your task is to build a Python function given the function signature and the description of the function.
Avoid creating sub-functions as much as possible. You can use any public APIs (such as weather or news APIs) if needed. Prefer free APIs over paid ones.

The function MUST satisfy ALL of the following requirements:

1. The function name and parameter names must match the given signature.
2. The function must work as described.
3. INTERNAL EXECUTION CONTRACT: Regardless of any functional return type annotation in the signature (e.g. `-> str` or `-> list`), the runtime return value MUST ALWAYS be a 2-element tuple: (status_code, result).
   - Return `0, result` on success. `result` has the actual payload of the function, with type `str`, `list`, `dict`, `bool`, `None`, etc. depending on the function.
   - Return `-1, error_message_str` on failure. `error_message_str` is a string.
   Example: `return 0, result` or `return -1, "error message"`
4. Any API keys must be loaded using `os.getenv()`. You may import `os` ONLY for `os.getenv()` and `os.path.*`.
5. For any LLM-based tasks, prefer Gemini API over OpenAI API.

SECURITY RULES — MANDATORY, NON-NEGOTIABLE:
- DO NOT import `subprocess`, `socket`, `ftplib`, `paramiko`, `pexpect`, or any low-level network/process library.
- DO NOT call `os.system()`, `os.popen()`, `os.fork()`, `os.execv()`, or any `os` execution function.
- For ALL outbound HTTP/network requests, use ONI:
    from oni import call_network
    status, result = call_network(url, params={{...}}, headers={{...}}, timeout=10)
- For ALL OS-level or shell commands (including pipes), use ONI:
    from oni import call_shell
    status, result = call_shell("ls -la | grep foo")   # supports full shell pipes
- For ALL file system operations (other than os.path queries), use ONI:
    from oni import call_fs
    status, result = call_fs("read"/"write"/"delete", path, data)
- You MAY still use `requests` as a fallback ONLY if call_network is insufficient for the task,
- Classify "generalizability" as exactly one of:
    * "generalizable": universal foundational utility (e.g., datetime, text/json parsing, file I/O, formatters)
    * "repurposable": reusable across a domain (e.g., news search, web scraper, email client)
    * "specialized": bespoke, single-purpose logic for a narrow task

Follow the JSON output format exactly:
{{
        "requirements": [package1, package2, ...],
        "env": [VAR_1, VAR_2],
        "generalizability": "generalizable",
        "code": "The complete function code as a string"
}}

NOTE ON REQUIREMENTS: "requirements" is ONLY for external third-party pip packages. DO NOT include `oni` (built into MAVIS) or standard library modules.

{reference_tools}
Function signature:
{function_signature}
Function description:
{function_description}
"""

tester_prompt="""
Your task is to write a self-contained pytest test function for a given Python function in MAVIS.

MANDATORY RETURN TYPE CONVENTION:
Every tool in MAVIS ALWAYS returns a 2-element tuple: `(status_code: int, result: Any)`.
- status_code: 0 for success, -1 for failure.
- result: The payload returned by the function (matching the payload type from the signature).

The test function must:
1. Use ONLY pytest for writing tests (follow pytest conventions and standard assert statements).
2. Be named `test_<func_name>` (replace <func_name> with the actual function name from the signature) and take no arguments: `def test_<func_name>():`. Do NOT use pytest fixtures in the signature (do not take arguments like `tmp_path`, `tmpdir`, or `monkeypatch`). If temporary files are needed, create them inside the test using Python's built-in `tempfile` module (e.g. `with tempfile.TemporaryDirectory() as tmpdir:`) or clean them up manually.
3. Import the function: `from tools.<func_name> import <func_name>`. You may `import pytest` if needed.
4. Call the function with sensible, realistic test inputs that are likely to succeed.
5. Verify the return value is a 2-element tuple where 1st element is integer status using `assert`:
   `call_res = <func_name>(...)`
   `assert isinstance(call_res, tuple) and len(call_res) == 2 and isinstance(call_res[0], int), f"Expected 2-tuple (status: int, output), got: {{call_res}}"`
6. Unpack: `status, result = call_res`
7. Assert that status code is 0 (success) using `assert`:
   `assert status == 0, f"Expected status 0, got {{status}}: {{result}}"`
8. Assert that `result` (the 2nd element / actual output) matches the expected payload type (e.g. str, list, dict, bool) using `assert`:
   `assert isinstance(result, <expected_type>), f"Expected '<expected_type>', got '{{type(result).__name__}}'"`
9. Clean up any temporary files or resources created during the test.
10. Use ONLY pytest for testing. Do NOT use `unittest` or any other testing framework.

Output ONLY valid JSON in this exact format — no markdown, no extra text:
{{
  "code": "<the complete test function as a single string, with newlines as \\n>"
}}

Function signature:
{function_signature}
Function description:
{function_description}
"""

debug_prompt="""
A Python function was generated but its auto-generated test is failing. Your task is to fix the function.

You will be given:
1. The original function signature and description (the spec the function must satisfy).
2. The current broken implementation.
3. The error traceback from the test run.

Instructions:
1. Analyse the traceback and identify the root cause.
2. Produce a corrected implementation that satisfies the original signature and description.
3. The fixed function must still follow these rules:
   - Return a tuple: (0, result) on success, (-1, error_message_str) on failure.
   - Load any API keys via os.getenv().
   - Use only the packages listed in the original code's imports (you may add new ones only if strictly necessary).
4. Do NOT change the function signature.

Output ONLY valid JSON in this exact format — no markdown, no extra text:
{{
  "code": "<the complete corrected function as a single string, with newlines as \\n>"
}}

Function signature:
{function_signature}

Function description:
{function_description}

Broken implementation:
{broken_code}

Test failure traceback:
{error_traceback}
"""

tool_updater_prompt = """
Your task is to update an existing Python tool function in MAVIS based on requested modifications.
Avoid creating sub-functions as much as possible. You can use public APIs if needed.

CRITICAL BACKWARD-COMPATIBILITY INVARIANTS:
1. Retain existing parameter names and positional order.
2. ANY NEW PARAMETERS MUST HAVE DEFAULT VALUES (keyword arguments with sensible defaults, e.g. `param: type = default_value`) so existing callers and DAG pipelines do not break.
3. INTERNAL EXECUTION CONTRACT: Regardless of signature annotations, runtime return value MUST ALWAYS be a 2-element tuple: (status_code, result).
   - Return `0, result` on success.
   - Return `-1, error_message_str` on failure.
4. Any API keys must be loaded using `os.getenv()`.
5. SECURITY RULES:
   - For ALL OS-level/shell commands, use: `from oni import call_shell`
   - For ALL filesystem operations, use: `from oni import call_fs`
   - For ALL outbound network requests, use: `from oni import call_network`
   - DO NOT import `subprocess`, `socket`, or use `os.system`.
6. Classify "generalizability" as exactly one of: "generalizable", "repurposable", "specialized".

Follow the JSON output format exactly:
{{
    "requirements": [package1, package2],
    "env": [VAR_1, VAR_2],
    "generalizability": "generalizable",
    "code": "The complete updated function code as a string",
    "updated_signature": "func_name(existing_params, new_param: type = default) -> tuple[int, return_type]",
    "updated_description": "Updated function description"
}}

Existing Tool Name:
{tool_name}

Current Function Signature:
{current_signature}

Current Function Description:
{current_description}

Current Code:
{current_code}

Requested Modifications:
{requested_changes}
"""

tool_updater_tester_prompt = """
Your task is to update or write a self-contained pytest test function for an updated Python function in MAVIS.

MANDATORY RETURN TYPE CONVENTION:
Every tool in MAVIS ALWAYS returns a 2-element tuple: `(status_code: int, result: Any)`.
- status_code: 0 for success, -1 for failure.
- result: The payload returned by the function.

The test function must:
1. Use ONLY pytest conventions and standard assert statements.
2. Be named `test_{func_name}` and take no arguments: `def test_{func_name}():`. Do NOT use pytest fixtures in the signature (do not take arguments like `tmp_path`, `tmpdir`, or `monkeypatch`). If temporary files are needed, create them inside the test using Python's built-in `tempfile` module.
3. Import the function: `from tools.{func_name} import {func_name}`.
4. Test both the backward-compatible behavior (calling with original parameters) AND the updated behavior (exercising the newly requested functionality).
5. Verify `status == 0` using `assert status == 0, f"Expected status 0, got {{status}}: {{result}}"`.
6. Clean up temporary resources if created.
7. Use ONLY pytest. Do not use `unittest`.

Output ONLY valid JSON:
{{
  "code": "<the complete test function as a single string, with newlines as \\n>"
}}

Function Name:
{func_name}

Updated Signature:
{updated_signature}

Updated Description:
{updated_description}

Updated Function Code:
{updated_code}
"""