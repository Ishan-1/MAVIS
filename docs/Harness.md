# MAVIS Harness

## Objective
- Prevent MAVIS from running unauthorized or unintended commands in a deterministic manner, especially since it will have complete control of system.

## Specs
- Able to intercept all network and system calls, similar to user and kernel mode.
- Able to set different trust levels during sessions, such as read-only, ask before execute and YOLO (Execute every command it can, without any checks).

---

## ONI (Operating System & Network Interface)
ONI will manage all network and system calls for MAVIS. It exposes **3 primitive functions** that are the only sanctioned I/O interfaces for all tools and internal components:

- `call_system_command(command, params)` — executes OS-level commands
- `call_network(url, data, method, params, headers, timeout)` — makes all outbound network requests
- `call_fs(operation, path, data, params)` — handles all file system operations (`read`, `write`, `delete`, `install_package`)

No tool, worker, or internal component may import `subprocess`, `os`, `socket`, `urllib`, `requests`, or any other system/network library directly. All I/O must go through ONI.

---

## Permissions System
ONI will have a permissions system that will be used to determine whether or not to execute a command.
- **Blacklist**: commands that are never allowed to execute. Pipeline aborts immediately if a blacklisted command is encountered.
- **Whitelist**: commands allowed to execute without any checks.
- **Greylist**: commands allowed to execute only with explicit user confirmation (see Greylist UX below).
- **YOLO mode**: all commands execute without checks. Intended for tool testing/debugging sessions only.
- **URL/IP Blacklist**: ONI blocks outbound network requests to blacklisted URLs and IP addresses.

---

## Trust Evaluation — Timing & Flow

Trust is evaluated in **two phases** to avoid interrupting a running pipeline mid-execution:

1. **Pre-flight scan**: Before `execute_pipeline()` starts, ONI scans the entire pipeline and classifies every command+params against the lists.
   - If anything is **blacklisted** → abort the whole pipeline immediately, before any command runs.
   - If anything is **greylisted** → collect all greylist items and ask the user upfront in a single batch prompt. Pipeline only starts after approval.
2. **Runtime check**: ONI validates again at actual call time as a second line of defense. This should never be the user's first notification — it is defense-in-depth only.

---

## Greylist UX — Confirmation Queue

A `ConfirmationGate` class manages greylist approvals:
- During pre-flight, greylisted commands are posted to a `pending_confirmations` queue and the pipeline is paused.
- The main input loop checks the queue and presents the user with the pending approvals.
- User responds yes/no; the gate unblocks and the pipeline either runs or aborts.
- **Timeout**: if no user response is received within a configurable timeout (default: deny), the request is auto-denied and logged.
- **Background tasks**: if a background task encounters a greylisted operation and no user session is active, it is auto-denied and recorded in the audit log. Background tasks never block waiting for user input.

---

## ToolBuilder — ONI Compliance

ToolBuilder currently writes files to disk, runs `pip install`, and modifies `.env` directly. All of these must route through ONI's `call_fs()` primitive.

### Approved write paths
ONI enforces that `call_fs(write, ...)` is only permitted to the following directories:
- `tools/` — generated tool code
- `tests/` — generated test code
- `memories/` — memory store files
- `logs/` — log files

Writes to any other path (including root, `data/`, `.env`) are blocked unless explicitly whitelisted.

### Package installation
`call_fs(install_package, ...)` maps to `pip install`. This operation is placed on the **greylist** — installing an unknown third-party package is a meaningful trust decision and requires user confirmation.

### Environment variables
The `.env` file is **write-protected by default**. Adding new environment variable stubs requires the trust level to explicitly permit it. This protects existing API keys from being overwritten by generated code.

### Enforcement via static analysis
Before a generated tool is executed (even in the test phase), `ToolTester` runs a static AST scan:
- Uses Python's `ast` module to inspect the tool file for forbidden imports (`subprocess`, `os`, `socket`, `urllib`, `requests`, etc.).
- If any forbidden import is found, the test fails immediately with a clear error message before any code runs.
- This error is fed back into the debug loop so the LLM can correct the violation.

The builder prompt is also updated with an explicit constraint: *"Your code MUST NOT import system or network libraries directly. Use `from oni import call_system_command, call_network, call_fs` for all I/O."*

---

## Tool Sandboxing

Generated tools run in a **child subprocess**, not in-process via `importlib`. A thin `run_tool.py` harness is called:

```
subprocess.run([sys.executable, "run_tool.py", func_name, json.dumps(params)],
               capture_output=True, timeout=30, env=restricted_env)
```

- `restricted_env` strips out API keys and environment variables not needed by the specific tool.
- A `timeout` hard limit (default: 30s) prevents runaway tools from hanging the system.
- A crashed or malicious tool cannot affect the main MAVIS process.
- The static AST import check (#Enforcement via static analysis) is the first defense; subprocess isolation is the second.

---

## Background Task Trust Isolation

Background tasks (scheduler workers) are assigned a **fixed trust level at registration time**, independent of the active session trust level.

```python
runner.register(lt_worker.run, interval_minutes=480, task_name="long_term_worker", trust_level="whitelist_only")
```

- Memory workers (`short_term_worker`, `long_term_worker`) are hardcoded to `whitelist_only`. They only need `call_fs()` access to `memories/`.
- Even if the active session is in YOLO mode, background tasks do **not** inherit YOLO. Session trust and background trust are fully isolated.

---

## Audit Log

ONI writes every call to a tamper-evident, append-only audit log at `logs/oni_audit.jsonl`:

```json
{"timestamp": "...", "trust_level": "ask", "type": "fs", "operation": "write", "path": "tools/foo.py", "decision": "allowed", "approved_by": "user"}
```

Properties:
- ONI opens this file in append (`a`) mode only — it never truncates or overwrites it.
- Separate from the general `helpers.py` log — the audit log is for accountability, not debugging.
- Records every call regardless of outcome (allowed, denied, timed out).
- Enables session replay: the exact sequence of everything MAVIS did can be reconstructed from this file.

---

## Controlled Restart

`restart_mav()` must not call `os.execv()` directly. Restart is treated as a named system command routed through ONI:

```python
call_system_command("restart_process", {})
```

- `restart_process` is placed on the **greylist** — it terminates the session and clears working memory, which is a significant action.
- The restart is recorded in the audit log before execution.
- This ensures the restart is subject to the same trust rules, confirmation flow, and audit trail as all other commands.

---

## Additional Protection
Toolbuilder will have explicit instructions in the LLM prompts that it can only use ONI exposed functions and that it should not try to run any other commands. Tool testing and debugging will have same instructions but will run in YOLO mode to make the process faster.

---

## Implementation Priority

| Priority | Feature |
|---|---|
| 🔴 1 | Subprocess sandboxing for tools |
| 🔴 2 | AST import scanner in ToolTester |
| 🟠 3 | `call_fs()` ONI primitive + approved write paths |
| 🟠 4 | Pre-flight trust evaluation before pipeline |
| 🟡 5 | Append-only audit log |
| 🟡 6 | Background task trust isolation |
| 🟢 7 | Greylist confirmation queue |
| 🟢 8 | Restart via ONI |
