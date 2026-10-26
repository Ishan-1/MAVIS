# MAVIS Harness (ONI: Operating System & Network Interface)

## Objective
Prevent MAVIS from running unauthorized, destructive, or unintended commands in a deterministic manner, enforcing strict OS-level process isolation, access control, and auditability.

---

## Architecture & I/O Primitives

ONI manages all network, filesystem, and system execution for MAVIS. It exposes **sanctioned primitive functions** that serve as the sole gateway for all tools and internal components:

- `call_system_command(command, params)` — executes OS-level commands.
- `call_network(url, data, method, params, headers, timeout)` — performs outbound network requests.
- `call_fs(operation, path, data, params)` — handles all filesystem operations (`read`, `write`, `delete`, `install_package`).
- `call_shell(command_line)` — streams shell commands with live selector-based terminal output.

**Golden Invariant**: No generated tool, worker, or internal component may import `subprocess`, `os`, `socket`, `urllib`, `requests`, or other network/system libraries directly. All I/O routes through ONI.

---

## Permissions & Trust Model

ONI enforces a tiered permissions engine configured via `data/mavis_config.json`:

- **Blacklist**: Commands permanently blocked from execution. If encountered, the entire pipeline aborts immediately.
- **Whitelist**: Commands permitted to execute without prompt interruption.
- **Greylist**: Potentially sensitive commands (`rm`, `mv`, `chmod`, `curl`, `pip`, `install_package`) that require explicit user confirmation.
- **Safe Read Whitelist (`SAFE_READ_COMMANDS`)**: Read-only inspection commands (`ls`, `find`, `read_file_contents`, `read_and_concatenate_files`, `git status`) execute without interactive prompt interruption.
- **Tiered Session Trust Levels**:
  - `ask`: Interactive user confirmation required for greylisted actions (default).
  - `yolo`: Unattended automated execution (used for goal execution `--yolo` and automated testing).
  - `whitelist_only`: Strict deny-by-default execution. All background daemons run locked at `whitelist_only`.

---

## Trust Evaluation Timing & Flow

Trust is evaluated in **two phases** to avoid mid-pipeline interruptions:

1. **Pre-flight Scan**: Before `execute_pipeline()` starts, ONI scans all DAG nodes:
   - If any node is **blacklisted** → aborts immediately before any step runs.
   - If any nodes are **greylisted** → batches all approvals into a single prompt for user decision.
2. **Runtime Verification**: Re-checked at dispatch time as defense-in-depth.

---

## Task-Scoped Execution Leases (`acquire_task_lease`)

Autonomous workflows like `/goal` (multi-wave DAGs) and `/reflect` (generator-critic exploration loops) require running successive tool steps without prompting the user on every wave:
- `oni.acquire_task_lease(task_id, trust_level, reason)` acquires a temporary, context-scoped lease.
- Single upfront approval unlocks execution for that specific goal or reflection run.
- When the task completes or errors out, the lease automatically expires.

---

## Child Process Environment Sanitization (`sanitize_env`)

To prevent third-party tools or dynamically synthesized code from leaking sensitive environment variables:
- `oni.sanitize_env(base_env)` scrubs all internal secrets, API keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `NEO4J_PASSWORD`), and internal configuration keys before spawning child subprocesses.
- Tools only receive the bare minimum runtime environment (`PATH`, `VIRTUAL_ENV`, `PYTHONPATH`, and specifically approved parameters).

---

## Protected Filesystem Write Paths

`call_fs(operation="write", ...)` enforces write boundaries strictly restricted to approved project paths:
- `tools/` — generated tool code
- `tests/` — generated test code
- `memories/` — memory store files
- `logs/` — log files
- `data/scratch/` — automatic disk scratchpad
- `requirements.txt` — approved package dependency manifest

Writes targeting project root, `.env`, or arbitrary filesystem locations outside the workspace are blocked by default.

---

## Static AST Code Guard & Subprocess Sandboxing

Before any synthesized tool code is tested or executed:
1. **AST Scan**: Inspects AST imports and forbidden `os` attributes (`system`, `popen`, `fork`, `execv`). Forbidden constructs trigger an immediate test failure and feed back into the debugger.
2. **Subprocess Sandboxing (`core/run_tool.py`)**: Tools run in isolated child processes with a 30-second execution timeout, ensuring crashes or memory leaks cannot affect the main MAVIS daemon.

---

## Audit Logging

Every security decision is recorded to an append-only JSONL log at `logs/oni_audit.jsonl`:
```json
{
  "timestamp": "2026-10-26T10:00:00Z",
  "trust_level": "ask",
  "type": "fs",
  "operation": "write",
  "path": "tools/search_news.py",
  "decision": "allowed",
  "approved_by": "user"
}
```
Properties:
- Opened in append (`a`) mode only; never truncated or overwritten.
- Telemetry summary emitted to `data/metrics/oni.csv`.
- Inspectable at runtime via `/config audit [N]`.
