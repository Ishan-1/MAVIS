# MAVIS User Experience (UX)

## Target User
The user has sufficient technical knowledge to operate Terminal, debug and review Python code, and is comfortable taking ownership of MAVIS's functionality.

---

## Interaction Style
Similar to a CLI coding agent like Codex, but with MAVIS also able to autonomously work on scheduled tasks, actively start initiatives, and proactively assist the user in their work.

---

## Input Modes

### Primary: Natural Language (Text)
Free-form natural language typed at the `What can I do for you?` prompt. MAVIS interprets the intent, builds any missing tools, and executes a DAG pipeline.

### Slash Commands (Power User)
Prefix commands for runtime control that bypass the LLM pipeline entirely and execute deterministically.

| Command | Description |
|---|---|
| `/help` | Print all slash commands |
| `/config` | Print current config |
| `/config set <section.key> <value>` | Change any config value live |
| `/config save` | Persist in-memory config to disk |
| `/config reload` | Reload config from disk (discard unsaved changes) |
| `/config audit [N]` | Tail the last N ONI audit log entries (default 10) |
| `/trust ask|yolo|whitelist` | Change ONI trust level for the session |
| `/allow <cmd>` | Add a command to the ONI whitelist |
| `/block <cmd>` | Add a command to the ONI blacklist |
| `/greylist <cmd>` | Add a command to the ONI greylist |
| `/unlist <cmd>` | Remove a command from all ONI lists |

**Tab completion** is implemented for all slash commands and their sub-commands using `readline`. The `/` character is excluded from completer delimiters so the entire `/command` token is treated as a single unit.

---

## CLI Output & Rendering

### Current State
All output is plain `print()` to stdout. Debug traces (LLM response JSON, pipeline execution steps, tool build logs) are printed inline, making it difficult to distinguish MAVIS's "answer" from internal scaffolding noise.

```
--- LLM Response ---
{ "pipeline": [...], ... }
--------------------
Executing node 'n1': get_current_datetime
Command 'get_current_datetime' executed successfully.
--- Pipeline Execution Finished ---
Final Result (from node 'n1'): 2026-08-30T07:44:00
```

### Required: Rich Markdown Rendering
MAVIS responses should be rendered with a library capable of rendering Markdown to the terminal (e.g. `rich`, `mdv`, or `mistletoe` + a terminal renderer). At minimum:
- Code blocks with syntax highlighting
- Tables
- Bold/italic emphasis
- Clear visual separation between MAVIS's answer and internal status logs

The `print(f"\nMAVIS: {direct}\n")` direct-response path is the easiest place to introduce this — wrap that output first, then progressively add it to pipeline results.

### Verbosity Levels
Internal scaffolding (LLM JSON, `Executing node`, `Command executed`) should be suppressible at a configurable verbosity level (e.g. `output.verbosity = quiet | normal | debug`). Default: `normal` (show MAVIS answers + high-level status; suppress raw LLM JSON).

---

## Status Visibility ("Working State")

### What the user needs to see
- What MAVIS is currently doing (interpreting, building a tool, running a pipeline node)
- What tasks are queued or scheduled in the background
- Whether background workers (short-term / long-term memory) are alive and when they last ran

### Current Gaps
There is no way to query running state from the CLI. Pipeline execution is synchronous and blocking. Background workers run as subprocesses (`worker_process.py`); the only indication they are alive is the PID printed at startup.

### Planned
- `/status` slash command: shows active pipeline step (if any), time of last short-term and long-term worker run, heartbeat age, and scheduler registered tasks.
- Non-blocking spinner or progress indicator during LLM calls and tool builds (the two main latency sources).
- Optional live task log: `tail -f logs/mavis.log` equivalent accessible via `/log` or similar.

---

## Notifications

### Desktop Notifications
MAVIS should send OS-level desktop notifications to inform the user of:
- Completion of a long-running pipeline
- Background worker errors or promotions above a threshold
- Tool build failures requiring manual attention

Implementation options (Linux):
- `notify-send` via `call_system_command()` through ONI (already whitelisted path)
- `plyer` Python library (cross-platform, wraps OS notification APIs)

`notify-send` is preferred since it routes through ONI, keeping all system I/O audited.

### Notification UX Rules
- Notifications must be non-intrusive: fire-and-forget, no blocking.
- Never send more than one notification per pipeline run.
- Background worker notifications only on error or notable events (not on every 15-minute cycle).

---

## Confirmation & Trust UX (ONI Integration)

### Greylist Confirmation Flow
When a pipeline contains greylisted commands, ONI surfaces a single batch prompt **before** any command executes:

```
[ONI] The following commands require your approval:
  1. restart_process()
  2. pip install requests
Approve all? (yes/no):
```

The user responds once. If denied, the entire pipeline aborts cleanly. If approved, execution proceeds.

### YOLO Mode Warning
Switching to YOLO mode prints an explicit warning and requires the user to type `YES` (uppercase) to confirm. This prevents accidental trust escalation.

### Feedback
Every ONI decision is reflected back to the user in plain language:
- `[ONI] Pipeline aborted: 'rm -rf /' is blacklisted.`
- `[MAVIS] 'pip_install' → greylist.`
- `[MAVIS] Trust level → whitelist_only`

---

## Tab Completion

### Current Implementation
`readline` tab completion is wired for slash commands. The completer handles three cases:
1. Completing the base command (e.g. `/c` → `/config`, `/trust`)
2. Completing sub-commands after a space (e.g. `/config ` → `set`, `save`, `reload`, `audit`)
3. Completing argument values for `/allow`, `/block`, `/greylist`, `/unlist` (offers names from `commands_list.json`)

### Planned
- **History-based completion**: `readline` history file (`.mavis_history`) persisted across sessions so previous commands can be recalled with ↑/↓.
- **Fuzzy / semantic completion for natural language**: Low priority; tab is less meaningful for free-form NL input.

---

## Session Lifecycle

### Startup
```
MAV v1.5 (ONI + unified config + worker subprocesses)
Say 'exit' or 'quit' to stop. Type /help for commands.
[ONI] Trust level: ask
[Workers] short_term_worker started (PID 12345, every 15 min).
[Workers] long_term_worker started (PID 12346, every 480 min).
```

### Graceful Exit
On `exit`, `quit`, `Ctrl+C`, or `Ctrl+D`:
1. Stop background worker subprocesses (SIGTERM → wait 5s → SIGKILL if needed).
2. Stop the `TaskRunner` scheduler.
3. Remove heartbeat file.
4. Return control to the terminal cleanly.

**Known issue**: `runner.stop()` currently raises a traceback on second `Ctrl+C` because the scheduler thread is still running. Fix: wrap `runner.stop()` in a `try/except KeyboardInterrupt` (already partially done) and ensure the scheduler's internal `threading.Event` is set before joining the thread.

---

## Error Handling & User Feedback

| Situation | Current Behaviour | Target Behaviour |
|---|---|---|
| LLM JSON parse error | Prints raw response + error | Show a friendly `[MAVIS] I had trouble understanding that. Try rephrasing.` then optionally show raw in debug mode |
| Tool build failure | Prints stack trace | `[MAVIS] I couldn't build the tool for '<signature>'. <reason>. You may need to build it manually.` |
| Tool execution timeout | Prints timeout message | Same message + notify-send if pipeline was user-initiated |
| Missing dependency node | Prints abort | `[MAVIS] Pipeline aborted: step '<id>' depends on '<dep>' which didn't run.` |
| ONI blacklist hit | Pipeline aborts | Already clear; no change needed |

---

## Memory UX

### What the user sees
- Memory is mostly invisible unless the user explicitly asks MAVIS to remember something.
- `direct_response` path surfaces memory: if MAVIS has the answer in context, it prints it directly without executing a pipeline.
- Long-term memory stores permanent behaviour rules and facts, so MAVIS's responses should reflect these across sessions silently.

### Transparency
- The user can inspect memory files directly (`memories/short_term/json/`, `memories/long_term/json/`).
- A `/memory` slash command (planned) could show: working memory size (tokens used / cap), last short-term promotion timestamp, and top-K retrieved entries for the current query.

---

## Open Design Questions

| # | Question | Decision |
|---|---|---|
| 1 | Markdown rendering library | `rich` |
| 2 | Verbosity config | Single `verbosity` enum: `quiet \| normal \| debug` |
| 3 | `/status` command scope | Scheduler state + memory token count |
| 4 | Notification backend | `notify-send` via ONI |
| 5 | History file location | `data/.mavis_history` |
