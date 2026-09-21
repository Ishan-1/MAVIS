# MAVIS User Experience (UX)

## Target User
The user has sufficient technical knowledge to operate the Terminal, review Python code, and take ownership of MAVIS's operational capabilities as an autonomous local-first assistant.

---

## Interaction Style
Similar to a CLI coding agent, but with MAVIS also capable of autonomously running multi-wave goals (`/goal`), executing background memory consolidation daemons, and providing rich observability.

---

## Input Modes

### Primary: Natural Language (Text)
Free-form natural language typed at the interactive prompt:
```
╭─ What can I do for you?
╰─❯ 
```
MAVIS interprets intent, checks the SQLite semantic cache, plans a heterogeneous DAG, dynamically builds missing tools or subagents, and executes the pipeline.

### Slash Commands (Runtime Control)
Prefix commands that execute deterministically without invoking the LLM planner:

| Command | Arguments | Description |
|---|---|---|
| `/help` | — | Display available slash commands and descriptions |
| `/goal` | `[--yolo] <description>` | Execute an autonomous multi-wave goal (`--yolo` for unattended execution) |
| `/status` | — | Inspect heartbeat age, background daemon workers, scheduler tasks, and memory pressure |
| `/metrics` | `[session]` | Render terminal performance, token economics, latency percentiles, and caching tables |
| `/dashboard` | — | Launch local Streamlit observability dashboard (`http://localhost:8501`) |
| `/tooldash` | — | Launch Tool, Subagent & MCP management studio (`http://localhost:8502`) |
| `/config` | `[set\|save\|reload\|audit]` | Inspect or update configuration keys live, persist to disk, or tail ONI audit logs |
| `/trust` | `ask\|yolo\|whitelist` | Change ONI security trust level for the active session |
| `/allow` | `<tool_name>` | Add a tool to the ONI whitelist (bypasses confirmation prompts) |
| `/block` | `<tool_name>` | Add a tool to the ONI blacklist (permanently blocks execution) |
| `/greylist` | `<tool_name>` | Add a tool to the ONI greylist (always requests user confirmation) |
| `/unlist` | `<tool_name>` | Remove a tool from all ONI lists |
| `/mcp` | `status\|list\|reload` | Inspect connected Model Context Protocol servers, list discovered tools, or hot-reload |
| `/save` | `[filename.md]` | Export the current session chat transcript to Markdown |
| `exit` / `quit` | — | Signal-safe clean shutdown terminating child processes and background daemons |

---

## Prompt & Autocompletion (`prompt_toolkit`)

### Dynamic Popup Completion
Powered by `prompt_toolkit`'s `MavisSlashCompleter`:
- Typing `/` triggers an interactive popup showing all available slash commands alongside descriptive docstrings.
- Typing subcommands (e.g. `/config `, `/trust `, `/mcp `) dynamically displays available sub-actions.
- Tool commands (`/allow `, `/block `, `/greylist `, `/unlist `) automatically suggest tools from `commands_list.json`.

### Persistent History
All CLI inputs are recorded to `data/.mavis_history` via `FileHistory`, allowing instant navigation through past queries using the ↑ and ↓ arrow keys across sessions.

### Real-Time Bottom Toolbar
A persistent status line rendered at the bottom of the terminal:
```
 [MAVIS v1.0]  Session: 14m • Tokens: 1,840 in / 420 out • Cache Hits: 3 • WM: 1,240/12,000 tokens • Type / for commands 
```
Automatically refreshes on each prompt render with up-to-date token economics, elapsed session time, and working memory token usage.

---

## CLI Output & Rich Rendering (`core/output.py`)

All terminal output is mediated through `core/output.py` backed by `rich`:

- **Themed Styles**:
  - `mavis.answer`: Clean white text for final responses.
  - `mavis.status`: Dim cyan text for internal progress notifications.
  - `mavis.warn`: Bold yellow text for non-fatal warnings.
  - `mavis.error`: Bold red text for failure diagnostics.
  - `mavis.ok`: Bold green text for successful operations.
- **Markdown & Code Highlighting**: Final answers synthesized by `Answerer` are rendered with full GitHub-flavored Markdown, including syntax-highlighted code fences and structured tables.
- **Spinners & Progress**: Indeterminate background operations (LLM calls, tool synthesis, pytest validation) run within `with spinner("...")` contexts so the terminal never feels frozen.
- **Live Output Streaming**: Subprocess tools executed by `main.py: call_command` stream stdout and stderr line-by-line using non-blocking OS `selectors`.
- **Verbosity Filtering**: Configurable via `cfg.output["verbosity"]` (`quiet`, `normal`, `debug`):
  - `quiet`: Only user-facing answers and critical warnings.
  - `normal`: Answers, high-level status messages, and approval gates (default).
  - `debug`: Detailed pipeline traces, raw LLM outputs, and diagnostic logs.

---

## Desktop Notifications

MAVIS sends desktop notifications (via Linux `notify-send` audited through ONI):
- When long-running autonomous goals (`/goal`) complete or encounter unrecoverable errors.
- When tool execution times out (default: 30s).
- When tool builds fail after retry exhaustion.

Notifications are strictly non-blocking and fire-and-forget.

---

## Confirmation & Trust UX (ONI Integration)

### Batch Pre-Flight Scan
Before any DAG pipeline executes, `oni.preflight_scan(pipeline)` analyzes all nodes:
- **Blacklist Hit**: Execution is immediately aborted before any step runs, informing the user of the forbidden command.
- **Greylist Confirmation**: If greylisted commands or unsafe operations are present, an interactive approval dialog (`ConfirmationGate`) prompts the user once upfront before execution begins.
- **Whitelist Operations**: Whitelisted and safe read commands (`get_current_datetime`, `ls`, `read_file_contents`) run silently without prompting.

### Single-Prompt Task Leases (`/goal`)
Autonomous multi-wave goals acquire an execution lease upfront, preventing repetitive approval prompts across waves while retaining session security.

---

## Session Lifecycle & Shutdown

### Graceful Signal-Safe Exit
When the user exits via `exit`, `quit`, `Ctrl+C`, or `Ctrl+D`:
1. `_safe_terminate` cleanly shuts down any running Streamlit dashboards (`_dashboard_proc`, `_tooldash_proc`) with `SIGTERM`, waiting up to 2 seconds before escalating to `SIGKILL`.
2. `_safe_shutdown_call` terminates the background daemon workers (`_stop_workers()`), stops the `TaskRunner` scheduler, and closes connected MCP server processes.
3. Removes the `.mavis_heartbeat` file.
4. Prints a formatted exit summary (`_print_exit_summary()`) displaying total session duration, total queries handled, and token economics.
