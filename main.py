from dotenv import load_dotenv
from prompts.prompt_templates import (
    interpreter_prompt,
    interpreter_system_prompt,
    format_interpreter_user_prompt,
)
from tool_builder import ToolBuilder
from core.scheduler import TaskRunner
from core.llm import get_llm_client
from core.tool_retriever import ToolRetriever
from memories.memory_store import MemoryStore
from memories.emotion_classifier import parse_classifier_fields
from core.output import (
    mavis_answer, mavis_status, mavis_ok, mavis_warn,
    mavis_error, mavis_debug, mavis_print, oni_print, spinner,
    rule, print_table, interactive_select_yes_no,
)
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style
import tasks.short_term_worker as st_worker
import tasks.long_term_worker as lt_worker
import os
import json
import selectors
import subprocess
import sys
import time
from datetime import datetime

load_dotenv()
llm = get_llm_client()
tool_builder = ToolBuilder(llm)
from agent_builder import AgentBuilder
from core.answerer import Answerer
from core.agents import load_agent
agent_builder = AgentBuilder(llm)
answerer = Answerer(llm)
memory_store = MemoryStore(llm, namespace="interpreter")
tool_retriever = ToolRetriever(llm)
from core.dag import validate_and_sort_dag, resolve_params
from core.caching import cache_manager

# ── Central config + ONI ────────────────────────────────────────────────────────────
from core.config import cfg
from oni import oni as _oni

commands_list = {}
with open("data/commands_list.json", "r") as file:
    commands_list = json.load(file)

agents_list = {}
if os.path.exists("data/agents_list.json"):
    try:
        with open("data/agents_list.json", "r") as file:
            agents_list = json.load(file)
    except Exception:
        agents_list = {}

import uuid
from core.metrics import MetricEmitter, get_metrics_summary, format_metrics_tables

_MAV_ROOT = os.path.dirname(os.path.abspath(__file__))
_session_chat: list[dict[str, str]] = []
_session_start_ts = time.time()
_interpreter_emitter = MetricEmitter("interpreter")
_dag_emitter = MetricEmitter("dag_execution")


def compute_dag_depth(pipeline: list[dict]) -> int:
    """Compute the longest critical dependency path in the DAG."""
    from core.dag import extract_node_dependencies
    depths: dict[str, int] = {}
    for node in pipeline:
        nid = node.get("id", "")
        deps = extract_node_dependencies(node)
        if not deps:
            depths[nid] = 1
        else:
            depths[nid] = 1 + max((depths.get(d, 0) for d in deps), default=0)
    return max(depths.values(), default=1) if depths else 0

# ── Readline tab-completion for slash commands ──────────────────────────────────
_SLASH_SUBCOMMANDS = {
    "/help": [],
    "/save": [],
    "/config": ["set", "save", "reload", "audit"],
    "/trust": ["ask", "yolo", "whitelist"],
    "/allow": [],
    "/block": [],
    "/greylist": [],
    "/unlist": [],
    "/status": [],
    "/metrics": [],
    "/dashboard": [],
}

_COMMAND_METAS = {
    "/help": "Show available slash commands",
    "/save": "Export current chat history to a markdown (.md) file",
    "/status": "Heartbeat, workers, scheduler, memory tokens",
    "/metrics": "Performance, latency, tokens & caching metrics",
    "/dashboard": "Launch local Streamlit web dashboard",
    "/config": "Inspect or edit configuration",
    "/trust": "Switch ONI trust level (ask | yolo | whitelist)",
    "/allow": "Add command to ONI whitelist",
    "/block": "Add command to ONI blacklist",
    "/greylist": "Add command to ONI greylist",
    "/unlist": "Remove command from all ONI lists",
}

_SUBCOMMAND_METAS = {
    "/config": {
        "set": "Set config value (e.g. /config set memory.top_k 10)",
        "save": "Persist current config to disk",
        "reload": "Reload config from disk",
        "audit": "Tail last N ONI audit log entries",
    },
    "/trust": {
        "ask": "Prompt on greylisted commands (default)",
        "yolo": "Allow all commands without prompting",
        "whitelist": "Deny all unlisted commands",
    },
}


class MavisSlashCompleter(Completer):
    """
    Dynamic slash command completer for prompt_toolkit.
    Pops down options and descriptions automatically when '/' is typed.
    """

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        line = text.lstrip()

        if not line.startswith("/"):
            return

        parts = line.split()

        # Completing the base command, e.g. '/' or '/con'
        if len(parts) == 0 or (len(parts) == 1 and not line.endswith(" ")):
            prefix = parts[0] if parts else line
            for cmd, meta in _COMMAND_METAS.items():
                if cmd.startswith(prefix):
                    yield Completion(
                        cmd,
                        start_position=-len(prefix),
                        display=cmd,
                        display_meta=meta,
                    )
        # Completing subcommands or command arguments
        elif len(parts) >= 1:
            cmd = parts[0]
            sub_prefix = parts[1] if (len(parts) == 2 and not line.endswith(" ")) else ""

            if cmd in _SUBCOMMAND_METAS:
                for sub, meta in _SUBCOMMAND_METAS[cmd].items():
                    if sub.startswith(sub_prefix):
                        yield Completion(
                            sub,
                            start_position=-len(sub_prefix),
                            display=sub,
                            display_meta=meta,
                        )
            elif cmd in ("/allow", "/block", "/greylist", "/unlist"):
                for sig, desc in commands_list.items():
                    func_name = sig.split("(")[0].strip()
                    if func_name.startswith(sub_prefix):
                        desc_str = desc.get("description", "") if isinstance(desc, dict) else str(desc)
                        yield Completion(
                            func_name,
                            start_position=-len(sub_prefix),
                            display=func_name,
                            display_meta=desc_str[:35] + ("..." if len(desc_str) > 35 else ""),
                        )


_PROMPT_STYLE = Style.from_dict(
    {
        "prompt.border": "#00d7ff bold",
        "prompt.label": "#ffffff bold",
        "prompt.arrow": "#00d7ff bold",
        "bottom-toolbar": "#888888 bg:#1a1a1a italic",
        "completion-menu.completion": "bg:#242424 #e0e0e0",
        "completion-menu.completion.current": "bg:#005f87 #ffffff bold",
        "completion-menu.meta.completion": "bg:#1a1a1a #888888 italic",
        "completion-menu.meta.completion.current": "bg:#004b6b #00d7ff bold italic",
    }
)


def _get_prompt_message():
    return [
        ("class:prompt.border", "╭─ "),
        ("class:prompt.label", "What can I do for you?"),
        ("class:prompt.border", "\n╰─❯ "),
    ]


def _get_bottom_toolbar():
    now = time.time()
    elapsed_m = int((now - _session_start_ts) / 60)
    wm = memory_store.get_working_memory()
    wm_tokens = sum(len(t.get("content", "")) // 4 for t in wm)
    wm_cap = cfg.memory.get("max_token", 12000)

    summary = get_metrics_summary(_session_start_ts)
    t_in = summary["tokens_total"]["input"]
    t_out = summary["tokens_total"]["output"]
    cache_hits = summary["caching"]["hits"]

    return [
        ("", f" [MAVIS v1.0]  Session: {elapsed_m}m • Tokens: {t_in:,} in / {t_out:,} out • Cache Hits: {cache_hits} • WM: {wm_tokens}/{wm_cap} tokens • Type / for commands "),
    ]


# ── Slash command handler ─────────────────────────────────────────────────────

_HELP_ROWS = [
    ("/help",                    "This help message."),
    ("/save [file.md]",          "Export current chat history to a Markdown file."),
    ("/status",                  "Heartbeat, worker PIDs, scheduler tasks, memory usage."),
    ("/metrics",                 "Display performance, latency, token, and cache analytics."),
    ("/dashboard",               "Launch local Streamlit web dashboard in browser."),
    ("/config",                  "Print current config."),
    ("/config save",             "Persist config to disk."),
    ("/config reload",           "Reload config from disk."),
    ("/config set <key> <val>",  "Set any config value live (JSON-parsed)."),
    ("/config audit [N]",        "Tail last N ONI audit entries (default 10)."),
    ("/trust ask|yolo|whitelist","Change ONI trust level for this session."),
    ("/allow <cmd>",             "Add to ONI whitelist."),
    ("/block <cmd>",             "Add to ONI blacklist."),
    ("/greylist <cmd>",          "Add to ONI greylist."),
    ("/unlist <cmd>",            "Remove from all ONI lists."),
]


def _save_chat_to_markdown(filename: str = "") -> None:
    """Save the current session chat history to a Markdown file."""
    filename = filename.strip()
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"chat_{timestamp}.md"
    elif not filename.endswith(".md"):
        filename = f"{filename}.md"

    entries = list(_session_chat)
    if not entries:
        wm = memory_store.get_working_memory()
        for t in wm:
            content = t.get("content", "")
            role = t.get("role", "unknown")
            ts = t.get("timestamp")
            dt_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, (int, float)) else ""
            if content.strip():
                entries.append({"role": role, "content": content, "timestamp": dt_str})

    if not entries:
        mavis_warn("No chat history available in this session to save.")
        return

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# MAVIS Chat Session",
        f"**Exported:** {now_str}\n",
        "---",
        "",
    ]

    for entry in entries:
        role_label = "User" if entry["role"] == "user" else "MAVIS"
        ts_label = f" *({entry['timestamp']})*" if entry.get("timestamp") else ""
        lines.append(f"### {role_label}{ts_label}")
        lines.append(entry["content"].strip())
        lines.append("")
        lines.append("---")
        lines.append("")

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).strip() + "\n")
        mavis_ok(f"Chat saved successfully to [bold]{filename}[/bold] ({len(entries)} turn(s)).")
    except Exception as e:
        mavis_error(f"Failed to save chat to '{filename}': {e}")


def _oni_list_add(list_name: str, command: str) -> None:
    """Add *command* to one ONI list and remove from all others."""
    oni_section = cfg.oni
    all_lists = ["whitelist", "blacklist", "greylist"]
    # Remove from any other list first
    for lst in all_lists:
        current = oni_section.get(lst, [])
        if command in current:
            oni_section[lst] = [c for c in current if c != command]
    # Add to target list
    target = oni_section.setdefault(list_name, [])
    if command not in target:
        target.append(command)


def handle_slash_command(raw: str) -> bool:
    """
    Handle a /command string. Returns True if it was a slash command (so the
    normal LLM pipeline is skipped), False if it should be passed through.
    """
    raw = raw.strip()
    if not raw.startswith("/"):
        return False

    parts = raw.split(maxsplit=2)
    verb = parts[0].lower()

    # ── /help ─────────────────────────────────────────────────────────────────
    if verb == "/help":
        rule("Slash Commands")
        print_table(_HELP_ROWS)
        mavis_print("  [dim]Tip: type [bold]/[/bold] then Tab to browse commands.[/dim]")
        rule()
        return True

    # ── /save [filename] ──────────────────────────────────────────────────────
    if verb == "/save":
        filename = parts[1] if len(parts) > 1 else ""
        _save_chat_to_markdown(filename)
        return True

    # ── /status ───────────────────────────────────────────────────────────────
    if verb == "/status":
        _print_status()
        return True

    # ── /metrics ──────────────────────────────────────────────────────────────
    if verb == "/metrics":
        tables = format_metrics_tables(_session_start_ts if (len(parts) > 1 and parts[1] == "session") else None)
        for t in tables:
            mavis_print(t)
        return True

    # ── /dashboard ────────────────────────────────────────────────────────────
    if verb == "/dashboard":
        mavis_status("Launching MAVIS Local Dashboard on http://localhost:8501 ...")
        try:
            import shutil
            import webbrowser
            dashboard_path = os.path.join(_MAV_ROOT, "scripts", "dashboard.py")
            
            # Find streamlit binary: check local venv first, then PATH, then sys.executable
            venv_streamlit = os.path.join(_MAV_ROOT, "bin", "streamlit")
            venv_python = os.path.join(_MAV_ROOT, "bin", "python")
            
            if os.path.exists(venv_streamlit):
                cmd = [venv_streamlit, "run", dashboard_path, "--server.headless", "true", "--server.port", "8501"]
            elif os.path.exists(venv_python):
                cmd = [venv_python, "-m", "streamlit", "run", dashboard_path, "--server.headless", "true", "--server.port", "8501"]
            elif shutil.which("streamlit"):
                cmd = [shutil.which("streamlit"), "run", dashboard_path, "--server.headless", "true", "--server.port", "8501"]
            else:
                cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path, "--server.headless", "true", "--server.port", "8501"]

            proc = subprocess.Popen(
                cmd,
                cwd=_MAV_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(1.0)
            if proc.poll() is not None:
                mavis_error(f"Dashboard process exited immediately with code {proc.returncode}. Try running: ./bin/streamlit run scripts/dashboard.py")
            else:
                mavis_ok(f"Dashboard running at [bold cyan]http://localhost:8501[/bold cyan] (PID {proc.pid})")
                try:
                    webbrowser.open("http://localhost:8501")
                except Exception:
                    pass
        except Exception as e:
            mavis_error(f"Could not launch dashboard: {e}")
        return True

    # ── /config ... ───────────────────────────────────────────────────────────
    if verb == "/config":
        sub = parts[1].lower() if len(parts) > 1 else "status"

        if sub in ("status", ""):
            _print_config()

        elif sub == "save":
            cfg.save()
            mavis_ok("Config saved to data/mavis_config.json.")

        elif sub == "reload":
            cfg.reload()
            mavis_ok("Config reloaded from disk.")
            mavis_status(f"Trust level: {cfg.oni.get('trust_level', 'ask')}")

        elif sub == "set":
            if len(parts) < 3:
                mavis_error("Usage: /config set <section.key> <value>")
                return True
            # parts[2] still has "key value" together — re-split
            kv = parts[2].split(maxsplit=1)
            if len(kv) < 2:
                mavis_error("Usage: /config set <section.key> <value>")
                return True
            dotted_key, raw_value = kv[0], kv[1]
            try:
                cfg.set_dotted(dotted_key, raw_value)
                section, key = dotted_key.split(".", 1)
                new_val = cfg.get(section, key)
                mavis_ok(f"{dotted_key} = {new_val!r}")
                if section == "scheduler":
                    mavis_warn("Scheduler changes require a restart to take effect.")
            except ValueError as e:
                mavis_error(f"Error: {e}")

        elif sub == "audit":
            n = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 10
            _print_audit(n)

        else:
            mavis_error(f"Unknown config sub-command: {sub!r}. Try /help.")

        return True

    # ── /trust <level> ────────────────────────────────────────────────────────
    if verb == "/trust":
        if len(parts) < 2:
            mavis_status(f"Current trust level: {cfg.oni.get('trust_level', 'ask')}")
            return True
        level = parts[1].lower()
        valid = {"ask", "yolo", "whitelist", "whitelist_only"}
        if level not in valid:
            mavis_error(f"Unknown trust level {level!r}. Valid: ask, yolo, whitelist.")
            return True
        # Normalise alias
        if level == "whitelist":
            level = "whitelist_only"

        if level == "yolo":
            mavis_warn("YOLO mode disables all ONI permission checks.")
            try:
                confirm = input("  Type YES to confirm: ").strip()
            except (EOFError, KeyboardInterrupt):
                confirm = ""
            if confirm != "YES":
                mavis_status("Cancelled.")
                return True

        cfg.set("oni", "trust_level", level)
        _oni.config.trust_level = level   # keep live ONI in sync
        mavis_ok(f"Trust level → {level}")
        return True

    # ── /allow, /block, /greylist, /unlist ───────────────────────────────────
    if verb in ("/allow", "/block", "/greylist", "/unlist"):
        if len(parts) < 2:
            mavis_error(f"Usage: {verb} <command_name>")
            return True
        command = parts[1]

        if verb == "/unlist":
            for lst in ["whitelist", "blacklist", "greylist"]:
                oni_section = cfg.oni
                if command in oni_section.get(lst, []):
                    oni_section[lst] = [c for c in oni_section[lst] if c != command]
            mavis_ok(f"'{command}' removed from all ONI lists.")
        else:
            list_map = {"/allow": "whitelist", "/block": "blacklist", "/greylist": "greylist"}
            target_list = list_map[verb]
            _oni_list_add(target_list, command)
            mavis_ok(f"'{command}' → {target_list}.")

        return True

    # Not a recognised slash command — pass through to LLM
    return False


def _print_config() -> None:
    """Pretty-print the current in-memory config as rich tables."""
    d = cfg.as_dict()
    oni = d.get("oni", {})
    mem = d.get("memory", {})
    tb  = d.get("toolbuilder", {})
    sch = d.get("scheduler", {})
    out = d.get("output", {})

    rule("Configuration  [dim]data/mavis_config.json[/dim]")

    print_table([
        ("trust_level",              oni.get("trust_level")),
        ("whitelist",                ", ".join(oni.get("whitelist", [])) or "(empty)"),
        ("greylist",                 ", ".join(oni.get("greylist", [])) or "(empty)"),
        ("blacklist",                ", ".join(oni.get("blacklist", [])) or "(empty)"),
        ("approved_write_paths",     ", ".join(oni.get("approved_fs_write_paths", []))),
        ("tool_timeout_s",           oni.get("tool_execution_timeout_seconds")),
    ], title="ONI")

    print_table([
        ("max_token",                mem.get("max_token")),
        ("top_k",                    mem.get("top_k")),
        ("short_term_ttl_days",      mem.get("short_term_ttl_days")),
        ("session_timeout_min",      mem.get("session_timeout_minutes")),
        ("repetition_window",        mem.get("repetition_window")),
        ("repetition_sim_threshold", mem.get("repetition_similarity_threshold")),
        ("emotion_threshold",        mem.get("emotion_strength_threshold")),
        ("intent_threshold",         mem.get("intent_strength_threshold")),
        ("lt_intent_threshold",      mem.get("lt_intent_threshold")),
    ], title="Memory")

    print_table([
        ("max_retries",      tb.get("max_retries")),
        ("forbidden_imports", ", ".join(tb.get("forbidden_imports", []))),
    ], title="ToolBuilder")

    print_table([
        ("tick_seconds",             sch.get("tick_seconds")),
        ("short_term_interval_min",  sch.get("short_term_worker_interval_minutes")),
        ("long_term_interval_min",   sch.get("long_term_worker_interval_minutes")),
    ], title="Scheduler  [dim](restart required to change)[/dim]")

    print_table([
        ("verbosity",         out.get("verbosity")),
        ("history_file",      out.get("history_file")),
        ("notify_threshold_s", out.get("notify_pipeline_threshold_s")),
    ], title="Output")

    rule()


def _print_status() -> None:
    """Print current MAVIS operational status as rich tables."""
    now = time.time()

    # Heartbeat age
    hb_age = "unknown"
    if os.path.exists(_HEARTBEAT_PATH):
        try:
            with open(_HEARTBEAT_PATH) as f:
                hb_ts = float(f.read().strip())
            hb_age = f"{int(now - hb_ts)}s ago"
        except Exception:
            pass

    rule("MAVIS Status")

    # System
    print_table([("heartbeat", hb_age)], title="System")

    # Workers
    worker_rows = []
    for proc in _worker_procs:
        alive = proc.poll() is None
        state = "[green]alive[/green]" if alive else f"[red]exited ({proc.returncode})[/red]"
        worker_rows.append((f"PID {proc.pid}", state))
    if not worker_rows:
        worker_rows = [("workers", "[dim]none started[/dim]")]
    print_table(worker_rows, title="Workers")

    # Scheduler tasks
    task_rows = []
    for task in runner.list_tasks():
        last = task.get("last_run_ts")
        last_str = f"{int(now - last)}s ago" if last else "[dim]not yet run[/dim]"
        task_rows.append((
            task["name"],
            f"every {task['interval_minutes']} min   last: {last_str}",
        ))
    print_table(task_rows, title="Scheduler")

    # Working memory
    wm = memory_store.get_working_memory()
    wm_tokens = sum(len(t.get("content", "")) // 4 for t in wm)
    wm_cap = cfg.memory.get("max_token", 12000)
    print_table([
        ("tokens", f"{wm_tokens} / {wm_cap} cap"),
        ("turns",  len(wm)),
    ], title="Working Memory")

    rule()



def _print_audit(n: int) -> None:
    """Print the last *n* lines of the ONI audit log."""
    log_path = "logs/oni_audit.jsonl"
    if not os.path.exists(log_path):
        mavis_warn("No audit log found.")
        return
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
        tail = lines[-n:]
        mavis_print(f"[MAVIS] Last {len(tail)} audit entries:", level="quiet")
        for line in tail:
            try:
                entry = json.loads(line)
                ts = entry.pop("timestamp", "?")
                mavis_print(f"  [{ts}] {entry}", level="quiet")
            except json.JSONDecodeError:
                mavis_print(f"  {line.rstrip()}", level="quiet")
    except Exception as e:
        mavis_error(f"Could not read audit log: {e}")


# ── Tool execution ────────────────────────────────────────────────────────────

def restart_mav():
    """Request a controlled restart through ONI (greylist — requires user approval)."""
    mavis_status("Requesting MAV restart via ONI...")
    _oni.call_system_command("restart_process", {})


def _notify(title: str, body: str = "") -> None:
    """Fire a non-blocking desktop notification via ONI (notify-send)."""
    try:
        _oni.call_system_command("notify-send", {"args": [title, body]})
    except Exception:
        pass  # Notifications are best-effort — never crash the main loop


def call_command(command_name, params_dict):
    """
    Execute a tool in an isolated subprocess via run_tool.py with two-way IPC.

    Sandboxing: a crashed or malicious tool cannot take down the MAVIS process.
    Tool prints go to stderr and are captured for debug logs.
    Bidirectional IPC handles interactive ONI approval prompts in the main terminal.
    A configurable timeout kills runaway tools.
    """
    env = os.environ.copy()
    env["MAVIS_TOOL_SUBPROCESS"] = "1"
    if hasattr(_oni, "session_allowances") and _oni.session_allowances:
        env["MAVIS_SESSION_ALLOWANCES"] = json.dumps(list(_oni.session_allowances))

    proc = subprocess.Popen(
        [
            sys.executable,
            os.path.join(_MAV_ROOT, "core", "run_tool.py"),
            command_name,
            json.dumps(params_dict),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=_MAV_ROOT,
        env=env,
    )

    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ, data="stdout")
    sel.register(proc.stderr, selectors.EVENT_READ, data="stderr")

    timeout_seconds = float(_oni.config.tool_execution_timeout)
    elapsed_tool_time = 0.0
    last_tick = time.time()
    final_output_str = ""
    stderr_lines = []

    try:
        while proc.poll() is None or sel.get_map():
            now = time.time()
            elapsed_tool_time += (now - last_tick)
            last_tick = now

            if elapsed_tool_time >= timeout_seconds:
                proc.kill()
                proc.wait()
                msg = f"'{command_name}' timed out after {int(timeout_seconds)}s."
                mavis_error(msg)
                _notify("MAVIS: Tool Timeout", msg)
                return -1, f"Tool '{command_name}' timed out."

            remaining = max(0.1, timeout_seconds - elapsed_tool_time)
            events = sel.select(timeout=min(remaining, 0.5))

            for key, mask in events:
                if key.data == "stdout":
                    line = proc.stdout.readline()
                    if not line:
                        sel.unregister(proc.stdout)
                        continue
                    try:
                        data = json.loads(line.strip())
                    except Exception:
                        data = None

                    if isinstance(data, dict) and data.get("__oni_ipc__"):
                        if data.get("type") == "approval_request":
                            desc = data.get("description", "")
                            approved = _oni.gate.request_approval(desc)
                            proc.stdin.write(json.dumps({"approved": approved}) + "\n")
                            proc.stdin.flush()
                            # Do not count user decision time towards tool execution timeout
                            last_tick = time.time()
                    else:
                        final_output_str = line.strip()

                elif key.data == "stderr":
                    line = proc.stderr.readline()
                    if not line:
                        sel.unregister(proc.stderr)
                        continue
                    stderr_lines.append(line)

        proc.wait()

        try:
            rem_err = proc.stderr.read()
            if rem_err:
                stderr_lines.append(rem_err)
        except Exception:
            pass

        full_stderr = "".join(stderr_lines).strip()
        if full_stderr:
            mavis_debug(full_stderr, entity=command_name)

        if proc.returncode != 0 and not final_output_str:
            mavis_error(f"Tool '{command_name}' exited with code {proc.returncode}.")
            return -1, full_stderr or f"Tool '{command_name}' exited with error."

        if not final_output_str:
            return -1, f"Tool '{command_name}' produced no output."

        status, output = json.loads(final_output_str)

        if status == 0:
            mavis_ok(f"'{command_name}' executed successfully.")
        else:
            mavis_error(f"'{command_name}' failed: {output}")

        return status, output

    except subprocess.TimeoutExpired:
        msg = f"'{command_name}' timed out after {_oni.config.tool_execution_timeout}s."
        mavis_error(msg)
        _notify("MAVIS: Tool Timeout", msg)
        return -1, f"Tool '{command_name}' timed out."
    except json.JSONDecodeError as e:
        mavis_error(f"'{command_name}' returned invalid output.")
        mavis_debug(f"Raw stdout: {final_output_str}", entity=command_name)
        return -1, f"Invalid output from tool '{command_name}': {e}"
    except Exception as e:
        mavis_error(f"Unexpected error with '{command_name}': {e}")
        return -1, str(e)
    finally:
        try:
            sel.close()
        except Exception:
            pass
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass


# ── Pipeline execution ────────────────────────────────────────────────────────

def execute_pipeline(pipeline, query: str = "", context: str = "", turn_id: str = ""):
    """
    Execute a list of pipeline nodes in topological sorted order, resolving dependencies.

    Validates DAG structure (cycle detection and dangling references) and sorts
    nodes topologically before execution.
    ONI pre-flight scans the topologically sorted pipeline before any node runs.
    """
    sorted_pipeline, dag_err = validate_and_sort_dag(pipeline)
    dag_size = len(pipeline) if isinstance(pipeline, list) else 0
    dag_depth = compute_dag_depth(sorted_pipeline) if sorted_pipeline else 1
    tool_nodes_count = sum(1 for n in (sorted_pipeline or []) if n.get("type", "tool") == "tool")
    subagent_nodes_count = sum(1 for n in (sorted_pipeline or []) if n.get("type") == "subagent")
    _pipeline_start = time.perf_counter()

    if dag_err:
        mavis_error(f"Pipeline validation failed: {dag_err}")
        _dag_emitter.log({
            "turn_id": turn_id,
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "latency_ms": 0.0,
            "status": "aborted",
            "dag_size": dag_size,
            "dag_depth": dag_depth,
            "tool_nodes_count": tool_nodes_count,
            "subagent_nodes_count": subagent_nodes_count,
            "failed_node_id": "dag_validation",
        })
        return

    ok, issues = _oni.preflight_scan(sorted_pipeline)
    if not ok:
        oni_print(f"Pipeline aborted: {'; '.join(issues)}")
        _dag_emitter.log({
            "turn_id": turn_id,
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "latency_ms": 0.0,
            "status": "aborted",
            "dag_size": dag_size,
            "dag_depth": dag_depth,
            "tool_nodes_count": tool_nodes_count,
            "subagent_nodes_count": subagent_nodes_count,
            "failed_node_id": "oni_preflight",
        })
        return

    node_results = {}
    mavis_status("Starting pipeline execution...")

    for node in sorted_pipeline:
        node_id = node["id"]
        node_type = node.get("type", "tool")
        command_name = node["function_name"]
        params = node.get("params", {})

        mavis_status(f"Running step '{node_id}' ({node_type}): {command_name}")

        try:
            resolved_params, res_err = resolve_params(params, node_results)
            if res_err:
                mavis_error(
                    f"Pipeline aborted: step '{node_id}' resolution error: {res_err}"
                )
                latency_ms = round((time.perf_counter() - _pipeline_start) * 1000, 2)
                _dag_emitter.log({
                    "turn_id": turn_id,
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "latency_ms": latency_ms,
                    "status": "node_failed",
                    "dag_size": dag_size,
                    "dag_depth": dag_depth,
                    "tool_nodes_count": tool_nodes_count,
                    "subagent_nodes_count": subagent_nodes_count,
                    "failed_node_id": node_id,
                })
                return

            if node_type == "subagent":
                agent = load_agent(command_name, llm)
                if not agent:
                    mavis_error(f"Pipeline aborted: subagent '{command_name}' not found.")
                    latency_ms = round((time.perf_counter() - _pipeline_start) * 1000, 2)
                    _dag_emitter.log({
                        "turn_id": turn_id,
                        "start_time": datetime.now().isoformat(),
                        "end_time": datetime.now().isoformat(),
                        "latency_ms": latency_ms,
                        "status": "node_failed",
                        "dag_size": dag_size,
                        "dag_depth": dag_depth,
                        "tool_nodes_count": tool_nodes_count,
                        "subagent_nodes_count": subagent_nodes_count,
                        "failed_node_id": node_id,
                    })
                    return
                status, result = agent.run(turn_id=turn_id, **resolved_params)
            else:
                status, result = call_command(command_name, resolved_params)

            # Contract verification: 1st element MUST be int status code, 2nd is output
            if not isinstance(status, int):
                mavis_error(f"Step '{node_id}' ({command_name}) returned invalid status type: {type(status).__name__}. Aborting pipeline.")
                latency_ms = round((time.perf_counter() - _pipeline_start) * 1000, 2)
                _dag_emitter.log({
                    "turn_id": turn_id,
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "latency_ms": latency_ms,
                    "status": "node_failed",
                    "dag_size": dag_size,
                    "dag_depth": dag_depth,
                    "tool_nodes_count": tool_nodes_count,
                    "subagent_nodes_count": subagent_nodes_count,
                    "failed_node_id": node_id,
                })
                return

            if status == 0:
                # 2nd element is the actual output received
                node_results[node_id] = result
            else:
                # Proper error handling: non-zero status aborts pipeline with error payload
                mavis_error(f"Step '{node_id}' ({command_name}) failed with status {status}: {result}. Aborting pipeline.")
                latency_ms = round((time.perf_counter() - _pipeline_start) * 1000, 2)
                _dag_emitter.log({
                    "turn_id": turn_id,
                    "start_time": datetime.now().isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "latency_ms": latency_ms,
                    "status": "node_failed",
                    "dag_size": dag_size,
                    "dag_depth": dag_depth,
                    "tool_nodes_count": tool_nodes_count,
                    "subagent_nodes_count": subagent_nodes_count,
                    "failed_node_id": node_id,
                })
                return

        except Exception as e:
            mavis_error(f"Critical error at step '{node_id}': {e}")
            latency_ms = round((time.perf_counter() - _pipeline_start) * 1000, 2)
            _dag_emitter.log({
                "turn_id": turn_id,
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "latency_ms": latency_ms,
                "status": "node_failed",
                "dag_size": dag_size,
                "dag_depth": dag_depth,
                "tool_nodes_count": tool_nodes_count,
                "subagent_nodes_count": subagent_nodes_count,
                "failed_node_id": node_id,
            })
            return

    elapsed_s = time.perf_counter() - _pipeline_start
    latency_ms = round(elapsed_s * 1000, 2)
    mavis_status("Pipeline finished.")

    _dag_emitter.log({
        "turn_id": turn_id,
        "start_time": datetime.now().isoformat(),
        "end_time": datetime.now().isoformat(),
        "latency_ms": latency_ms,
        "status": "success",
        "dag_size": dag_size,
        "dag_depth": dag_depth,
        "tool_nodes_count": tool_nodes_count,
        "subagent_nodes_count": subagent_nodes_count,
        "failed_node_id": "",
    })

    if node_results and pipeline:
        final_answer = answerer.synthesize(
            query=query,
            pipeline_results=node_results,
            memory_context=context,
            turn_id=turn_id,
        )
        mavis_answer(final_answer)
        _session_chat.append({
            "role": "assistant",
            "content": final_answer,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    # Fire notification if the pipeline took longer than the configured threshold
    threshold = cfg.output.get("notify_pipeline_threshold_s", 5)
    if elapsed_s >= threshold:
        _notify("MAVIS: Pipeline complete", f"Finished in {elapsed_s:.1f}s")

    return node_results


# ── Interpreter ───────────────────────────────────────────────────────────────

def interpret_command(command: str) -> bool:
    """Interpret user input, build missing tools, and execute the pipeline.
    Returns False if user requested exit, True otherwise.
    """
    if command.lower() in ["exit", "quit"]:
        return False

    # ── Slash command check ────────────────────────────────────────────────────
    if handle_slash_command(command):
        return True

    turn_id = uuid.uuid4().hex[:8]
    _session_chat.append({
        "role": "user",
        "content": command,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    # 0. Retrieve memory context (bounded and truncated)
    try:
        context = memory_store.retrieve_context(command)
    except Exception as exc:
        mavis_debug(f"Memory retrieval failed: {exc}", entity="interpreter")
        context = ""

    # 0.5 Check Pipeline Cache
    cache_hit = cache_manager.check_cache(command, turn_id=turn_id)
    if cache_hit:
        if cache_hit["ttl_valid"] and cache_hit["result"] is not None:
            mavis_status("Cache hit (Result valid). Serving from cache.")
            final_answer = answerer.synthesize(
                query=command,
                pipeline_results=cache_hit["result"],
                memory_context=context,
                turn_id=turn_id,
            )
            mavis_answer(final_answer)
            _session_chat.append({
                "role": "assistant",
                "content": final_answer,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            return True
        else:
            mavis_status("Cache hit (Pipeline valid, result expired). Re-running pipeline.")
            res = execute_pipeline(cache_hit["pipeline"], query=command, context=context, turn_id=turn_id)
            if res:
                cache_manager.save_cache(
                    command, 
                    cache_hit["pipeline"], 
                    res, 
                    cache_hit.get("ttl", 300), 
                    cache_hit.get("generalizability", "specialized")
                )
            return True

    # 1. Filter tools dynamically (Strategy 3)
    try:
        active_tools = tool_retriever.get_relevant_tools(command, commands_list)
    except Exception as exc:
        mavis_debug(f"Tool retrieval failed: {exc}", entity="interpreter")
        active_tools = commands_list
    commands_str = json.dumps(active_tools, indent=2)
    agents_str = json.dumps(agents_list, indent=2)

    # 2. Build user turn adhering to stable prefix ordering (Tools -> Agents -> Context -> Input)
    user_prompt = format_interpreter_user_prompt(
        commands_list_str=commands_str,
        agents_list_str=agents_str,
        memory_context=context,
        user_input=command,
    )

    t0_plan = time.perf_counter()
    try:
        with spinner("Interpreting..."):
            response = llm.generate(
                user_prompt,
                json_mode=True,
                system_instruction=interpreter_system_prompt,
            )
        mavis_debug(response, entity="interpreter")
        response_dict = json.loads(response)
    except json.JSONDecodeError:
        latency_ms = round((time.perf_counter() - t0_plan) * 1000, 2)
        _interpreter_emitter.log({
            "turn_id": turn_id,
            "latency_ms": latency_ms,
            "status": "error",
            "input_tokens": len(user_prompt) // 4 + len(interpreter_system_prompt) // 4,
            "output_tokens": 0,
            "tools_retrieved_count": len(active_tools),
        })
        mavis_error("I had trouble understanding that — try rephrasing.")
        mavis_debug(f"Raw LLM response: {response}", entity="interpreter")
        return True
    except Exception as e:
        latency_ms = round((time.perf_counter() - t0_plan) * 1000, 2)
        _interpreter_emitter.log({
            "turn_id": turn_id,
            "latency_ms": latency_ms,
            "status": "error",
            "input_tokens": len(user_prompt) // 4 + len(interpreter_system_prompt) // 4,
            "output_tokens": 0,
            "tools_retrieved_count": len(active_tools),
        })
        mavis_error(f"LLM call failed: {e}")
        return True

    plan_latency_ms = round((time.perf_counter() - t0_plan) * 1000, 2)
    input_tokens = len(user_prompt) // 4 + len(interpreter_system_prompt) // 4
    output_tokens = len(response) // 4

    emotion, emotion_strength, directive = parse_classifier_fields(response_dict)

    memory_store.add_turn(
        role="user",
        content=command,
        emotion=emotion,
        emotion_strength=emotion_strength,
        directive=directive,
    )

    # 0. Check for direct context response (memory recall, stored facts, preferences)
    direct = response_dict.get("direct_response")
    if direct and direct.strip():
        _interpreter_emitter.log({
            "turn_id": turn_id,
            "latency_ms": plan_latency_ms,
            "status": "direct_response",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tools_retrieved_count": len(active_tools),
        })
        mavis_answer(direct)
        _session_chat.append({
            "role": "assistant",
            "content": direct,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })
        memory_store.add_turn(
            role="assistant",
            content=direct,
            emotion=emotion,
            emotion_strength=emotion_strength,
            directive=directive,
        )
        return True

    _interpreter_emitter.log({
        "turn_id": turn_id,
        "latency_ms": plan_latency_ms,
        "status": "pipeline",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tools_retrieved_count": len(active_tools),
    })

    # 2a. Build missing commands (tools)
    missing = response_dict.get("missing_commands", [])
    tool_failure = False
    if missing:
        mavis_status(f"Building {len(missing)} missing tool(s)...")
        all_tools_built = True
        for new_tool in missing:
            signature = new_tool['signature']
            description = new_tool['description']
            func_name = signature.split('(')[0].strip()

            mavis_status(f"Building '{func_name}': {signature}")
            try:
                gen_score = tool_builder.build_tool(signature, description) or "repurposable"
                commands_list[func_name] = {
                    "description": description,
                    "generalizability": gen_score,
                }
                tool_retriever.index_tool(signature, description, generalizability=gen_score)
                mavis_ok(f"Built '{func_name}'.")
            except Exception as e:
                mavis_error(f"Couldn't build '{func_name}': {e}. See logs/tool_builder.log.")
                all_tools_built = False

        if not all_tools_built:
            tool_failure = True
            memory_store.add_turn(
                role="user",
                content=command,
                emotion=emotion,
                emotion_strength=emotion_strength,
                directive=directive,
                tool_failure=True,
            )
            _notify("MAVIS: Tool Build Failed", "One or more tools could not be built. See logs/tool_builder.log.")
            mavis_error("Some tools failed to build. Pipeline execution skipped.")
            return True
        else:
            mavis_ok("All missing tools built successfully.")
            mavis_status("Proceeding with pipeline execution.")

    # 2b. Build missing agents
    missing_agents = response_dict.get("missing_agents", [])
    if missing_agents:
        mavis_status(f"Building {len(missing_agents)} missing cognitive agent(s)...")
        all_agents_built = True
        for new_agent in missing_agents:
            if isinstance(new_agent, str):
                agent_name = new_agent.split("(")[0].strip()
                description = f"Cognitive sub-agent {agent_name}"
                input_schema = {"content": "Any"}
                output_schema = None
            elif isinstance(new_agent, dict):
                agent_name = (
                    new_agent.get("name")
                    or new_agent.get("agent_name")
                    or new_agent.get("function_name")
                    or (new_agent.get("signature", "").split("(")[0].strip() if new_agent.get("signature") else None)
                )
                description = new_agent.get("description", "")
                input_schema = new_agent.get("input_schema", {"content": "Any"})
                output_schema = new_agent.get("output_schema")
            else:
                agent_name = None

            if not agent_name:
                mavis_error(f"Could not determine agent name from specification: {new_agent}")
                all_agents_built = False
                continue

            mavis_status(f"Building agent '{agent_name}'...")
            try:
                gen_score = agent_builder.build_agent(
                    agent_name=agent_name,
                    agent_description=description,
                    input_schema=input_schema,
                    output_schema=output_schema,
                )
                sig = f"{agent_name}(...) -> tuple[int, Any]"
                agents_list[sig] = {
                    "description": description,
                    "generalizability": gen_score,
                }
                mavis_ok(f"Built agent '{agent_name}'.")
            except Exception as e:
                mavis_error(f"Couldn't build agent '{agent_name}': {e}. See logs/oni_audit.jsonl.")
                all_agents_built = False

        if not all_agents_built:
            memory_store.add_turn(
                role="user",
                content=command,
                emotion=emotion,
                emotion_strength=emotion_strength,
                directive=directive,
                tool_failure=True,
            )
            _notify("MAVIS: Agent Build Failed", "One or more sub-agents could not be built.")
            mavis_error("Some agents failed to build. Pipeline execution skipped.")
            return True
        else:
            mavis_ok("All missing agents built successfully.")

    # 3. Execute the pipeline (ONI pre-flight inside execute_pipeline)
    pipeline = response_dict.get("pipeline", [])
    if not pipeline:
        mavis_status("No executable pipeline in the response.")
        return True

    res = execute_pipeline(pipeline, query=command, context=context, turn_id=turn_id)
    if res:
        ttl = response_dict.get("ttl", 300)
        gen = response_dict.get("generalizability", "specialized")
        cache_manager.save_cache(command, pipeline, res, ttl, gen)

    # 4. Store assistant response in working memory
    last_assistant_text = _session_chat[-1]["content"] if (_session_chat and _session_chat[-1]["role"] == "assistant") else ""
    pipeline_summary = ", ".join(
        f"{n.get('function_name')}({n.get('params', {})})"
        for n in pipeline
    )
    memory_store.add_turn(
        role="assistant",
        content=last_assistant_text or f"Executed pipeline: {pipeline_summary}",
    )
    return True


# ── Worker subprocess management ──────────────────────────────────────────────────
_worker_procs: list[subprocess.Popen] = []
_HEARTBEAT_PATH = os.path.join(_MAV_ROOT, "data", ".mavis_heartbeat")
_HEARTBEAT_INTERVAL = 30  # seconds


def _start_workers() -> None:
    """Spawn each memory worker as an independent subprocess."""
    for name, interval_key in [
        ("short_term", "short_term_worker_interval_minutes"),
        ("long_term", "long_term_worker_interval_minutes"),
    ]:
        interval = cfg.scheduler.get(interval_key, 15 if name == "short_term" else 480)
        proc = subprocess.Popen(
            [
                sys.executable,
                os.path.join(_MAV_ROOT, "tasks", "worker_process.py"),
                name,
                str(interval),
            ],
            cwd=_MAV_ROOT,
        )
        _worker_procs.append(proc)
        mavis_status(f"{name}_worker started (PID {proc.pid}, every {interval} min).")


def _stop_workers() -> None:
    """Terminate all worker subprocesses gracefully."""
    for proc in _worker_procs:
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except BaseException:
            try:
                proc.kill()
            except Exception:
                pass
    if os.path.exists(_HEARTBEAT_PATH):
        try:
            os.remove(_HEARTBEAT_PATH)
        except Exception:
            pass


def _write_heartbeat() -> None:
    """Write current timestamp to heartbeat file (workers monitor this)."""
    try:
        with open(_HEARTBEAT_PATH, "w") as f:
            f.write(str(time.time()))
    except Exception:
        pass


def _print_exit_summary() -> None:
    now = time.time()
    elapsed_m = round((now - _session_start_ts) / 60, 1)
    summary = get_metrics_summary(_session_start_ts)
    t_in = summary["tokens_total"]["input"]
    t_out = summary["tokens_total"]["output"]
    queries = summary["interpreter"]["total_queries"]
    cache_hits = summary["caching"]["hits"]
    tokens_saved = summary["caching"]["tokens_saved"]

    if queries > 0 or t_in > 0:
        rule("Session Performance Summary")
        print_table([
            ("Session Duration", f"{elapsed_m} min"),
            ("Queries Processed", str(queries)),
            ("Tokens Consumed", f"{t_in + t_out:,} ({t_in:,} in / {t_out:,} out)"),
            ("Cache Hits", f"{cache_hits} hits ({tokens_saved:,} tokens saved)"),
        ])
        rule()


# ── Entry point ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rule()
    mavis_print("  [bold cyan]MAVIS[/bold cyan]  [dim]v1.0 [/dim]")
    mavis_print("  Type [bold]/help[/bold] for commands, [bold]/[/bold] to browse suggestions. Say [dim]exit[/dim] to quit.")
    rule()
    mavis_status(f"ONI trust level: {_oni.config.trust_level}")

    runner = TaskRunner(tick_seconds=cfg.scheduler.get("tick_seconds", 30))
    runner.register(_write_heartbeat, interval_minutes=1, task_name="heartbeat")
    runner.register(cache_manager.evict_expired, interval_minutes=5, task_name="cache_ttl_eviction")
    runner.register(cache_manager.lru_evict, interval_minutes=60, task_name="cache_lru_eviction")
    runner.start()

    _write_heartbeat()
    _start_workers()
    _last_heartbeat = time.time()

    # ── Interactive prompt session with dynamic pop-down & history ──────────────
    _HISTORY_FILE = cfg.output.get("history_file", "data/.mavis_history")
    os.makedirs(os.path.dirname(_HISTORY_FILE), exist_ok=True)

    session = PromptSession(
        history=FileHistory(_HISTORY_FILE),
        completer=MavisSlashCompleter(),
        complete_while_typing=True,
        complete_style=CompleteStyle.COLUMN,
        style=_PROMPT_STYLE,
        bottom_toolbar=_get_bottom_toolbar,
    )

    try:
        while True:
            try:
                command = session.prompt(_get_prompt_message()).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not command:
                continue
            if command.lower() in ["exit", "quit"]:
                break
            try:
                if not interpret_command(command):
                    break
            except Exception as e:
                import traceback
                traceback.print_exc()
                mavis_error(f"Error processing command: {e}")

            # Refresh heartbeat after each user interaction if interval reached
            now = time.time()
            if now - _last_heartbeat >= _HEARTBEAT_INTERVAL:
                _write_heartbeat()
                _last_heartbeat = now
    except BaseException as e:
        if not isinstance(e, (KeyboardInterrupt, SystemExit)):
            import traceback
            traceback.print_exc()
    finally:
        try:
            _stop_workers()
        except BaseException:
            pass
        try:
            runner.stop()
        except BaseException:
            pass
        try:
            _print_exit_summary()
        except BaseException:
            pass
        mavis_print("[dim]Goodbye.[/dim]", level="quiet")