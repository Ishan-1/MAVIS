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
import subprocess
import sys
import time

load_dotenv()
llm = get_llm_client()
tool_builder = ToolBuilder(llm)
memory_store = MemoryStore(llm, namespace="interpreter")
tool_retriever = ToolRetriever(llm)

# ── Central config + ONI ────────────────────────────────────────────────────────────
from core.config import cfg
from oni import oni as _oni

commands_list = {}
with open("data/commands_list.json", "r") as file:
    commands_list = json.load(file)

_MAV_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Readline tab-completion for slash commands ──────────────────────────────────
_SLASH_SUBCOMMANDS = {
    "/help": [],
    "/config": ["set", "save", "reload", "audit"],
    "/trust": ["ask", "yolo", "whitelist"],
    "/allow": [],
    "/block": [],
    "/greylist": [],
    "/unlist": [],
    "/status": [],
}

_COMMAND_METAS = {
    "/help": "Show available slash commands",
    "/status": "Heartbeat, workers, scheduler, memory tokens",
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
    return [
        ("", " [MAVIS v1.0]  Type / for commands • ↑/↓ for history • 'exit' to quit "),
    ]


# ── Slash command handler ─────────────────────────────────────────────────────

_HELP_ROWS = [
    ("/help",                    "This help message."),
    ("/status",                  "Heartbeat, worker PIDs, scheduler tasks, memory usage."),
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

    # ── /status ───────────────────────────────────────────────────────────────
    if verb == "/status":
        _print_status()
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
    Execute a tool in an isolated subprocess via run_tool.py.

    Sandboxing: a crashed or malicious tool cannot take down the MAVIS process.
    Tool prints go to stderr and are not mixed with the JSON result.
    A configurable timeout kills runaway tools.
    """
    try:
        result = subprocess.run(
            [
                sys.executable,
                os.path.join(_MAV_ROOT, "core", "run_tool.py"),
                command_name,
                json.dumps(params_dict),
            ],
            capture_output=True,
            text=True,
            timeout=_oni.config.tool_execution_timeout,
            cwd=_MAV_ROOT,
        )

        if result.stderr.strip():
            mavis_debug(result.stderr.strip(), entity=command_name)

        if result.returncode != 0 and not result.stdout.strip():
            mavis_error(f"Tool '{command_name}' exited with code {result.returncode}.")
            return -1, result.stderr.strip() or f"Tool '{command_name}' exited with error."

        status, output = json.loads(result.stdout.strip())

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
        raw = result.stdout.strip() if 'result' in dir() else ''
        mavis_error(f"'{command_name}' returned invalid output.")
        mavis_debug(f"Raw stdout: {raw}", entity=command_name)
        return -1, f"Invalid output from tool '{command_name}': {e}"
    except Exception as e:
        mavis_error(f"Unexpected error with '{command_name}': {e}")
        return -1, str(e)


# ── Pipeline execution ────────────────────────────────────────────────────────

def execute_pipeline(pipeline):
    """
    Execute a list of pipeline nodes in order, resolving dependencies.

    ONI pre-flight scans the full pipeline before any node runs.
    Blacklisted commands abort immediately. Greylisted commands are
    presented to the user as a single batch prompt upfront.
    """
    ok, issues = _oni.preflight_scan(pipeline)
    if not ok:
        oni_print(f"Pipeline aborted: {'; '.join(issues)}")
        return

    node_results = {}
    mavis_status("Starting pipeline execution...")
    _pipeline_start = time.time()

    # Topological sort for dependencies
    try:
        from graphlib import TopologicalSorter
        ts = TopologicalSorter()
        for node in pipeline:
            deps = [v for v in node["params"].values() if isinstance(v, str) and v.startswith("$")]
            ts.add(node["id"], *[d.split(".")[0][1:] for d in deps])
        pipeline_order = list(ts.static_order())
        sorted_pipeline = [n for node_id in pipeline_order for n in pipeline if n["id"] == node_id]
    except ImportError:
        sorted_pipeline = pipeline

    for node in sorted_pipeline:
        node_id = node["id"]
        command_name = node["function_name"]
        params = node["params"]
        resolved_params = {}

        mavis_status(f"Running step '{node_id}': {command_name}")

        try:
            for param_name, param_value in params.items():
                if isinstance(param_value, str) and param_value.startswith("$"):
                    parts = param_value[1:].split(".", 1)
                    dep_node_id = parts[0]
                    if dep_node_id not in node_results:
                        mavis_error(
                            f"Pipeline aborted: step '{node_id}' depends on "
                            f"'{dep_node_id}' which didn't complete."
                        )
                        return
                    dep_result = node_results[dep_node_id]
                    if len(parts) == 2 and isinstance(dep_result, dict):
                        resolved_params[param_name] = dep_result.get(parts[1], dep_result)
                    else:
                        resolved_params[param_name] = dep_result
                else:
                    resolved_params[param_name] = param_value

            status, result = call_command(command_name, resolved_params)

            if status == 0:
                node_results[node_id] = result
            else:
                mavis_error(f"Step '{node_id}' ({command_name}) failed. Aborting pipeline.")
                return

        except Exception as e:
            mavis_error(f"Critical error at step '{node_id}': {e}")
            return

    elapsed = time.time() - _pipeline_start
    mavis_status("Pipeline finished.")
    if node_results and pipeline:
        final_node_id = pipeline[-1]["id"]
        final_result = node_results.get(final_node_id)
        if final_result is not None:
            mavis_answer(str(final_result))

    # Fire notification if the pipeline took longer than the configured threshold
    threshold = cfg.output.get("notify_pipeline_threshold_s", 5)
    if elapsed >= threshold:
        _notify("MAVIS: Pipeline complete", f"Finished in {elapsed:.1f}s")


# ── Interpreter ───────────────────────────────────────────────────────────────

def interpret_command(command: str) -> bool:
    """Interpret user input, build missing tools, and execute the pipeline.
    Returns False if user requested exit, True otherwise.
    """
    if command.lower() in ["exit", "quit"]:
        return False

    # ── Slash command check ────────────────────────────────────────────────────
    if handle_slash_command(command):
        return

    # 0. Retrieve memory context (bounded and truncated)
    context = memory_store.retrieve_context(command)

    # 1. Filter tools dynamically (Strategy 3)
    active_tools = tool_retriever.get_relevant_tools(command, commands_list)
    commands_str = json.dumps(active_tools, indent=2)

    # 2. Build user turn adhering to stable prefix ordering (Tools -> Context -> Input)
    user_prompt = format_interpreter_user_prompt(
        commands_list_str=commands_str,
        memory_context=context,
        user_input=command,
    )

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
        mavis_error("I had trouble understanding that — try rephrasing.")
        mavis_debug(f"Raw LLM response: {response}", entity="interpreter")
        return
    except Exception as e:
        mavis_error(f"LLM call failed: {e}")
        return

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
        mavis_answer(direct)
        memory_store.add_turn(
            role="assistant",
            content=direct,
            emotion=emotion,
            emotion_strength=emotion_strength,
            directive=directive,
        )
        return

    # 2. Build missing commands
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
            return
        else:
            mavis_ok("All missing tools built successfully.")
            mavis_status("Proceeding with pipeline execution.")

    # 3. Execute the pipeline (ONI pre-flight inside execute_pipeline)
    pipeline = response_dict.get("pipeline", [])
    if not pipeline:
        mavis_status("No executable pipeline in the response.")
        return

    execute_pipeline(pipeline)

    # 4. Store assistant response in working memory
    pipeline_summary = ", ".join(
        f"{n.get('function_name')}({n.get('params', {})})"
        for n in pipeline
    )
    memory_store.add_turn(
        role="assistant",
        content=f"Executed pipeline: {pipeline_summary}",
    )


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


# ── Entry point ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rule()
    mavis_print("  [bold cyan]MAVIS[/bold cyan]  [dim]v1.0 [/dim]")
    mavis_print("  Type [bold]/help[/bold] for commands, [bold]/[/bold] to browse suggestions. Say [dim]exit[/dim] to quit.")
    rule()
    mavis_status(f"ONI trust level: {_oni.config.trust_level}")

    runner = TaskRunner(tick_seconds=cfg.scheduler.get("tick_seconds", 30))
    runner.register(_write_heartbeat, interval_minutes=1, task_name="heartbeat")
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
            if not interpret_command(command):
                break

            # Refresh heartbeat after each user interaction if interval reached
            now = time.time()
            if now - _last_heartbeat >= _HEARTBEAT_INTERVAL:
                _write_heartbeat()
                _last_heartbeat = now
    except BaseException:
        pass
    finally:
        try:
            _stop_workers()
        except BaseException:
            pass
        try:
            runner.stop()
        except BaseException:
            pass
        mavis_print("[dim]Goodbye.[/dim]", level="quiet")