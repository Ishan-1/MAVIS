# MAVIS - Antigravity Agent Memory & Guidelines

This document persists the architectural decisions, operational conventions, and core invariants for pair programming on the MAVIS repository.

---

## 1. Golden Invariants & Constraints

- **Never modify `tools/`**: All tool files inside `tools/` are user-owned baseline primitives. Do not edit them. Any adjustments, debugging repairs, or wrapping must be performed in `core/`, `main.py`, `oni/`, or specialized agents.
- **Python Environment & Tests**:
  - Virtualenv binary: `./bin/python`
  - Pytest runner: `./bin/pytest tests/<test_file>.py`
  - Note: `tests/` is gitignored; do not stage or commit test files unless explicitly requested.
- **Project Direction**: Feature scope is complete for personal use. Focus strictly on **hardening, robustness, edge-case recovery, token/latency optimization, and UX polish** on existing systems.

---

## 2. Core Architecture & Subsystems

### A. DAG Parameter Resolution & Branching (`core/dag.py`)
- **Parameter Interpolation**:
  - Standalone references (e.g. `"$node_id.output"` or `"$node_id"`) return native Python types (`dict`, `list`, `int`) to preserve type safety for subagents.
  - Embedded references inside composite strings (e.g. `'git commit -m "$n2.output"'`) are interpolated via `DEP_SEARCH_RE.sub()`, converting values to strings (or JSON serialization for dicts/lists).
- **Gate Nodes (`type: "gate"`) & Pruning**:
  - Evaluates deterministic expressions (e.g. `$node.status == 0`, `len($node.output) > 0`) via restricted AST in `<1ms` at 0 token cost, or fuzzy conditions via NLP LLM Judge (`mode: "nlp"`).
  - Automatically topologically sorted before branch targets. Untaken subtrees are dynamically pruned via `compute_pruned_nodes()`.

### B. Autonomous Goal Runner (`core/goal_runner.py`, `/goal`)
- **Wave DAG Unrolling**: High-level autonomous tasks loop without breaking acyclicity by generating iterative successive DAG waves up to `max_iterations = 8`.
- **ONI Permission Safeguards**:
  - Default `/goal <desc>` runs at session trust (`ask`), prompting per wave for greylisted commands.
  - `/goal --yolo <desc>` requests unattended mode with an upfront confirmation gate (`_oni.gate.request_approval`) before Wave 1 begins.

### C. Live Output Streaming & Shell Execution (`main.py`, `oni/oni.py`)
- `call_command()` streams subprocess output line-by-line using non-blocking `selectors` so users see real-time progress during long tasks.
- Shell commands run via ONI's `call_shell()`.

### D. Observability Dashboard (`scripts/dashboard.py`, `/dashboard`)
- **Web Dashboard**: Local Streamlit app served at `http://localhost:8501`.
- **Detached Execution Invariant**: Must always run completely detached:
  ```python
  subprocess.Popen(
      cmd,
      cwd=_MAV_ROOT,
      stdin=subprocess.DEVNULL,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
      start_new_session=True,  # Mandatory to avoid TTY conflict with prompt_toolkit
  )
  ```
- **Interception**: Natural language queries (`run dashboard`, `open dashboard`) and pipeline steps containing `streamlit run` are intercepted and routed to the background daemon rather than running in the foreground.
- **Active Instance Detection**: Before spawning, check `_is_port_open(8501)` to prevent duplicate exit code 1 failures.
- **Quiet Browser Opener**: Suppress `xdg-open` lookup noise when attempting to open browsers in headless environments.

### E. Large Payload Offloading (`core/scratchpad.py`)
- Tool outputs $> 4,000$ bytes are offloaded to disk at `data/scratch/<turn_id>_<node_id>.<ext>`.
- Downstream steps receive compact head/tail digests with absolute file pointers to preserve context window space.

### F. Pipeline Self-Debugging (`core/pipeline_debugger.py`)
- Catches non-zero step failures and diagnoses errors using LLM reflection.
- Emits repair actions (`patch_params`, `replace_node`, or `unrecoverable`).
- Automatically learns and records fixes into namespaced memory (`memories/pipeline_debugger/`).

---

## 3. Telemetry & Metrics Files
- All telemetry writes to isolated single-CSV tables in `data/metrics/` (`caching.csv`, `dag_execution.csv`, `interpreter.csv`, `memory.csv`, `tool_usage.csv`, `gate_evaluator.csv`, `goal_runner.csv`).
- Independent CSV reads prevent CPU/IO bottlenecks in the dashboard.
