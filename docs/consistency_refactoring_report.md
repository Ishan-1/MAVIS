# Code Consistency & Architecture Report

This document outlines findings related to code consistency, style, naming conventions, error handling, and structural intuitiveness within MAVIS, along with their resolution status.

---

## Resolved Findings

### 1. Inconsistent `from __future__ import annotations` Usage
* **Type:** `code_style`
* **Status:** **RESOLVED**
* **Description:** Inconsistent use of `from __future__ import annotations`. Found missing in `memories/embedding.py`, `core/helpers.py`, `tasks/__init__.py`, and `main.py`.
* **Resolution:** Added `from __future__ import annotations` across `memories/embedding.py`, `core/helpers.py`, `tasks/__init__.py`, and `main.py` to ensure unified forward reference type hinting across Python 3.10+.

### 2. Ambiguous Naming for Base Paths
* **Type:** `naming_convention`
* **Status:** **RESOLVED**
* **Description:** The `_BASE` constant in `memories/memory_store.py` referred to the `memories/` directory, while `_MAV_ROOT` in `main.py` and `core/` referred to the absolute project root.
* **Resolution:** Renamed `_BASE` to `_MEMORIES_DIR` in `memories/memory_store.py` (with a backward-compatibility alias). Exported canonical `MAV_ROOT` from `core/helpers.py` to serve as the unified root path anchor.

### 3. Inconsistent Error Handling for Process Termination
* **Type:** `error_handling`
* **Status:** **RESOLVED**
* **Description:** Inconsistent error handling for `subprocess.Popen` termination during shutdown in `main.py`. Bare `try...except BaseException: pass` blocks suppressed signals like `KeyboardInterrupt` and `SystemExit`.
* **Resolution:** Created `_safe_terminate(proc, name, timeout=2.0)` with a clean `terminate() -> wait(timeout) -> kill()` fallback ladder catching `Exception` and logging failures via `log_it` rather than masking `BaseException`.

### 4. Duplicated Shutdown Logic
* **Type:** `duplication`
* **Status:** **RESOLVED**
* **Description:** Six consecutive `try: ... except BaseException: pass` blocks duplicated suppression logic for terminating dashboards, workers, scheduler, and printing summaries.
* **Resolution:** Replaced with `_safe_shutdown_call(name, fn)` which catches standard `Exception`, logs failures with `[WARN]` level, and guarantees all cleanup steps execute without swallowing termination signals.

### 5. Misplaced & Scattered Token Count Utility
* **Type:** `code_style` / `duplication`
* **Status:** **RESOLVED**
* **Description:** The heuristic `len(text) // 4` was duplicated across 15+ files and 20+ locations.
* **Resolution:** Implemented a centralized, null-safe `estimate_tokens(text: str | None) -> int` in `core/helpers.py`. Integrated across `memories/memory_store.py`, `core/answerer.py`, `core/agents/base.py`, `core/agents/subagent.py`, and `main.py`.

### 6. Inconsistent Logging Entity Definition & Latent Signature Bug
* **Type:** `naming_convention` / `defect`
* **Status:** **RESOLVED**
* **Description:** `main.py` lacked an `_ENTITY` constant. Furthermore, investigation revealed that `core/scratchpad.py` and `core/mcp_client.py` invoked `log_it(_ENTITY, message, level="...")` with inverted arguments and an unhandled `level=` keyword, causing file path mangling and runtime `TypeError` crashes.
* **Resolution:**
  1. Defined `_ENTITY = "main"` in `main.py`.
  2. Upgraded `log_it` in `core/helpers.py` to support `level: str | None = None` and default `entity_name="main"`, with heuristic detection to safely handle inverted argument signatures.
  3. Fixed all `log_it` call sites in `core/scratchpad.py` and `core/mcp_client.py`.

### 7. Duplicated Streamlit App Launchers in `main.py`
* **Type:** `duplication` / `spaghetti_reduction`
* **Status:** **RESOLVED**
* **Description:** `/dashboard` (port 8501) and `/tooldash` (port 8502) in `main.py` shared ~140 lines of duplicate socket connection checks, file-descriptor browser silencing, 4-tier Streamlit binary fallback, and detached subprocess spawning.
* **Resolution:** Consolidated into a unified `_launch_streamlit_app(script_path, port, app_name)` helper, cutting ~100 lines of duplicated spaghetti while retaining complete headless/detached session guarantees.

### 8. Misplaced DAG Depth Calculation
* **Type:** `cohesion` / `architecture`
* **Status:** **RESOLVED**
* **Description:** `compute_dag_depth` was implemented inside `main.py` rather than within `core/dag.py`, splitting the DAG engine domain.
* **Resolution:** Relocated `compute_dag_depth` into `core/dag.py` and imported it into `main.py`.

### 9. Structural Symmetry: ToolBuilder & AgentBuilder
* **Type:** `file_structure` / `symmetry`
* **Status:** **RESOLVED**
* **Description:** `agent_builder/` followed PEP-8 with `agent_builder.py`, while `tool_builder/` used `toolbuilder.py`.
* **Resolution:** Created `tool_builder/tool_builder.py` alias and updated exports so both builders have symmetrical module structures and import semantics (`ToolBuilder`, `ToolTester`, `AgentBuilder`, `AgentTester`, `AgentDebugger`).

### 10. Background Subsystem Clarification: `tasks/`
* **Type:** `file_structure` / `documentation`
* **Status:** **RESOLVED**
* **Description:** The root directory `tasks/` contained background memory consolidation workers (`short_term_worker.py`, `long_term_worker.py`, `worker_process.py`), which caused confusion with general user tasks.
* **Resolution:** Documented the background daemon architecture in `tasks/__init__.py` and the main `README.md` directory layout.