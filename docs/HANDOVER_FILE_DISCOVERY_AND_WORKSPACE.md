# Handover: File Discovery Architecture & Workspace Scoping

**Date**: 2026-09-20  
**Status**: Pinned for pickup in next session  
**Reference Areas**: `oni/oni.py`, `benchmarks/harness/runner.py`, `core/dag.py`, `prompts/prompt_templates.py`

---

## 1. Context & Motivation

During benchmark runs with the `oss_maintainer` persona (`oss_task_001`), two key behaviors were analyzed:
1. **Workspace Boundary Leaks**:
   MAVIS previously wrote files (e.g., `CHANGELOG.md`) directly into the repository root instead of the isolated benchmark workspace (`benchmarks/workspaces/oss_maintainer/oss_task_001/`).
2. **File Path Locating (LLM Backbone vs. Architecture)**:
   Other coding agents (Cursor, Claude Code, Antigravity) first locate a file path cleanly before editing. In MAVIS, this appeared to struggle when given a bare filename like `CHANGELOG.md` vs `./CHANGELOG.md`.

### Key Takeaway from Architectural Discussion
This is **not an LLM backbone issue**. The difference stems from:
- **Execution Loop**: Standard agents use a step-by-step ReAct loop (`Thought -> Tool Call -> Observation`), naturally selecting discovery tools (`find_files`) as low-entropy first steps. MAVIS generates ahead-of-time DAGs (`core/dag.py`), forcing the planner to guess parameters for Node 1 before any step executes.
- **Context Injection vs. Tools**: Standard coding agents often inject a shallow repository tree into every prompt. However, because MAVIS is a multi-domain agent (emails, academic papers, scheduling, diagnostics), **blanket context injection is wasteful and pollutes attention** on non-coding turns.

---

## 2. Completed Work & Immediate Fixes

### A. Workspace-Aware Path Resolution (`oni/oni.py`)
- Added `ONI.get_active_workspace()` and `ONI.resolve_path(raw_path)`.
- Normalizes all filesystem operations (`call_fs`, `call_shell` cwd) to `MAVIS_ACTIVE_WORKSPACE`.
- Strips redundant `"workspace/"` and `"./workspace/"` path prefixes without filesystem mutations.
- Leaked root files (`CHANGELOG.md`, `pyproject.toml`) were removed from the repo root.

### B. Removal of Recursive Workspace Symlink (`benchmarks/harness/runner.py`)
- Removed `internal_ws_symlink = ws / "workspace" -> ws.resolve()`.
- **Reason**: Pytest traverses directory symlinks during test collection, causing infinite recursion (`ws/workspace/workspace/...`) and failing with exit code 2. Path normalization in ONI eliminates the need for any internal symlinks.

### C. Workspace Prompt Rules (`prompts/prompt_templates.py`)
- Updated `interpreter_system_prompt` with explicit guidance on workspace isolation and relative path clean resolution.

---

## 3. Pinned Topic: On-Demand File Discovery Tool

Instead of shallow context injection, the proposed design is an **on-demand file discovery capability**:

### A. Proposed Primitives
1. **`locate_file(name_or_pattern: str, search_root: Optional[str] = None) -> str`**:
   - Locates a file within the active workspace and returns its clean relative path.
   - **DAG Compatibility**: Can be piped into subsequent steps via MAVIS's native parameter interpolation:
     ```json
     { "node_id": "find", "tool": "locate_file", "params": { "name_or_pattern": "CHANGELOG.md" } },
     { "node_id": "edit", "tool": "write_file_contents", "params": { "file_path": "$find.output", "contents": "..." } }
     ```
2. **`list_directory_tree(max_depth: int = 2, include_hidden: bool = False) -> str`**:
   - Returns a compact tree structure of the active workspace.
   - Ideal for subagent cognitive nodes (`core/symmetrical_agent_builder.py`) and `/goal` Wave 1 reconnaissance.

### B. Golden Invariant Constraint
- **Rule**: Never modify `tools/` (user-owned primitives).
- **Implementation Strategy**: Provide these discovery capabilities via ONI built-in actions (`call_fs`), or register them as subagent tools within `core/` / `oni/`.

---

## 4. Verification & Resumption Guide

### Current Test Health
```bash
./bin/pytest -v tests/test_control_node.py tests/test_dag.py tests/test_symmetrical_agent_builder.py
# Expected: 27 passed
```

### Running the Benchmark
```bash
./bin/python -m benchmarks --persona oss_maintainer --task oss_task_001
```
Verify that:
1. `CHANGELOG.md` is generated inside `benchmarks/workspaces/oss_maintainer/oss_task_001/`.
2. Pytest passes without symlink recursion loops.
3. The simulated user evaluates actual workspace artifacts rather than just textual claims.
