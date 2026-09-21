# Code Consistency Report

This document outlines findings related to code consistency, style, naming conventions, and error handling within the codebase. The goal is to identify areas for improvement to enhance maintainability, readability, and robustness.

---

## Findings

### 1. Inconsistent `from __future__ import annotations` Usage

*   **Type:** `code_style`
*   **Description:** Inconsistent use of `from __future__ import annotations`. It is present in `memories/emotion_classifier.py` and `memories/memory_store.py` but missing in `memories/embedding.py`.
*   **Files Involved:**
    *   `./memories/embedding.py`
    *   `./memories/emotion_classifier.py`
    *   `./memories/memory_store.py`
*   **Recommendation:** Add `from __future__ import annotations` to the top of all Python files, especially those using type hints, for consistent forward reference support and cleaner type hint syntax.

### 2. Ambiguous Naming for Base Paths

*   **Type:** `naming_convention`
*   **Description:** The `_BASE` constant in `memories/memory_store.py` is defined relative to `__file__`, while `_MAV_ROOT` in `main.py` is defined as the absolute project root. While functionally different, using similar names (`_BASE`, `_ROOT`) for different base paths can be confusing.
*   **Files Involved:**
    *   `./memories/memory_store.py`
    *   `./main.py`
*   **Recommendation:** Rename `_BASE` in `memories/memory_store.py` to something more specific like `_MEMORIES_BASE_DIR` to clearly indicate its scope and avoid confusion with a potential global project root constant like `_MAV_ROOT`.

### 3. Inconsistent Error Handling for Process Termination

*   **Type:** `error_handling`
*   **Description:** Inconsistent error handling for `subprocess.Popen` termination during shutdown in `main.py`. Some `try...except BaseException` blocks simply `pass`, suppressing all errors, while others attempt `kill()` after `terminate()` and `wait()`. This broad `BaseException` catch with `pass` can hide critical issues.
*   **Files Involved:**
    *   `./main.py`
*   **Recommendation:** Standardize error handling during shutdown. Instead of `except BaseException: pass`, consider logging the exception or catching more specific exceptions. For process termination, ensure a consistent pattern of `terminate()`, `wait()`, and then `kill()` if necessary, with appropriate logging for each step.

### 4. Duplicated Shutdown Logic

*   **Type:** `duplication`
*   **Description:** The pattern of `try...except BaseException` followed by `pass` is repeated multiple times in the `finally` block of `main.py` for shutting down different components (`_dashboard_proc`, `_tooldash_proc`, `_stop_workers`, `runner.stop`, `mcp_manager.shutdown`, `_print_exit_summary`).
*   **Files Involved:**
    *   `./main.py`
*   **Recommendation:** Refactor the shutdown logic into a helper function (e.g., `_safe_shutdown_component(component, name)`) that encapsulates the `terminate/wait/kill` logic and error handling. This would reduce duplication and make the shutdown sequence clearer and more robust.

### 5. Misplaced Token Count Utility

*   **Type:** `code_style`
*   **Description:** The `_token_count` function in `memories/memory_store.py` uses a simple heuristic (`len(text) // 4`) for token estimation. If this is a general utility, it might be better placed in a shared `core.llm.helpers` or similar module for consistency and reusability across the project.
*   **Files Involved:**
    *   `./memories/memory_store.py`
*   **Recommendation:** Move `_token_count` to a `core.llm.utils` or `core.helpers` module if it's intended to be a general-purpose token estimation utility, and import it where needed. This centralizes common logic.

### 6. Inconsistent Logging Entity Definition

*   **Type:** `naming_convention`
*   **Description:** The `_ENTITY` constant is used in `memories/embedding.py`, `memories/emotion_classifier.py`, and `memories/memory_store.py` for logging. This is a good pattern, but it's not explicitly defined or used in `main.py` for its own logging, which uses `mavis_status`, `mavis_print`, `mavis_error` directly.
*   **Files Involved:**
    *   `./memories/embedding.py`
    *   `./memories/emotion_classifier.py`
    *   `./memories/memory_store.py`
    *   `./main.py`
*   **Recommendation:** Consider introducing a `_ENTITY` constant or a similar mechanism in `main.py` (e.g., `_ENTITY = 'main'`) and using the `log_it` helper consistently, or ensure `mavis_status`/`mavis_print`/`mavis_error` functions internally use a consistent entity identifier if they are the preferred logging mechanism for `main.py`.