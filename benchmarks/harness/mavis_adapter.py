"""
benchmarks/harness/mavis_adapter.py
Live MAVIS execution adapter connecting the benchmark harness to MAVIS's
interpreter, ONI permission harness, DAG pipeline runner, and answerer.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from oni import oni as _oni


class MavisBenchmarkAdapter:
    """
    Adapter that executes benchmark turns directly through MAVIS's live
    interpret_command pipeline with workspace scoping and ONI task leasing.
    """

    def __init__(self, lease_trust: str = "yolo") -> None:
        self.lease_trust = lease_trust
        self._mavis_module = None

    def _get_mavis(self):
        if self._mavis_module is None:
            import main
            self._mavis_module = main
        return self._mavis_module

    @property
    def llm_client(self):
        return self._get_mavis().llm

    def execute_turn(self, user_msg: str, workspace_dir: Path) -> dict[str, Any]:
        """
        Execute a single user turn in MAVIS scoped to the task workspace.
        """
        main_mod = self._get_mavis()
        ws_abs = str(workspace_dir.resolve())

        # 1. Authorize workspace in ONI approved filesystem write paths
        cfg_paths = _oni.config.approved_fs_write_paths
        if ws_abs not in cfg_paths:
            cfg_paths.append(ws_abs)

        # 2. Add relative workspace prefix if not already present
        ws_rel = os.path.relpath(ws_abs, os.getcwd())
        if ws_rel not in cfg_paths:
            cfg_paths.append(ws_rel)

        # 3. Contextualize the prompt with the target workspace so MAVIS tools target it
        contextual_prompt = f"[Workspace: {ws_abs}]\n{user_msg}"

        start_chat_len = len(main_mod._session_chat)

        # 4. Execute within ONI task lease to avoid interactive terminal prompt blocks
        with _oni.task_lease("benchmark_task", lease_trust=self.lease_trust):
            main_mod.interpret_command(contextual_prompt)

        # 5. Extract assistant's final synthesized answer
        reply = "Task processing complete."
        new_messages = main_mod._session_chat[start_chat_len:]
        for m in reversed(new_messages):
            if m.get("role") == "assistant":
                reply = m.get("content", "")
                break

        return {
            "reply": reply,
            "waves": 1,
            "trace": new_messages,
            "commands": [],
        }
