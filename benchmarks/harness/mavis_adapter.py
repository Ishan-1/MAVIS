"""
benchmarks/harness/mavis_adapter.py
Live MAVIS execution adapter connecting the benchmark harness to MAVIS's
interpreter, ONI permission harness, DAG pipeline runner, and answerer.
"""
from __future__ import annotations

import json
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
        self.clear_working_memory()

    def clear_working_memory(self) -> None:
        """Clear MAVIS working memory across all namespaces and reset session chat."""
        from memories.memory_store import clear_all_working_memories
        clear_all_working_memories()

        main_mod = self._get_mavis()
        if hasattr(main_mod, "memory_store") and main_mod.memory_store:
            main_mod.memory_store.clear_working_memory()
        if hasattr(main_mod, "_session_chat") and isinstance(main_mod._session_chat, list):
            main_mod._session_chat.clear()

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
        ws_rel = os.path.relpath(ws_abs, os.getcwd())
        cfg_paths = _oni.config.approved_fs_write_paths
        for p in (ws_abs, ws_rel, "workspace", "./workspace"):
            if p not in cfg_paths:
                cfg_paths.append(p)

        # 3. Contextualize the prompt with the target workspace so MAVIS tools target it
        contextual_prompt = f"[Workspace: {ws_abs}]\n{user_msg}"

        start_chat_len = len(main_mod._session_chat)
        audit_file = "logs/oni_audit.jsonl"
        start_offset = os.path.getsize(audit_file) if os.path.exists(audit_file) else 0

        # 4. Execute within ONI task lease to avoid interactive terminal prompt blocks
        os.environ["MAVIS_ACTIVE_WORKSPACE"] = ws_abs
        with _oni.task_lease("benchmark_task", lease_trust=self.lease_trust):
            main_mod.interpret_command(contextual_prompt)

        # 5. Extract newly executed commands from ONI audit log
        executed_commands: list[dict[str, Any]] = []
        if os.path.exists(audit_file):
            try:
                with open(audit_file, "r", encoding="utf-8") as f:
                    f.seek(start_offset)
                    for line in f:
                        line = line.strip()
                        if line:
                            entry = json.loads(line)
                            cmd = entry.get("command") or entry.get("operation") or entry.get("event") or ""
                            params = entry.get("params") or entry.get("path") or {}
                            executed_commands.append({
                                "command": str(cmd),
                                "params": params,
                                "type": entry.get("type", "unknown"),
                                "decision": entry.get("decision", "unknown"),
                                "is_greylisted": entry.get("reason") == "greylisted" or entry.get("approved_by") == "greylist",
                                "user_confirmed": entry.get("approved_by") == "user",
                            })
            except Exception:
                pass

        # 6. Extract assistant's final synthesized answer
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
            "commands": executed_commands,
        }
