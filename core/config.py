"""
config.py
Central MAVIS configuration manager — single source of truth for all settings.

All modules import the process-wide singleton:

    from config import cfg

    # Read
    retries = cfg.get("toolbuilder", "max_retries", default=3)
    trust   = cfg.oni["trust_level"]

    # Write (runtime, in-memory only)
    cfg.set("oni", "trust_level", "yolo")

    # Persist to disk
    cfg.save()

    # Reload from disk (discards unsaved changes)
    cfg.reload()
"""
from __future__ import annotations

import json
import os
from typing import Any

_MAV_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_MAV_ROOT, "data", "mavis_config.json")

_DEFAULTS: dict = {
    "version": "1.3",

    "oni": {
        "trust_level": "ask",
        "blacklist": [],
        "whitelist": [
            "get_current_datetime",
            "extract_date_from_datetime",
            "parse_natural_date_to_yyyymmdd",
            "search_news",
            "echo",
            "date",
            "pwd",
            "notify-send",
        ],
        "greylist": [
            "restart_process",
            "install_package",
            "rm", "mv", "cp",
            "chmod", "chown",
            "curl", "wget", "pip",
        ],
        "url_blacklist": [],
        "approved_fs_write_paths": ["tools/", "tests/", "memories/", "logs/", "data/", "agents/"],
        "approved_fs_files": ["requirements.txt"],
        "tool_execution_timeout_seconds": 30,
    },

    "agentbuilder": {
        "max_retries": 3,
    },

    "memory": {
        "max_token": 12000,
        "context_window": 1000000,
        "top_k": 5,
        "short_term_ttl_days": 7,
        "repetition_window": 50,
        "repetition_similarity_threshold": 0.85,
        "repetition_min_count": 3,
        "session_timeout_minutes": 30,
        "emotion_strength_threshold": 0.75,
        "intent_strength_threshold": 0.85,
        "lt_intent_threshold": 0.90,
        "working_memory_active_turns": 8,
        "compact_token_threshold": 1500,
        "max_memory_entry_chars": 600,
        "tool_retrieval_threshold": 8,
        "tool_retrieval_top_k": 6,
        "general_tool_threshold": 0.75,
        "specific_tools_top_k": 5,
    },

    "toolbuilder": {
        "max_retries": 3,
        "forbidden_imports": [
            "subprocess", "socket", "ftplib", "paramiko", "pexpect", "httplib2",
        ],
        "forbidden_os_attrs": [
            "system", "popen", "fork", "execv", "execl", "execle",
            "execlp", "execlpe", "execvp", "execvpe", "spawn", "spawnl",
        ],
    },

    "scheduler": {
        "tick_seconds": 30,
        "short_term_worker_interval_minutes": 15,
        "long_term_worker_interval_minutes": 480,
    },

    "output": {
        "verbosity": "normal",              # quiet | normal | debug
        "history_file": "data/.mavis_history",
        "notify_pipeline_threshold_s": 5,   # min elapsed seconds to fire a desktop notification
    },

    "llm": {
        "provider": "gemini",               # gemini | openai | ollama
        "model": "gemini-2.5-flash",
        "embedding_model": "text-embedding-004",
        "temperature": 0.2,
        "vertexai": True,
        "base_url": None,
    },
}


class MAVISConfig:
    """
    Process-wide MAVIS configuration singleton.

    Loads from data/mavis_config.json, deep-merging with built-in defaults
    so newly added keys always have a safe fallback value.

    Section dicts (.oni, .memory, .toolbuilder, .scheduler) are live references
    to the internal _data dict — mutating them via set() is immediately visible
    to all modules reading from those dicts.
    """

    def __init__(self) -> None:
        self._data: dict = {}
        self.reload()

    # ── Persistence ───────────────────────────────────────────────────────────

    def reload(self) -> None:
        """Reload from disk, deep-merging with defaults for any missing keys."""
        try:
            with open(_CONFIG_PATH, "r") as f:
                loaded = json.load(f)
            self._data = self._deep_merge(_DEFAULTS, loaded)
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = json.loads(json.dumps(_DEFAULTS))  # safe deep copy

    def save(self) -> None:
        """Persist current in-memory config to disk."""
        os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
        with open(_CONFIG_PATH, "w") as f:
            json.dump(self._data, f, indent=4)

    # ── Section accessors ─────────────────────────────────────────────────────

    @property
    def oni(self) -> dict:
        return self._data.setdefault("oni", dict(_DEFAULTS["oni"]))

    @property
    def memory(self) -> dict:
        return self._data.setdefault("memory", dict(_DEFAULTS["memory"]))

    @property
    def toolbuilder(self) -> dict:
        return self._data.setdefault("toolbuilder", dict(_DEFAULTS["toolbuilder"]))

    @property
    def scheduler(self) -> dict:
        return self._data.setdefault("scheduler", dict(_DEFAULTS["scheduler"]))

    @property
    def output(self) -> dict:
        return self._data.setdefault("output", dict(_DEFAULTS["output"]))

    @property
    def llm(self) -> dict:
        return self._data.setdefault("llm", dict(_DEFAULTS["llm"]))

    # ── Generic get / set ─────────────────────────────────────────────────────

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Safe read: cfg.get('memory', 'top_k', default=5)."""
        return self._data.get(section, {}).get(key, default)

    def set(self, section: str, key: str, value: Any) -> None:
        """Runtime in-memory update. Call save() to persist."""
        self._data.setdefault(section, {})[key] = value

    def set_dotted(self, dotted_key: str, raw_value: str) -> None:
        """
        Parse and apply a 'section.key value' update from a slash command.

        The raw_value string is JSON-parsed so numbers, booleans, and lists
        all work naturally. Falls back to treating it as a plain string.

        Example:
            cfg.set_dotted("memory.top_k", "10")
            cfg.set_dotted("oni.trust_level", '"yolo"')
        """
        parts = dotted_key.split(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Key must be 'section.key', got: {dotted_key!r}")
        section, key = parts
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            value = raw_value  # plain string fallback
        self.set(section, key, value)

    # ── Convenience ───────────────────────────────────────────────────────────

    def as_dict(self) -> dict:
        """Return a deep copy of the full config (safe to mutate)."""
        return json.loads(json.dumps(self._data))

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge *override* into *base*. Override wins on conflicts."""
        result = dict(base)
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = MAVISConfig._deep_merge(result[key], val)
            else:
                result[key] = val
        return result


# ── Process-wide singleton ────────────────────────────────────────────────────
cfg = MAVISConfig()
