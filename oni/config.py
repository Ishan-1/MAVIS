from __future__ import annotations

from typing import Literal

from core.config import cfg

TrustLevel = Literal["whitelist_only", "ask", "yolo"]


class ONIConfig:
    """
    Adapter that surfaces the 'oni' section of MAVISConfig as typed properties.
    No file I/O here — all persistence is handled by cfg.save().
    """

    # ── Trust level ───────────────────────────────────────────────────────────

    @property
    def trust_level(self) -> str:
        return cfg.oni.get("trust_level", "ask")

    @trust_level.setter
    def trust_level(self, value: str) -> None:
        cfg.set("oni", "trust_level", value)

    # ── Permission lists ──────────────────────────────────────────────────────

    @property
    def blacklist(self) -> set[str]:
        return set(cfg.oni.get("blacklist", []))

    @property
    def whitelist(self) -> set[str]:
        return set(cfg.oni.get("whitelist", []))

    @property
    def greylist(self) -> set[str]:
        return set(cfg.oni.get("greylist", []))

    @property
    def url_blacklist(self) -> list[str]:
        return cfg.oni.get("url_blacklist", [])

    # ── Filesystem ────────────────────────────────────────────────────────────

    @property
    def approved_fs_write_paths(self) -> list[str]:
        return cfg.oni.get("approved_fs_write_paths", [])

    @property
    def approved_fs_files(self) -> list[str]:
        return cfg.oni.get("approved_fs_files", [])

    # ── Execution ─────────────────────────────────────────────────────────────

    @property
    def tool_execution_timeout(self) -> int:
        return cfg.oni.get("tool_execution_timeout_seconds", 30)

    # ── Persistence helpers (delegate to central config) ──────────────────────

    def load(self) -> None:
        """Reload from disk (delegates to central MAVISConfig)."""
        cfg.reload()

    def save(self) -> None:
        """Persist to disk (delegates to central MAVISConfig)."""
        cfg.save()
