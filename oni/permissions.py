"""
oni/permissions.py
Permission classification logic for commands, URLs, and filesystem paths.
"""
from __future__ import annotations

import os
from typing import Literal

from oni.config import ONIConfig

Decision = Literal["allow", "greylist", "deny"]


def classify_command(name: str, config: ONIConfig, trust_level: str) -> Decision:
    """
    Classify a pipeline command name or OS command against the permission lists.

    Resolution order:
      yolo         → always allow
      blacklist    → deny
      whitelist    → allow
      greylist     → greylist (requires user confirmation)
      unlisted + whitelist_only → deny
      unlisted + ask            → greylist (unknown = ask)
    """
    if trust_level == "yolo":
        return "allow"
    if name in config.blacklist:
        return "deny"
    if name in config.whitelist:
        return "allow"
    if name in config.greylist:
        return "greylist"
    # Not explicitly listed
    if trust_level == "whitelist_only":
        return "deny"
    # Default for "ask" mode: treat unknown as greylist
    return "greylist"


def classify_url(url: str, config: ONIConfig, trust_level: str) -> Decision:
    """
    Classify an outbound network URL.

    Blocks any URL whose hostname or prefix appears in url_blacklist.
    In yolo mode all URLs are allowed. Otherwise unknown URLs are allowed
    (URL blacklist is opt-in rather than opt-out).
    """
    if trust_level == "yolo":
        return "allow"
    for blocked in config.url_blacklist:
        if blocked and blocked in url:
            return "deny"
    return "allow"


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _is_protected_path(path: str) -> bool:
    """
    Check if path targets protected system files/directories unconditionally.

    Protected paths:
      - .env (any .env file or reference)
      - data/mavis_config.json (ONI configuration / permission policy)
      - oni/ (entire ONI gatekeeper subsystem)
    """
    if not path or not isinstance(path, str):
        return False

    norm_path = os.path.normpath(path)
    base_name = os.path.basename(norm_path)

    # 1. Any file named .env
    if base_name == ".env" or norm_path == ".env":
        return True

    # 2. Determine relative path against repo root if absolute
    if os.path.isabs(norm_path):
        try:
            rel = os.path.relpath(norm_path, _REPO_ROOT).replace("\\", "/")
        except ValueError:
            rel = norm_path.replace("\\", "/")
    else:
        rel = norm_path.replace("\\", "/")

    if rel.startswith("./"):
        rel = rel[2:]

    # 3. Check .env relative
    if rel == ".env" or rel.endswith("/.env"):
        return True

    # 4. Check data/mavis_config.json
    if rel == "data/mavis_config.json" or base_name == "mavis_config.json":
        return True

    # 5. Check oni/ directory or files within oni/
    if rel == "oni" or rel.startswith("oni/") or "/oni/" in f"/{rel}/":
        return True

    return False


def classify_fs_write(path: str, config: ONIConfig, trust_level: str) -> Decision:
    """
    Classify a filesystem write or delete operation.

    Unconditional protection:
      - .env, data/mavis_config.json, and oni/ always return "deny",
        regardless of trust level (including yolo mode).

    Allowed if:
      - yolo mode (for non-protected paths), OR
      - path matches an entry in approved_fs_files (exact match), OR
      - path starts with an entry in approved_fs_write_paths

    In ask mode, unlisted paths trigger a greylist prompt.
    In whitelist_only mode, unlisted paths are denied.
    """
    if _is_protected_path(path):
        return "deny"

    if trust_level == "yolo":
        return "allow"

    norm_path = os.path.normpath(path)

    # Exact-file allowances (e.g. requirements.txt)
    for allowed_file in config.approved_fs_files:
        if norm_path == os.path.normpath(allowed_file):
            return "allow"

    # Active workspace allowance
    active_ws = os.environ.get("MAVIS_ACTIVE_WORKSPACE")
    if active_ws:
        norm_ws = os.path.normpath(active_ws)
        if norm_path == norm_ws or norm_path.startswith(norm_ws + os.sep):
            return "allow"

    # Directory-prefix allowances
    for approved_dir in config.approved_fs_write_paths:
        norm_approved = os.path.normpath(approved_dir)
        if norm_path == norm_approved or norm_path.startswith(norm_approved + os.sep):
            return "allow"

    # Not in any approved location
    if trust_level == "whitelist_only":
        return "deny"
    return "greylist"
