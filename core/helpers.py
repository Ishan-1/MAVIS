"""
core/helpers.py
Central utility helpers for logging, token estimation, path discovery, and prompt sanitization.
"""
from __future__ import annotations

import os
from typing import Any

_MAV_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAV_ROOT = _MAV_ROOT


def estimate_tokens(text: str | None) -> int:
    """Return an estimated token count (~4 characters per token), null-safe."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def sanitize_delimiter(text: str) -> str:
    """Escape <tool_input> closing tag to prevent prompt injection delimiter breakout."""
    if not isinstance(text, str):
        return text
    return text.replace("</tool_input>", "&lt;/tool_input&gt;")


def log_it(
    message: str,
    entity_name: str = "main",
    level: str | None = None,
) -> None:
    """
    Log a message to the entity-specific log and the central main.log.

    Robust to inverted argument order (e.g. log_it(entity_name, message)).
    """
    # Guard against inverted signature: log_it(entity, message)
    # If message is a short identifier without spaces/newlines and entity_name has whitespace
    if isinstance(message, str) and isinstance(entity_name, str):
        if (" " in entity_name or "\n" in entity_name) and (" " not in message and "\n" not in message and len(message) <= 30):
            message, entity_name = entity_name, message

    if not entity_name:
        entity_name = "main"

    prefix = f"[{level.upper()}] " if level else ""
    formatted_msg = f"{prefix}{message}"

    log_file = os.path.join(_MAV_ROOT, "logs", f"{entity_name}.log")
    main_log = os.path.join(_MAV_ROOT, "logs", "main.log")
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as file:
            file.write(f"{formatted_msg}\n")
        with open(main_log, "a", encoding="utf-8") as file:
            file.write(f"{entity_name}: {formatted_msg}\n")
    except Exception:
        pass

