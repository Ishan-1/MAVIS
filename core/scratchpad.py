"""
core/scratchpad.py

Scratchpad pointer pattern for MAVIS execution pipelines.
Offloads large tool outputs (>4KB) to disk (data/scratch/) and returns
a compact head/tail digest with file pointer to prevent token bloat
in subsequent LLM prompts while retaining full fidelity on disk.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.helpers import log_it

_ENTITY = "scratchpad"
SCRATCH_DIR = Path("data/scratch")


def _ensure_scratch_dir() -> Path:
    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    return SCRATCH_DIR


def spill_if_large(
    turn_id: str,
    node_id: str,
    raw_output: Any,
    threshold_bytes: int = 4000,
    head_lines: int = 20,
    tail_lines: int = 20,
) -> tuple[Any, str | None]:
    """
    Check if raw_output exceeds threshold_bytes.
    If so, writes full raw content to data/scratch/<turn_id>_<node_id>.<ext>
    and returns (compact_digest, scratch_file_path).
    If within threshold, returns (raw_output, None).
    """
    if raw_output is None:
        return None, None

    is_json = False
    if isinstance(raw_output, (dict, list)):
        try:
            content_str = json.dumps(raw_output, indent=2, default=str)
            is_json = True
        except Exception:
            content_str = str(raw_output)
    elif isinstance(raw_output, str):
        content_str = raw_output
    else:
        content_str = str(raw_output)

    encoded = content_str.encode("utf-8", errors="replace")
    byte_len = len(encoded)

    if byte_len <= threshold_bytes:
        return raw_output, None

    # Spill to scratchpad file
    scratch_dir = _ensure_scratch_dir()
    safe_turn = "".join(c for c in turn_id if c.isalnum() or c in ("-", "_")) or "turn"
    safe_node = "".join(c for c in node_id if c.isalnum() or c in ("-", "_")) or "node"
    ext = "json" if is_json else "txt"
    scratch_file = scratch_dir / f"{safe_turn}_{safe_node}.{ext}"

    try:
        scratch_file.write_text(content_str, encoding="utf-8")
        log_it(
            _ENTITY,
            f"Offloaded {byte_len} bytes from step '{node_id}' to scratchpad: {scratch_file}",
        )
    except Exception as e:
        log_it(_ENTITY, f"Failed to spill to scratchpad: {e}", level="WARN")
        return raw_output, None

    lines = content_str.splitlines()
    total_lines = len(lines)

    if total_lines <= (head_lines + tail_lines + 5):
        # Even if byte count is high, line count is short (e.g. wide lines)
        head_part = "\n".join(lines[:head_lines])
        tail_part = "\n".join(lines[-tail_lines:]) if total_lines > head_lines else ""
        middle_notice = f"\n... [{byte_len} bytes offloaded to {scratch_file}] ...\n"
        digest = head_part + middle_notice + tail_part
    else:
        head_part = "\n".join(lines[:head_lines])
        tail_part = "\n".join(lines[-tail_lines:])
        hidden_lines = total_lines - (head_lines + tail_lines)
        middle_notice = (
            f"\n... [{hidden_lines} lines / {byte_len} bytes offloaded to file: {scratch_file}] ...\n"
        )
        digest = head_part + middle_notice + tail_part

    return digest, str(scratch_file)


def read_scratch_file(file_path: str | Path) -> str:
    """Read the full content of a spilled scratchpad file."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Scratchpad file not found: {p}")
    return p.read_text(encoding="utf-8")
