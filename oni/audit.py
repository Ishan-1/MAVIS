"""
oni/audit.py
Append-only, tamper-evident audit log for all ONI operations.

Every call — allowed or denied — is recorded to logs/oni_audit.jsonl as a
JSON Lines entry. The file is opened in append ('a') mode only and is
never truncated, making it a reliable accountability trail.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

_AUDIT_LOG_PATH = "logs/oni_audit.jsonl"


def record(entry: dict) -> None:
    """
    Append *entry* to the ONI audit log.

    Automatically stamps with a UTC ISO-8601 timestamp.
    Creates the logs/ directory if it does not exist.
    Silently swallows write errors so audit failures never crash MAVIS.
    """
    try:
        os.makedirs("logs", exist_ok=True)
        entry = dict(entry)  # don't mutate caller's dict
        entry["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
        # Append-only — never 'w' or truncate
        with open(_AUDIT_LOG_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        # Audit log failures must not bring down the system
        pass
