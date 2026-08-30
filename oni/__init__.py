"""
oni/__init__.py
ONI package — Operating System & Network Interface.

Creates the process-wide singleton on first import, then exposes
module-level convenience functions so any tool or module can use:

    from oni import call_network, call_fs, call_system_command

The singleton is also available directly as `oni.oni` for callers that
need access to preflight_scan, set_trust_level, or set_context_trust.
"""
from oni.oni import ONI

# Process-wide singleton — created once, shared across all modules
oni = ONI()


# ── Module-level convenience wrappers ────────────────────────────────────────

def call_system_command(command: str, params: dict):
    """Execute an OS-level command through ONI."""
    return oni.call_system_command(command, params)


def call_network(url, data=None, method="GET", params=None, headers=None, timeout=10):
    """Make an outbound HTTP request through ONI."""
    return oni.call_network(
        url, data=data, method=method, params=params,
        headers=headers, timeout=timeout,
    )


def call_fs(operation, path, data=None, params=None):
    """Perform a filesystem operation through ONI."""
    return oni.call_fs(operation, path, data=data, params=params)


def call_shell(shell_string: str):
    """Execute a shell command or pipeline through ONI."""
    return oni.call_shell(shell_string)


def set_context_trust(level):
    """Set thread-local trust level override (for background tasks)."""
    oni.set_context_trust(level)


__all__ = [
    "oni",
    "call_system_command",
    "call_shell",
    "call_network",
    "call_fs",
    "set_context_trust",
]
