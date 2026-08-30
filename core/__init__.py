"""
core package
Provides foundational configuration, logging, scheduling, and UI output layers for MAVIS.
"""
from core.config import cfg
from core.helpers import log_it
from core.scheduler import TaskRunner
from core.output import (
    mavis_answer,
    mavis_status,
    mavis_ok,
    mavis_warn,
    mavis_error,
    mavis_debug,
    mavis_print,
    oni_print,
    oni_gate_panel,
    interactive_select_yes_no,
    spinner,
    rule,
    print_table,
)

__all__ = [
    "cfg",
    "log_it",
    "TaskRunner",
    "mavis_answer",
    "mavis_status",
    "mavis_ok",
    "mavis_warn",
    "mavis_error",
    "mavis_debug",
    "mavis_print",
    "oni_print",
    "oni_gate_panel",
    "interactive_select_yes_no",
    "spinner",
    "rule",
    "print_table",
]
