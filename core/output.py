"""
output.py
Rich-backed output layer for MAVIS.

All user-facing output goes through this module instead of bare print().
Verbosity is controlled by cfg.output["verbosity"]:
  quiet  — only mavis_answer() and mavis_print(level="quiet") are shown
  normal — quiet + mavis_status() (default)
  debug  — normal + mavis_debug() + raw LLM JSON / pipeline traces

Usage:
    from output import mavis_answer, mavis_status, mavis_debug, spinner

    mavis_answer("Here is the result:\n\n```python\nprint('hi')\n```")
    mavis_status("Building tool 'get_user_name'...")
    mavis_debug("Raw LLM JSON", entity="interpreter")
    with spinner("Calling LLM..."):
        response = client.models.generate_content(...)
"""

from __future__ import annotations

from contextlib import contextmanager

from rich.columns import Columns
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.theme import Theme

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.styles import Style

from core.config import cfg

# ── Console setup ─────────────────────────────────────────────────────────────

_theme = Theme(
    {
        "mavis.answer": "bold white",
        "mavis.status": "dim cyan",
        "mavis.warn": "bold yellow",
        "mavis.error": "bold red",
        "mavis.debug": "dim magenta",
        "mavis.ok": "bold green",
    }
)

_console = Console(theme=_theme, highlight=False)

# ── Verbosity helpers ──────────────────────────────────────────────────────────

_LEVELS = {"quiet": 0, "normal": 1, "debug": 2}


def _verbosity() -> int:
    raw = cfg.output.get("verbosity", "normal")
    return _LEVELS.get(str(raw).lower(), 1)


def _at_least(level: str) -> bool:
    return _verbosity() >= _LEVELS.get(level, 1)


# ── Public API ─────────────────────────────────────────────────────────────────


def mavis_answer(text: str) -> None:
    """
    Render MAVIS's primary answer to the user.

    Text is rendered as Markdown (code blocks, tables, bold, etc.).
    The panel width is capped at 100 columns so it never spans the full terminal
    on wide displays, but always fits the content when content is short.
    Always visible regardless of verbosity.
    """
    md = Markdown(text, code_theme="monokai")
    _console.print(
        Panel(
            md,
            title="[mavis.answer]MAVIS[/mavis.answer]",
            border_style="cyan",
            expand=False,
            width=min(_console.width, 100),
        )
    )


def mavis_status(text: str) -> None:
    """
    Print a dim informational status line (e.g. 'Building tool foo...').
    Shown in normal and debug modes; suppressed in quiet mode.
    """
    if _at_least("normal"):
        _console.print(f"[mavis.status]▸ {text}[/mavis.status]")


def mavis_ok(text: str) -> None:
    """Print a success confirmation line. Shown in normal and debug modes."""
    if _at_least("normal"):
        _console.print(f"[mavis.ok]✓ {text}[/mavis.ok]")


def mavis_warn(text: str) -> None:
    """Print a warning line. Always visible."""
    _console.print(f"[mavis.warn]⚠ {text}[/mavis.warn]")


def mavis_error(text: str) -> None:
    """Print a user-facing error line. Always visible."""
    _console.print(f"[mavis.error]✗ {text}[/mavis.error]")


def mavis_debug(text: str, entity: str = "") -> None:
    """
    Print a debug-only line (raw JSON, pipeline traces, etc.).
    Only visible when verbosity = debug.
    """
    if _at_least("debug"):
        prefix = f"[{entity}] " if entity else ""
        _console.print(f"[mavis.debug]{prefix}{text}[/mavis.debug]")


def mavis_print(text: str, level: str = "normal") -> None:
    """
    General-purpose print at a given verbosity level.
    level: 'quiet' (always), 'normal', or 'debug'.
    """
    if _at_least(level):
        _console.print(text)


def oni_print(text: str) -> None:
    """Print an ONI gate message. Always visible."""
    _console.print(f"[bold yellow][ONI][/bold yellow] {text}")


def oni_gate_panel(description: str) -> None:
    """Render a styled ONI approval alert box."""
    _console.print(
        Panel(
            f"[bold yellow]⚠  Approval Required[/bold yellow]\n\n{description.strip()}",
            title="[bold yellow]ONI Gate[/bold yellow]",
            border_style="yellow",
            padding=(1, 2),
            expand=False,
            width=min(_console.width, 90),
        )
    )


_SELECT_STYLE = Style.from_dict(
    {
        "prompt": "bold yellow",
        "selected": "bold green reverse",
        "unselected": "#888888",
        "hint": "cyan italic",
    }
)


def interactive_select_yes_no(prompt_text: str = "Allow?", default: bool = False) -> bool:
    """
    Inline interactive selector for Yes/No with arrow key navigation,
    y/n hotkeys, and Enter to confirm.
    """
    selected = [1 if default else 0]  # 0: No, 1: Yes
    options = [("no", "No (Deny)"), ("yes", "Yes (Allow)")]

    kb = KeyBindings()

    @kb.add("left")
    @kb.add("up")
    @kb.add("tab")
    def _(event):
        selected[0] = 1 - selected[0]
        event.app.invalidate()

    @kb.add("right")
    @kb.add("down")
    @kb.add("s-tab")
    def _(event):
        selected[0] = 1 - selected[0]
        event.app.invalidate()

    @kb.add("y")
    @kb.add("Y")
    def _(event):
        selected[0] = 1
        event.app.exit(result=True)

    @kb.add("n")
    @kb.add("N")
    def _(event):
        selected[0] = 0
        event.app.exit(result=False)

    @kb.add("enter")
    def _(event):
        event.app.exit(result=(selected[0] == 1))

    @kb.add("c-c")
    @kb.add("escape")
    def _(event):
        event.app.exit(result=False)

    def get_text() -> FormattedText:
        res = [("class:prompt", f"  {prompt_text}  ")]
        for i, (val, label) in enumerate(options):
            if i == selected[0]:
                res.append(("class:selected", f" [● {label}] "))
            else:
                res.append(("class:unselected", f" [○ {label}] "))
            res.append(("", "  "))
        res.append(("class:hint", "(←/→ or y/n, Enter to confirm)"))
        return FormattedText(res)

    layout = Layout(Window(content=FormattedTextControl(get_text), height=1))
    app = Application(layout=layout, key_bindings=kb, style=_SELECT_STYLE)
    result = app.run()

    if result:
        _console.print("  [bold green]✓ Decision: Allowed[/bold green]\n")
    else:
        _console.print("  [bold red]✗ Decision: Denied[/bold red]\n")
    return bool(result)


def rule(title: str = "") -> None:
    """Print a horizontal rule separator, optionally with a centred title."""
    _console.print(Rule(title, style="dim cyan"))


def print_table(rows: list[tuple[str, str]], title: str = "") -> None:
    """
    Print a two-column key/value table.
    Used for /config and /status structured output.
    """
    tbl = Table(
        show_header=bool(title),
        header_style="bold cyan",
        box=None,
        padding=(0, 2),
        title=title or None,
        title_style="bold cyan",
        show_edge=False,
    )
    tbl.add_column("Key", style="dim", no_wrap=True)
    tbl.add_column("Value", style="white")
    for k, v in rows:
        tbl.add_row(k, str(v))
    _console.print(tbl)


@contextmanager
def spinner(label: str):
    """
    Context manager that shows a Rich spinner while work is in progress.
    Only rendered in normal/debug modes; falls through silently in quiet mode.

    Usage:
        with spinner("Calling LLM..."):
            result = client.models.generate_content(...)
    """
    if _at_least("normal"):
        with _console.status(f"[mavis.status]{label}[/mavis.status]", spinner="dots"):
            yield
    else:
        yield
