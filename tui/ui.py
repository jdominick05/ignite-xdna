"""Rendering and input.

rich is the only TUI-capable library installed (15.0.0, in both conda envs), and
nothing else is available: textual and prompt_toolkit are absent, and `import curses`
raises ModuleNotFoundError because the env has the stdlib wrapper but no _curses
extension. So: rich for everything drawn, msvcrt for single keypresses.

msvcrt is stdlib on Windows but does not work under Git Bash/mintty, which is where
scripts/ are documented to run. Rather than degrade silently, detect it and fall back
to numbered menus that work identically everywhere.
"""
import os
import sys

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except ImportError:                                       # pragma: no cover
    # rich 15.0.0 is present in resnet_env17 and resnet_env on Desktop 2, but it
    # arrived as somebody's transitive dependency: this repo has no
    # requirements.txt, no pyproject.toml, and mentions rich nowhere. So the
    # laptop and Desktop 1 may not have it. Fail with the fix rather than with a
    # traceback, and rather than half-rendering through a shim that would be one
    # more thing to keep correct.
    raise SystemExit(
        "the TUI needs `rich`, which is not installed in this environment.\n"
        "  conda activate resnet_env17 && pip install rich\n"
        "Everything else in the repo runs without it; see docs/SETUP.md.")

console = Console(highlight=False)

# Arrow keys need a real Windows console. MSYSTEM is set by Git Bash/MSYS2, where
# msvcrt.getch() does not see keystrokes.
_INTERACTIVE = sys.stdin.isatty() and not os.environ.get("MSYSTEM")

try:
    import msvcrt
except ImportError:                                       # not Windows
    msvcrt = None
    _INTERACTIVE = False


def can_use_keys(plain=False) -> bool:
    return bool(_INTERACTIVE and msvcrt and not plain)


def rule(text):
    console.rule(f"[bold]{text}[/bold]")


def panel(body, title="", style="cyan"):
    console.print(Panel(body, title=title, border_style=style, expand=False))


def _getkey():
    """One keypress, normalised. Arrow keys arrive as a two-byte sequence."""
    ch = msvcrt.getch()
    if ch in (b"\x00", b"\xe0"):
        code = msvcrt.getch()
        return {b"H": "up", b"P": "down", b"K": "left", b"M": "right"}.get(code, "")
    if ch in (b"\r", b"\n"):
        return "enter"
    if ch == b"\x1b":
        return "esc"
    if ch == b"\x03":
        raise KeyboardInterrupt
    try:
        return ch.decode("utf-8", "ignore").lower()
    except Exception:                                     # noqa: BLE001
        return ""


def choose(title, options, describe=None, plain=False, allow_back=True):
    """Pick one of `options`. Returns the chosen item, or None for back/quit.

    `describe` maps an option to (label, detail). Both input modes render the same
    table, so the only thing that differs is how a row gets selected.
    """
    describe = describe or (lambda o: (str(o), ""))
    rows = [describe(o) for o in options]

    if not can_use_keys(plain):
        return _choose_numbered(title, options, rows, allow_back)
    return _choose_keys(title, options, rows, allow_back)


def _render(title, rows, cursor=None, allow_back=True):
    table = Table(title=title, title_justify="left", header_style="bold",
                  show_header=False, box=None, pad_edge=False)
    table.add_column("", width=2)
    table.add_column("", style="cyan", no_wrap=True)
    table.add_column("", style="dim")
    for i, (label, detail) in enumerate(rows):
        marker = ">" if cursor == i else " "
        style = "bold" if cursor == i else ""
        table.add_row(marker, f"[{style}]{label}[/{style}]" if style else label, detail)
    console.print(table)
    if allow_back:
        console.print("[dim]  (b) back   (q) quit[/dim]")


def _choose_numbered(title, options, rows, allow_back):
    numbered = [(f"{i + 1}. {label}", detail) for i, (label, detail) in enumerate(rows)]
    _render(title, numbered, allow_back=allow_back)
    while True:
        try:
            raw = input("select> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw in ("q", "quit", "b", "back", ""):
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        console.print(f"[red]enter 1-{len(options)}, or b/q[/red]")


def _choose_keys(title, options, rows, allow_back):
    cursor = 0
    while True:
        console.clear()
        _render(title, rows, cursor, allow_back)
        key = _getkey()
        if key == "up":
            cursor = (cursor - 1) % len(options)
        elif key == "down":
            cursor = (cursor + 1) % len(options)
        elif key == "enter":
            return options[cursor]
        elif key in ("q", "esc", "b"):
            return None
        elif key.isdigit() and 1 <= int(key) <= len(options):
            return options[int(key) - 1]


def confirm(question, default=True):
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        raw = input(f"{question} {suffix} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if not raw:
        return default
    return raw.startswith("y")


def ask(question, default=""):
    hint = f" [{default}]" if default else ""
    try:
        raw = input(f"{question}{hint} ").strip()
    except (EOFError, KeyboardInterrupt):
        return default
    return raw.strip('"').strip("'") or default


STATUS_STYLE = {"ok": "green", "warn": "yellow", "fail": "red", "unknown": "dim"}


def render_checks(checks, title="Device"):
    table = Table(title=title, title_justify="left", header_style="bold",
                  box=None, pad_edge=False)
    table.add_column("", style="cyan", no_wrap=True)
    table.add_column("")
    for c in checks:
        style = STATUS_STYLE.get(c.status, "")
        table.add_row(c.name, f"[{style}]{c.detail}[/{style}]")
    console.print(table)
