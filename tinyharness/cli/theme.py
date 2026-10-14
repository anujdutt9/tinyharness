"""The CLI's look: one palette, shared by rich (output) and prompt_toolkit (input).

Change a color here and it changes everywhere. There's a dark palette and a light one,
chosen by the theme setting: "system" (the default) asks the terminal what its
background color is, "light" or "dark" pick one. First found wins:

    TINYHARNESS_THEME=light      environment
    theme = "light"              ~/.tinyharness/config.toml
    /theme light                 remembered by the CLI

NO_COLOR turns color off (rich and prompt_toolkit both honor it).
"""

from __future__ import annotations

import contextlib
import os
import re
import select
import sys
import time
import tomllib
from pathlib import Path
from urllib.parse import quote

from prompt_toolkit.styles import Style
from rich.style import Style as RichStyle
from rich.text import Text
from rich.theme import Theme

from tinyharness.permissions import PermissionMode

# -- palette ------------------------------------------------------------------


THEMES = ("system", "light", "dark")


def theme_setting() -> tuple[str, str]:
    """(the theme asked for, where that came from)."""
    env = os.environ.get("TINYHARNESS_THEME", "").lower()
    if env in THEMES:
        return env, "TINYHARNESS_THEME"
    from tinyharness.config import saved_theme
    from tinyharness.paths import default_config_path

    with contextlib.suppress(OSError, tomllib.TOMLDecodeError, UnicodeDecodeError):
        configured = str(tomllib.loads(default_config_path().read_text()).get("theme", "")).lower()
        if configured in THEMES:
            return configured, "config"
    saved = saved_theme()
    return (saved, "/theme") if saved in THEMES else ("system", "default")


def terminal_is_light() -> bool | None:
    """Ask the terminal for its background color (OSC 11); None if it doesn't say.
    Most terminals answer at once; if none does in 100ms, we don't wait longer."""
    if os.name != "posix" or not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    import termios
    import tty

    fd = sys.stdin.fileno()
    try:
        saved = termios.tcgetattr(fd)
    except termios.error:
        return None
    reply = b""
    try:
        tty.setcbreak(fd)
        os.write(sys.stdout.fileno(), b"\033]11;?\033\\")
        deadline = time.monotonic() + 0.1
        while time.monotonic() < deadline and not reply.endswith((b"\007", b"\033\\")):
            ready, _, _ = select.select([fd], [], [], max(deadline - time.monotonic(), 0))
            if not ready:
                break
            reply += os.read(fd, 64)
    except OSError:
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    m = re.search(rb"rgb:([0-9a-f]+)/([0-9a-f]+)/([0-9a-f]+)", reply, re.IGNORECASE)
    if not m:
        return None
    r, g, b = (int(x, 16) / (16 ** len(x) - 1) for x in m.groups())
    return 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5


def _is_light(choice: str) -> bool:
    if choice != "system":
        return choice == "light"
    if (answer := terminal_is_light()) is not None:
        return answer
    background = os.environ.get("COLORFGBG", "").rpartition(";")[2]  # set by some terminals
    return background in ("7", "15")


THEME_SETTING, THEME_SOURCE = theme_setting()
LIGHT = _is_light(THEME_SETTING)

# "Field Notes": ink on paper, with a highlighter for what matters. BRAND is the ink,
# BRAND_2 a second ink (keys, code), HIGHLIGHT the marker, used as a background only.
if LIGHT:
    BRAND, BRAND_2, ON_BRAND = "#1f5f99", "#127a6a", "#ffffff"
    HIGHLIGHT, ON_HIGHLIGHT = "#ffe15a", "#2a2300"
    MUTED, SUBTLE = "#5b6876", "#9aa6b2"
    SUCCESS, WARNING, ERROR = "#1d7a45", "#a86400", "#b42318"
    ADDED, REMOVED = "#1d7a45", "#b42318"
    TOOL_COLORS = {
        "read": "#1f5f99",
        "edit": "#8a5a00",
        "run": "#b4532a",
        "web": "#127a6a",
        "agent": "#7a4bb5",
    }
    MENU_BG, MENU_FG = "#eef2f6", "#17212c"
else:
    BRAND, BRAND_2, ON_BRAND = "#8ec2ec", "#8fd8c4", "#0b1016"
    HIGHLIGHT, ON_HIGHLIGHT = "#e8c93e", "#1f1a00"
    MUTED, SUBTLE = "#93a1af", "#5d6b78"
    SUCCESS, WARNING, ERROR = "#79d29a", "#f2bd5c", "#ff8a7a"
    ADDED, REMOVED = "#79d29a", "#ff8a7a"
    TOOL_COLORS = {
        "read": "#8ec2ec",
        "edit": "#f2d16b",
        "run": "#f0a37f",
        "web": "#8fd8c4",
        "agent": "#c9aaf0",
    }
    MENU_BG, MENU_FG = "#18212b", "#d6e0e9"

# Each tool's family, which gives it its color: looking, changing, running, the web, agents.
TOOL_KIND = {
    "Read": "read",
    "Glob": "read",
    "Grep": "read",
    "Pdf": "read",
    "Edit": "edit",
    "Write": "edit",
    "Bash": "run",
    "WebFetch": "web",
    "WebSearch": "web",
    "Agent": "agent",
    "Team": "agent",
}

LOGO = "◌"  # the agent loop: a circle with a gap (the README's mark)
BULLET = "●"
ELBOW = "⎿"
PROMPT = "❯"
DONE, FAILED, DENIED = "✓", "✗", "⊘"
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

THEME = Theme(
    {
        "brand": f"bold {BRAND}",
        "muted": MUTED,
        "subtle": SUBTLE,
        "ok": SUCCESS,
        "warn": WARNING,
        "err": ERROR,
        "tool": "bold",
        **{f"tool.{kind}": f"bold {color}" for kind, color in TOOL_COLORS.items()},
        "path": "underline",
        "thinking": f"italic {MUTED}",
        "diff.add": ADDED,
        "diff.del": REMOVED,
        "key": f"bold {BRAND_2}",
        "highlight": f"{ON_HIGHLIGHT} on {HIGHLIGHT}",
        # Markdown tweaks for model output.
        "markdown.code": f"bold {BRAND_2}",
        "markdown.link": f"underline {BRAND_2}",
        "markdown.item.bullet": BRAND,
        "markdown.item.number": BRAND,
        "markdown.block_quote": MUTED,
        "markdown.hr": SUBTLE,
        "markdown.h1": f"bold {BRAND}",
        "markdown.h2": f"bold {BRAND}",
        "markdown.h3": "bold",
        "markdown.h4": "bold",
    }
)

# -- permission modes -----------------------------------------------------------

MODE_STYLE = {
    PermissionMode.ASK: MUTED,
    PermissionMode.DEFAULT: BRAND,
    PermissionMode.PLAN: BRAND_2,
    PermissionMode.AUTO: WARNING,
    PermissionMode.BYPASS: ERROR,
}

MODE_LABEL = {
    PermissionMode.ASK: "ask",
    PermissionMode.DEFAULT: "default",
    PermissionMode.PLAN: "plan",
    PermissionMode.AUTO: "auto",
    PermissionMode.BYPASS: "bypass",
}

MODE_HINT = {
    PermissionMode.ASK: "asks before every edit and command",
    PermissionMode.DEFAULT: "works in this project and the sandbox; asks before anything else",
    PermissionMode.PLAN: "read-only: looks, doesn't touch",
    PermissionMode.AUTO: "works on its own; asks before risky actions",
    PermissionMode.BYPASS: "routine prompts off; hard boundaries still ask",
}

# -- prompt_toolkit -------------------------------------------------------------

PT_STYLE = Style.from_dict(
    {
        "prompt": f"{BRAND} bold",
        "placeholder": SUBTLE,
        "continuation": SUBTLE,
        "bottom-toolbar": "noreverse bg:default",
        "toolbar": MUTED,
        "toolbar.dim": SUBTLE,
        "toolbar.key": f"{MUTED} bold",
        "toolbar.warn": f"{WARNING} bold",
        "toolbar.ok": SUCCESS,
        "toolbar.err": ERROR,
        "toolbar.brand": BRAND_2,
        "completion-menu": f"bg:{MENU_BG} {MENU_FG}",
        "completion-menu.completion.current": f"bg:{BRAND} {ON_BRAND} bold",
        "completion-menu.meta.completion": f"bg:{MENU_BG} {MUTED}",
        "completion-menu.meta.completion.current": f"bg:{BRAND} {ON_BRAND}",
        # While typing a search: the rest of each name stays readable (prompt_toolkit's
        # default is a near-invisible #444444) and the typed letters stand out.
        "completion-menu.completion fuzzymatch.outside": MENU_FG,
        "completion-menu.completion fuzzymatch.inside": f"{BRAND_2} bold",
        "completion-menu.completion.current fuzzymatch.outside": ON_BRAND,
        "completion-menu.completion.current fuzzymatch.inside": f"{ON_BRAND} bold",
        # Shared searchable pickers and approval prompts.
        "palette.selected": f"{BRAND} bold",
        "palette.detail": SUBTLE,
        "selected-option": f"{BRAND} bold",
        "number": SUBTLE,
    }
)


# -- helpers --------------------------------------------------------------------


def wordmark(text: str = "tinyharness") -> Text:
    """The name as the README sets it: "tiny" under a highlighter, the rest in ink. Any
    other text (a greeting) gets the ink alone."""
    if text.startswith("tiny"):
        return Text.assemble(("tiny", f"bold {ON_HIGHLIGHT} on {HIGHLIGHT}"), (text[4:], f"bold {BRAND}"))
    return Text(text, style=f"bold {BRAND}")


def percent(fraction: float) -> str:
    """A context-use percentage that never looks stuck at 0% on a huge window:
    "0%", "<0.1%", "0.5%", "3.2%", "45%"."""
    if fraction <= 0:
        return "0%"
    if fraction < 0.001:
        return "<0.1%"
    if fraction < 0.1:
        return f"{fraction:.1%}"
    return f"{fraction:.0%}"


def split_summary(summary: str) -> tuple[str, str]:
    """ "Edit(src/app.py, -1/+4 lines)" -> ("Edit", "src/app.py, -1/+4 lines")."""
    name, paren, rest = summary.partition("(")
    if not paren:
        return summary, ""
    return name, rest[:-1] if rest.endswith(")") else rest


def tool_style(name: str) -> str:
    kind = TOOL_KIND.get(name)
    return f"tool.{kind}" if kind else "tool"


def ellipsize(text: str, width: int) -> str:
    """Shorten from the middle, so both ends of a path stay: src/…/handlers/auth.py."""
    if len(text) <= width:
        return text
    if width <= 1:
        return "…"[:width]
    head = (width - 1) // 2
    return f"{text[:head]}…{text[len(text) - (width - 1 - head) :]}"


def file_link(path: str | Path) -> RichStyle:
    """A clickable link to a file, for terminals that support them (others show plain text)."""
    resolved = Path(path).expanduser().resolve()
    return RichStyle(link=f"file://{quote(str(resolved))}")


def tool_line(summary: str, bullet_style: str) -> Text:
    """A tool call as one line: `● Edit  src/app.py, -1/+4 lines`."""
    name, detail = split_summary(summary)
    line = Text()
    line.append(f"{BULLET} ", style=bullet_style)
    line.append(name, style=tool_style(name))
    if detail:
        line.append(f"  {detail}", style="muted")
    return line
