"""The CLI's look: one palette, shared by rich (output) and prompt_toolkit (input).

Change a color here and it changes everywhere.
"""

from __future__ import annotations

from prompt_toolkit.styles import Style
from rich.text import Text
from rich.theme import Theme

from tinyharness.permissions import PermissionMode

# -- palette ------------------------------------------------------------------

BRAND = "#a78bfa"  # violet: the wordmark, prompt, spinner
BRAND_2 = "#5eead4"  # mint: the other end of the wordmark gradient
MUTED = "#8b8fa3"
SUBTLE = "#5c6070"
SUCCESS = "#4ade80"
WARNING = "#fbbf24"
ERROR = "#f87171"
ADDED = "#86efac"
REMOVED = "#fca5a5"

LOGO = "◆"
BULLET = "●"
ELBOW = "⎿"
PROMPT = "❯"

THEME = Theme(
    {
        "brand": f"bold {BRAND}",
        "muted": MUTED,
        "subtle": SUBTLE,
        "ok": SUCCESS,
        "warn": WARNING,
        "err": ERROR,
        "tool": "bold",
        "path": "underline",
        "diff.add": ADDED,
        "diff.del": REMOVED,
        "key": f"bold {BRAND_2}",
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
    PermissionMode.DEFAULT: MUTED,
    PermissionMode.ACCEPT_EDITS: BRAND,
    PermissionMode.PLAN: BRAND_2,
    PermissionMode.AUTO: WARNING,
    PermissionMode.BYPASS: ERROR,
}

MODE_LABEL = {
    PermissionMode.DEFAULT: "default",
    PermissionMode.ACCEPT_EDITS: "accept edits",
    PermissionMode.PLAN: "plan",
    PermissionMode.AUTO: "auto",
    PermissionMode.BYPASS: "bypass",
}

MODE_HINT = {
    PermissionMode.DEFAULT: "asks before edits",
    PermissionMode.ACCEPT_EDITS: "edits in this project run without asking",
    PermissionMode.PLAN: "read-only: looks, doesn't touch",
    PermissionMode.AUTO: "a classifier approves safe actions",
    PermissionMode.BYPASS: "never asks",
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
        "completion-menu": "bg:#1f2130 #c9cbd6",
        "completion-menu.completion.current": f"bg:{BRAND} #111111 bold",
        "completion-menu.meta.completion": "bg:#1f2130 #8b8fa3",
        "completion-menu.meta.completion.current": f"bg:{BRAND} #111111",
        # While typing a search: the rest of each name stays readable (prompt_toolkit's
        # default is a near-invisible #444444) and the typed letters stand out.
        "completion-menu.completion fuzzymatch.outside": "#c9cbd6",
        "completion-menu.completion fuzzymatch.inside": f"{BRAND_2} bold",
        "completion-menu.completion.current fuzzymatch.outside": "#111111",
        "completion-menu.completion.current fuzzymatch.inside": "#111111 bold",
        # The approval prompt (ChoiceInput).
        "selected-option": f"{BRAND} bold",
        "number": SUBTLE,
    }
)


# -- helpers --------------------------------------------------------------------


def gradient(text: str, start: str = BRAND, end: str = BRAND_2, bold: bool = True) -> Text:
    """Color `text` with a left-to-right gradient between two hex colors."""
    a, b = _rgb(start), _rgb(end)
    out = Text()
    steps = max(len(text) - 1, 1)
    for i, char in enumerate(text):
        t = i / steps
        r, g, bl = (round(x + (y - x) * t) for x, y in zip(a, b, strict=True))
        out.append(char, style=f"{'bold ' if bold else ''}#{r:02x}{g:02x}{bl:02x}")
    return out


def _rgb(color: str) -> tuple[int, int, int]:
    h = color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


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


def tool_line(summary: str, bullet_style: str) -> Text:
    """A tool call as one line: `● Edit  src/app.py, -1/+4 lines`."""
    name, detail = split_summary(summary)
    line = Text()
    line.append(f"{BULLET} ", style=bullet_style)
    line.append(name, style="tool")
    if detail:
        line.append(f"  {detail}", style="muted")
    return line
