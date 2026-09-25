"""Rendering the model's streamed Markdown.

Re-rendering the whole reply on every chunk flickers and breaks once the reply is
taller than the terminal. Instead the stream is split at block boundaries (blank
lines outside code fences): finished blocks are printed once and never touched
again, and only the block still being written stays live.
"""

from __future__ import annotations

from typing import ClassVar

from rich.console import Console, ConsoleOptions, RenderableType, RenderResult
from rich.markdown import Heading, Markdown
from rich.segment import Segment
from rich.text import Text

CODE_THEME = "ansi_dark"  # follows the terminal's own palette


class _LeftHeading(Heading):
    """rich centers some headings; in a chat transcript they read better left-aligned.
    Their colors come from the theme (markdown.h1, markdown.h2, ...)."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        text = self.text.copy()
        text.justify = "left"
        yield text


class ChatMarkdown(Markdown):
    elements: ClassVar = {**Markdown.elements, "heading_open": _LeftHeading}

    def __init__(self, text: str) -> None:
        super().__init__(text, code_theme=CODE_THEME, hyperlinks=True)


def split_finished(text: str) -> tuple[str, str]:
    """Split streamed Markdown into (finished blocks, block still being written)."""
    cut = 0
    in_fence = False
    pos = 0
    for line in text.splitlines(keepends=True):
        pos += len(line)
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            in_fence = not in_fence
        elif not stripped and not in_fence and line.endswith("\n"):
            cut = pos
    return text[:cut], text[cut:]


class _Trimmed:
    """rich ends some Markdown blocks with a blank line; spacing is the renderer's job."""

    def __init__(self, renderable: ChatMarkdown) -> None:
        self.renderable = renderable

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        lines = console.render_lines(self.renderable, options, pad=False)
        while lines and _blank(lines[-1]):
            lines.pop()
        while lines and _blank(lines[0]):
            lines.pop(0)
        for line in lines:
            yield from line
            yield Segment.line()


def _blank(line: list[Segment]) -> bool:
    return not "".join(seg.text for seg in line).strip()


def render(text: str) -> RenderableType:
    return _Trimmed(ChatMarkdown(text)) if text.strip() else Text()
