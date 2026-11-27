"""The interactive read-eval-print loop.

╭──────────────────────────────────────────────────────────────╮
│ ❯ fix the failing test in tests/test_app.py                  │
╰──────────────────────────────────────────────────────────────╯
  ⏵⏵ accept edits · shift+tab to switch     sonnet-5 · sandbox · ⎇ main ✱ · 12% context · $0.42

The frame's color follows the permission mode, so what the agent may do without
asking is always visible.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import inspect
import io
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

from PIL import Image as PILImage
from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application, get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, AnyFormattedText, FormattedText, StyleAndTextTuples
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import ConditionalContainer, FormattedTextControl, HSplit, Layout, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import DynamicStyle, Style, merge_styles
from prompt_toolkit.utils import get_cwidth
from rich.console import Console, RenderableType
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

from tinyharness.agent.history import Image
from tinyharness.agent.transcript import SessionBusyError
from tinyharness.cli.commands import COMMANDS
from tinyharness.cli.markdown import render as render_markdown
from tinyharness.cli.render import Renderer, dollars, tokens, without_links
from tinyharness.cli.theme import (
    MODE_HINT,
    MODE_LABEL,
    MODE_STYLE,
    PROMPT,
    PT_STYLE,
    SUBTLE,
    ellipsize,
    percent,
    tool_style,
)
from tinyharness.cli.widgets import pick
from tinyharness.events import (
    AssistantText,
    Done,
    Event,
    Reasoning,
    SubagentUpdate,
    TokenUsage,
    ToolInputStreaming,
    ToolUse,
)
from tinyharness.ext import CommandContext
from tinyharness.llm.client import LLMError, ProviderConnectionError
from tinyharness.llm.usage import Usage
from tinyharness.paths import auth_path, home_dir, pretty_path
from tinyharness.permissions import PermissionMode
from tinyharness.security import find_api_key
from tinyharness.session import MissingAPIKeyError, Session
from tinyharness.tools.base import ToolError
from tinyharness.tools.open import launch
from tinyharness.tools.read import load_image

log = logging.getLogger(__name__)

EXIT_CONFIRM_SECONDS = 2.0
RESIZE_SETTLE_SECONDS = 0.15  # a resize this long ago with none since: redraw for it
_CONFIRM_AUTO = "\x00confirm-auto-mode"  # returned by the prompt when Shift+Tab reaches auto mode
_PAUSE = "\x00pause"  # returned by the prompt when a question (approval) needs the keyboard
BUILD_PLAN = "Build the approved plan."  # sent for the user who picks "build it" at the hand-off
# Built-in commands run at once while the agent works (seen live: /effort "has to wait"),
# except these, which replace or rewrite the conversation the running turn is adding to:
# they run as soon as the turn ends.
_AFTER_TURN = {"/clear", "/sessions", "/compact"}
# A new model takes over from the next turn: the running one finishes on the model it began with.
_NEXT_TURN = {"/model", "/provider"}
# Commands that wait on the model run under the input box like a turn, so what's typed
# meanwhile waits for them. Seen live: a message typed during /compact landed on its
# spinner line, and Enter added a newline instead of sending it.
_IN_BACKGROUND = {"/compact"}

_MODE_ICON = {
    PermissionMode.ASK: "○",
    PermissionMode.DEFAULT: "⏵⏵",
    PermissionMode.PLAN: "⏸",
    PermissionMode.AUTO: "◑",
    PermissionMode.BYPASS: "⚠",
}


class SlashCompleter(Completer):
    """Completes `/commands` and `/skills` at the start of the input."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not is_command(text) or " " in text:
            return
        entries = [(cmd.name, cmd.help) for cmd in dict.fromkeys(COMMANDS.values())]
        entries += [(f"/{s.name}", f"skill · {s.description}") for s in self.session.skills.list()]
        entries += [(f"/{c.name}", c.help) for e in self.session.extensions for c in e.commands]
        for name, help_text in entries:
            if name.startswith(text):
                yield Completion(name, start_position=-len(text), display_meta=_short(help_text, 60))


def _cells(fragments: StyleAndTextTuples) -> int:
    return sum(get_cwidth(t) for _, t, *_ in fragments)


def _join(parts: list[tuple[int, StyleAndTextTuples]]) -> StyleAndTextTuples:
    out: StyleAndTextTuples = []
    for i, (_, items) in enumerate(parts):
        if i:
            out.append(("class:toolbar.dim", " · "))
        out.extend(items)
    return out


def git_status(root: Path) -> tuple[str, bool] | None:
    """(branch, has uncommitted changes), or None outside a repo or if git is slow."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--branch"],
            capture_output=True,
            text=True,
            timeout=1.5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    first, *changes = proc.stdout.splitlines() or [""]
    branch = first.removeprefix("## ").split("...")[0]
    if branch.startswith("No commits yet on "):
        branch = branch.removeprefix("No commits yet on ")
    return branch or "(detached)", bool(changes)


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


_COMMAND = re.compile(r"/[\w.:-]*(\s|$)")


def is_command(line: str) -> bool:
    """`/model`, `/key anthropic`: a slash and a name. A path pasted at the start of a
    message (`/Users/me/shot.png what's wrong here?`) has more slashes, and is a message."""
    return bool(_COMMAND.match(line))


@dataclass
class Aside:
    """A /btw question, and its answer once it arrives (None until then)."""

    question: str
    answer: str | None = None
    # Asked while the agent worked: it's answered from the steps finished by then. Seen
    # live: "I'm not editing any file yet" while an edit was being written.
    mid_turn: bool = False


class _CompactPrompt(PromptSession[str]):
    """The input box, which stays up while the agent works.

    Above it, `live` (the running tools, agents and spinner) is redrawn in place;
    finished output scrolls above everything (via patch_stdout). prompt_toolkit also
    stretches the input to the bottom of the terminal when there is a bottom toolbar:
    its height is capped at the text it holds so the frame hugs the input.

    The box follows the output rather than being pinned to the terminal's last rows:
    pinning needs the terminal's cursor position before every redraw, and waiting for it
    shows as a flash each time output is printed. Once the screen fills, the box is at
    the bottom anyway."""

    live: Callable[[], str] | None = None
    _active_dumb_app: Application[str] | None = None

    @contextlib.contextmanager
    def _dumb_prompt(self, message: AnyFormattedText = "") -> Iterator[Application[str]]:
        """Remember prompt-toolkit's temporary app so permission questions can close it.

        On a dumb terminal, ``PromptSession.prompt_async`` runs this application instead
        of ``self.app``. Without tracking it, ``Repl._close_input`` targets the idle app
        and waits forever for the temporary one to release the UI lock.
        """
        with super()._dumb_prompt(message) as app:
            self._active_dumb_app = app
            try:
                yield app
            finally:
                self._active_dumb_app = None

    def _create_layout(self) -> Layout:
        layout = super()._create_layout()
        text = lambda: self.live() if self.live is not None else ""  # noqa: E731
        above = ConditionalContainer(
            Window(FormattedTextControl(lambda: ANSI(text())), dont_extend_height=True, wrap_lines=False),
            filter=Condition(lambda: bool(text())),
        )
        return Layout(HSplit([above, layout.container]), focused_element=layout.current_window)

    def _create_application(self, *args: Any, **kwargs: Any) -> Application[str]:
        """Redraw once a resize settles. Seen live in iTerm2: dragging the window edge sent
        a resize every few milliseconds, and each redraw (or a spinner's, meanwhile) was
        for a size the window had already left, so old boxes piled up. Held back until the
        size stops changing, the box is erased and redrawn once, from the width it was
        drawn at to the one it ends at (`_count_reflowed_rows`)."""
        app = super()._create_application(*args, **kwargs)
        resize, render = app._on_resize, app.renderer.render
        waiting: list[asyncio.TimerHandle] = []

        def settled() -> None:
            waiting.clear()
            if not (pushed_up := _count_reflowed_rows(app)):
                resize()
                return
            # prompt_toolkit's resize, with a move back down to where the box was drawn
            # between the erase and the redraw (see `_count_reflowed_rows`). Scrolling the
            # screen down first (SD) brings the conversation down with it, so no gap opens
            # between them; the blank rows go to the top of the screen instead.
            app.renderer.erase(leave_alternate_screen=False)
            app.output.write_raw(f"\x1b[{pushed_up}T" + "\n" * pushed_up)
            app.output.flush()
            app._request_absolute_cursor_position()
            app._redraw()

        def on_resize() -> None:
            for timer in waiting:
                timer.cancel()
            waiting[:] = [asyncio.get_running_loop().call_later(RESIZE_SETTLE_SECONDS, settled)]

        def render_when_settled(app: Application[Any], layout: Layout, is_done: bool = False) -> None:
            if not waiting or is_done:
                render(app, layout, is_done)

        app._on_resize = on_resize  # type: ignore[method-assign]
        app.renderer.render = render_when_settled  # type: ignore[method-assign]
        return app

    def _get_default_buffer_control_height(self) -> Dimension:
        width = max(get_app().output.get_size().columns - 6, 10)  # frame + prompt
        lines = sum(len(line) // width + 1 for line in self.default_buffer.document.lines)
        # The completion menu opens below the cursor, inside the frame, so the frame has to
        # make room for it. prompt_toolkit reserves `reserve_space_for_menu` rows whenever a
        # menu *might* show; room for just the entries showing keeps the box from growing
        # blank rows (a longer list scrolls within the cap).
        state = self.default_buffer.complete_state
        menu = 0 if state is None or get_app().is_done else len(state.completions)
        height = lines + min(menu, self.reserve_space_for_menu)
        return Dimension(min=height, max=height, preferred=height)


def _count_reflowed_rows(app: Application[Any]) -> int:
    """Before prompt_toolkit erases the box for a narrower terminal: count the rows its last
    drawing takes now. Terminals rewrap each line to the new width, so a full-width frame
    line becomes two rows, but the erase moves up by the rows it drew. Seen live: dragging
    the window narrower left a stack of broken frame lines above the box.

    Returns how far the rewrap pushed the drawing up, for the redraw to move back down.
    Terminals differ, and both cases were measured:
    - iTerm2 keeps the bottom row fixed: rows added anywhere in the drawing push it all
      up, so only the rows above the cursor lie between it and the drawing's top. Left
      alone, the box climbed a few rows per resize until its top line went into the
      scrollback, out of the erase's reach, and came back into view on widening.
    - xterm.js (VS Code's terminal) keeps the top fixed and moves the cursor down by every
      row added. Counting those elsewhere would erase transcript lines, so this count is
      used only where the terminal says it's xterm.js."""
    renderer = app.renderer
    screen, size = renderer._last_screen, renderer._last_size
    columns = app.output.get_size().columns
    if screen is None or size is None or columns <= 0 or columns >= size.columns:
        return 0  # nothing drawn yet, or wider: nothing rewraps onto more rows
    cursor = renderer._cursor_pos
    added = []
    for y in range(screen.height):
        row = screen.data_buffer.get(y)
        used = max((x + 1 for x, cell in row.items() if cell.char.strip()), default=0) if row else 0
        added.append(max(-(-used // columns) - 1, 0))  # ceil(used / columns) rows, less the one it had
    if os.environ.get("TERM_PROGRAM") == "vscode":
        renderer._cursor_pos = Point(x=cursor.x % columns, y=cursor.y + sum(added))
        return 0
    above = sum(added[: cursor.y]) + cursor.x // columns  # the cursor's own line rewraps around it
    renderer._cursor_pos = Point(x=cursor.x % columns, y=cursor.y + above)
    return sum(added)


@cache
def _style_for(mode: PermissionMode) -> Style:
    return Style.from_dict({"frame.border": MODE_STYLE[mode], "mode": f"{MODE_STYLE[mode]} bold"})


class Repl:
    def __init__(self, session: Session, console: Console, renderer: Renderer) -> None:
        self.session = session
        self.console = console
        self.renderer = renderer
        self.running = True
        self.context_used = 0.0  # fraction of the window, refreshed after model calls and turns
        self._context_approximate = False
        self._exit_armed_until = 0.0  # a second Ctrl-C before this time exits
        self._draft = ""  # pre-filled into the next prompt (a message that wasn't sent)
        self._after_turn: list[str] = []  # commands typed mid-turn that wait for it to end
        self.expanded = False  # ctrl+o: the last output / running subagents, above the input box
        # Started with --mode auto counts as agreeing to it; otherwise it's asked once.
        self.auto_confirmed = session.mode == PermissionMode.AUTO
        self._plan_ready = False  # the last turn ended with a plan to review (hand_off_plan)
        self._last_done = ""  # why the last turn ended (events.Done.reason)
        self.auto_declined = False  # said no once: Shift+Tab skips auto mode (/mode auto still asks)
        self.git = git_status(session.workspace.root)  # refreshed after each turn, not on every redraw
        self._prompt: _CompactPrompt | None = None
        self._turn: asyncio.Task[None] | None = None  # the agent's turn, running under the input box
        self._ui = asyncio.Lock()  # who has the keyboard: the input box, or a question (approval)
        self._questions = 0  # questions waiting for the keyboard, or holding it
        self._sending_now = False  # Esc stopped the turn to send the waiting messages
        self._pasted: dict[str, Image] = {}  # images named in pasted text, taken as pasted
        self.suggestion = ""  # the likely next message, shown in the empty input box (Tab takes it)
        self._suggesting: asyncio.Task[None] | None = None
        self._offering: asyncio.Task[None] | None = None  # questions from a prose reply (_offer_questions)
        self.aside: Aside | None = None  # the /btw answer shown above the input box (Esc closes it)
        self._answering: asyncio.Task[None] | None = None
        self._live_input_tokens = 0  # estimated current call, until its TokenUsage arrives
        self._live_output_chars = 0
        self._live_tool_chars = 0
        self._last_usage_refresh = 0.0

    # -- input ----------------------------------------------------------------

    def _toolbar(self) -> StyleAndTextTuples:
        if time.monotonic() < self._exit_armed_until:
            return [("class:toolbar.warn", "  Press Ctrl+C again to exit")]
        left = self._mode_part()
        width = get_app().output.get_size().columns
        right = self._status(width - _cells(left) - 4)
        gap = max(width - _cells(left) - _cells(right) - 2, 2)
        return [*left, ("", " " * gap), *right, ("", "  ")]

    def _mode_part(self) -> StyleAndTextTuples:
        """`⏸ plan · Token refresh · shift+tab to approve`: the mode, and what the key does."""
        mode = self.session.mode
        # In plan mode with a plan written, the footer names it: leaving plan mode approves it.
        plan = self.session.plan_tool.current if mode == PermissionMode.PLAN else None
        key, hint = ("ctrl+c", " to stop") if self.turn_running else ("shift+tab", " to switch")
        if plan is not None and not self.turn_running:
            hint = " to approve"
        left: StyleAndTextTuples = [("class:mode", f"  {_MODE_ICON[mode]} {MODE_LABEL[mode]}")]
        if plan is not None:
            title = plan.title if len(plan.title) <= 40 else plan.title[:39] + "…"
            left += [("class:toolbar.dim", " · "), ("", title + (" (edited)" if plan.edited() else ""))]
        left += [("class:toolbar.dim", " · "), ("class:toolbar.key", key), ("class:toolbar.dim", hint)]
        return left

    def _status(self, room: int) -> StyleAndTextTuples:
        """`sonnet-5 · sandbox · ⎇ main ✱ · 38% context · 12.3k out · $0.42`, dropping the
        least important parts first when the terminal is narrow."""
        session = self.session
        parts: list[tuple[int, StyleAndTextTuples]] = []  # (keep priority: lower is dropped first, items)
        model = f"{session.config.provider_info.label} / {session.model_label}"
        if session.config.effort:
            model += f" · {session.config.effort}"
        parts.append((5, [("class:toolbar.dim", model)]))
        if session.sandbox is not None:
            parts.append((3, [("class:toolbar.ok", "sandbox")]))
        elif "Bash" in session.tools.names():
            parts.append((4, [("class:toolbar.warn", "sandbox off")]))
        if running := len(session.jobs.running()):
            parts.append(
                (4, [("class:toolbar.brand", f"{running} job{'s' if running != 1 else ''} · /jobs")])
            )
        if self.git is not None:
            branch, dirty = self.git
            parts.append(
                (2, [("class:toolbar.brand", f"⎇ {branch}"), ("class:toolbar.warn", " ✱" if dirty else "")])
            )
        level = (
            "toolbar.err"
            if self.context_used >= 0.8
            else "toolbar.warn"
            if self.context_used >= 0.5
            else "toolbar.dim"
        )
        live = bool(self._live_input_tokens or self._live_output_chars)
        context_approx = "~" if self._context_approximate or live else ""
        live_approx = "~" if live else ""
        parts.append((6, [(f"class:{level}", f"{context_approx}{percent(self.context_used)} context")]))
        meter = session.ctx.usage
        live_output = self._live_output_tokens
        if meter.total.output or live_output:  # what the model wrote; /cost has the full breakdown
            parts.append(
                (1, [("class:toolbar.dim", f"{live_approx}{tokens(meter.total.output + live_output)} out")])
            )
        cost = self._live_cost
        if cost is not None:
            parts.append((5, [("class:toolbar", f"{live_approx}{dollars(cost)}")]))
        while parts and _cells(_join(parts)) > room:
            parts.remove(min(parts, key=lambda p: p[0]))
        return _join(parts)

    def _prompt_session(self) -> _CompactPrompt:
        """Enter submits; Alt+Enter or Ctrl-J inserts a newline; Tab accepts a suggestion;
        Shift+Tab cycles the mode.
        Ctrl-C clears the input, or on an empty prompt, exits when pressed twice."""
        keys = KeyBindings()

        @keys.add("c-c")
        def _interrupt(event: KeyPressEvent) -> None:
            if event.current_buffer.text:
                event.current_buffer.reset()
            elif self._turn is not None and not self._turn.done():
                self._turn.cancel()
            elif time.monotonic() < self._exit_armed_until:
                event.app.exit(exception=EOFError)
            else:
                self._exit_armed_until = time.monotonic() + EXIT_CONFIRM_SECONDS

        @keys.add("enter")
        def _submit(event: KeyPressEvent) -> None:
            buffer = event.current_buffer
            if buffer.complete_state and buffer.complete_state.current_completion:
                buffer.apply_completion(buffer.complete_state.current_completion)
            buffer.validate_and_handle()

        @keys.add("escape", filter=Condition(lambda: self.aside is not None))
        def _close_aside(event: KeyPressEvent) -> None:
            """Esc closes a /btw answer first (a second Esc then sends waiting messages)."""
            self.close_aside()

        @keys.add(
            "escape",
            filter=Condition(
                lambda: self.aside is None and self.turn_running and bool(self.session.pending_steering)
            ),
        )
        def _send_now(event: KeyPressEvent) -> None:
            """Stop the agent's current step; the waiting messages go to it at once."""
            self._sending_now = True
            if self._turn is not None:
                self._turn.cancel()

        @keys.add("escape", "/")
        def _escape_then_slash(event: KeyPressEvent) -> None:
            """Esc, then / before Esc's wait for a second key ran out (Alt+key arrives as
            Esc and the key). Seen in review: prompt_toolkit's Meta+/ took both keys, so
            "/cost" went to the agent as "cost". Esc does what it does, then / is typed."""
            if self.aside is not None:
                _close_aside(event)
            elif self.turn_running and self.session.pending_steering:
                _send_now(event)
            event.current_buffer.insert_text("/")

        @keys.add("escape", "enter")
        @keys.add("c-j")
        def _newline(event: KeyPressEvent) -> None:
            event.current_buffer.insert_text("\n")

        @keys.add("tab")
        def _accept(event: KeyPressEvent) -> None:
            """Tab takes the gray suggestion (the likely next message in an empty box, else
            the rest of an earlier message), or the highlighted (else first) menu entry."""
            buffer = event.current_buffer
            if not buffer.text and self.suggestion and not self.turn_running:
                self.session.record_suggestion("accepted", chars=len(self.suggestion))
                buffer.insert_text(self.suggestion)
            elif buffer.suggestion and buffer.suggestion.text:
                buffer.insert_text(buffer.suggestion.text)
            elif buffer.complete_state:
                completion = buffer.complete_state.current_completion or next(
                    iter(buffer.complete_state.completions), None
                )
                if completion:
                    buffer.apply_completion(completion)
            elif is_command(buffer.text):
                buffer.start_completion(select_first=False)

        @keys.add("c-b", filter=Condition(lambda: self.session.jobs.has_foreground()))
        def _to_background(event: KeyPressEvent) -> None:
            """Move the running command to the background: it keeps going as a job, and the
            agent carries on (Ctrl+B otherwise moves the cursor back)."""
            self.session.jobs.detach_foreground()

        @keys.add("c-g", filter=Condition(lambda: self.renderer.offer is not None))
        def _open_offer(event: KeyPressEvent) -> None:
            """Open what the agent offered. Seen in review: a bare "o" on an empty box ate the
            first letter of messages like "ok, now…"; a chord can't be typed by accident."""
            if (target := self.renderer.take_offer()) is not None:
                event.app.create_background_task(self._launch(target))

        @keys.add(Keys.BracketedPaste)
        def _paste(event: KeyPressEvent) -> None:
            """Paste (a dragged-in file arrives as one), taking any images it names right away."""
            text = event.data.replace("\r\n", "\n").replace("\r", "\n")
            event.current_buffer.insert_text(text)
            self._pasted.update(capture_pasted_images(text, self.session.workspace.root))

        @keys.add("c-t")
        def _toggle_thinking(event: KeyPressEvent) -> None:
            """Show or hide the model's reasoning (turning it on shows the latest)."""
            self.show_thinking(not self.renderer.show_thinking)

        @keys.add("c-o")
        def _toggle_output(event: KeyPressEvent) -> None:
            """Expand a running subagent's calls, or the last tool call's output, above the
            input box; press again to close it (it leaves nothing in the scrollback)."""
            self.expanded = not self.expanded and self._has_expansion()
            event.app.invalidate()

        @keys.add("s-tab")
        def _cycle_mode(event: KeyPressEvent) -> None:
            mode = self.next_mode()
            if mode == PermissionMode.AUTO and not self.auto_confirmed:
                # Leave the prompt to ask; what was typed comes back afterwards.
                self._draft = event.current_buffer.text
                event.app.exit(result=_CONFIRM_AUTO)
                return
            if notice := self.session.set_mode(mode):
                self.console.print(f"\n  [ok]✓[/] {notice}\n")

        history_file = home_dir() / "repl_history"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        typing_command = Condition(lambda: is_command(prompt.default_buffer.text))
        prompt = _CompactPrompt(
            message=FormattedText([("class:prompt", f"{PROMPT} ")]),
            multiline=True,
            prompt_continuation="  ",
            key_bindings=keys,
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=SlashCompleter(self.session),
            complete_while_typing=typing_command,
            reserve_space_for_menu=6,
            placeholder=lambda: FormattedText(self._placeholder()),
            bottom_toolbar=self._toolbar,
            refresh_interval=0.1,  # the live region's spinner, and the "press again to exit" hint
            erase_when_done=True,
            style=DynamicStyle(lambda: merge_styles([PT_STYLE, _style_for(self.session.mode)])),
        )
        prompt.live = self._live
        return prompt

    def _live(self) -> str:
        size = get_app().output.get_size()
        parts = (
            [self.renderer.live_ansi(size.columns), self._pending(size.columns)] if self.turn_running else []
        )
        if self.expanded:
            used = sum(part.count("\n") + 1 for part in parts if part)
            # Leave room for the input box and toolbar below (about 8 rows).
            parts.append(self._expansion_ansi(size.columns, max(size.rows - used - 8, 6)))
        if self.aside is not None:
            used = sum(part.count("\n") + 1 for part in parts if part)
            parts.append(self._aside_ansi(self.aside, size.columns, max(size.rows - used - 8, 6)))
        return "\n".join(part for part in parts if part)

    def _has_expansion(self) -> bool:
        return bool(self._running_agents()) or self.renderer.last_output is not None

    def _running_agents(self) -> list[SubagentUpdate]:
        return [agent for agent in self.renderer._agents.values() if agent.log]

    def _expansion_ansi(self, width: int, max_lines: int) -> str:
        """Ctrl+O's view, fitted to `max_lines`: running subagents' latest calls (newest
        kept), else the last tool call's output (its start kept). /output prints it all."""
        if agents := self._running_agents():
            title = " · ".join(f"{a.agent}: {' '.join(a.label.split())}" for a in agents)
            body = Text("\n").join(self._subagent_calls(a, width - 4) for a in agents)
            keep_end = True
        elif self.renderer.last_output is not None:
            summary, output = self.renderer.last_output
            title = " ".join(self.renderer._relative(summary).split())
            body = Text(output.rstrip() or "(no output)")
            keep_end = False
        else:
            self.expanded = False
            return ""
        lines: list[Text] = list(body.split("\n"))
        room = max(max_lines - 2, 3)  # the panel's border takes two rows
        if len(lines) > room:
            hidden = len(lines) - room + 1
            note = Text(
                f"… {hidden} {'earlier' if keep_end else 'more'} lines (/output prints it all)",
                style="subtle",
            )
            lines = [note, *lines[-(room - 1) :]] if keep_end else [*lines[: room - 1], note]
        panel = Panel(
            Text("\n").join(lines),
            title=Text(ellipsize(title, width - 30), style="bold"),
            title_align="left",
            subtitle=Text("ctrl+o to close", style="subtle"),
            subtitle_align="right",
            border_style="subtle",
            padding=(0, 1),
        )
        with self.console.capture() as captured:
            self.console.print(panel, width=width, highlight=False)
        return without_links(captured.get().rstrip("\n"))

    def _pending(self, width: int) -> str:
        """Messages waiting for the next step, and how to send them immediately."""
        pending = self.session.pending_steering
        if not pending:
            return ""
        text = Text("• ", style="subtle").append("Sent after the agent's current step ", style="muted")
        text.append("(esc: stop it and send now)", style="subtle")
        for message in pending:
            text.append("\n  ↳ ", style="subtle").append(ellipsize(" ".join(message.split()), width - 6))
        with self.console.capture() as captured:
            self.console.print(text, width=width, highlight=False)
        return captured.get().rstrip("\n")

    def _placeholder(self) -> StyleAndTextTuples:
        if self.turn_running:
            return [(SUBTLE, "Message the agent: it gets this before its next step")]
        if self.suggestion:
            return [(SUBTLE, self.suggestion)]
        return [(SUBTLE, "Ask anything, or type / for commands")]

    # -- side calls -------------------------------------------------------------

    def ask_aside(self, question: str) -> None:
        """/btw: answered in the background, shown above the input box until Esc, so the
        input box and a running turn carry on and nothing is left in the scrollback."""
        self.close_aside()
        aside = Aside(question, mid_turn=self.turn_running)
        if self._prompt is not None:
            self.aside = aside
        else:  # no input box to show it above: print it when it comes
            self.console.print(Text("  ⎿ answering on the side…", style="subtle"))

        async def answer() -> None:
            try:
                aside.answer = await self.session.aside(question)
            except Exception as e:
                if self.aside is aside:
                    self.aside = None
                report_error(self.console, e)
                return
            if self._prompt is None:
                self.console.print(self._aside_panel(aside, render_markdown(aside.answer), "not kept"))
            elif self._prompt.app.is_running:
                self._prompt.app.invalidate()

        self._answering = asyncio.create_task(answer())

    def close_aside(self) -> None:
        """Esc, or the next message: the /btw answer goes, unanswered or not."""
        self.aside = None
        if self._answering is not None and not self._answering.done():
            self._answering.cancel()
        self._answering = None

    def show_aside(self) -> bool:
        """/output while a /btw answer is open: print it in full, and close it."""
        aside = self.aside
        if aside is None or aside.answer is None:
            return False
        self.console.print(self._aside_panel(aside, render_markdown(aside.answer), "not kept"))
        self.close_aside()
        return True

    def _aside_panel(self, aside: Aside, body: RenderableType, note: str) -> Panel:
        return Panel(
            body,
            title=Text(f"btw: {ellipsize(' '.join(aside.question.split()), 60)}", style="bold"),
            title_align="left",
            subtitle=Text(note, style="subtle"),
            subtitle_align="right",
            border_style="subtle",
            padding=(0, 1),
        )

    def _aside_ansi(self, aside: Aside, width: int, max_lines: int) -> str:
        """The /btw panel above the input box, fitted to `max_lines` (its start kept)."""
        if aside.answer is None:
            body: RenderableType = Text("answering…", style="subtle")
            note = "esc to cancel"
        else:
            with self.console.capture() as captured:
                self.console.print(render_markdown(aside.answer), width=width - 4)
            lines: list[Text] = list(Text.from_ansi(captured.get().rstrip("\n")).split("\n"))
            room = max(max_lines - 2, 3)  # the panel's border takes two rows
            if len(lines) > room:
                lines = [*lines[: room - 1], Text(f"… {len(lines) - room + 1} more lines", style="subtle")]
                note = "esc to close · /output prints it all"
            else:
                note = "esc to close · not added to the conversation"
            if aside.mid_turn:
                note = f"as of its last finished step · {note}"
            body = Text("\n").join(lines)
        with self.console.capture() as captured:
            self.console.print(self._aside_panel(aside, body, note), width=width, highlight=False)
        return without_links(captured.get().rstrip("\n"))

    def _offer_questions(self) -> None:
        """A plan-mode turn that ended asking in prose: the same picker Ask shows, once the
        questions are read (`Session.questions_in_reply`), and the picks go as the next
        message. Read in the background: typing or sending something first skips it, and
        with no questions found the next message is suggested as usual."""

        async def offer() -> None:
            try:
                questions = await self.session.questions_in_reply()
            except Exception as e:  # a missed picker isn't worth an error panel
                log.debug("no questions read from the reply: %s", e)
                questions = []
            typed = self._prompt is not None and bool(self._prompt.default_buffer.text.strip())
            if not questions or typed or self.turn_running or not self.running:
                if not questions and not typed and not self.turn_running:
                    self._suggest()
                return
            answers = await self.renderer.ask_questions(questions)
            if answers is None or self.turn_running:
                return  # Esc: they'll answer in chat
            line = "\n".join(f"{q.header}: {a}" for q, a in zip(questions, answers, strict=True))
            self._echo(line)
            await self._run(self.session.run(line))

        self._offering = asyncio.create_task(offer())

    def _suggest(self) -> None:
        """Once a turn ends: guess the next message, to show in the empty input box."""
        if not self.session.config.prompt_suggestions or not self.running:
            return

        async def suggest() -> None:
            try:
                self.suggestion = await self.session.suggest_next()
                if self.suggestion:
                    self.session.record_suggestion("shown", chars=len(self.suggestion))
            except Exception as e:  # a missing suggestion isn't worth an error panel
                log.debug("no prompt suggestion: %s", e)
                return
            if self._prompt is not None and self._prompt.app.is_running:
                self._prompt.app.invalidate()

        self._suggesting = asyncio.create_task(suggest())

    def _forget_suggestion(self) -> None:
        self.suggestion = ""
        if self._suggesting is not None:
            self._suggesting.cancel()
            self._suggesting = None

    @property
    def turn_running(self) -> bool:
        return self._turn is not None and not self._turn.done()

    @contextlib.asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        """Take the keyboard from the input box (for a permission prompt), keeping what was
        typed so far; the input box comes back afterwards.

        A question can arrive while the input box is between runs: just closed for the
        previous question, its next run about to start. Closing it then does nothing, so
        the count below also stops the loop from opening it, and a run that starts anyway
        closes itself at once (`_make_way`). Otherwise the question waits for a keyboard
        the new input box holds, and both wait forever."""
        self._questions += 1
        try:
            self._close_input()
            async with self._ui:
                yield
        finally:
            self._questions -= 1

    def _close_input(self) -> None:
        """Close the input box if it's up, keeping what was typed."""
        prompt = self._prompt
        app = (prompt._active_dumb_app or prompt.app) if prompt is not None else None
        if app is None or app.future is None or app.future.done():
            return  # between runs: the next one sees the waiting question (_make_way)
        self._draft = app.current_buffer.text
        app.exit(result=_PAUSE)

    def _make_way(self) -> None:
        """At the start of each input box run: step aside if a question is waiting."""
        if self._questions:
            self._close_input()

    def _echo(self, line: str, *, queued: bool = False) -> None:
        """Print the submitted message back into the transcript (the input box is erased)."""
        first, *rest = line.splitlines()
        text = Text().append(f"{PROMPT} ", style="brand").append(first, style="bold")
        for more in rest:
            text.append(f"\n  {more}", style="bold")
        if queued and not is_command(line):
            text.append("  · for the agent's next step", style="subtle")
        self.console.print(text, highlight=False)

    async def loop(self) -> None:
        """The input box stays up the whole time: turns run under it, and anything typed
        while one runs is passed to the agent."""
        self._prompt = prompt = self._prompt_session()
        self.renderer.embedded = True  # the prompt draws the live region now
        self.renderer.can_detach = self.session.jobs.has_foreground
        self.renderer.can_offer = True
        self.renderer.exclusive = self.exclusive
        with patch_stdout(raw=True):  # output scrolls above the input box
            plan = self.session.plan_tool.current
            if self.session.resume_report is not None and plan is not None and not plan.approved:
                await self.hand_off_plan()  # left while it waited for review: ask again
            while self.running:
                async with self._ui:
                    if self._questions:
                        continue  # a question is waiting: it gets the keyboard first
                    try:
                        # show_frame must be passed here: prompt_async() resets it (prompt_toolkit 3.0.52).
                        draft, self._draft = self._draft, ""
                        line = await prompt.prompt_async(
                            show_frame=True, default=draft, pre_run=self._make_way
                        )
                        line = line.strip()
                    except (EOFError, KeyboardInterrupt):
                        break  # Ctrl-D, or Ctrl-C twice
                    if line == _PAUSE:
                        continue  # a question needs the keyboard; it's released on the way round
                    if line == _CONFIRM_AUTO:
                        await self.shift_tab_to_auto()
                    elif line:
                        self._echo(line, queued=self.turn_running)
                        await self.handle(line)
            if self._turn is not None and not self._turn.done():
                self._turn.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._turn
            self.close_aside()
            self._forget_suggestion()
        self.console.print(Text("\n  bye 👋\n", style="muted"))

    async def handle(self, line: str) -> None:
        if not line.startswith("/btw"):
            self._forget_suggestion()
            if not line.startswith("/output"):
                self.close_aside()
        if self.turn_running:
            await self._while_running(line)
            return
        if not is_command(line):
            if await self._safe_to_send(line):
                text, images = self._attach(line)
                await self._run(self.session.run(text, images))
            return

        name, _, arg = line.partition(" ")
        arg = arg.strip()
        if name in _IN_BACKGROUND and name in COMMANDS and self._prompt is not None:
            self.console.print()
            self._turn = asyncio.create_task(self._command_in_background(line))
        elif cmd := COMMANDS.get(name):
            self.console.print()
            result = cmd.handler(self, arg)
            if inspect.isawaitable(result):
                await result
        elif ext_cmd := self.session.extension_command(name[1:]):
            self.console.print()
            context = CommandContext(
                say=lambda text: self.console.print(Text(f"  {text}")),
                send=lambda text: self._run(self.session.run(text)),
                session=self.session,
            )
            result = ext_cmd.handler(arg, context)
            if inspect.isawaitable(result):
                await result
        elif self.session.skills.get(name[1:]):
            if await self._safe_to_send(line):
                await self._run(self.session.run_skill(name[1:], arg))
        else:
            self.console.print(
                f"\n  Unknown command [bold]{escape(name)}[/]. Type [key]/[/] to see what's available."
            )

    async def _launch(self, target: str) -> None:
        """Open an offered artifact; if that fails, say so where the user is looking."""
        try:
            await launch(target)
        except ToolError as e:
            self.console.print(f"  [warn]Couldn't open it: {escape(str(e))}[/]")
        else:
            self.renderer.opened_offer(target)

    def show_thinking(self, show: bool) -> None:
        self.renderer.show_thinking = show
        self.session.set_show_thinking(show)
        # Reasoning already printed is in the terminal's scrollback, which nothing can take
        # back; the switch is for what comes next, including a thought still being written.
        if not show:
            self.console.print("  [muted]∴ Reasoning hidden from here on (ctrl+t shows it again)[/]")
            return
        self.console.print("  [muted]∴ Reasoning shown from here on (ctrl+t hides it again)[/]")
        # The latest reasoning, if it was hidden when it happened; once is enough.
        renderer = self.renderer
        if renderer.last_thought and not renderer.last_thought_shown and not self.turn_running:
            self.console.print(Padding(Text(renderer.last_thought, style="thinking"), (0, 0, 0, 2)))
            renderer.last_thought_shown = True

    def show_last_output(self) -> None:
        """/output: a running subagent's calls so far, if one is running; else the last tool
        call's full output, printed in full (ctrl+o shows the same, fitted to the screen)."""
        if running := [agent for agent in self.renderer._agents.values() if agent.log]:
            for agent in running:
                self._show_subagent(agent)
            return
        if self.renderer.last_output is None:
            return
        summary, output = self.renderer.last_output
        self.console.print()
        self.console.print(
            Panel(
                Text(output.rstrip() or "(no output)"),
                title=Text(self.renderer._relative(summary), style="bold"),
                title_align="left",
                border_style="subtle",
                padding=(0, 1),
            ),
            highlight=False,
        )

    def switch_session(self, session_id: str) -> bool:
        """Atomically replace the idle conversation, keeping the old one on any error."""
        if session_id == self.session.id:
            self.console.print("  [muted]That conversation is already open here.[/]")
            return False
        old = self.session
        options = replace(
            old.options,
            working_dir=old.workspace.root,
            provider=None,
            model=None,
            permission_mode=old.mode,
            resume=session_id,
            trust_project=old.project_trusted,
            use_last_model=True,
        )
        try:
            selected = Session(options)
        except SessionBusyError as error:
            self.console.print(Text(f"  ✗ {error}", style="err"))
            return False
        except Exception as error:
            report_error(self.console, error)
            return False

        self.session = selected
        old.close()
        self.renderer.use_session(
            str(selected.workspace.root),
            selected.ctx.usage,
            show_thinking=selected.config.show_thinking,
            scratch=str(selected.scratch.dir),
        )
        self.context_used = 0.0
        self.git = git_status(selected.workspace.root)
        self.expanded = False
        self.close_aside()
        self._forget_suggestion()
        if self._prompt is not None:
            self._prompt.completer = SlashCompleter(selected)
        from tinyharness.cli.app import print_resumed

        print_resumed(self.console, selected)
        self._measure_context()
        return True

    def _show_subagent(self, agent: SubagentUpdate) -> None:
        """What a running subagent has done so far, printed in full (/output)."""
        body = self._subagent_calls(agent, self.console.width)
        title = f"{agent.agent} · {agent.label} · {agent.tools_used} tools so far"
        self.console.print()
        self.console.print(
            Panel(
                body or Text("(no tool calls yet)", style="muted"),
                title=Text(title, style="bold"),
                title_align="left",
                border_style="subtle",
                padding=(0, 1),
            ),
            highlight=False,
        )

    def _subagent_calls(self, agent: SubagentUpdate, width: int) -> Text:
        """Each call a subagent made and the start of its result, from its log (written as
        it goes)."""
        body = Text()
        for line in _read_lines(Path(agent.log)):
            if line.get("event") == "tool_use":
                tool = str(line.get("tool", "?"))
                body.append("\n" if body else "").append(tool, style=tool_style(tool))
                body.append(
                    f"  {ellipsize(_main_arg(line.get('args')), width - len(tool) - 8)}", style="muted"
                )
            elif line.get("event") == "tool_result":
                text = str(line.get("error") or line.get("result") or "")
                style = "err" if line.get("error") else "subtle"
                for out in [out for out in text.splitlines() if out.strip()][:3]:
                    body.append(f"\n  ⎿ {ellipsize(out.strip(), width - 10)}", style=style)
        return body

    def next_mode(self) -> PermissionMode:
        """Where Shift+Tab goes: the next mode, skipping auto once the user has said no to it."""
        mode = self.session.mode.next()
        if mode == PermissionMode.AUTO and self.auto_declined:
            mode = mode.next()
        return mode

    async def shift_tab_to_auto(self) -> None:
        """Shift+Tab reached auto mode for the first time: ask. A "no" keeps the current
        mode, and later Shift+Tabs go past auto instead of asking again."""
        if not await self.enter_auto_mode():
            self.auto_declined = True
            self.console.print(
                "  [muted]Auto mode stays off; Shift+Tab skips it now. [/][key]/mode auto[/][muted] turns it on.[/]\n"
            )

    async def enter_auto_mode(self) -> bool:
        """Switch to auto mode, asking once per session first. False if the user said no."""
        if not self.auto_confirmed:
            self.console.print(
                "\n  [bold]Auto mode[/] works without asking: reading, searching and editing files in "
                "this project, and reading public web pages.\n"
                "  It still asks before anything risky: changing files outside the project or in "
                ".git, replacing most of a file, deleting a large block. Unclear cases go to a "
                "quick safety check.\n"
            )
            choice = await pick(
                "Turn on auto mode for this session?",
                [("yes", "Yes, work on its own"), ("no", "No, keep asking")],
                "yes",
                mark_current=False,
            )
            if choice != "yes":
                return False
            self.auto_confirmed = True
        self.session.set_mode(PermissionMode.AUTO)
        return True

    async def _safe_to_send(self, line: str) -> bool:
        """Catch an API key pasted into the chat before it reaches the model."""
        if (key := find_api_key(line, self.session.config.credential_values())) is None:
            return True
        self.console.print(
            f"\n  [warn]⚠ Your message contains what looks like an API key[/] [muted]({key})[/]\n"
            f"  [muted]Anything you send goes to the model. Add keys with[/] /key [muted](stored in[/] "
            f"{pretty_path(auth_path())}[muted], where the model can't read them)[/]\n"
        )
        choice = await pick(
            "Send it anyway?",
            [("edit", "No, let me edit it"), ("send", "Yes, send it")],
            "edit",
            mark_current=False,
        )
        if choice == "send":
            return True
        self._draft = line  # back into the input box, to edit
        return False

    def _attach(self, line: str) -> tuple[str, list[Image]]:
        """The images a message names, shown as attached, and the message as the model gets it."""
        images, problems, text = attachments(line, self.session.workspace.root, self._pasted)
        self._pasted = {}
        for image in images:
            size = len(image.data) * 3 // 4 // 1024
            self.console.print(
                Text(f"  ⎿ attached {image.name}{_dimensions(image)} · {size:,} KB", style="subtle")
            )
        for problem in problems:
            self.console.print(Text(f"  ⎿ not attached: {problem}", style="warn"))
        return text, images

    async def _while_running(self, line: str) -> None:
        """Input while the agent works: a message goes to it; built-in commands run now, or
        when the turn ends if they replace the conversation; other commands wait."""
        name = line.partition(" ")[0]
        if not is_command(line):
            if await self._safe_to_send(line):
                text, images = self._attach(line)
                self.session.steer(text, images)
            return
        if name in _AFTER_TURN and name in COMMANDS:
            self._after_turn.append(line)
            self.console.print(f"  [muted]{name} runs when the agent finishes (ctrl+c stops it now).[/]")
            return
        if name in COMMANDS:
            await self._command(line)
            if name in _NEXT_TURN:
                self.console.print("  [muted]The current turn finishes on the model it started with.[/]")
            return
        # Skills and extension commands talk to the agent, so they wait for it.
        self.console.print(f"  [muted]{name} has to wait until the agent finishes (ctrl+c stops it).[/]")
        self._draft = line

    async def _command(self, line: str) -> None:
        name, _, arg = line.partition(" ")
        result = COMMANDS[name].handler(self, arg.strip())
        if inspect.isawaitable(result):
            await result

    # -- turns ----------------------------------------------------------------

    async def _run(self, events: AsyncGenerator[Event, None]) -> None:
        """Start a turn: under the input box when the REPL is running, else in the foreground."""
        if self._prompt is None:
            await self.run_turn(events)
        else:
            self._turn = asyncio.create_task(self._turn_in_background(events))

    async def _turn_in_background(self, events: AsyncGenerator[Event, None]) -> None:
        finished = False
        try:
            await self.renderer.render(self._status_events(events))
            finished = True
        except asyncio.CancelledError:
            if self._sending_now:  # Esc: the waiting messages go to the agent right away
                self.console.print("\n[warn]■ Stopped.[/] [muted]Sending your message now.[/]")
            else:
                self.console.print("\n[warn]■ Interrupted.[/] [muted]What should it do instead?[/]")
        except Exception as e:
            report_error(self.console, e)
        finally:
            self._sending_now = False
            self._measure_context()
            self.console.print()
        await self._after_work(finished)

    async def _command_in_background(self, line: str) -> None:
        """A command that waits on the model (`_IN_BACKGROUND`), run as a turn is."""
        finished = False
        try:
            await self._command(line)
            finished = True
        except asyncio.CancelledError:
            self.console.print("\n[warn]■ Stopped.[/]")
        except Exception as e:
            report_error(self.console, e)
        await self._after_work(finished, suggest=False)

    async def _after_work(self, finished: bool, *, suggest: bool = True) -> None:
        """Once a turn (or a command run like one) ends: the commands that waited for it,
        then the messages that arrived meanwhile, else the plan hand-off or a suggestion."""
        waiting, self._after_turn = self._after_turn, []
        for line in waiting:
            if self.running:
                await self._command(line)
        if (left := self.session.take_steering()) and self.running:
            text = "\n\n".join(m.content or "" for m in left)
            images = [image for m in left for image in m.images]
            self._turn = asyncio.create_task(self._turn_in_background(self.session.run(text, images)))
        elif finished and self.running and self._plan_ready:
            await self.hand_off_plan()
        elif finished and suggest and self._last_done == "stop" and self.session.mode == PermissionMode.PLAN:
            self._offer_questions()
        elif finished and suggest:
            self._suggest()

    async def run_turn(self, events: AsyncGenerator[Event, None]) -> None:
        """Render one agent turn. Ctrl-C cancels the turn but keeps the REPL alive."""
        task = asyncio.create_task(self.renderer.render(self._status_events(events)))
        loop = asyncio.get_running_loop()
        with contextlib.suppress(NotImplementedError):  # not available on Windows
            loop.add_signal_handler(signal.SIGINT, task.cancel)
        finished = False
        try:
            await task
            finished = True
        except asyncio.CancelledError:
            self.console.print("\n[warn]■ Interrupted.[/] [muted]What should it do instead?[/]")
        except Exception as e:
            report_error(self.console, e)
        finally:
            with contextlib.suppress(NotImplementedError):
                loop.remove_signal_handler(signal.SIGINT)
            self._measure_context()
        if finished and self._plan_ready:
            await self.hand_off_plan()

    async def hand_off_plan(self) -> None:
        """A plan is ready for review: ask once whether to build it, and in which mode.

        Picking a mode approves the plan on screen and starts the build; that pick is the
        user's "build it". Esc or "keep planning" stays in plan mode, and whatever they type
        next is their feedback. Shift+Tab still approves without building."""
        self._plan_ready = False
        session = self.session
        if session.mode != PermissionMode.PLAN or session.plan_tool.current is None:
            return  # a command that waited for the turn (/clear, /mode) already moved on
        before = session.mode_before_plan
        modes = [
            m
            for m in dict.fromkeys([before, PermissionMode.AUTO, PermissionMode.DEFAULT])
            if m != PermissionMode.PLAN
        ]
        options = [(m.value, f"Yes, build it in {MODE_LABEL[m]} mode · {MODE_HINT[m]}") for m in modes]
        options.append((PermissionMode.PLAN.value, "No, keep planning"))
        # The turn's last lines (plan ready, the footer) may still be in patch_stdout's
        # queue, which writes in batches; seen live: the picker showed up above them.
        await asyncio.sleep(getattr(sys.stdout, "sleep_between_writes", 0) + 0.05)
        async with self.exclusive():
            choice = await pick("Build this plan?", options, modes[0].value, mark_current=False)
        if choice in (None, PermissionMode.PLAN.value):
            self.console.print(
                "  [muted]Still planning. Say what to change, or press shift+tab to approve it later.[/]\n"
            )
            return
        mode = PermissionMode(choice)
        if mode == PermissionMode.AUTO:
            self.auto_confirmed = True  # chosen with its description on screen
        notice = session.set_mode(mode)
        plan = session.plan_tool.current
        if plan is not None and plan.approved:
            notice = f"Plan approved · building in {MODE_LABEL[mode]} mode"
        self.console.print(f"  [ok]✓[/] {notice}")
        self._echo(BUILD_PLAN)
        await self._run(session.run(BUILD_PLAN))

    async def _status_events(self, events: AsyncGenerator[Event, None]) -> AsyncGenerator[Event, None]:
        """Keep the toolbar moving during a model call, then replace estimates with usage."""
        self._clear_live_usage()
        self._plan_ready = False
        async for event in events:
            if isinstance(event, Done):
                self._plan_ready = event.reason == "plan_ready"
                self._last_done = event.reason
            elif isinstance(event, TokenUsage):
                if self.session.context_window:
                    self.context_used = (event.input + event.cached) / self.session.context_window
                    self._context_approximate = False
                self._clear_live_usage()
                self._refresh_status(force=True)
            elif isinstance(event, AssistantText | Reasoning):
                self._start_live_usage()
                self._live_output_chars += len(event.text)
                self._refresh_live_context()
                self._refresh_status()
            elif isinstance(event, ToolInputStreaming):
                self._start_live_usage()
                # `chars` is cumulative for the current tool call, so only add its growth.
                self._live_output_chars += max(event.chars - self._live_tool_chars, 0)
                self._live_tool_chars = event.chars
                self._refresh_live_context()
                self._refresh_status()
            elif isinstance(event, ToolUse):
                self._live_tool_chars = 0
            yield event

    @property
    def _live_output_tokens(self) -> int:
        return self._live_output_chars // 4

    @property
    def _live_cost(self) -> float | None:
        base = self.session.ctx.usage.cost
        if not (self._live_input_tokens or self._live_output_chars):
            return base
        cfg = self.session.config
        info = self.session.llm.model_info(cfg.model)
        price = cfg.price(cfg.provider, cfg.model) or (info.price if info else None)
        if price is None:
            return base
        estimated = price.cost(Usage(input=self._live_input_tokens, output=self._live_output_tokens))
        return (base or 0.0) + estimated

    def _start_live_usage(self) -> None:
        if self._live_input_tokens:
            return
        with contextlib.suppress(Exception):
            self._live_input_tokens = self.session.compactor.used(self.session.history)

    def _refresh_live_context(self) -> None:
        if self.session.context_window:
            self.context_used = (
                self._live_input_tokens + self._live_output_tokens
            ) / self.session.context_window
            self._context_approximate = True

    def _clear_live_usage(self) -> None:
        self._live_input_tokens = 0
        self._live_output_chars = 0
        self._live_tool_chars = 0

    def _refresh_status(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_usage_refresh < 0.1:
            return
        self._last_usage_refresh = now
        if self._prompt is not None and self._prompt.app.is_running:
            self._prompt.app.invalidate()

    def _measure_context(self) -> None:
        self._clear_live_usage()
        with contextlib.suppress(Exception):  # the toolbar is a nicety; never fail a turn over it
            used = self.session.compactor.used(self.session.history)
            self.context_used = used / self.session.context_window
            self._context_approximate = True
        self.git = git_status(self.session.workspace.root)


def report_error(console: Console, error: Exception) -> None:
    hint = ""
    if isinstance(error, ProviderConnectionError) and not error.slow_reply:
        hint = "Check your network, VPN or proxy, then try again. /ping tests the connection to the provider."
    elif (
        isinstance(error, TimeoutError)
        or "Timeout" in type(error).__name__
        or (isinstance(error, ProviderConnectionError) and error.slow_reply)
    ):
        hint = (
            "The model took longer than `request_timeout` to answer. Files already written are kept.\n"
            "Try a faster model (/model), less reasoning (/effort low), or a larger request_timeout."
        )
    elif isinstance(error, LLMError) and "auth" in str(error):
        hint = "Check your API key with /whoami and /ping."
    body = Text(str(error) or type(error).__name__)
    if hint:
        body.append(f"\n\n{hint}", style="muted")
    if not log.isEnabledFor(logging.DEBUG) and not isinstance(error, MissingAPIKeyError):  # setup, not a bug
        body.append("\n\nRun with --debug for the full traceback.", style="subtle")
    console.print()
    console.print(
        Panel(body, title=f"[err]✗ {type(error).__name__}[/]", title_align="left", border_style="err")
    )
    if log.isEnabledFor(logging.DEBUG):
        console.print_exception()


def _read_lines(path: Path) -> list[dict[str, Any]]:
    """A JSONL log's records; a line still being written (or unreadable) is skipped."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    records = []
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _main_arg(args: object) -> str:
    """The argument that says what a call was about: its path, command, URL, query..."""
    if not isinstance(args, dict):
        return ""
    for key in ("path", "command", "url", "query", "pattern", "target", "prompt"):
        if args.get(key):
            return " ".join(str(args[key]).split())
    return " ".join(json.dumps(args).split())


_IMAGE_NAMES = (".png", ".jpg", ".jpeg", ".gif", ".webp")


def _image_words(text: str, cwd: Path) -> list[tuple[str, Path]]:
    """The words of `text` that name image files, and their paths. Words are split as a shell
    would, so a pasted `Screenshot\\ 2026-09-30\\ at\\ 3.11.03 PM.png` or a quoted
    drag-and-drop path is one path. Only names ending like an image count."""
    found = []
    for word in _words(text, windows=os.name == "nt"):
        if word.lower().endswith(_IMAGE_NAMES):
            path = Path(word).expanduser()
            found.append((word, path if path.is_absolute() else cwd / path))
    return found


def _words(text: str, *, windows: bool) -> list[str]:
    """`text` split as the terminal wrote it. A macOS or Linux terminal escapes the spaces of a
    dropped path with backslashes; a Windows one quotes the path, whose backslashes are its
    separators (seen in CI: POSIX splitting turned `C:\\Users\\shot.png` into `C:Usersshot.png`)."""
    if windows:
        words = re.findall(r'"[^"]*"|\'[^\']*\'|\S+', text)
        return [w[1:-1] if len(w) > 1 and w[0] == w[-1] and w[0] in "\"'" else w for w in words]
    try:
        return shlex.split(text)
    except ValueError:
        # An unmatched quote ("what's wrong here?"): split on spaces, but not escaped ones,
        # which a dropped file's path is full of.
        return [re.sub(r"\\(.)", r"\1", word) for word in re.findall(r"(?:\\.|\S)+", text)]


def capture_pasted_images(text: str, cwd: Path) -> dict[str, Image]:
    """The images named in pasted or dropped text, loaded now. Seen live: macOS deletes a
    screenshot dragged in from its thumbnail (a copy under /var/folders/…/TemporaryItems)
    within seconds, so by the time the message was sent the file was gone, and the model's
    attempt to Read the path was refused."""
    captured = {}
    for _, path in _image_words(text, cwd):
        with contextlib.suppress(OSError, ValueError):
            if path.is_file():
                captured[str(path)] = load_image(path)
    return captured


def attachments(
    line: str, cwd: Path, captured: dict[str, Image] | None = None
) -> tuple[list[Image], list[str], str]:
    """Image files named in a message, to show the model: (images, why others couldn't be,
    the message with each attached path replaced by a mention of its image). Images captured
    when they were pasted come first; the file's bytes decide the rest. Seen live: with the
    path left in, the model tried to Read the screenshot too, and auto review refused it."""
    images: list[Image] = []
    problems: list[str] = []
    text = line
    seen: set[Path] = set()
    for word, path in _image_words(line, cwd):
        if path in seen:
            continue
        image = (captured or {}).get(str(path))
        if image is None:
            if not path.is_file():
                if Path(word).expanduser().is_absolute():  # a file named outright: say it's gone
                    problems.append(
                        f"{path.name} isn't there any more (macOS removes a screenshot dragged "
                        "from its thumbnail within seconds); drop it in again, or save it first"
                    )
                continue
            try:
                image = load_image(path)
            except (OSError, ValueError) as e:
                problems.append(str(e))
                continue
        seen.add(path)
        images.append(image)
        text = _mention(text, word, f"[image {len(images)} attached: {image.name}{_dimensions(image)}]")
    return images, problems, text


def _mention(text: str, word: str, mention: str) -> str:
    """`text` with the path `word` replaced by `mention`, however it was written."""
    for written in (f'"{word}"', f"'{word}'", word.replace(" ", "\\ "), shlex.quote(word), word):
        if written in text:
            return text.replace(written, mention)
    return text


def _dimensions(image: Image) -> str:
    try:
        with PILImage.open(io.BytesIO(base64.b64decode(image.data))) as picture:
            return f", {picture.width}×{picture.height}"
    except (OSError, ValueError):
        return ""
