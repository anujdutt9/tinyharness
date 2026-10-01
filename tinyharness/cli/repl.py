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
import contextlib
import inspect
import json
import logging
import re
import shlex
import signal
import subprocess
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable
from functools import cache
from pathlib import Path
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, FormattedText, StyleAndTextTuples
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import ConditionalContainer, FormattedTextControl, HSplit, Layout, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import DynamicStyle, Style, merge_styles
from prompt_toolkit.utils import get_cwidth
from rich.console import Console
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

from tinyharness.agent.history import Image
from tinyharness.cli.commands import COMMANDS
from tinyharness.cli.render import Renderer, dollars, tokens, without_links
from tinyharness.cli.theme import (
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
from tinyharness.events import Event, SubagentUpdate
from tinyharness.llm.client import LLMError, ProviderConnectionError
from tinyharness.llm.tokens import count_message_tokens
from tinyharness.paths import auth_path, home_dir, pretty_path
from tinyharness.permissions import PermissionMode
from tinyharness.security import find_api_key
from tinyharness.session import MissingAPIKeyError, Session
from tinyharness.tools.read import load_image

log = logging.getLogger(__name__)

EXIT_CONFIRM_SECONDS = 2.0
_CONFIRM_AUTO = "\x00confirm-auto-mode"  # returned by the prompt when Shift+Tab reaches auto mode
_PAUSE = "\x00pause"  # returned by the prompt when a question (approval) needs the keyboard
# Commands that don't touch the conversation, so they can run while the agent works.
_SAFE_WHILE_RUNNING = {
    "/help", "/mode", "/context", "/theme", "/thinking", "/cwd", "/tools", "/skills", "/output",
}  # fmt: skip

_MODE_ICON = {
    PermissionMode.DEFAULT: "○",
    PermissionMode.ACCEPT_EDITS: "⏵⏵",
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

    def _create_layout(self) -> Layout:
        layout = super()._create_layout()
        text = lambda: self.live() if self.live is not None else ""  # noqa: E731
        above = ConditionalContainer(
            Window(FormattedTextControl(lambda: ANSI(text())), dont_extend_height=True, wrap_lines=False),
            filter=Condition(lambda: bool(text())),
        )
        return Layout(HSplit([above, layout.container]), focused_element=layout.current_window)

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


@cache
def _style_for(mode: PermissionMode) -> Style:
    return Style.from_dict({"frame.border": MODE_STYLE[mode], "mode": f"{MODE_STYLE[mode]} bold"})


class Repl:
    def __init__(self, session: Session, console: Console, renderer: Renderer) -> None:
        self.session = session
        self.console = console
        self.renderer = renderer
        self.running = True
        self.context_used = 0.0  # fraction of the context window, refreshed after each turn
        self._exit_armed_until = 0.0  # a second Ctrl-C before this time exits
        self._draft = ""  # pre-filled into the next prompt (a message that wasn't sent)
        self.expanded = False  # ctrl+o: the last output / running subagents, above the input box
        # Started with --mode auto counts as agreeing to it; otherwise it's asked once.
        self.auto_confirmed = session.mode == PermissionMode.AUTO
        self.auto_declined = False  # said no once: Shift+Tab skips auto mode (/mode auto still asks)
        self.git = git_status(session.workspace.root)  # refreshed after each turn, not on every redraw
        self._prompt: _CompactPrompt | None = None
        self._turn: asyncio.Task[None] | None = None  # the agent's turn, running under the input box
        self._ui = asyncio.Lock()  # who has the keyboard: the input box, or a question (approval)
        self._questions = 0  # questions waiting for the keyboard, or holding it
        self._sending_now = False  # Esc stopped the turn to send the waiting messages

    # -- input ----------------------------------------------------------------

    def _toolbar(self) -> StyleAndTextTuples:
        if time.monotonic() < self._exit_armed_until:
            return [("class:toolbar.warn", "  Press Ctrl+C again to exit")]
        mode = self.session.mode
        key, hint = ("ctrl+c", " to stop") if self.turn_running else ("shift+tab", " to switch")
        left: StyleAndTextTuples = [
            ("class:mode", f"  {_MODE_ICON[mode]} {MODE_LABEL[mode]}"),
            ("class:toolbar.dim", " · "),
            ("class:toolbar.key", key),
            ("class:toolbar.dim", hint),
        ]
        width = get_app().output.get_size().columns
        right = self._status(width - _cells(left) - 4)
        gap = max(width - _cells(left) - _cells(right) - 2, 2)
        return [*left, ("", " " * gap), *right, ("", "  ")]

    def _status(self, room: int) -> StyleAndTextTuples:
        """`sonnet-5 · sandbox · ⎇ main ✱ · 38% context · 12.3k out · $0.42`, dropping the
        least important parts first when the terminal is narrow."""
        session = self.session
        parts: list[tuple[int, StyleAndTextTuples]] = []  # (keep priority: lower is dropped first, items)
        model = session.model_label + (f" · {session.config.effort}" if session.config.effort else "")
        parts.append((5, [("class:toolbar.dim", model)]))
        if session.sandbox is not None:
            parts.append((3, [("class:toolbar.ok", "sandbox")]))
        elif "Bash" in session.tools.names():
            parts.append((4, [("class:toolbar.err", "no sandbox")]))
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
        parts.append((6, [(f"class:{level}", f"{percent(self.context_used)} context")]))
        meter = session.ctx.usage
        if meter.total.output:  # what the model wrote; /cost has the full breakdown
            parts.append((1, [("class:toolbar.dim", f"{tokens(meter.total.output)} out")]))
        if meter.cost is not None:
            parts.append((5, [("class:toolbar", dollars(meter.cost))]))
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

        @keys.add(
            "escape", filter=Condition(lambda: self.turn_running and bool(self.session.pending_steering))
        )
        def _send_now(event: KeyPressEvent) -> None:
            """Stop the agent's current step; the waiting messages go to it at once."""
            self._sending_now = True
            if self._turn is not None:
                self._turn.cancel()

        @keys.add("escape", "enter")
        @keys.add("c-j")
        def _newline(event: KeyPressEvent) -> None:
            event.current_buffer.insert_text("\n")

        @keys.add("tab")
        def _accept(event: KeyPressEvent) -> None:
            """Tab takes the gray suggestion, or the highlighted (else first) menu entry."""
            buffer = event.current_buffer
            if buffer.suggestion and buffer.suggestion.text:
                buffer.insert_text(buffer.suggestion.text)
            elif buffer.complete_state:
                completion = buffer.complete_state.current_completion or next(
                    iter(buffer.complete_state.completions), None
                )
                if completion:
                    buffer.apply_completion(completion)
            elif is_command(buffer.text):
                buffer.start_completion(select_first=False)

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
            self.session.set_mode(mode)

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
            placeholder=lambda: FormattedText([(SUBTLE, self._placeholder())]),
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
        """Messages waiting for the agent's next step, and how to send them now (as Codex
        shows them): `• Sent after the agent's current step (esc: stop it and send now)`."""
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

    def _placeholder(self) -> str:
        if self.turn_running:
            return "Message the agent: it gets this before its next step"
        return "Ask anything, or type / for commands"

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
        app = self._prompt.app if self._prompt is not None else None
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
        self.renderer.exclusive = self.exclusive
        with patch_stdout(raw=True):  # output scrolls above the input box
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
        self.console.print(Text("\n  bye 👋\n", style="muted"))

    async def handle(self, line: str) -> None:
        if self.turn_running:
            await self._while_running(line)
            return
        if not is_command(line):
            if await self._safe_to_send(line):
                images, problems = attachments(line, self.session.workspace.root)
                for image in images:
                    size = len(image.data) * 3 // 4 // 1024
                    self.console.print(Text(f"  ⎿ attached {image.name} · {size:,} KB", style="subtle"))
                for problem in problems:
                    self.console.print(Text(f"  ⎿ not attached: {problem}", style="warn"))
                await self._run(self.session.run(line, images))
            return

        name, _, arg = line.partition(" ")
        arg = arg.strip()
        if cmd := COMMANDS.get(name):
            self.console.print()
            result = cmd.handler(self, arg)
            if inspect.isawaitable(result):
                await result
        elif self.session.skills.get(name[1:]):
            if await self._safe_to_send(line):
                await self._run(self.session.run_skill(name[1:], arg))
        else:
            self.console.print(
                f"\n  Unknown command [bold]{escape(name)}[/]. Type [key]/[/] to see what's available."
            )

    def show_thinking(self, show: bool) -> None:
        self.renderer.show_thinking = show
        self.session.set_show_thinking(show)
        if not show:
            self.console.print("  [muted]∴ Reasoning hidden (ctrl+t shows it)[/]")
            return
        self.console.print("  [muted]∴ Reasoning shown (ctrl+t hides it)[/]")
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
        if (key := find_api_key(line)) is None:
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

    async def _while_running(self, line: str) -> None:
        """Input while the agent works: a message goes to it; commands that don't touch the
        conversation run; the rest wait."""
        name = line.partition(" ")[0]
        if not is_command(line):
            if await self._safe_to_send(line):
                self.session.steer(line)
            return
        if name in _SAFE_WHILE_RUNNING and (cmd := COMMANDS.get(name)):
            result = cmd.handler(self, line.partition(" ")[2].strip())
            if inspect.isawaitable(result):
                await result
            return
        self.console.print(f"  [muted]{name} has to wait until the agent finishes (ctrl+c stops it).[/]")
        self._draft = line

    # -- turns ----------------------------------------------------------------

    async def _run(self, events: AsyncGenerator[Event, None]) -> None:
        """Start a turn: under the input box when the REPL is running, else in the foreground."""
        if self._prompt is None:
            await self.run_turn(events)
        else:
            self._turn = asyncio.create_task(self._turn_in_background(events))

    async def _turn_in_background(self, events: AsyncGenerator[Event, None]) -> None:
        try:
            await self.renderer.render(events)
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
        # Messages that arrived after the agent's last step: they start the next turn.
        if (left := self.session.take_steering()) and self.running:
            self._turn = asyncio.create_task(self._turn_in_background(self.session.run("\n\n".join(left))))

    async def run_turn(self, events: AsyncGenerator[Event, None]) -> None:
        """Render one agent turn. Ctrl-C cancels the turn but keeps the REPL alive."""
        task = asyncio.create_task(self.renderer.render(events))
        loop = asyncio.get_running_loop()
        with contextlib.suppress(NotImplementedError):  # not available on Windows
            loop.add_signal_handler(signal.SIGINT, task.cancel)
        try:
            await task
        except asyncio.CancelledError:
            self.console.print("\n[warn]■ Interrupted.[/] [muted]What should it do instead?[/]")
        except Exception as e:
            report_error(self.console, e)
        finally:
            with contextlib.suppress(NotImplementedError):
                loop.remove_signal_handler(signal.SIGINT)
            self._measure_context()

    def _measure_context(self) -> None:
        with contextlib.suppress(Exception):  # the toolbar is a nicety; never fail a turn over it
            used = count_message_tokens(self.session.history.to_dicts())
            self.context_used = used / self.session.context_window
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


def attachments(line: str, cwd: Path) -> tuple[list[Image], list[str]]:
    """Image files named in a message, to show the model: (images, why others couldn't be).
    Words are split as a shell would, so a pasted `Screenshot\\ 2026-09-30\\ at\\ 3.11.03 PM.png`
    or a quoted drag-and-drop path is one path. Only names ending like an image are
    looked at; the file's bytes decide."""
    try:
        words = shlex.split(line)
    except ValueError:  # an unmatched quote: fall back to spaces
        words = line.split()
    images: list[Image] = []
    problems: list[str] = []
    seen: set[Path] = set()
    for word in words:
        if not word.lower().endswith(_IMAGE_NAMES):
            continue
        path = Path(word).expanduser()
        path = path if path.is_absolute() else cwd / path
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            images.append(load_image(path))
        except (OSError, ValueError) as e:
            problems.append(str(e))
    return images, problems
