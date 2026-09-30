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
import logging
import signal
import subprocess
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable
from functools import cache
from pathlib import Path

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
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

from tinyharness.cli.commands import COMMANDS
from tinyharness.cli.render import Renderer, dollars, tokens
from tinyharness.cli.theme import MODE_LABEL, MODE_STYLE, PROMPT, PT_STYLE, SUBTLE, percent
from tinyharness.cli.widgets import pick
from tinyharness.events import Event
from tinyharness.llm.client import LLMError, ProviderConnectionError
from tinyharness.llm.tokens import count_message_tokens
from tinyharness.paths import auth_path, home_dir, pretty_path
from tinyharness.permissions import PermissionMode
from tinyharness.security import find_api_key
from tinyharness.session import MissingAPIKeyError, Session

log = logging.getLogger(__name__)

EXIT_CONFIRM_SECONDS = 2.0
_CONFIRM_AUTO = "\x00confirm-auto-mode"  # returned by the prompt when Shift+Tab reaches auto mode
_SHOW_OUTPUT = "\x00show-last-output"  # returned by the prompt on Ctrl-O
_PAUSE = "\x00pause"  # returned by the prompt when a question (approval) needs the keyboard
# Commands that don't touch the conversation, so they can run while the agent works.
_SAFE_WHILE_RUNNING = {"/help", "/mode", "/context", "/theme", "/thinking", "/cwd", "/tools", "/skills"}

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
        if not text.startswith("/") or " " in text:
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


class _CompactPrompt(PromptSession[str]):
    """The input box, which stays up while the agent works.

    Above it, `live` (the running tools, agents and spinner) is redrawn in place;
    finished output scrolls above everything (via patch_stdout). prompt_toolkit also
    stretches the input to the bottom of the terminal when there is a bottom toolbar:
    its height is capped at the text it holds so the frame hugs the input."""

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
        # Started with --mode auto counts as agreeing to it; otherwise it's asked once.
        self.auto_confirmed = session.mode == PermissionMode.AUTO
        self.auto_declined = False  # said no once: Shift+Tab skips auto mode (/mode auto still asks)
        self.git = git_status(session.workspace.root)  # refreshed after each turn, not on every redraw
        self._prompt: _CompactPrompt | None = None
        self._turn: asyncio.Task[None] | None = None  # the agent's turn, running under the input box
        self._ui = asyncio.Lock()  # who has the keyboard: the input box, or a question (approval)

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
            elif buffer.text.startswith("/"):
                buffer.start_completion(select_first=False)

        @keys.add("c-t")
        def _toggle_thinking(event: KeyPressEvent) -> None:
            """Show or hide the model's reasoning (turning it on shows the latest)."""
            self.show_thinking(not self.renderer.show_thinking)

        @keys.add("c-o")
        def _show_output(event: KeyPressEvent) -> None:
            """Show the last tool call's full output (the transcript shows one line of it)."""
            if self.renderer.last_output is not None:
                self._draft = event.current_buffer.text
                event.app.exit(result=_SHOW_OUTPUT)

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
        typing_command = Condition(lambda: prompt.default_buffer.text.startswith("/"))
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
        if not self.turn_running:
            return ""
        return self.renderer.live_ansi(get_app().output.get_size().columns)

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
        typed so far; the input box comes back afterwards."""
        app = self._prompt.app if self._prompt is not None else None
        if app is not None and app.is_running:
            self._draft = app.current_buffer.text
            app.exit(result=_PAUSE)
        async with self._ui:
            yield

    def _echo(self, line: str, *, queued: bool = False) -> None:
        """Print the submitted message back into the transcript (the input box is erased)."""
        first, *rest = line.splitlines()
        text = Text().append(f"{PROMPT} ", style="brand").append(first, style="bold")
        for more in rest:
            text.append(f"\n  {more}", style="bold")
        if queued and not line.startswith("/"):
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
                    try:
                        # show_frame must be passed here: prompt_async() resets it (prompt_toolkit 3.0.52).
                        draft, self._draft = self._draft, ""
                        line = (await prompt.prompt_async(show_frame=True, default=draft)).strip()
                    except (EOFError, KeyboardInterrupt):
                        break  # Ctrl-D, or Ctrl-C twice
                    if line == _PAUSE:
                        continue  # a question needs the keyboard; it's released on the way round
                    if line == _CONFIRM_AUTO:
                        await self.shift_tab_to_auto()
                    elif line == _SHOW_OUTPUT:
                        self.show_last_output()
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
        if not line.startswith("/"):
            if await self._safe_to_send(line):
                await self._run(self.session.run(line))
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
                f"\n  Unknown command [bold]{name}[/]. Type [key]/[/] to see what's available."
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
        if not line.startswith("/"):
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
            self.console.print("\n[warn]■ Interrupted.[/] [muted]What should it do instead?[/]")
        except Exception as e:
            report_error(self.console, e)
        finally:
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
