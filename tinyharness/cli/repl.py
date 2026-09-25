"""The interactive read-eval-print loop.

╭──────────────────────────────────────────────────────────────╮
│ ❯ fix the failing test in tests/test_app.py                  │
╰──────────────────────────────────────────────────────────────╯
  ⏵⏵ accept edits · shift+tab to switch       gpt-5 · 12% context

The frame's color follows the permission mode, so what the agent may do without
asking is always visible.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import signal
import time
from collections.abc import AsyncGenerator, Iterable
from functools import cache

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText, StyleAndTextTuples
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.styles import DynamicStyle, Style, merge_styles
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from tinyharness.cli.commands import COMMANDS
from tinyharness.cli.render import Renderer
from tinyharness.cli.theme import MODE_LABEL, MODE_STYLE, PROMPT, PT_STYLE, SUBTLE
from tinyharness.events import Event
from tinyharness.llm.client import LLMError
from tinyharness.llm.tokens import count_message_tokens
from tinyharness.paths import home_dir
from tinyharness.permissions import PermissionMode
from tinyharness.session import Session

log = logging.getLogger(__name__)

EXIT_CONFIRM_SECONDS = 2.0

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


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class _CompactPrompt(PromptSession[str]):
    """prompt_toolkit stretches the input to the bottom of the terminal when there is a
    bottom toolbar. Cap its height at the text it holds so the frame hugs the input and
    the status line sits right under it."""

    def _get_default_buffer_control_height(self) -> Dimension:
        reserved = super()._get_default_buffer_control_height().min
        width = max(get_app().output.get_size().columns - 6, 10)  # frame + prompt
        lines = sum(len(line) // width + 1 for line in self.default_buffer.document.lines)
        return Dimension(min=max(reserved, 1), max=max(reserved, lines), preferred=lines)


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

    # -- input ----------------------------------------------------------------

    def _toolbar(self) -> StyleAndTextTuples:
        if time.monotonic() < self._exit_armed_until:
            return [("class:toolbar.warn", "  Press Ctrl+C again to exit")]
        mode = self.session.mode
        left: StyleAndTextTuples = [
            ("class:mode", f"  {_MODE_ICON[mode]} {MODE_LABEL[mode]}"),
            ("class:toolbar.dim", " · "),
            ("class:toolbar.key", "shift+tab"),
            ("class:toolbar.dim", " to switch"),
        ]
        right_text = f"{self.session.config.model_info.display} · {self.context_used:.0%} context  "
        width = get_app().output.get_size().columns
        used = sum(len(t) for _, t, *_ in left)
        gap = max(width - used - len(right_text), 2)
        return [*left, ("", " " * gap), ("class:toolbar.dim", right_text)]

    def _prompt_session(self) -> PromptSession[str]:
        """Enter submits; Alt+Enter or Ctrl-J inserts a newline; Shift+Tab cycles the mode.
        Ctrl-C clears the input, or on an empty prompt, exits when pressed twice."""
        keys = KeyBindings()

        @keys.add("c-c")
        def _interrupt(event: KeyPressEvent) -> None:
            if event.current_buffer.text:
                event.current_buffer.reset()
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

        @keys.add("s-tab")
        def _cycle_mode(event: KeyPressEvent) -> None:
            self.session.set_mode(self.session.mode.next())

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
            placeholder=FormattedText([(SUBTLE, "Ask anything, or type / for commands")]),
            bottom_toolbar=self._toolbar,
            refresh_interval=0.5,  # so the "press again to exit" hint times out
            erase_when_done=True,
            style=DynamicStyle(lambda: merge_styles([PT_STYLE, _style_for(self.session.mode)])),
        )
        return prompt

    def _echo(self, line: str) -> None:
        """Print the submitted message back into the transcript (the input box is erased)."""
        first, *rest = line.splitlines()
        text = Text().append(f"{PROMPT} ", style="brand").append(first, style="bold")
        for more in rest:
            text.append(f"\n  {more}", style="bold")
        self.console.print(text, highlight=False)

    async def loop(self) -> None:
        prompt = self._prompt_session()
        while self.running:
            try:
                # show_frame must be passed here: prompt_async() resets it (prompt_toolkit 3.0.52).
                line = (await prompt.prompt_async(show_frame=True)).strip()
            except (EOFError, KeyboardInterrupt):
                break  # Ctrl-D, or Ctrl-C twice
            if line:
                self._echo(line)
                await self.handle(line)
                self.console.print()
        self.console.print(Text("\n  bye 👋\n", style="muted"))

    async def handle(self, line: str) -> None:
        if not line.startswith("/"):
            await self.run_turn(self.session.run(line))
            return

        name, _, arg = line.partition(" ")
        arg = arg.strip()
        if cmd := COMMANDS.get(name):
            self.console.print()
            result = cmd.handler(self, arg)
            if inspect.isawaitable(result):
                await result
        elif self.session.skills.get(name[1:]):
            await self.run_turn(self.session.run_skill(name[1:], arg))
        else:
            self.console.print(
                f"\n  Unknown command [bold]{name}[/]. Type [key]/[/] to see what's available."
            )

    # -- turns ----------------------------------------------------------------

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
            used = count_message_tokens(self.session.history.to_openai())
            self.context_used = used / self.session.config.context_window


def report_error(console: Console, error: Exception) -> None:
    hint = ""
    if isinstance(error, TimeoutError) or "Timeout" in type(error).__name__:
        hint = (
            "The model took longer than `request_timeout` to answer. Files already written are kept.\n"
            "Try a faster model (/model), less reasoning (/think off), or a larger request_timeout."
        )
    elif isinstance(error, LLMError) and "auth" in str(error):
        hint = "Check your API key with /whoami and /ping."
    body = Text(str(error) or type(error).__name__)
    if hint:
        body.append(f"\n\n{hint}", style="muted")
    if not log.isEnabledFor(logging.DEBUG):
        body.append("\n\nRun with --debug for the full traceback.", style="subtle")
    console.print()
    console.print(
        Panel(body, title=f"[err]✗ {type(error).__name__}[/]", title_align="left", border_style="err")
    )
    if log.isEnabledFor(logging.DEBUG):
        console.print_exception()
