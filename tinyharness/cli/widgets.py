"""Small interactive inputs shared by slash commands.

Each shows on the bottom rows of the terminal, where the input box is (see `at_bottom`),
and is erased once answered; the input box then comes back in the same place."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.completion import FuzzyWordCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.shortcuts.choice_input import ChoiceInput

from tinyharness.cli.theme import BRAND, PROMPT, PT_STYLE, SUBTLE

T = TypeVar("T")
R = TypeVar("R")


def bottom_spacer() -> Window:
    """Takes the rows a prompt doesn't need. prompt_toolkit gives a prompt every row below
    the cursor (it asks the terminal where that is), so with this on top the prompt sits
    on the terminal's last rows, and output above it stays where it is."""
    return Window(height=Dimension(min=0, preferred=0, weight=1))


def at_bottom(app: Application[R]) -> Application[R]:
    """`app`, shown on the bottom rows of the terminal (see `bottom_spacer`)."""
    layout = app.layout
    app.layout = Layout(HSplit([bottom_spacer(), layout.container]), focused_element=layout.current_window)
    return app


def prompt_at_bottom(**kwargs: object) -> PromptSession[str]:
    """A one-line PromptSession shown on the bottom rows (see `at_bottom`)."""
    session = PromptSession[str](**kwargs)  # type: ignore[arg-type]
    at_bottom(session.app)
    return session


async def pick(
    title: str, options: Sequence[tuple[T, str]], current: T | None = None, *, mark_current: bool = True
) -> T | None:
    """An arrow-key picker starting on `current`, cleared from the screen once answered.
    Returns None on Esc / Ctrl-C."""
    keys = KeyBindings()

    @keys.add("escape", eager=True)
    def _cancel(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    picker = ChoiceInput[T | None](
        message=FormattedText([("bold", f" {title}")]),
        options=[
            (value, label + ("  · current" if mark_current and value == current else ""))
            for value, label in options
        ],
        default=current if any(v == current for v, _ in options) else None,
        style=PT_STYLE,
        symbol=PROMPT,
        key_bindings=keys,
    )
    # Clear the list once answered: the caller prints what was chosen. (ChoiceInput has
    # no option for this, so set it on the application it builds.)
    app = at_bottom(picker._create_application())
    app.erase_when_done = True
    try:
        return await app.run_async()
    except (KeyboardInterrupt, EOFError):
        return None


async def ask(
    question: str,
    *,
    choices: Sequence[str] = (),
    meta: dict[str, str] | None = None,
    hint: str = "",
    secret: bool = False,
) -> str:
    """One line of input. `choices` enables fuzzy autocompletion (free text still allowed);
    `secret` hides what is typed. Cleared from the screen once answered.
    Returns "" when cancelled with Ctrl-C / Ctrl-D."""
    completer = (
        FuzzyWordCompleter(list(choices), meta_dict=meta, WORD=True) if choices else None
    )  # ids contain - / . :
    try:
        text = await prompt_at_bottom(erase_when_done=True).prompt_async(
            FormattedText([(f"{BRAND} bold", f" {PROMPT} "), ("bold", question), ("", " ")]),
            completer=completer,
            complete_while_typing=bool(choices),
            placeholder=FormattedText([(SUBTLE, hint)]) if hint else None,
            style=PT_STYLE,
            is_password=secret,
        )
    except (KeyboardInterrupt, EOFError):
        return ""
    return text.strip()
