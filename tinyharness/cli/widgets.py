"""Small interactive inputs shared by slash commands."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import FuzzyWordCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.shortcuts.choice_input import ChoiceInput

from tinyharness.cli.theme import BRAND, PROMPT, PT_STYLE, SUBTLE

T = TypeVar("T")


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
    app = picker._create_application()
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
        text = await PromptSession[str](erase_when_done=True).prompt_async(
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
