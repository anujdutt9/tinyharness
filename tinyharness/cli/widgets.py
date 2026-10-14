"""Small interactive inputs shared by slash commands."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from prompt_toolkit import PromptSession
from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import FuzzyWordCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout import BufferControl, FormattedTextControl, HSplit, Layout, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.shortcuts.choice_input import ChoiceInput

from tinyharness.cli.theme import BRAND, PROMPT, PT_STYLE, SUBTLE

T = TypeVar("T")


@dataclass(frozen=True)
class PaletteItem(Generic[T]):
    """One searchable command-palette row."""

    value: T
    title: str
    detail: str = ""
    keywords: str = ""


async def palette(
    title: str,
    items: Sequence[PaletteItem[T]],
    current: T | None = None,
    *,
    query: str = "",
    mark_current: bool = True,
    rows: int = 10,
) -> T | None:
    """A searchable, reusable overlay. Arrow keys move; typing filters; Esc cancels."""
    keys = KeyBindings()
    search = Buffer(multiline=False)
    selected = 0

    def matches() -> list[PaletteItem[T]]:
        words = search.text.casefold().split()
        if not words:
            return list(items)

        def score(item: PaletteItem[T]) -> tuple[int, int]:
            name = item.title.casefold()
            haystack = f"{item.title} {item.detail} {item.keywords}".casefold()
            if not all(word in haystack for word in words):
                return (99, 0)
            phrase = " ".join(words)
            return (0 if name.startswith(phrase) else 1 if phrase in name else 2, len(name))

        return sorted((item for item in items if score(item)[0] < 99), key=score)

    def visible() -> FormattedText:
        found = matches()
        if not found:
            return FormattedText([("class:placeholder", "  No matches")])
        fragments: list[tuple[str, str]] = []
        start = min(max(selected - rows + 1, 0), max(len(found) - rows, 0))
        for index, item in enumerate(found[start : start + rows], start):
            active = index == selected
            marker = f"{PROMPT} " if active else "  "
            style = "class:palette.selected" if active else ""
            suffix = "  · current" if mark_current and item.value == current else ""
            fragments.append((style, f" {marker}{item.title}{suffix}"))
            if item.detail:
                fragments.append(("class:palette.detail", f"  {item.detail}"))
            fragments.append(("", "\n"))
        return FormattedText(fragments)

    results = FormattedTextControl(visible)

    def changed(_: Buffer) -> None:
        nonlocal selected
        selected = 0
        app.invalidate()

    search.on_text_changed += changed

    @keys.add("up")
    @keys.add("c-p")
    def up(event: KeyPressEvent) -> None:
        nonlocal selected
        selected = max(0, selected - 1)
        event.app.invalidate()

    @keys.add("down")
    @keys.add("c-n")
    def down(event: KeyPressEvent) -> None:
        nonlocal selected
        selected = min(max(len(matches()) - 1, 0), selected + 1)
        event.app.invalidate()

    @keys.add("enter")
    def accept(event: KeyPressEvent) -> None:
        found = matches()
        event.app.exit(result=found[selected].value if found else None)

    @keys.add("escape", eager=True)
    @keys.add("c-c")
    def cancel(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    header = Window(FormattedTextControl(FormattedText([("bold", f" {title}")])), height=1)
    search_window = Window(
        BufferControl(
            buffer=search,
            input_processors=[BeforeInput(FormattedText([(f"{BRAND} bold", f" {PROMPT} ")]))],
        ),
        height=1,
    )
    result_window = Window(
        results,
        height=Dimension(min=1, max=rows),
        dont_extend_height=True,
        wrap_lines=False,
    )
    footer = Window(
        FormattedTextControl(
            FormattedText([(SUBTLE, " ↑↓ move  ·  type to filter  ·  enter choose  ·  esc close")])
        ),
        height=1,
    )
    app: Application[T | None] = Application(
        layout=Layout(HSplit([header, search_window, result_window, footer]), focused_element=search_window),
        key_bindings=keys,
        style=PT_STYLE,
        erase_when_done=True,
        full_screen=False,
    )
    if query:
        search.text = query
        search.cursor_position = len(query)
    elif current is not None:
        selected = next((i for i, item in enumerate(items) if item.value == current), 0)
    try:
        return await app.run_async()
    except (KeyboardInterrupt, EOFError):
        return None


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
        default=current if any(value == current for value, _ in options) else None,
        style=PT_STYLE,
        symbol=PROMPT,
        key_bindings=keys,
    )
    # Clear the list once answered: the caller prints what was chosen. ChoiceInput has
    # no public option for this, so set it on the application it builds.
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
