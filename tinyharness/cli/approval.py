"""The interactive permission prompt.

╭─ Edit  src/app.py, -1/+4 lines ───────────────────────╮
│ - return x                                            │
│ + if verbose:                                         │
│ +     log(x)                                          │
│ + return x                                            │
╰───────────────────────────────────────────────────────╯
 Make this edit?
 ❯ 1. Yes
   2. Yes, allow all edits in this project for this session
   3. No, and tell it what to do instead  esc
"""

from __future__ import annotations

from typing import Any, Literal

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.shortcuts.choice_input import ChoiceInput
from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.text import Text

from tinyharness.cli.theme import BRAND, PROMPT, PT_STYLE, SUBTLE, split_summary
from tinyharness.permissions import Decision, PermissionRequest

Choice = Literal["yes", "remember", "no"]
PREVIEW_LINES = 12

_QUESTIONS = {"Edit": "Make this edit?", "Write": "Write this file?"}


def preview(request: PermissionRequest) -> Text | None:
    """A short look at what the call will change, so it can be judged at a glance."""
    args = request.args
    lines = Text()
    if request.tool == "Edit":
        for sign, key, style in (("-", "old_string", "diff.del"), ("+", "new_string", "diff.add")):
            for line in str(args.get(key, "")).splitlines()[:PREVIEW_LINES]:
                lines.append(f"{sign} {line}\n", style=style)
    elif request.tool == "Write":
        content = str(args.get("content", "")).splitlines()
        for line in content[:PREVIEW_LINES]:
            lines.append(f"+ {line}\n", style="diff.add")
        if len(content) > PREVIEW_LINES:
            lines.append(f"… {len(content) - PREVIEW_LINES} more lines\n", style="muted")
    lines.rstrip()
    return lines if lines.plain else None


def card(request: PermissionRequest, summary: str) -> RenderableType:
    """The tool call being asked about, framed so it stands out from the transcript."""
    name, detail = split_summary(summary)
    title = Text.assemble((name, "bold"), (f"  {detail}" if detail else "", "muted"))
    body: list[RenderableType] = []
    if (diff := preview(request)) is not None:
        body.append(diff)
    elif request.path:
        body.append(Text(request.path, style="path"))
    return Panel(
        Group(*body) if body else Text(summary, style="muted"),
        title=title,
        title_align="left",
        box=box.ROUNDED,
        border_style=BRAND,
        padding=(0, 1),
    )


def _choice_keys() -> KeyBindings:
    """Digits answer immediately; Esc means "no"."""
    keys = KeyBindings()

    def answer(key: str, result: Choice) -> None:
        @keys.add(key, eager=True)
        def _(event: KeyPressEvent) -> None:
            event.app.exit(result=result)

    shortcuts: list[tuple[str, Choice]] = [("1", "yes"), ("2", "remember"), ("3", "no"), ("escape", "no")]
    for key, result in shortcuts:
        answer(key, result)
    return keys


async def ask(console: Console, request: PermissionRequest, summary: str) -> Decision:
    console.print(card(request, summary))
    question = _QUESTIONS.get(request.tool, f"Allow {request.tool}?")

    options: list[tuple[Choice, Any]] = [
        ("yes", "Yes"),
        ("remember", f"Yes, {request.remember_label}"),
        ("no", FormattedText([("", "No, and tell it what to do instead  "), (SUBTLE, "esc")])),
    ]
    try:
        choice: Choice = await ChoiceInput[Choice](
            message=FormattedText([("bold", f" {question}")]),
            options=options,
            default="yes",
            style=PT_STYLE,
            symbol=PROMPT,
            key_bindings=_choice_keys(),
        ).prompt_async()
    except (KeyboardInterrupt, EOFError):
        choice = "no"

    if choice != "no":
        return Decision(True, "user approved", "user", remember=choice == "remember")

    feedback = await _ask_feedback()
    return Decision(False, "user rejected", "user", interrupt=True, feedback=feedback)


async def _ask_feedback() -> str:
    try:
        text: Any = await PromptSession[str]().prompt_async(
            FormattedText([(f"{BRAND} bold", f" {PROMPT} "), ("", "What should it do instead? ")]),
            placeholder=FormattedText([(SUBTLE, "enter to just stop")]),
            style=PT_STYLE,
            key_bindings=_escape_means_empty(),
        )
    except (KeyboardInterrupt, EOFError):
        return ""
    return str(text).strip()


def _escape_means_empty() -> KeyBindings:
    keys = KeyBindings()

    @keys.add("escape", eager=True)
    def _empty(event: KeyPressEvent) -> None:
        event.app.exit(result="")

    return keys
