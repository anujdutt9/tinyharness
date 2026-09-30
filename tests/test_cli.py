import pytest

from tinyharness.cli.markdown import split_finished
from tinyharness.cli.render import format_todos, summarize_output
from tinyharness.cli.theme import split_summary


@pytest.mark.parametrize(
    ("text", "done", "rest"),
    [
        ("still typing", "", "still typing"),
        ("para one\n\npara two", "para one\n\n", "para two"),
        ("```py\nx = 1\n\ny = 2\n", "", "```py\nx = 1\n\ny = 2\n"),  # never split inside a fence
        ("```py\nx = 1\n```\n\nafter", "```py\nx = 1\n```\n\n", "after"),
    ],
)
def test_streamed_markdown_is_split_at_finished_blocks(text, done, rest):
    assert split_finished(text) == (done, rest)


def test_split_summary():
    assert split_summary("Edit(src/app.py, -1/+4 lines)") == ("Edit", "src/app.py, -1/+4 lines")
    assert split_summary("Todos") == ("Todos", "")


@pytest.mark.parametrize(
    ("tool", "output", "expected"),
    [
        ("Read", "a\nb\nc", "Read 3 lines"),
        ("Glob", "2 match(es):\na.py\nb.py", "2 match(es)"),
        ("Grep", "3 total match(es) across 2 file(s):\na.py:3\nb.py:1", "3 total match(es) across 2 file(s)"),
        ("Grep", "(no matches for 'x' under /p)", "no matches"),
        ("Grep", "(0 matches for 'x' under /p)", "no matches"),
        ("Write", "wrote a.py (3 bytes)", "wrote a.py (3 bytes)"),
        ("Write", "", "done"),
    ],
)
def test_summarize_output(tool, output, expected):
    assert summarize_output(tool, output) == expected


def test_todos_render_as_a_checklist():
    todos = [
        {"content": "Read", "activeForm": "Reading", "status": "completed"},
        {"content": "Edit", "activeForm": "Editing", "status": "in_progress"},
        {"content": "Test", "activeForm": "Testing", "status": "pending"},
    ]
    assert format_todos(todos).plain.splitlines() == ["✔ Read", "    ◼ Editing", "    ◻ Test"]


def test_long_markdown_paragraphs_wrap_without_losing_text():
    from rich.console import Console

    from tinyharness.cli.render import Renderer

    console = Console(width=40, record=True)
    renderer = Renderer(console)
    renderer._streaming = "word " * 30 + "THE-END"
    renderer._flush_text(everything=True)
    assert "THE-END" in console.export_text()


def test_spinner_says_thinking_only_when_the_model_reasons():
    from rich.console import Console

    from tinyharness.cli.render import WAITING, Renderer
    from tinyharness.events import AssistantText, Reasoning

    renderer = Renderer(Console(width=40, record=True))
    renderer.wait(WAITING)
    renderer.handle(Reasoning())
    assert renderer._activity == "Thinking"

    plain = Renderer(Console(width=40, record=True))
    plain.wait(WAITING)
    plain.handle(AssistantText("hello"))
    assert plain._activity is None  # text arrived; never claimed to be thinking


@pytest.mark.parametrize(
    ("fraction", "shown"),
    [
        (0, "0%"),
        (0.0004, "<0.1%"),
        (4_839 / 1_048_576, "0.5%"),
        (0.032, "3.2%"),
        (0.45, "45%"),
        (1.0, "100%"),
    ],
)
def test_context_percent_never_looks_stuck_at_zero(fraction, shown):
    from tinyharness.cli.theme import percent

    assert percent(fraction) == shown


# ---- the input box and the / menu ----


@pytest.mark.parametrize(("matches", "rows"), [(0, 1), (2, 3), (6, 7), (20, 7)])
async def test_the_input_box_grows_only_by_the_menu_it_shows(matches, rows):
    """One row for the input, plus one per entry showing, up to the menu's cap. prompt_toolkit
    alone reserves the full cap whenever a menu might show, leaving blank rows in the box."""
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.buffer import CompletionState
    from prompt_toolkit.completion import Completion, WordCompleter
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from tinyharness.cli.repl import _CompactPrompt

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        prompt = _CompactPrompt(completer=WordCompleter([]), reserve_space_for_menu=6, multiline=True)
        prompt.default_buffer.text = "/mode"
        if matches:
            prompt.default_buffer.complete_state = CompletionState(
                prompt.default_buffer.document, [Completion(f"/c{i}") for i in range(matches)]
            )
        height = prompt._get_default_buffer_control_height()
    assert (height.min, height.max) == (rows, rows)
