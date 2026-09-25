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
        ("Glob", "a.py\nb.py", "2 results"),
        ("Grep", "(no matches for 'x' under /p)", "no matches"),
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
