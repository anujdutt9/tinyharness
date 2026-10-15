import pytest

from tinyharness.cli.markdown import split_finished
from tinyharness.cli.render import format_todos, summarize_output
from tinyharness.cli.theme import split_summary


async def test_resume_picker_uses_titles_and_search_instead_of_ids(monkeypatch):
    """Seen live: an opaque timestamp id made resuming feel unusable even though a picker existed."""
    import io
    from datetime import datetime, timedelta

    from rich.console import Console

    from tinyharness import SessionInfo
    from tinyharness.cli import app

    now = datetime.now()
    sessions = [
        SessionInfo("1791-opaque", now, "Build the Starbucks presentation", 45, "anthropic:sonnet"),
        SessionInfo("1790-opaque", now - timedelta(hours=2), "Review the API", 12, "openai:gpt"),
    ]
    monkeypatch.setattr(app, "list_sessions", lambda options: sessions)
    seen = {}

    async def choose(title, options, **kwargs):
        seen["title"], seen["options"] = title, options
        return options[0][0]

    monkeypatch.setattr(app, "pick", choose)
    console = Console(file=io.StringIO(), color_system=None)
    assert await app.pick_session(object(), console) == "1791-opaque"
    assert "1791-opaque" not in seen["options"][0][1]
    assert "Build the Starbucks presentation" in seen["options"][0][1]
    assert "45 messages" in seen["options"][0][1]
    assert "anthropic / sonnet" in seen["options"][0][1]

    assert await app.pick_session(object(), console, "starbucks") == "1791-opaque"
    assert 'Resuming "Build the Starbucks presentation"' in console.file.getvalue()


async def test_ambiguous_resume_search_does_not_guess_without_a_terminal(monkeypatch):
    import io
    from datetime import datetime

    from rich.console import Console

    from tinyharness import SessionInfo
    from tinyharness.cli import app

    sessions = [
        SessionInfo("one", datetime.now(), "Build presentation", 4),
        SessionInfo("two", datetime.now(), "Review presentation", 6),
    ]
    monkeypatch.setattr(app, "list_sessions", lambda options: sessions)
    console = Console(file=io.StringIO(), color_system=None)
    assert await app.pick_session(object(), console, "presentation") is None
    assert "More than one conversation matches" in console.file.getvalue()


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


@pytest.mark.parametrize(
    ("line", "command"),
    [
        ("/model", True),
        ("/key anthropic", True),
        ("/claude-design-skill make a deck", True),
        ("/var/folders/w4/T/Screenshot\\ 2026-09-30.png  - This is slide 1", False),  # a pasted path
        ("/Users/me/shot.png what's wrong here?", False),
        ("fix /model's help text", False),
    ],
)
def test_only_a_slash_and_a_name_is_a_command(line, command):
    from tinyharness.cli.repl import is_command

    assert is_command(line) is command


# ---- attaching screenshots ----

from conftest import png  # noqa: E402

PNG = png(40, 20)


def test_pasted_screenshot_paths_are_attached(tmp_path):
    """As macOS pastes them: spaces escaped, a note after each."""
    from tinyharness.cli.repl import attachments

    shots = tmp_path / "NSIRD_screencaptureui_OZHpGn"
    shots.mkdir()
    for n in (1, 2):
        (shots / f"Screenshot 2026-09-30 at 3.11.0{n} PM.png").write_bytes(PNG)
    import shlex

    line = "\n".join(
        f"{shlex.quote(str(shots / f'Screenshot 2026-09-30 at 3.11.0{n} PM.png'))}  - This is slide {n}"
        for n in (1, 2)
    )
    images, problems = attachments(line, tmp_path)
    assert [i.name for i in images] == [
        "Screenshot 2026-09-30 at 3.11.01 PM.png",
        "Screenshot 2026-09-30 at 3.11.02 PM.png",
    ]
    assert problems == []


def test_only_real_images_are_attached(tmp_path):
    from tinyharness.cli.repl import attachments

    (tmp_path / "fake.png").write_text("not an image")
    (tmp_path / "broken.png").write_bytes(PNG[:20])
    (tmp_path / "notes.txt").write_text("hi")
    images, problems = attachments("look at fake.png broken.png notes.txt missing.png", tmp_path)
    assert images == []
    assert problems[0] == "fake.png isn't a PNG, JPEG, GIF or WebP image"
    assert problems[1].startswith("broken.png: not a readable image")


def test_a_pasted_retina_screenshot_is_shrunk(tmp_path):
    """It used to go in at full size, and a provider rejected every later request."""
    import base64
    import io

    from PIL import Image as PILImage

    from tinyharness.cli.repl import attachments

    (tmp_path / "shot.png").write_bytes(png(3024, 1964))
    [image], _ = attachments("look at shot.png", tmp_path)
    assert max(PILImage.open(io.BytesIO(base64.b64decode(image.data))).size) == 2000


@pytest.mark.parametrize("compact", [False, True])
def test_pip_greets_you_on_every_launch_when_the_terminal_can_show_him(tmp_path, monkeypatch, compact):
    import io

    from rich.console import Console

    from tinyharness import Session, SessionOptions
    from tinyharness.cli.app import print_banner
    from tinyharness.cli.theme import THEME

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    session = Session(SessionOptions(config_path=tmp_path / "none.toml", working_dir=tmp_path))
    try:
        color = Console(
            file=io.StringIO(), force_terminal=True, color_system="truecolor", width=100, theme=THEME
        )
        print_banner(color, session, compact=compact)
        assert "▀" in color.file.getvalue()  # Scout, in half-blocks
        plain = Console(file=io.StringIO(), width=100, color_system=None, theme=THEME)
        print_banner(plain, session, compact=compact)
        assert "▀" not in plain.file.getvalue()  # no colors, no Scout
    finally:
        session.close()


async def test_a_run_without_a_terminal_denies_approvals_instead_of_prompting(monkeypatch):
    """Seen live: `tinyharness -p ...` from a script ended with "OSError: [Errno 22]" the
    first time the model asked for something that needs approval (a host-boundary action):
    the prompt tried to open a terminal that wasn't there. With no terminal there's no
    one to ask: the gate denies it and tells the model why."""
    import argparse
    import io

    from rich.console import Console

    from tinyharness.cli import app

    seen = {}

    def fake_open(options, console, interactive):
        seen["callback"], seen["interactive"] = options.permission_callback, interactive
        return None

    monkeypatch.setattr(app, "open_session", fake_open)
    monkeypatch.setattr(app.sys, "stdin", io.StringIO(""))  # not a TTY
    args = argparse.Namespace(
        config=None, cwd=None, provider=None, model=None, mode=None, no_skill_tool=False,
        prompt="hi", setup=False, resume_latest=False, resume=None,
    )  # fmt: skip
    await app.run(args, Console(file=io.StringIO()))
    assert seen == {"callback": None, "interactive": False}


def test_sandbox_and_extensions_commands_show_config_names_literally(tmp_path, monkeypatch):
    """`[sandbox]` in a Rich string is markup and vanished from the hint."""
    import io

    from rich.console import Console

    from tinyharness import Session, SessionOptions
    from tinyharness.cli.commands import COMMANDS
    from tinyharness.cli.theme import THEME

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class Repl:
        pass

    repl = Repl()
    repl.session = Session(SessionOptions(working_dir=tmp_path / "w", config_path=tmp_path / "c.toml"))
    repl.console = Console(file=io.StringIO(), width=200, theme=THEME, color_system=None)
    try:
        COMMANDS["/sandbox"].handler(repl, "")
        COMMANDS["/extensions"].handler(repl, "on todo")
        repl.session.extensions[0].needs = ["network:api.example.com"]
        COMMANDS["/extensions"].handler(repl, "")
        out = repl.console.file.getvalue()
        if repl.session.sandbox is not None:
            assert "[sandbox] allow_hosts" in out and "pypi.org" in out
        assert "[extensions] enabled" in out and "needs network:api.example.com" in out
    finally:
        repl.session.close()


def test_cost_shows_the_latest_calls_cache_hits_and_the_fixed_prefix(tmp_path, monkeypatch):
    """Whether the prompt cache is working shows on the latest call, not in a session total."""
    import io

    from rich.console import Console

    from tinyharness import Session, SessionOptions
    from tinyharness.cli.commands import COMMANDS
    from tinyharness.cli.theme import THEME
    from tinyharness.llm.usage import Usage

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    class Repl:
        pass

    repl = Repl()
    repl.session = Session(SessionOptions(working_dir=tmp_path / "w", config_path=tmp_path / "c.toml"))
    repl.console = Console(file=io.StringIO(), width=200, theme=THEME, color_system=None)
    try:
        repl.session.ctx.usage.add(Usage(input=9000, cached=0, output=100))
        repl.session.ctx.usage.add(Usage(input=200, cached=9000, output=50))
        COMMANDS["/cost"].handler(repl, "")
        out = repl.console.file.getvalue()
        assert "98% of its 9,200 input tokens read from the cache" in out
        assert "fixed prefix" in out and "system prompt and tools" in out
    finally:
        repl.session.close()
