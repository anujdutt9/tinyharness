"""What the CLI shows: tool rows, the turn footer, the status bar, tokens and cost."""

import json
import re
import subprocess
import time

import pytest
from rich.console import Console

from conftest import ScriptedLLM
from tinyharness import Session, SessionOptions
from tinyharness.cli.render import Renderer, dollars, tokens
from tinyharness.cli.repl import Repl, git_status
from tinyharness.cli.theme import ellipsize
from tinyharness.config import note_launch
from tinyharness.events import (
    AssistantText,
    Done,
    Reasoning,
    SubagentUpdate,
    TokenUsage,
    ToolResult,
    ToolsDispatched,
    ToolUse,
)
from tinyharness.llm.types import StreamChunk
from tinyharness.llm.usage import Price, Usage, UsageMeter, price_from_listing, read_usage

# ---- reading usage from each API ----


def test_anthropic_usage_counts_cache_reads_and_writes():
    usage = read_usage(
        {
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 900,
                    "cache_creation_input_tokens": 50,
                    "output_tokens": 1,
                }
            },
        },
        None,
    )
    usage = read_usage({"type": "message_delta", "usage": {"output_tokens": 42}}, usage)
    assert usage == Usage(input=150, cached=900, output=42)


def test_openai_usage_separates_cached_input():
    event = {
        "type": "response.completed",
        "response": {
            "usage": {
                "input_tokens": 1000,
                "input_tokens_details": {"cached_tokens": 800},
                "output_tokens": 50,
            }
        },
    }
    assert read_usage(event, None) == Usage(input=200, cached=800, output=50)


def test_chat_usage_and_a_reported_cost():
    event = {"choices": [], "usage": {"prompt_tokens": 500, "completion_tokens": 20, "cost": 0.0021}}
    assert read_usage(event, None) == Usage(input=500, cached=0, output=20, cost=0.0021)
    assert read_usage({"type": "content_block_delta"}, None) is None


def test_prices():
    price = price_from_listing(
        {"prompt": "0.000003", "completion": "0.000015", "input_cache_read": "0.0000003"}
    )
    assert price == Price(3.0, 15.0, 0.3)
    assert price.cost(Usage(input=1_000_000, cached=1_000_000, output=100_000)) == pytest.approx(4.8)
    assert price_from_listing({"prompt": "free"}) is None
    assert Price(2.0, 8.0).cost(Usage(cached=1_000_000)) == 2.0  # no cache price: full rate


# ---- the loop accounts for every model call ----


class MeteredLLM(ScriptedLLM):
    async def stream(self, messages, tools=None, **kw):
        async for chunk in super().stream(messages, tools, **kw):
            yield chunk
        yield StreamChunk(usage=Usage(input=1000, cached=0, output=100))


async def test_usage_is_priced_from_the_config_and_totaled(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = tmp_path / "config.toml"
    config.write_text('[pricing."openai:gpt-6-sol"]\ninput = 2.0\noutput = 10.0\n')
    session = Session(
        SessionOptions(config_path=config, working_dir=tmp_path / "w", model="openai:gpt-6-sol")
    )
    try:
        session.llm.stream = MeteredLLM([[("Glob", '{"pattern": "*"}')], "done"]).stream
        events = [e async for e in session.run("go")]
        usages = [e for e in events if isinstance(e, TokenUsage)]
        assert len(usages) == 2 and usages[0].cost == pytest.approx(0.003)
        assert session.ctx.usage.total.tokens == 2200
        assert session.ctx.usage.cost == pytest.approx(0.006)
    finally:
        session.close()


def test_without_a_price_the_cost_is_unknown_not_zero():
    meter = UsageMeter()
    meter.add(Usage(input=10, output=5))
    assert meter.cost is None and meter.unpriced == 1


# ---- tool rows and the footer ----


def replay(renderer: Renderer, *calls):
    for summary, name, args, output, is_error in calls:
        renderer.handle(ToolUse(name, args, summary))
        renderer.handle(ToolsDispatched([name], False))
        renderer.handle(ToolResult(name, output, is_error))


@pytest.fixture
def renderer():
    r = Renderer(Console(width=100, record=True))
    r.root = "/repo/"
    r.meter = UsageMeter()
    r._turn.started = time.monotonic()
    return r


def shown(renderer: Renderer) -> list[str]:
    return [line.rstrip() for line in renderer.console.export_text().splitlines() if line.strip()]


def test_each_call_is_one_row_with_its_result_on_the_right(renderer):
    replay(
        renderer,
        ("Read(/repo/src/app.py)", "Read", {"path": "/repo/src/app.py"}, "a\nb\nc", False),
        ("Edit(/repo/src/app.py, -1/+4 lines)", "Edit", {"path": "/repo/src/app.py"}, "edited", False),
        ("Grep(x)", "Grep", {}, "7 total match(es) across 3 file(s):\na", False),
    )
    read, edit, grep = shown(renderer)
    assert read.startswith("✓ Read    src/app.py") and read.endswith("3 lines")
    assert edit.startswith("✓ Edit    src/app.py") and edit.endswith("+4 −1")
    assert grep.endswith("7 matches in 3 files")
    assert len({len(read), len(edit), len(grep)}) == 1  # right-aligned to the terminal's width


def test_failed_commands_show_why_underneath(renderer):
    output = "> build\nsrc/app.ts(12,5): error TS2322: bad type\n(exit code 2)"
    replay(renderer, ("Bash(npm run build)", "Bash", {"command": "npm run build"}, output, False))
    row, why = shown(renderer)
    assert row.startswith("✗ Bash    npm run build") and row.endswith("exit 2")
    assert why.strip() == "⎿ src/app.ts(12,5): error TS2322: bad type"


def test_rejected_and_denied_calls(renderer):
    replay(
        renderer,
        ("Bash(rm -rf /)", "Bash", {}, "The user rejected this tool call.", True),
        ("Write(/etc/x, 1 lines)", "Write", {}, "denied by rule: blocked by a deny rule", True),
    )
    rejected, denied = shown(renderer)
    assert rejected.startswith("⊘ Bash") and rejected.endswith("rejected")
    assert denied.startswith("⊘ Write") and "denied by rule" in denied


def test_the_footer_sums_up_the_turn(renderer):
    replay(
        renderer,
        ("Edit(/repo/a.py, -3/+14 lines)", "Edit", {"path": "/repo/a.py"}, "ok", False),
        ("Write(/repo/tests/test_a.py, 41 lines)", "Write", {"path": "/repo/tests/test_a.py"}, "ok", False),
        ("Bash(uv run pytest -q)", "Bash", {"command": "uv run pytest -q"}, "212 passed in 4.1s", False),
    )
    renderer.meter.add(Usage(input=8200, cached=3100, output=1050, cost=0.034))
    renderer.meter.add(Usage(input=900, cached=11300, output=450, cost=0.002))  # re-sends it all
    renderer.handle(Done())
    footer = shown(renderer)[-1]
    assert footer.startswith("✻ ")
    # The largest call and the output; not 24.9k, the sum of both calls' re-sent input.
    assert "3 tools · 2 files +55 −3 · tests ✓ · 12.2k context · 1.5k out · $0.04" in footer


def test_a_failing_test_run_is_a_red_cross(renderer):
    replay(renderer, ("Bash(pytest)", "Bash", {"command": "pytest -x"}, "1 failed\n(exit code 1)", False))
    renderer.handle(Done())
    assert "tests ✗" in shown(renderer)[-1]


def test_plain_answers_get_no_footer(renderer):
    renderer.handle(AssistantText("Hi."))
    renderer.handle(Done())
    assert shown(renderer) == ["Hi."]


def test_subagents_are_shown_under_their_call(renderer):
    args = {"subagent_type": "reviewer", "description": "check the retry logic"}
    renderer.handle(ToolUse("Agent", args, "Agent(reviewer: check the retry logic)"))
    renderer.handle(ToolsDispatched(["Agent"], False))
    update = SubagentUpdate(
        "a1", "reviewer", "check the retry logic", "done", "3 tool calls", 3, 3.2, "haiku"
    )
    renderer.handle(update)
    renderer.handle(ToolResult("Agent", "Looks right."))
    [row] = shown(renderer)  # the finished agent folds into its call's row
    assert row.startswith("✓ Agent   reviewer: check the retry logic") and row.endswith("haiku · 3 tools")


def test_a_team_collapses_to_one_row_unless_someone_failed(renderer):
    renderer.handle(ToolUse("Team", {}, "Team(compare laptops · 3 tasks)"))
    renderer.handle(ToolsDispatched(["Team"], False))
    for run_id, status, tools in (("a", "done", 12), ("b", "failed", 3), ("c", "done", 9)):
        label = f"{run_id}: Laptop {run_id.upper()}"
        renderer.handle(SubagentUpdate(run_id, "researcher", label, status, "", tools, 20.0, "haiku"))
    renderer.handle(ToolResult("Team", "report"))
    row, failed = shown(renderer)
    assert row.startswith("✗ Team    compare laptops · 3 tasks") and row.endswith("3 agents · 24 tools")
    assert failed.startswith("  └ ✗ researcher  b: Laptop B")


def test_last_output_is_kept_in_full_for_ctrl_o(renderer):
    replay(renderer, ("Bash(ls)", "Bash", {"command": "ls"}, "a\nb\nc", False))
    assert renderer.last_output == ("Bash(ls)", "a\nb\nc")


def test_numbers():
    assert [tokens(n) for n in (950, 12_300, 1_000, 1_200_000)] == ["950", "12.3k", "1k", "1.2M"]
    assert [dollars(x) for x in (0.001, 0.034, 12.5)] == ["<$0.01", "$0.03", "$12.50"]
    assert ellipsize("src/very/long/path/to/handlers/auth.py", 20) == "src/very/…rs/auth.py"
    assert ellipsize("short.py", 20) == "short.py"


# ---- the status bar ----


def git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_git_status(tmp_path):
    assert git_status(tmp_path) is None
    git(tmp_path, "init", "-q", "-b", "trunk")
    assert git_status(tmp_path) == ("trunk", False)
    (tmp_path / "new.py").write_text("x")
    assert git_status(tmp_path) == ("trunk", True)


@pytest.fixture
def repl(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    work = tmp_path / "w"
    work.mkdir()
    git(work, "init", "-q", "-b", "main")
    session = Session(SessionOptions(config_path=tmp_path / "none.toml", working_dir=work))
    console = Console(width=120, record=True)
    yield Repl(session, console, Renderer(console))
    session.close()


def text(fragments) -> str:
    return "".join(t for _, t, *_ in fragments)


def test_the_status_bar_says_what_matters(repl):
    repl.session.ctx.usage.add(Usage(input=12_000, output=300, cost=0.42))
    repl.context_used = 0.38
    bar = text(repl._status(200))
    assert "⎇ main" in bar and "38% context" in bar and "300 out" in bar and "$0.42" in bar
    assert ("sandbox" in bar) == (repl.session.sandbox is not None)


def test_the_status_bar_drops_the_least_important_parts_first(repl):
    repl.session.ctx.usage.add(Usage(input=12_000, output=300, cost=0.42))
    narrow = text(repl._status(30))
    assert "context" in narrow and " out" not in narrow and "⎇" not in narrow
    assert len(narrow) <= 30


# ---- theme and banner ----


def test_light_theme_is_picked_from_the_terminal(monkeypatch):
    import importlib

    import tinyharness.cli.theme as theme

    monkeypatch.setenv("COLORFGBG", "0;15")
    monkeypatch.delenv("TINYHARNESS_THEME", raising=False)
    assert importlib.reload(theme).LIGHT
    monkeypatch.setenv("TINYHARNESS_THEME", "dark")
    assert not importlib.reload(theme).LIGHT
    monkeypatch.delenv("COLORFGBG")
    monkeypatch.delenv("TINYHARNESS_THEME")
    importlib.reload(theme)


def test_launches_are_counted(isolated_home):
    assert note_launch() == 1 and note_launch() == 2
    assert json.loads((isolated_home / "state.json").read_text())["launches"] == 2


# ---- theme: system, light, dark ----


def fake_terminal(monkeypatch, reply: bytes | None):
    """A pseudo-terminal that answers the background-color query like a real one."""
    import os
    import pty
    import sys
    import threading

    parent, child = pty.openpty()
    stdin, stdout = os.fdopen(child, "r"), os.fdopen(os.dup(child), "w")
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", stdout)
    asked = []

    def terminal():
        asked.append(os.read(parent, 64))
        if reply is not None:
            os.write(parent, reply)

    threading.Thread(target=terminal, daemon=True).start()
    return asked


@pytest.mark.parametrize(
    ("reply", "light"),
    [
        (b"\x1b]11;rgb:ffff/ffff/ffff\x1b\\", True),
        (b"\x1b]11;rgb:fafa/f9f9/f5f5\x07", True),
        (b"\x1b]11;rgb:1e1e/1e1e/2e2e\x1b\\", False),
        (b"\x1b]11;rgb:16/17/1d\x07", False),  # 2 hex digits per channel
        (None, None),  # a terminal that doesn't answer
    ],
)
def test_the_terminal_is_asked_for_its_background(monkeypatch, reply, light):
    from tinyharness.cli.theme import terminal_is_light

    asked = fake_terminal(monkeypatch, reply)
    start = time.monotonic()
    assert terminal_is_light() is light
    assert time.monotonic() - start < 0.5  # never holds up startup
    assert asked == [b"\x1b]11;?\x1b\\"]


def test_theme_setting_precedence(monkeypatch, tmp_path):
    from tinyharness.cli.theme import theme_setting
    from tinyharness.config import save_theme
    from tinyharness.paths import default_config_path

    monkeypatch.delenv("TINYHARNESS_THEME", raising=False)
    assert theme_setting() == ("system", "default")
    save_theme("light")
    assert theme_setting() == ("light", "/theme")
    default_config_path().parent.mkdir(parents=True, exist_ok=True)
    default_config_path().write_text('theme = "dark"\n')
    assert theme_setting() == ("dark", "config")
    monkeypatch.setenv("TINYHARNESS_THEME", "light")
    assert theme_setting() == ("light", "TINYHARNESS_THEME")


# ---- talking to the agent while it works ----


async def test_a_message_sent_mid_turn_reaches_the_agent_at_its_next_step(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    session = Session(SessionOptions(config_path=tmp_path / "none.toml", working_dir=tmp_path / "w"))
    llm = ScriptedLLM([[("Glob", '{"pattern": "*"}')], "done"])
    session.llm.stream = llm.stream
    try:
        async for event in session.run("look around"):
            if isinstance(event, ToolUse):
                session.steer("also check the docs")  # typed while the tool runs
        second_request = llm.requests[1]
        assert second_request[-1].role == "user" and second_request[-1].content == "also check the docs"
        assert session.take_steering() == []  # delivered, so nothing left over
        session.steer("too late for this turn")
        assert session.take_steering() == ["too late for this turn"]  # the REPL starts the next turn with it
    finally:
        session.close()


def test_the_live_region_never_shows_raw_link_codes():
    from tinyharness.cli.render import without_links

    linked = "\x1b]8;id=1;file:///repo/app.py\x1b\\app.py\x1b]8;;\x1b\\ done"
    assert without_links(linked) == "app.py done"


# ---- the model's reasoning ----


def think(renderer: Renderer, *pieces: str, seconds: float = 3.0):
    for piece in pieces:
        renderer.handle(Reasoning(piece))
    renderer._thought_since -= seconds  # as if it took that long


def test_reasoning_is_hidden_by_default_but_kept(renderer):
    think(renderer, "The bug is ", "in the retry loop.")
    renderer.handle(AssistantText("Found it."))
    assert shown(renderer)[0] == "∴ Thought for 3s · ctrl+t to show"
    assert renderer.last_thought == "The bug is in the retry loop."


def test_reasoning_can_be_shown(renderer):
    renderer.show_thinking = True
    think(renderer, "The bug is in the retry loop.")
    assert "The bug is in the retry loop." in renderer.live_ansi(100)  # while it thinks
    renderer.handle(AssistantText("Found it."))
    assert shown(renderer)[:2] == ["∴ Thinking", "  The bug is in the retry loop."]


def test_a_model_that_keeps_its_reasoning_private_just_shows_the_time(renderer):
    think(renderer, "")  # it said it's reasoning, but not what
    renderer.handle(Done())
    assert shown(renderer) == ["∴ Thought for 3s"]


def test_showing_reasoning_asks_the_provider_for_it(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    session = Session(SessionOptions(config_path=tmp_path / "none.toml", working_dir=tmp_path / "w"))
    try:
        assert not session.llm.show_reasoning
        session.set_show_thinking(True)
        assert session.llm.show_reasoning and session.config.show_thinking
    finally:
        session.close()


def test_a_sandboxed_test_run_that_fails_shows_as_failed(renderer):
    """The sandbox note used to come after `(exit code 1)`, hiding it: 8 failed tests showed ✓."""
    from tinyharness.sandbox import SandboxPolicy
    from tinyharness.tools.bash import BashTool

    output = BashTool(None, SandboxPolicy(write_roots=()))._report(
        "E   PermissionError: Operation not permitted\n8 failed, 462 passed in 10.29s",
        code=1,
        timed_out=False,
        timeout=120,
        sandboxed=True,
    )
    assert output.endswith("(exit code 1)") and "(note: the sandbox" in output
    replay(renderer, ("Bash(pytest -q)", "Bash", {"command": "pytest -q"}, output, False))
    renderer.handle(Done())
    row, under, footer = shown(renderer)[-3:]
    assert row.startswith("✗ Bash") and "exit 1" in row
    assert under.strip() == "⎿ 8 failed, 462 passed in 10.29s"  # the note isn't taken for the error
    assert "tests ✗" in footer


def test_cost_explains_where_the_tokens_went():
    from types import SimpleNamespace

    from tinyharness.cli.commands import cost
    from tinyharness.cli.theme import THEME

    meter = UsageMeter()
    meter.add(Usage(input=3164, cached=0, output=126, cost=0.001))
    meter.add(Usage(input=934, cached=82128, output=1137, cost=0.003))
    console = Console(width=120, record=True, theme=THEME)
    cost(SimpleNamespace(console=console, session=SimpleNamespace(ctx=SimpleNamespace(usage=meter))), "")
    text = console.export_text()
    assert "2 (subagents included)" in text
    assert "83,062 tokens (the most any one call sent)" in text
    assert "1,263 tokens (what the model wrote" in text
    assert "86,226 tokens · 95% of it cache reads" in text
    assert "<$0.01" in text


def test_ctrl_t_shows_the_last_reasoning_once(repl):
    """Pressing it again used to print the same reasoning block every time."""
    from tinyharness.events import Reasoning

    repl.renderer.handle(Reasoning("Weighing the edge cases."))
    repl.renderer.handle(AssistantText("Done."))
    repl.renderer.handle(Done())
    for _ in range(3):
        repl.show_thinking(True)
        repl.show_thinking(False)
    assert repl.console.export_text().count("Weighing the edge cases.") == 1


def test_ctrl_t_does_not_repeat_reasoning_already_shown_live(repl):
    from tinyharness.events import Reasoning

    repl.show_thinking(True)
    repl.renderer.handle(Reasoning("Checking the tests."))
    repl.renderer.handle(AssistantText("Done."))
    repl.renderer.handle(Done())
    repl.show_thinking(False)
    repl.show_thinking(True)
    assert repl.console.export_text().count("Checking the tests.") == 1


def test_clear_keeps_the_sessions_spend_and_says_so(repl):
    from tinyharness.cli.commands import clear, cost
    from tinyharness.cli.theme import THEME

    repl.console = Console(width=120, record=True, theme=THEME)
    meter = repl.session.ctx.usage
    meter.add(Usage(input=20_000, cached=60_000, output=14_100, cost=0.11))
    clear(repl, "")
    assert "Session so far: 14.1k out · $0.11 (/cost for details)" in repl.console.export_text()
    assert meter.total.output == 14_100 and meter.cost == 0.11  # the status bar keeps them

    meter.add(Usage(input=900, cached=0, output=40, cost=0.001))  # "hello world" after /clear
    cost(repl, "")
    text = repl.console.export_text()
    assert "This session (since launch; /clear doesn't reset it)" in text
    assert re.search(r"^  since /clear +1 call · 40 out · <\$0\.01", text, re.M)


def test_clear_before_any_calls_says_nothing_about_spend(repl):
    from tinyharness.cli.commands import clear
    from tinyharness.cli.theme import THEME

    repl.console = Console(width=120, record=True, theme=THEME)
    clear(repl, "")
    assert "Session so far" not in repl.console.export_text()
