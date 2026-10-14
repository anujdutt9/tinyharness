"""What the CLI shows: tool rows, the turn footer, the status bar, tokens and cost."""

import json
import re
import subprocess
import time

import pytest
from rich.console import Console
from rich.text import Text

from conftest import ScriptedLLM
from tinyharness import Session, SessionOptions
from tinyharness.cli.render import Renderer, _preview_line, _RunningTool, dollars, tokens
from tinyharness.cli.repl import Repl, git_status
from tinyharness.cli.theme import ellipsize
from tinyharness.config import note_launch
from tinyharness.events import (
    AssistantText,
    Done,
    PermissionDecision,
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
        ("Grep(x)", "Grep", {}, "7 total match(es) across 3 file(s):\na.py:1: x", False),
    )
    read, edit, grep, match = shown(renderer)
    assert read.startswith("✓ Read    src/app.py") and read.endswith("3 lines")
    assert edit.startswith("✓ Edit    src/app.py") and edit.endswith("+4 −1")
    assert grep.endswith("7 matches in 3 files")
    assert len({len(read), len(edit), len(grep)}) == 1  # right-aligned to the terminal's width
    assert match.strip() == "⎿ a.py:1: x"  # what a search found shows under it


def test_command_output_is_previewed_under_the_row(renderer):
    output = "\n".join(f"row {i}" for i in range(1, 9))
    replay(renderer, ("Bash(ls)", "Bash", {"command": "ls"}, output, False))
    row, *preview = shown(renderer)
    assert row.rstrip() == "✓ Bash    ls"  # nothing crammed on the right
    assert [line.strip() for line in preview] == ["⎿ row 1", "row 2", "row 3", "… 5 more lines"]


def test_command_output_that_mentions_timeout_is_not_reported_as_timed_out(renderer):
    """Seen live: a source diff containing the timeout marker was shown as a failed command."""
    output = '+if "(timed out after" in text:\n+    handle_timeout()'
    replay(renderer, ("Bash(git diff)", "Bash", {"command": "git diff"}, output, False))
    assert shown(renderer)[0].startswith("✓ Bash")


def test_sequential_tool_duration_excludes_time_waiting_for_the_previous_tool(renderer, monkeypatch):
    """Seen live: a 33ms Git diff displayed the preceding pytest run's 58-second duration."""
    now = 10.0
    monkeypatch.setattr(time, "monotonic", lambda: now)
    renderer.handle(ToolUse("Bash", {"command": "pytest"}, "Bash(pytest)", id="tests"))
    renderer.handle(ToolUse("Bash", {"command": "git diff"}, "Bash(git diff)", id="diff"))
    renderer.handle(ToolsDispatched(["Bash", "Bash"], False))
    now = 68.0
    renderer.handle(ToolResult("Bash", "1 passed", False, id="tests"))
    now = 68.03
    renderer.handle(ToolResult("Bash", "+change", False, id="diff"))
    diff_row = next(line for line in shown(renderer) if "git diff" in line)
    assert "58s" not in diff_row


def test_git_diff_previews_color_additions_and_removals():
    """Seen live: a Git diff stat was rendered as one muted block despite its useful +/- signal."""
    tool = _RunningTool("Bash(git diff --stat)", {"command": "git diff --stat"})
    stat = _preview_line(tool, " app.py | 4 ++--", False)
    added = _preview_line(tool, "+new line", False)
    removed = _preview_line(tool, "-old line", False)
    assert [span.style for span in stat.spans] == ["diff.add", "diff.add", "diff.del", "diff.del"]
    assert added.style == "diff.add" and removed.style == "diff.del"


def test_who_allowed_a_call_without_asking_is_credited_under_it(renderer):
    renderer.handle(ToolUse("Bash", {"command": "ls"}, "Bash(ls)", id="c1"))
    renderer.handle(PermissionDecision("Bash", True, "classifier", "read-only listing"))
    renderer.handle(ToolResult("Bash", "(no output)", False, id="c1"))
    row, credit = shown(renderer)
    assert row.endswith("no output") and credit.strip() == "⎿ auto mode · read-only listing"


def test_failed_commands_show_why_underneath(renderer):
    output = "> build\nsrc/app.ts(12,5): error TS2322: bad type\n(exit code 2)"
    replay(renderer, ("Bash(npm run build)", "Bash", {"command": "npm run build"}, output, False))
    row, *why = shown(renderer)
    assert row.startswith("✗ Bash    npm run build") and row.endswith("exit 2")
    assert [line.strip() for line in why] == ["⎿ > build", "src/app.ts(12,5): error TS2322: bad type"]


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
    assert footer.startswith("── ")
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
    # "no sandbox" when there's none (as when tinyharness itself runs in one: they don't nest)
    assert ("no sandbox" in bar) == (repl.session.sandbox is None)


async def test_context_status_refreshes_after_each_model_call(repl):
    """Seen live: context stayed at 0% throughout a long, multi-step first turn."""

    async def events():
        yield TokenUsage("model", input=20_000, cached=10_000, output=100, cost=0.01)

    received = [event async for event in repl._status_events(events())]
    assert len(received) == 1
    assert repl.context_used == pytest.approx(30_000 / repl.session.context_window)
    assert "~" not in text(repl._status(200))


async def test_status_estimates_usage_while_the_model_is_streaming(repl):
    """Seen live: context, output tokens and cost stayed stale until a reply completed."""
    cfg = repl.session.config
    cfg.pricing[f"{cfg.provider}:{cfg.model}"] = Price(input=1.0, output=10.0)

    async def events():
        yield AssistantText("x" * 800)
        yield TokenUsage("model", input=20_000, cached=10_000, output=210, cost=0.04)

    stream = repl._status_events(events())
    assert isinstance(await anext(stream), AssistantText)
    live = text(repl._status(200))
    assert "~" in live and "200 out" in live and "~<" in live
    assert repl.context_used > 0

    assert isinstance(await anext(stream), TokenUsage)
    exact = text(repl._status(200))
    assert "~" not in exact
    assert repl.context_used == pytest.approx(30_000 / repl.session.context_window)


def test_idle_status_marks_next_request_context_as_estimated(repl):
    """The idle context is locally tokenized, while completed output and cost are exact."""
    repl.session.ctx.usage.add(Usage(input=8_000, output=741, cost=0.07))
    repl._measure_context()
    status = text(repl._status(200))
    assert "~" in status.split(" context")[0]
    assert "741 out" in status and "~741 out" not in status
    assert "$0.07" in status and "~$0.07" not in status


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
    from tinyharness.tools.bash import BashInput, BashTool

    output = BashTool(None, SandboxPolicy(write_roots=()))._report(
        "E   PermissionError: Operation not permitted\n8 failed, 462 passed in 10.29s",
        code=1,
        timed_out=False,
        args=BashInput(command="pytest -q"),
        sandboxed=True,
    )
    assert output.endswith("(exit code 1)")
    assert "tinyharness:filesystem_or_process_denied" in output
    replay(renderer, ("Bash(pytest -q)", "Bash", {"command": "pytest -q"}, output, False))
    renderer.handle(Done())
    lines = shown(renderer)
    row = next(line for line in lines if line.startswith("✗ Bash"))
    first = next(line for line in lines if "PermissionError" in line)
    footer = lines[-1]
    assert row.startswith("✗ Bash") and "exit 1" in row
    assert first.strip() == "⎿ E   PermissionError: Operation not permitted"
    assert any(line.strip() == "8 failed, 462 passed in 10.29s" for line in lines)
    assert "tests ✗" in footer


def test_an_app_that_crashes_in_the_sandbox_says_to_run_it_outside():
    """Chrome.app aborts under Seatbelt (it needs LaunchServices); the model used to guess."""
    from tinyharness.sandbox import SandboxPolicy
    from tinyharness.tools.bash import BashInput, BashTool

    chrome = (
        '"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --screenshot=a.png x.html'
    )
    output = BashTool(None, SandboxPolicy(write_roots=()))._report(
        "/bin/bash: line 1: 48216 Abort trap: 6  ...",
        code=134,
        timed_out=False,
        args=BashInput(command=chrome),
        sandboxed=True,
    )
    assert "tinyharness:host_application_blocked" in output
    assert output.endswith("(exit code 134)")


def test_cost_explains_where_the_tokens_went():
    from types import SimpleNamespace

    from tinyharness.cli.commands import cost
    from tinyharness.cli.theme import THEME

    meter = UsageMeter()
    meter.add(Usage(input=3164, cached=0, output=126, cost=0.001))
    meter.add(Usage(input=934, cached=82128, output=1137, cost=0.003))
    console = Console(width=120, record=True, theme=THEME)
    session = SimpleNamespace(ctx=SimpleNamespace(usage=meter), prefix_tokens=lambda: 2900)
    cost(SimpleNamespace(console=console, session=session), "")
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


def test_writing_a_file_is_not_called_thinking(renderer):
    """After the model reasons it may spend minutes writing a file as a tool call; the
    spinner said "Thinking" all along, and the thought's time included the writing."""
    from tinyharness.events import Reasoning, ToolInputStreaming

    renderer.wait("Working")
    renderer.handle(Reasoning("Planning the deck."))
    assert renderer._activity == "Thinking"
    renderer.handle(ToolInputStreaming("Write", 0))
    assert renderer._activity == "Writing a file"
    assert any(line.startswith("∴ Thought for") for line in shown(renderer))  # ended when writing began
    renderer.handle(ToolInputStreaming("Write", 18_400, "/repo/out/deck.html"))
    assert (renderer._activity, renderer._activity_detail) == ("Writing out/deck.html", "18.4k chars")
    spinner = Text.from_ansi(renderer.live_ansi(100)).plain  # colored or not, depending on the terminal
    assert "Writing out/deck.html · 18.4k chars…" in spinner and "Thinking" not in spinner
    renderer.handle(ToolUse("Write", {"path": "/repo/out/deck.html"}, "Write(/repo/out/deck.html, 1 lines)"))
    assert renderer._activity == "Working"


def test_other_tools_being_written_get_their_own_label(renderer):
    from tinyharness.events import ToolInputStreaming

    for event, label in [
        (ToolInputStreaming("Edit", 10, "/repo/src/app.py"), "Editing src/app.py"),
        (ToolInputStreaming("Bash", 10, "pytest -q"), "Writing a command"),
        (ToolInputStreaming("WebFetch", 10, "https://x.example"), "Preparing WebFetch"),
    ]:
        renderer.handle(event)
        assert renderer._activity == label


def test_parallel_results_land_on_their_own_rows_with_their_own_times(renderer):
    """Three searches share a name; the one that finishes first isn't always the first."""
    for i in range(3):
        renderer.handle(ToolUse("WebSearch", {"query": f"q{i}"}, f"WebSearch(query {i})", id=f"c{i}"))
    renderer.handle(ToolsDispatched(["WebSearch"] * 3, True))
    for tool in renderer._running:
        tool.started -= 10  # dispatched 10s ago
    renderer._running[2].started += 7  # ...but the third only 3s ago
    renderer.handle(ToolResult("WebSearch", "Search results for 'query 2'", id="c2"))
    renderer.handle(ToolResult("WebSearch", "Search results for 'query 0'", id="c0"))
    renderer.handle(ToolResult("WebSearch", "Search results for 'query 1'", id="c1"))
    rows = [line for line in shown(renderer) if line.startswith("✓ WebSearch")]
    assert [row.split()[2] for row in rows] == ["query", "query", "query"]
    for row, n, seconds in zip(rows, (2, 0, 1), ("3s", "10s", "10s"), strict=True):
        assert f"query {n}" in row.split("  ")[1] and f"'query {n}'" in row and row.endswith(seconds)


def test_a_subagent_folds_into_its_row_when_the_agent_was_left_to_default(renderer):
    args = {"description": "check the retry logic", "prompt": "..."}  # no subagent_type
    renderer.handle(ToolUse("Agent", args, "Agent(general-purpose: check the retry logic)"))
    renderer.handle(ToolsDispatched(["Agent"], False))
    renderer.handle(
        SubagentUpdate("a1", "general-purpose", "check the retry logic", "done", "", 3, 3.2, "haiku")
    )
    renderer.handle(ToolResult("Agent", "Looks right."))
    [row] = shown(renderer)  # folded, not a dangling line of its own
    assert row.startswith("✓ Agent   general-purpose: check the retry logic")


def test_ctrl_o_shows_a_running_subagents_calls_so_far(repl, tmp_path):
    import json

    from tinyharness.cli.theme import THEME
    from tinyharness.events import SubagentUpdate

    log = tmp_path / "sidechain.jsonl"
    log.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"event": "tool_use", "tool": "Bash", "args": {"command": "node measure.mjs"}},
                {
                    "event": "tool_result",
                    "tool": "Bash",
                    "result": "TypeError: x is undefined\n(exit code 1)",
                },
                {"event": "tool_use", "tool": "Edit", "args": {"path": "measure.mjs"}},
            ]
        )
        + '\n{"event": "tool_res'  # a line still being written
    )
    repl.console = Console(width=100, record=True, theme=THEME)
    repl.renderer.handle(
        SubagentUpdate("a1", "general-purpose", "Verify deck", "working", "", 2, 30.0, log=str(log))
    )
    repl.show_last_output()
    text = repl.console.export_text()
    assert "general-purpose · Verify deck · 2 tools so far" in text
    assert "Bash  node measure.mjs" in text and "⎿ TypeError: x is undefined" in text
    assert "Edit  measure.mjs" in text


async def test_back_to_back_questions_both_get_the_keyboard(repl, monkeypatch):
    """Two approvals in a row (a batch of Read + Grep outside the project): the second used
    to arrive while the input box was coming back, found it not yet running, and waited
    forever for a keyboard the new input box held. A dumb terminal uses prompt-toolkit's
    separate fallback application, which must be closed instead of PromptSession.app."""
    import asyncio

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    answered = []
    monkeypatch.setenv("TERM", "dumb")

    async def questions() -> None:
        await asyncio.sleep(0.3)  # the input box is up
        for name in ("Read", "Grep"):
            async with repl.exclusive():
                answered.append(name)

    with create_pipe_input() as keys, create_app_session(input=keys, output=DummyOutput()):
        loop = asyncio.create_task(repl.loop())
        try:
            await asyncio.wait_for(questions(), 3)
        finally:
            repl.running = False
            keys.send_text("\x04")  # Ctrl-D ends the input loop
            await asyncio.wait_for(loop, 3)
    assert answered == ["Read", "Grep"]


async def test_a_half_typed_message_survives_back_to_back_questions(repl):
    import asyncio

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as keys, create_app_session(input=keys, output=DummyOutput()):
        loop = asyncio.create_task(repl.loop())
        try:
            await asyncio.sleep(0.3)
            keys.send_text("half-typed")
            await asyncio.sleep(0.2)
            for _ in range(2):
                async with repl.exclusive():
                    pass
            await asyncio.sleep(0.3)  # the input box is back
            assert repl._prompt.app.current_buffer.text == "half-typed"
        finally:
            repl.running = False
            keys.send_text("\x15\x04")  # clear the line, then Ctrl-D
            await asyncio.wait_for(loop, 3)


def test_reading_a_missing_file_is_a_quiet_not_found(renderer):
    """The model checking whether a file exists isn't an error worth red text."""
    replay(
        renderer,
        (
            "Read(/repo/AGENTS.md)",
            "Read",
            {"path": "/repo/AGENTS.md"},
            "file not found: /repo/AGENTS.md",
            True,
        ),
    )
    [row] = shown(renderer)
    assert row.startswith("✗ Read") and row.endswith("not found")
    assert "error" not in row


def test_the_footer_counts_the_agents_work_too(renderer):
    """A team review showed "17 tools" after 128: the agents' 111 were left out."""
    from tinyharness.events import SubagentUpdate

    replay(renderer, ("Read(/repo/a.py)", "Read", {"path": "/repo/a.py"}, "x", False))
    for run_id, tools in (("a", 60), ("b", 51)):
        renderer.handle(SubagentUpdate(run_id, "reviewer", "review", "done", "", tools, 30.0))
    assert "112 tools (111 by 2 agents)" in renderer.footer().plain


def test_a_delivered_message_says_so(renderer):
    from tinyharness.events import MessageDelivered

    renderer.handle(MessageDelivered("/data/nova_ai.db  Here is the sqlite database file"))
    assert shown(renderer) == [
        "  ↳ delivered to the agent: /data/nova_ai.db Here is the sqlite database file"
    ]


def test_waiting_messages_show_above_the_input_with_how_to_send_them_now(repl):
    repl.session.steer("use the db in demo_data/")
    pending = repl._pending(100)
    assert "Sent after the agent's current step (esc: stop it and send now)" in pending
    assert "↳ use the db in demo_data/" in pending
    repl.session.take_steering()
    assert repl._pending(100) == ""


async def test_esc_stops_the_turn_and_sends_the_waiting_message_now(repl):
    """A message typed mid-step waits for the next step unless Esc sends it immediately."""
    import asyncio

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from tinyharness.llm.types import StreamChunk

    prompts = []

    async def stream(messages, tools=None, **kw):
        prompts.append(messages[-1].content)
        if len(prompts) == 1:
            await asyncio.sleep(30)  # a long first step
        yield StreamChunk(text="ok", finish_reason="stop")

    repl.session.llm.stream = stream
    with create_pipe_input() as keys, create_app_session(input=keys, output=DummyOutput()):
        loop = asyncio.create_task(repl.loop())
        try:
            await asyncio.sleep(0.3)
            keys.send_text("start\r")
            await asyncio.sleep(0.3)
            keys.send_text("use the db in demo_data/\r")  # queued for the next step
            await asyncio.sleep(0.3)
            assert repl.session.pending_steering == ("use the db in demo_data/",)
            keys.send_text("\x1b")  # Esc
            for _ in range(40):
                if len(prompts) >= 2:
                    break
                await asyncio.sleep(0.1)
            assert prompts[:2] == ["start", "use the db in demo_data/"]  # then the next-message suggestion
            assert "Sending your message now." in repl.console.export_text()
        finally:
            repl.running = False
            keys.send_text("\x04")
            await asyncio.wait_for(loop, 5)


async def test_alt_enter_still_makes_a_new_line_with_a_message_waiting(repl):
    """Esc now sends waiting messages; Alt+Enter starts with Esc too, and must still work."""
    import asyncio

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    with create_pipe_input() as keys, create_app_session(input=keys, output=DummyOutput()):
        loop = asyncio.create_task(repl.loop())
        try:
            repl._turn = asyncio.create_task(asyncio.sleep(30))  # a turn is running
            repl.session.steer("waiting")
            await asyncio.sleep(0.3)
            keys.send_text("a\x1b\rb")  # a, Alt+Enter, b
            await asyncio.sleep(0.3)
            assert repl._prompt.app.current_buffer.text == "a\nb"
            assert not repl._turn.cancelled()
        finally:
            repl._turn.cancel()
            repl.session.take_steering()
            repl.running = False
            repl._prompt.app.current_buffer.reset()  # Ctrl-D only exits an empty input
            keys.send_text("\x04")
            await asyncio.wait_for(loop, 5)


def test_the_spinner_clock_keeps_counting_through_blank_text(renderer, monkeypatch):
    """Some models stream blank lines between pieces of a tool call; the clock restarted
    each time (0s, 1s, 0s, 1s)."""
    import tinyharness.cli.render as render
    from tinyharness.events import ToolInputStreaming

    now = [100.0]
    monkeypatch.setattr(render.time, "monotonic", lambda: now[0])
    renderer.handle(ToolInputStreaming("Bash", 10))
    for _ in range(5):
        now[0] += 1
        renderer.handle(AssistantText("\n"))
        renderer.handle(ToolInputStreaming("Bash", 2000))
    assert renderer._activity == "Writing a command"
    assert int(now[0] - renderer._activity_since) == 5


def test_a_multi_line_command_takes_one_row(renderer):
    """A heredoc shows its first line and how many follow, not its body over several rows."""
    command = "cat > /tmp/q.sql << 'EOF'\n.mode json\nSELECT endpoint FROM usage\nEOF"
    replay(renderer, (f"Bash({command})", "Bash", {"command": command}, "", False))
    (row,) = shown(renderer)
    assert row.startswith("✓ Bash    cat > /tmp/q.sql << 'EOF' (+3 lines)")


def test_calls_allowed_by_the_users_own_rules_get_no_credit_line(renderer):
    renderer.handle(ToolUse("Bash", {"command": "ls"}, "Bash(ls)", id="c1"))
    renderer.handle(PermissionDecision("Bash", True, "rule", "matches allow rule Bash(ls)"))
    renderer.handle(ToolResult("Bash", "(no output)", False, id="c1"))
    assert len(shown(renderer)) == 1


def test_a_passing_test_run_shows_its_result_on_the_row():
    from tinyharness.cli.render import _test_summary

    assert (
        _test_summary(
            "...\n----------------------------------------------------------------------\nRan 3 tests in 0.001s\n\nOK\n"
        )
        == "3 passed"
    )
    assert _test_summary("tests/a.py ....\n======= 212 passed in 4.10s =======") == "212 passed in 4.10s"
    assert _test_summary("all good") == ""


def test_a_passing_test_row_needs_no_output_under_it():
    console = Console(width=100, record=True)
    renderer = Renderer(console)
    renderer.handle(ToolUse("Bash", {"command": "python3 -m unittest"}, "Bash(python3 -m unittest)", id="t"))
    renderer.handle(ToolResult("Bash", "...\n------\nRan 3 tests in 0.001s\n\nOK", False, id="t"))
    renderer._stop_live()
    out = console.export_text()
    assert "3 passed" in out and "Ran 3 tests" not in out
