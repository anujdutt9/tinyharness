"""What the CLI shows: tool rows, the turn footer, the status bar, tokens and cost."""

import inspect
import json
import re
import subprocess
import sys
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


def test_nonzero_commands_are_neutral_and_keep_their_evidence(renderer):
    """Seen live: every nonzero subprocess looked like a tinyharness crash to new users."""
    output = "> build\nsrc/app.ts(12,5): error TS2322: bad type\n(exit code 2)"
    replay(renderer, ("Bash(npm run build)", "Bash", {"command": "npm run build"}, output, False))
    row, *why = shown(renderer)
    assert row.startswith("· Bash    npm run build") and row.endswith("exit 2")
    assert [line.strip() for line in why] == ["⎿ > build", "src/app.ts(12,5): error TS2322: bad type"]


def test_no_output_nonzero_command_is_neutral(renderer):
    """Seen live: a grep with no match showed as a red `exit 1` though the tool worked."""
    command = 'cd sglang && grep -rn "apple silicon" docs'
    output = "(no output)\n(exit code 1)"
    replay(renderer, (f"Bash({command})", "Bash", {"command": command}, output, False))
    row, *preview = shown(renderer)
    assert row.startswith("· Bash") and row.endswith("exit 1")
    assert preview == []


def test_status_queries_need_no_command_specific_outcome_parser(renderer):
    for command in ("which missing && which python", "grep -rn x docs"):
        replay(renderer, (f"Bash({command})", "Bash", {"command": command}, "(exit code 1)", False))
    rows = shown(renderer)
    assert all(row.startswith("· Bash") and row.endswith("exit 1") for row in rows)


def test_installing_pytest_is_not_a_test_run(renderer):
    """Seen live: `uv pip install -e ".[srt_mps]" pytest pre-commit` counted as a test run,
    so a failed build showed `exit 1` (not `build failed`) and the footer said tests ✗."""
    command = 'cd python && uv pip install -e ".[srt_mps]" pytest pre-commit 2>&1 | tail -15'
    output = "hint: Build failures usually indicate a problem with the package\n(exit code 1)"
    replay(renderer, (f"Bash({command})", "Bash", {"command": command}, output, False))
    renderer.handle(Done())
    lines = shown(renderer)
    assert lines[0].endswith("build failed") and "tests" not in lines[-1]


@pytest.mark.parametrize(
    "command",
    ["pytest -q", "uv run pytest -x", ".venv/bin/python -m pytest tests", "cd py && python3.12 -m unittest",
     "FOO=1 npx jest", "npm test", "cargo test", "ls && ../.venv/bin/pytest --co | tail -3"],
)  # fmt: skip
def test_a_step_that_runs_tests_is_a_test_run(command):
    from tinyharness.cli.render import _is_test_command, _RunningTool

    assert _is_test_command(_RunningTool(f"Bash({command})", {"command": command}))


def test_a_sandbox_block_in_a_command_that_runs_tests_is_still_a_block(renderer):
    """Seen live: `import sglang` hit ~/.cache/sglang in a command that also ran
    `pytest --co`; the tests never ran, so the row should say the sandbox blocked it."""
    from pathlib import Path

    blocked = Path.home() / ".cache" / "sglang"
    command = ".venv/bin/python -c 'import sglang'; .venv/bin/python -m pytest test --co"
    output = (
        f"PermissionError: Operation not permitted: '{blocked}'\n"
        f"[tinyharness:filesystem_or_process_denied path={blocked}] This command stayed sandboxed.\n"
        "(exit code 1)"
    )
    replay(renderer, (f"Bash({command})", "Bash", {"command": command}, output, False))
    assert shown(renderer)[0].replace("\\", "/").endswith("sandbox blocked a write to ~/.cache/sglang")


def test_a_protected_path_block_says_it_needs_the_users_approval(renderer):
    command = ".venv/bin/pre-commit install"
    output = (
        "PermissionError: [Errno 1] Operation not permitted: '.git/hooks/pre-commit'\n"
        "[tinyharness:protected_path_denied path=.git/hooks/pre-commit] This command stayed sandboxed.\n"
        "(exit code 3)"
    )
    replay(renderer, (f"Bash({command})", "Bash", {"command": command}, output, False))
    row, *why = shown(renderer)
    row = row.replace("\\", "/")  # Windows shows its own separator
    assert row.startswith("⊘ Bash") and row.endswith("sandbox blocked a write to .git/hooks/pre-commit")
    assert "changing them needs your approval" in " ".join(line.strip() for line in why)


def test_tests_that_passed_before_a_later_step_failed_count_as_passed(renderer):
    """Seen live: `...; .venv/bin/pytest | tail -15; ls -a | grep pre-commit` printed
    `1 passed`, then the grep found nothing, and the footer said tests ✗."""
    command = ".venv/bin/pytest 2>&1 | tail -15; ls -a | grep pre-commit"
    output = "1 passed in 0.01s\n(exit code 1)"
    replay(renderer, (f"Bash({command})", "Bash", {"command": command}, output, False))
    renderer.handle(Done())
    lines = shown(renderer)
    assert lines[0].startswith("· Bash") and lines[0].endswith("exit 1")
    assert "tests ✓" in lines[-1]


def test_failing_tests_piped_into_tail_still_show_as_failed(renderer):
    """Seen live: Sonnet ran `uv run pytest -q 2>&1 | tail -15`, which exits 0, and the row
    said "3 passed" in green and the footer "tests ✓" over `1 failed, 3 passed`."""
    command = "uv run pytest -q 2>&1 | tail -15"
    output = "FAILED tests/test_stock.py::test_remove - DID NOT RAISE\n1 failed, 3 passed, 1 warning in 0.58s"
    replay(renderer, (f"Bash({command})", "Bash", {"command": command}, output, False))
    renderer.handle(Done())
    lines = shown(renderer)
    assert lines[0].startswith("✗ Bash") and lines[0].endswith("1 failed, 3 passed, 1 warning in 0.58s")
    assert "tests ✗" in lines[-1]


def test_a_zero_error_count_is_not_a_failure(renderer):
    command = "uv run pytest -q"
    replay(renderer, (f"Bash({command})", "Bash", {"command": command}, "12 passed, 0 errors in 1.2s", False))
    assert shown(renderer)[0].startswith("✓ Bash")


def test_a_failed_setup_step_keeps_its_error_without_becoming_a_harness_failure(renderer):
    command = "cd /missing && grep -rn x docs"
    output = "bash: line 1: cd: /missing: No such file or directory\n(exit code 1)"
    replay(renderer, (f"Bash({command})", "Bash", {"command": command}, output, False))
    row, preview = shown(renderer)
    assert row.startswith("· Bash") and preview.strip().startswith("⎿ bash: line 1: cd:")


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
    lines = shown(renderer)
    assert lines[0].endswith("1 failed") and "exit 1" not in lines[0]
    assert "tests ✗" in lines[-1]


def test_a_direct_python_test_file_failure_has_a_semantic_outcome(renderer):
    """Seen live: a unittest file piped through grep showed raw `exit 120` four rows into
    diagnosing the same macOS-only failure, although the meaningful outcome was test failure."""
    command = "python test/registered/unit/mem_cache/test_radix_cache_unit.py 2>&1 | grep -m1 Error"
    output = "RuntimeError: native hash is only supported on Linux\n(exit code 120)"
    replay(renderer, (f"Bash({command})", "Bash", {"command": command}, output, False))
    row, detail = shown(renderer)
    assert row.endswith("tests failed") and "exit 120" not in row
    assert "RuntimeError" in detail


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
    assert "OpenAI · GPT / gpt-6-sol" in bar
    assert "⎇ main" in bar and "38% context" in bar and "300 out" in bar and "$0.42" in bar
    # "sandbox off" when there's none (as when tinyharness itself runs in one: they don't nest)
    assert ("sandbox off" in bar) == (repl.session.sandbox is None)


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
@pytest.mark.skipif(sys.platform == "win32", reason="Windows has no POSIX pseudo-terminal")
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


def test_scratch_paths_are_shown_short_and_link_to_the_real_file(renderer):
    """Seen live: rows read `/Users/…/.tinyharness/projects/app-63fe…/tmp/1793…/make_nb.py`,
    burying the file name and making scratch work look like it happened somewhere odd."""
    renderer.scratch = "/home/u/.tinyharness/projects/app/tmp/1793-abc/"
    shown = renderer._relative("cd /home/u/.tinyharness/projects/app/tmp/1793-abc && python3 make_nb.py")
    assert shown == "cd (scratch) && python3 make_nb.py"
    assert (
        renderer._relative("/home/u/.tinyharness/projects/app/tmp/1793-abc/make_nb.py")
        == "(scratch)/make_nb.py"
    )
    assert renderer._relative("/repo/notebook.ipynb") == "notebook.ipynb"  # project paths as before
    assert (
        renderer._absolute("(scratch)/make_nb.py")
        == "/home/u/.tinyharness/projects/app/tmp/1793-abc/make_nb.py"
    )


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


def test_a_thought_under_a_second_gets_no_row(renderer):
    """Seen in review: "∴ Thought for 0s" rows, which say nothing; ctrl+t still shows it."""
    think(renderer, "Easy.", seconds=0.2)
    renderer.handle(AssistantText("Done."))
    assert not any("Thought" in line for line in shown(renderer))
    assert renderer.last_thought == "Easy."


def test_reasoning_can_be_shown(renderer):
    renderer.show_thinking = True
    think(renderer, "The bug is in the retry loop.")
    assert "The bug is in the retry loop." in renderer.live_ansi(100)  # while it thinks
    renderer.handle(AssistantText("Found it."))
    assert shown(renderer)[:2] == ["∴ Thinking", "  The bug is in the retry loop."]


def test_long_reasoning_streams_line_by_line_and_keeps_its_top(renderer):
    """Seen live: the live region showed the last 8 lines of a thought, so the top of a long
    one was never readable while it streamed. Finished lines now go to the scrollback."""
    renderer.show_thinking = True
    lines = [f"Step {i}: check the part numbered {i}." for i in range(1, 21)]
    think(renderer, *(line + "\n" for line in lines), "Now the last idea, still")
    printed = shown(renderer)
    assert printed[0] == "∴ Thinking" and printed[1] == "  Step 1: check the part numbered 1."
    assert "  Step 20: check the part numbered 20." in printed
    live = renderer.live_ansi(100)
    assert "Now the last idea, still" in live and "Step 1:" not in live
    renderer.handle(AssistantText("Found it."))
    text = "\n".join(printed + shown(renderer))  # shown() returns what's new since it was last called
    assert text.count("Step 1: check") == 1 and text.count("Now the last idea, still") == 1


def test_a_very_long_reasoning_line_shows_its_end_while_it_streams(renderer):
    renderer.show_thinking = True
    think(renderer, "start " + "word " * 2000 + "the latest words")
    live = renderer.live_ansi(100)
    assert live.rstrip().endswith("words") and "start" not in live and "…" in live


def test_ctrl_t_says_it_applies_from_now_on():
    """Reasoning already printed stays in the terminal's scrollback: the notice says so,
    instead of promising to hide what's on screen."""
    from tinyharness.cli.repl import Repl

    source = inspect.getsource(Repl.show_thinking)
    assert "hidden from here on" in source and "shown from here on" in source


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
    assert row.startswith("✗ Bash") and "8 failed" in row and "exit 1" not in row
    assert first.strip() == "⎿ E   PermissionError: Operation not permitted"
    assert any(line.strip() == "8 failed, 462 passed in 10.29s" for line in lines)
    assert "tests ✗" in footer


def test_a_sandbox_block_is_its_own_row_not_a_crash(renderer):
    """Seen live: a blocked write to ~/.cache/sglang showed as a red `exit 1` under a
    40-line traceback, which read as SGLang being broken."""
    from pathlib import Path

    blocked = Path.home() / ".cache" / "sglang"
    output = (
        f"Traceback (most recent call last):\n  ...\nPermissionError: Operation not permitted: '{blocked}'\n"
        f"[tinyharness:filesystem_or_process_denied path={blocked}] This command stayed sandboxed.\n"
        "(exit code 1)"
    )
    command = "python -c 'import sglang'"
    for _ in range(2):
        replay(renderer, (f"Bash({command})", "Bash", {"command": command}, output, False))
    lines = shown(renderer)
    rows = [line for line in lines if line.startswith("⊘ Bash")]
    assert len(rows) == 2 and rows[0].replace("\\", "/").endswith(
        "sandbox blocked a write to ~/.cache/sglang"
    )
    said = " ".join(line.strip() for line in lines if not line.startswith("⊘"))
    assert said.count("/sandbox") == 1 and "runs later, unsandboxed" in said  # said once
    assert "Traceback" not in said  # folded: the model has it


def test_a_nonzero_pipeline_status_is_neutral(renderer):
    command = "git branch -a | head -3"
    output = "main\n(exit code 141)"
    replay(renderer, (f"Bash({command})", "Bash", {"command": command}, output, False))
    text = "\n".join(shown(renderer))
    assert "· Bash" in text and "exit 141" in text and "pipeline stopped" not in text


def test_package_build_failure_has_a_semantic_outcome(renderer):
    """Seen live: a missing Cargo toolchain appeared only as a scary raw `exit 1`."""
    command = "cd python && uv pip install -e ."
    output = (
        "RuntimeError: cargo is required to discover the Rust extension modules\n"
        "hint: Build failures usually indicate a problem with the package or the build environment\n"
        "[tinyharness:venv_uses_sandbox_python path=.venv] Tell the user about the cache.\n"
        "(exit code 1)"
    )
    replay(renderer, (f"Bash({command})", "Bash", {"command": command}, output, False))
    text = "\n".join(shown(renderer))
    assert "build failed" in text and "exit 1" not in text


def test_github_cli_auth_failure_has_a_semantic_outcome(renderer):
    """Seen live: probing public issues with an unauthenticated gh CLI showed raw exit 4."""
    command = 'gh issue list -R org/repo -l "good first issue"'
    output = "To get started with GitHub CLI, please run:  gh auth login\n(exit code 4)"
    replay(renderer, (f"Bash({command})", "Bash", {"command": command}, output, False))
    row, *preview = shown(renderer)
    assert row.endswith("GitHub login required") and "exit 4" not in row
    assert preview == ["  ⎿ To get started with GitHub CLI, please run:  gh auth login"]


def test_the_spinner_shows_minutes_and_seconds(renderer, monkeypatch):
    import tinyharness.cli.render as render
    from tinyharness.events import ToolInputStreaming

    now = [100.0]
    monkeypatch.setattr(render.time, "monotonic", lambda: now[0])
    renderer.handle(ToolInputStreaming("Bash", 10))
    now[0] += 515
    assert "8m 35s · ctrl+c to interrupt" in Text.from_ansi(renderer.live_ansi(100)).plain


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
    renderer._thought_since -= 3  # as if it took that long
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


@pytest.mark.parametrize(
    ("verdict", "label"),
    [
        (
            "auto review held this call back (high risk: it posts project data). It did not run",
            "auto review held back",
        ),
        ("auto review denied this call (it sends .env away). It did not run", "auto review denied"),
        (
            "auto review couldn't run (the review couldn't run: TimeoutError). It did not run",
            "auto review couldn't run",
        ),
    ],
)
def test_a_call_auto_review_stopped_says_how_and_why(renderer, verdict, label):
    """Seen live: "denied by classifier: Pattern…" for calls that were only held back, with
    the reason cut off at 60 characters."""
    replay(renderer, ("WebFetch(https://x.dev)", "WebFetch", {}, f"denied by classifier: {verdict}", True))
    row, reason = shown(renderer)
    assert row.startswith("⊘ WebFetch") and row.rstrip().endswith(label)
    assert verdict.split("(", 1)[1].split(")")[0] in reason


def test_a_saved_plan_is_shown_in_full_with_where_to_edit_it(renderer):
    from tinyharness.events import PlanSaved

    renderer.handle(ToolUse("Plan", {"title": "Token refresh"}, "Plan(Token refresh)"))
    renderer.handle(ToolsDispatched(["Plan"], False))
    renderer.handle(ToolResult("Plan", "Saved the plan to ~/x.md (2 steps).", False))
    text = "# Token refresh\n\n## Goal\n\nRefresh tokens.\n\n## Steps\n\n### S1: Add refresh\n"
    renderer.handle(PlanSaved("Token refresh", "/home/u/.tinyharness/plans/token-refresh.md", text))
    out = "\n".join(shown(renderer))
    assert "Plan  Token refresh" in out and "token-refresh.md · edit it to change the plan" in out
    assert "Refresh tokens." in out and "S1: Add refresh" in out
    assert "Saved the plan to" not in out  # the plan itself, not the tool's note to the model


async def test_plan_mode_names_the_plan_in_the_footer(repl):
    """Shift+tab is how a plan gets approved and built, so the footer says which plan and
    what the key does."""
    from conftest import call, make_ctx

    assert text(repl._mode_part()).endswith("shift+tab to switch")
    repl.session.set_mode("plan")
    assert "plan · shift+tab to switch" in text(repl._mode_part())
    ctx = make_ctx(repl.session.workspace.root)
    await call(repl.session.plan_tool, {"plan": "# Token refresh\n\n1. a\n"}, ctx)
    assert "⏸ plan · Token refresh · shift+tab to approve" in text(repl._mode_part())
    current = repl.session.plan_tool.current
    current.path.write_text(current.path.read_text() + "\nmore\n")
    assert "Token refresh (edited)" in text(repl._mode_part())
    current.path.write_text(current.text().removesuffix("\nmore\n"))  # undo the edit
    await call(repl.session.plan_tool, {"plan": "# " + "x" * 60 + "\n\n1. a\n"}, ctx)
    assert f"{'x' * 39}… · shift+tab" in text(repl._mode_part())
    repl.session.set_mode("default")
    assert "xxx" not in text(repl._mode_part())


def test_a_plan_ready_for_review_leaves_the_choice_to_the_hand_off(renderer):
    renderer.handle(Done(reason="plan_ready"))
    out = "\n".join(shown(renderer))
    assert "Plan ready for review" in out and "build it" not in out


def test_a_plan_from_the_reply_shows_where_it_is_without_repeating_it(renderer):
    from tinyharness.cli import markdown
    from tinyharness.events import PlanSaved

    renderer.handle(
        PlanSaved("Token refresh", "/home/u/plans/token-refresh.md", "# Token refresh\n\nBODY\n", True)
    )
    out = "\n".join(shown(renderer))
    assert "Plan  Token refresh" in out and "token-refresh.md" in out and "BODY" not in out
    console = Console(width=60, record=True)
    console.print(markdown.render("<plan>\n# Token refresh\n</plan>"))
    text = console.export_text()
    assert "<plan>" not in text and "Token refresh" in text


def test_a_job_row_shows_where_the_job_got_to():
    """Seen live: a finished 18-epoch run showed "epoch 1/18 loss=1.7000" on its row."""
    from tinyharness.cli.render import summarize_output

    out = "epoch 1/18 loss=1.7000\nepoch 18/18 loss=0.1073\nfinal loss: 0.1073\n(job j1 running for 3m00s: python3 train.py)"
    assert summarize_output("Job", out) == "final loss: 0.1073"
    assert summarize_output("Job", "(job j1 finished: exit 0, no output)").startswith("(job j1")


def test_a_stopped_job_row_says_stopped():
    """Seen in review: a stopped server's row showed the last request it logged."""
    from tinyharness.cli.render import summarize_output

    out = "GET / 200\n(job j3 stopped: python3 -m http.server · stopped it)"
    assert summarize_output("Job", out) == "stopped"


def test_a_command_that_moved_to_the_background_does_not_look_finished(renderer):
    """Seen in review: a command still running at its 5s timeout became job j3, and its row
    read "✓ Bash … 5s" as if it had finished."""
    out = (
        "building...\n(still running after 5s, so it moved to the background as job j3 instead "
        "of being killed; you'll get a note when it ends. Job(id='j3', action='wait') waits.)"
    )
    replay(renderer, ("Bash(make)", "Bash", {"command": "make"}, out, False))
    text = renderer.console.export_text()
    assert "running in the background as j3" in text and "✓" not in text


def test_files_changed_by_subagents_count_in_the_footer(renderer):
    """Seen live: two subagents changed files, and the footer said "1 file +3 −1"."""
    from tinyharness.events import SubagentUpdate

    replay(renderer, ("Edit(a.py, -1/+3 lines)", "Edit", {"path": "a.py"}, "edited a.py", False))
    renderer.handle(
        SubagentUpdate(
            "r1", "general-purpose", "tests", "done", tools_used=3,
            edits=("Edit(a.py, -2/+2 lines)", "Write(tests/test_b.py, 10 lines)"),
        )
    )  # fmt: skip
    renderer.handle(Done())
    footer = shown(renderer)[-1]
    assert "2 files +15 −3" in footer


def test_narrowing_the_terminal_erases_the_box_as_it_rewrapped(monkeypatch):
    """Seen live in iTerm2: dragging the window narrower left broken frame lines above the
    input box. Each full-width line rewraps onto more rows, which the erase has to cover,
    and the rows added push the drawing up, which the redraw undoes."""
    from types import SimpleNamespace

    from prompt_toolkit.data_structures import Point, Size
    from prompt_toolkit.layout.screen import Char, Screen

    from tinyharness.cli.repl import _count_reflowed_rows

    screen = Screen()
    for y, width in enumerate([100, 100, 100, 40]):  # frame top, input, frame bottom, toolbar
        for x in range(width):
            screen.data_buffer[y][x] = Char("─")
    screen.height = 4
    renderer = SimpleNamespace(
        _last_screen=screen, _last_size=Size(rows=30, columns=100), _cursor_pos=Point(3, 1)
    )

    def app(columns):
        renderer._cursor_pos = Point(3, 1)  # on the input line, below the frame's top
        return SimpleNamespace(renderer=renderer, output=SimpleNamespace(get_size=lambda: Size(30, columns)))

    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    # iTerm2 keeps the bottom row fixed: the frame's top line now takes two rows above the
    # cursor, and the three frame lines' extra rows pushed the drawing up by three.
    assert _count_reflowed_rows(app(60)) == 3 and renderer._cursor_pos == Point(3, 2)
    assert _count_reflowed_rows(app(30)) == 10 and renderer._cursor_pos == Point(3, 4)
    assert _count_reflowed_rows(app(120)) == 0 and renderer._cursor_pos == Point(3, 1)  # wider
    # xterm.js (VS Code) keeps the top fixed and moves the cursor down by every row added.
    monkeypatch.setenv("TERM_PROGRAM", "vscode")
    assert _count_reflowed_rows(app(60)) == 0 and renderer._cursor_pos == Point(3, 4)


async def test_a_burst_of_resizes_redraws_once_it_settles(monkeypatch):
    """Seen live in iTerm2: dragging the window edge sent a resize every few milliseconds,
    and redraws for sizes the window had already left piled up old boxes."""
    import asyncio

    from prompt_toolkit.application import Application
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.renderer import Renderer as PTRenderer

    from tinyharness.cli import repl as repl_module

    resized, rendered = [], []
    monkeypatch.setattr(Application, "_on_resize", lambda self: resized.append(1))
    monkeypatch.setattr(
        PTRenderer, "render", lambda self, app, layout, is_done=False: rendered.append(is_done)
    )
    monkeypatch.setattr(repl_module, "RESIZE_SETTLE_SECONDS", 0.05)
    with create_pipe_input() as pipe:
        app = repl_module._CompactPrompt(input=pipe, output=DummyOutput()).app
        for _ in range(3):
            app._on_resize()
            app.renderer.render(app, app.layout)  # e.g. a spinner redrawing meanwhile
        assert resized == [] and rendered == []  # held back while the size keeps changing
        app.renderer.render(app, app.layout, is_done=True)  # leaving is never held back
        await asyncio.sleep(0.15)
        assert resized == [1] and rendered == [True]
        app.renderer.render(app, app.layout)
        assert rendered == [True, False]


def test_printed_lines_end_where_their_text_ends():
    """Seen live in iTerm2: Scout and every reply broke into rows with gaps between them
    when the window narrowed. rich padded their lines with spaces to the full width, and
    the terminal rewrapped the spaces like text."""
    import io

    from rich.padding import Padding
    from rich.text import Text

    from tinyharness.cli.theme import TrimmedConsole

    out = io.StringIO()
    console = TrimmedConsole(file=out, width=80, color_system="truecolor", force_terminal=True)
    console.print(Padding(Text("hello"), (0, 0, 0, 2)))
    console.print(Text("code   ", style="on #202020"))  # a background shows: kept
    console.print("partial ", end="")  # the line isn't finished: more may follow
    console.print("rest")
    lines = out.getvalue().split("\n")
    assert lines[0] == "  hello"  # was "  hello" and 73 spaces
    assert lines[1] == "\x1b[48;2;32;32;32mcode   \x1b[0m"
    assert lines[2] == "partial rest"


def test_auto_modes_reason_stays_on_one_line():
    """Seen in review: auto mode wrote two lines of review reasoning under each call."""
    from tinyharness.events import PermissionDecision

    renderer = Renderer(Console(width=60, record=True))
    renderer.handle(ToolUse("Bash", {"command": "ls"}, "Bash(ls)"))
    reason = "listing files in the project is read-only and matches what the user asked for here"
    renderer.handle(PermissionDecision("Bash", True, "classifier", reason))
    renderer.handle(ToolsDispatched(["Bash"], False))
    renderer.handle(ToolResult("Bash", "a.py", False))
    lines = renderer.console.export_text().splitlines()
    [note] = [line for line in lines if "auto mode" in line]
    assert "…" in note and len(lines) == 4  # blank, the row, its output, the one-line note
