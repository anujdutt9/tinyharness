"""Side calls that stay out of the conversation: /btw questions and the next-message
suggestion shown in the input box after a turn."""

import asyncio
import contextlib
import io
import json
from types import SimpleNamespace

import pytest
from rich.console import Console

from tinyharness.agent.history import Message
from tinyharness.cli.render import Renderer
from tinyharness.cli.repl import _SAFE_WHILE_RUNNING, Repl
from tinyharness.cli.theme import THEME
from tinyharness.llm.types import StreamChunk
from tinyharness.llm.usage import Usage
from tinyharness.testing import ScriptedLLM, scripted_session


def scripted(tmp_path, turns):
    session = scripted_session(tmp_path / "w", [])
    llm = ScriptedLLM(turns)
    session.llm.stream = llm.stream  # type: ignore[method-assign]
    return session, llm


async def converse(session, prompt="fix the bug"):
    async for _ in session.run(prompt):
        pass


async def test_a_side_question_is_answered_from_the_conversation_and_not_kept(tmp_path):
    session, llm = scripted(
        tmp_path, ["Fixed: the loop was off by one.", "Because range() excludes the end."]
    )
    await converse(session)
    before = list(session.history.messages)

    answer = await session.aside("why was it off by one?")

    assert answer == "Because range() excludes the end."
    assert session.history.messages == before  # nothing added
    sent = llm.requests[-1]
    assert sent[: len(before)] == before  # the same prefix as the agent's calls, so it's cached
    assert "why was it off by one?" in sent[-1].content and "/btw" in sent[-1].content


async def test_a_side_question_that_needs_a_tool_says_so(tmp_path):
    session, _ = scripted(tmp_path, ["ok", [("Read", '{"path": "a.py"}')]])
    await converse(session)
    answer = await session.aside("what's in a.py?")
    assert "needs a tool" in answer
    assert all(not m.tool_calls for m in session.history.messages[-1:])  # the call wasn't kept or run


async def test_a_side_question_at_the_start_of_a_turn_joins_the_waiting_message(tmp_path):
    """Mid-turn, before the model has answered, the history ends with the user's message:
    the question goes in the same message rather than a second one in a row."""
    session, llm = scripted(tmp_path, ["ok", "noted"])
    await converse(session)
    session.history.append(Message(role="user", content="build it"))
    await session.aside("which file first?")
    last = llm.requests[-1][-1]
    assert last.role == "user" and last.content.startswith("build it") and "which file first?" in last.content
    assert llm.requests[-1][-2].role != "user"


async def test_an_empty_side_question_is_refused(tmp_path):
    session, llm = scripted(tmp_path, [])
    with pytest.raises(ValueError, match="/btw"):
        await session.aside("  ")
    assert llm.requests == []


async def test_side_calls_count_toward_the_session_cost(tmp_path):
    usage = Usage(input=100, cached=9_000, output=20, cost=0.003)
    session, _ = scripted(tmp_path, [[StreamChunk(text="sure", finish_reason="stop", usage=usage)]])
    await session.aside("quick one?")
    assert session.usage.calls == 1 and session.usage.cost == pytest.approx(0.003)


async def test_the_next_message_is_suggested_after_a_turn(tmp_path):
    session, _ = scripted(tmp_path, ["All tests pass.", '"run the full test suite"\n'])
    await converse(session)
    before = list(session.history.messages)
    assert await session.suggest_next() == "run the full test suite"
    assert session.history.messages == before


async def test_no_suggestion_without_a_conversation_or_a_short_one(tmp_path):
    session, llm = scripted(tmp_path, ["done", "x" * 300])
    assert await session.suggest_next() == ""
    assert llm.requests == []  # nothing to go on: no call made
    await converse(session)
    assert await session.suggest_next() == ""  # an explanation, not a message to send


# ---------------------------------------------------------------------------- the CLI


@pytest.fixture
def repl(tmp_path):
    session, llm = scripted(tmp_path, [])
    console = Console(file=io.StringIO(), width=100, theme=THEME)
    repl = Repl(session, console, Renderer(console))
    repl.llm = llm
    yield repl
    session.close()


def test_btw_can_be_asked_while_the_agent_works():
    assert "/btw" in _SAFE_WHILE_RUNNING


def test_the_suggestion_shows_in_the_empty_input_box(repl):
    repl.suggestion = "commit it"
    shown = "".join(text for _, text in repl._placeholder())
    assert shown.startswith("commit it") and "tab" in shown


async def test_a_new_message_drops_the_suggestion(repl):
    repl.suggestion = "commit it"
    await repl.handle("/cwd")
    assert repl.suggestion == ""


async def test_btw_without_an_input_box_prints_the_answer(repl):
    repl.llm.turns = ["Because it was simpler."]
    await repl.handle("/btw why that way?")
    await repl._answering
    out = repl.console.file.getvalue()
    assert "btw: why that way?" in out and "Because it was simpler." in out
    assert repl.session.history.messages == []


@pytest.fixture
def boxed(repl):
    """The REPL as it runs: with the input box up, a /btw answer is shown above it."""
    repl._prompt = SimpleNamespace(app=SimpleNamespace(is_running=False))
    return repl


async def test_btw_shows_above_the_input_box_until_esc(boxed):
    boxed.llm.turns = ["Because it was simpler."]
    await boxed.handle("/btw why that way?")
    assert "answering…" in boxed._aside_ansi(boxed.aside, 80, 20)
    await boxed._answering
    shown = boxed._aside_ansi(boxed.aside, 80, 20)
    assert "Because it was simpler." in shown and "esc to close" in shown
    assert "Because" not in boxed.console.file.getvalue()  # nothing in the scrollback

    boxed.close_aside()  # what Esc does
    assert boxed.aside is None


async def test_the_next_message_closes_the_answer_and_output_prints_it(boxed):
    boxed.llm.turns = ["one", "two"]
    await boxed.handle("/btw first?")
    await boxed._answering
    await boxed.handle("/cwd")
    assert boxed.aside is None

    await boxed.handle("/btw second?")
    await boxed._answering
    await boxed.handle("/output")
    assert "two" in boxed.console.file.getvalue() and boxed.aside is None


async def test_closing_an_unanswered_question_cancels_it(boxed):
    boxed.llm.turns = ["slow"]
    await boxed.handle("/btw anything?")
    task = boxed._answering
    boxed.close_aside()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_a_long_answer_is_cut_to_fit(boxed):
    from tinyharness.cli.repl import Aside

    shown = boxed._aside_ansi(Aside("q", "\n\n".join(f"line {i}" for i in range(40))), 80, 12)
    assert shown.count("\n") < 12 and "more lines" in shown and "/output" in shown


async def test_a_finished_turn_brings_a_suggestion(repl):
    repl.llm.turns = ["Done: tests pass.", "commit it"]
    await repl._turn_in_background(repl.session.run("fix it"))
    await repl._suggesting
    assert repl.suggestion == "commit it"
    records = [json.loads(line) for line in repl.session.transcript.path.read_text().splitlines()]
    assert any(
        record["event"] == "suggestion" and record["action"] == "shown" and record["chars"] == 9
        for record in records
    )


async def test_suggestions_can_be_turned_off(repl):
    repl.session.config.prompt_suggestions = False
    repl.llm.turns = ["Done."]
    await repl._turn_in_background(repl.session.run("fix it"))
    assert repl._suggesting is None and len(repl.llm.requests) == 1


async def test_esc_closes_the_answer_in_the_running_input_box(tmp_path):
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    session, _ = scripted(tmp_path, ["Because it was simpler."])
    console = Console(file=io.StringIO(), width=100, theme=THEME)
    repl = Repl(session, console, Renderer(console))
    with create_pipe_input() as keys, create_app_session(input=keys, output=DummyOutput()):
        loop = asyncio.create_task(repl.loop())
        try:
            await asyncio.sleep(0.3)
            keys.send_text("/btw why that way?\r")
            for _ in range(30):
                if repl.aside is not None and repl.aside.answer:
                    break
                await asyncio.sleep(0.1)
            assert "Because it was simpler." in repl._live()
            keys.send_text("\x1b")
            for _ in range(30):
                if repl.aside is None:
                    break
                await asyncio.sleep(0.1)
            assert repl.aside is None and "Because" not in repl._live()
            assert "Because" not in console.file.getvalue()
        finally:
            repl.running = False
            keys.send_text("\x04")
            with contextlib.suppress(Exception):
                await asyncio.wait_for(loop, 3)
    session.close()
