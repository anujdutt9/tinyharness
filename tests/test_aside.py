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
from tinyharness.cli.repl import _AFTER_TURN, Repl
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


async def test_a_suggestion_never_shows_the_models_reasoning(tmp_path):
    """Seen live: qwen3:4b on Ollama wrote its reasoning into the reply, and the input box
    suggested "<think>". Reasoning in the text is dropped, and so is a reply cut off."""
    session, _ = scripted(
        tmp_path,
        [
            "done",
            "<think>\nThe user wants to try it.\n</think>\n\nrun wc.py -l notes.txt",
            "<think>\nThey might want to",  # ended mid-thought
            [StreamChunk(text="run the tests and then"), StreamChunk(finish_reason="length")],
        ],
    )
    await converse(session)
    assert await session.suggest_next() == "run wc.py -l notes.txt"
    assert await session.suggest_next() == ""
    assert await session.suggest_next() == ""


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
    assert "/btw" not in _AFTER_TURN


def test_the_suggestion_shows_in_the_empty_input_box(repl):
    repl.suggestion = "commit it"
    shown = "".join(text for _, text in repl._placeholder())
    assert shown == "commit it"


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
    assert "as of its last finished step" not in shown  # asked between turns


async def test_btw_asked_mid_turn_says_what_it_answered_from(boxed):
    """Seen live: asked during a turn, /btw said "I'm not editing any file yet" while an
    edit was being written. It answers from the finished steps, and says so."""
    boxed.llm.turns = ["Only stock.py so far."]
    boxed._turn = asyncio.get_running_loop().create_future()  # a turn is running
    await boxed.handle("/btw which file are you on?")
    await boxed._answering
    assert "as of its last finished step" in boxed._aside_ansi(boxed.aside, 100, 20)
    boxed._turn.cancel()


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


async def test_built_in_commands_run_while_the_agent_works(repl):
    """Seen live: "/effort has to wait until the agent finishes". Settings apply from the
    next model call; only commands that replace the conversation wait for the turn."""
    repl._turn = asyncio.get_running_loop().create_future()  # a turn is running
    repl.auto_confirmed = True  # no confirmation picker: Windows CI has no console for one
    await repl.handle("/effort none")
    await repl.handle("/mode auto")
    out = repl.console.file.getvalue()
    assert "has to wait" not in out and repl.session.config.effort == "none"
    assert repl._draft == ""
    repl._turn.cancel()


async def test_conversation_commands_typed_mid_turn_run_when_it_ends(repl):
    await converse(repl.session, "hello")
    assert len(repl.session.history.messages) > 1
    repl._turn = asyncio.get_running_loop().create_future()
    await repl.handle("/clear")
    assert "runs when the agent finishes" in repl.console.file.getvalue()
    assert len(repl.session.history.messages) > 1  # not yet: the turn still adds to it

    async def finished_turn():
        return
        yield

    repl._turn.cancel()
    await repl._turn_in_background(finished_turn())
    assert [m.role for m in repl.session.history.messages] in ([], ["system"])


async def test_a_message_typed_during_compact_is_sent_when_it_finishes(boxed, monkeypatch):
    """Seen live: a message typed during /compact landed on its spinner line, and Enter
    added a newline instead of sending it. /compact runs under the input box now."""
    release = asyncio.Event()

    async def compact():
        await release.wait()
        return True

    sent = []

    async def run(text, images=None):
        sent.append(text)
        return
        yield

    monkeypatch.setattr(boxed.session, "compact", compact)
    monkeypatch.setattr(boxed.session, "run", run)
    await boxed.handle("/compact")
    compacting = boxed._turn
    assert boxed.turn_running
    await boxed.handle("is the suite green?")
    assert sent == []  # it waits for the compaction
    release.set()
    await compacting
    await boxed._turn
    assert sent == ["is the suite green?"]
    assert "Compacted" in boxed.console.file.getvalue()


async def test_esc_then_a_slash_still_types_a_command(tmp_path):
    """Seen in review: Esc, then "/cost" 0.4s later, and the agent got "cost" as a prompt:
    prompt_toolkit's Meta+/ (Esc waits for a second key) took the slash."""
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    session, _ = scripted(tmp_path, [])
    console = Console(file=io.StringIO(), width=100, theme=THEME)
    repl = Repl(session, console, Renderer(console))
    try:
        with create_pipe_input() as keys, create_app_session(input=keys, output=DummyOutput()):
            prompt = repl._prompt_session()
            keys.send_text("\x1b/cost\r")
            assert await asyncio.wait_for(prompt.prompt_async(), 3) == "/cost"
    finally:
        session.close()
