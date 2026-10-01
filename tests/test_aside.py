"""Side calls that stay out of the conversation: /btw questions and the next-message
suggestion shown in the input box after a turn."""

import io

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


async def test_btw_prints_the_answer_aside(repl):
    repl.llm.turns = ["Because it was simpler."]
    await repl.handle("/btw why that way?")
    await next(iter(repl._asides))
    out = repl.console.file.getvalue()
    assert "btw: why that way?" in out and "Because it was simpler." in out
    assert "not added to the conversation" in out
    assert repl.session.history.messages == []


async def test_a_finished_turn_brings_a_suggestion(repl):
    repl.llm.turns = ["Done: tests pass.", "commit it"]
    await repl._turn_in_background(repl.session.run("fix it"))
    await repl._suggesting
    assert repl.suggestion == "commit it"


async def test_suggestions_can_be_turned_off(repl):
    repl.session.config.prompt_suggestions = False
    repl.llm.turns = ["Done."]
    await repl._turn_in_background(repl.session.run("fix it"))
    assert repl._suggesting is None and len(repl.llm.requests) == 1
