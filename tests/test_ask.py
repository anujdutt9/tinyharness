"""The Ask tool: the agent's questions, answered one at a time by picking an option."""

import io
import json

from rich.console import Console

from tinyharness.events import Done
from tinyharness.testing import scripted_session
from tinyharness.tools.ask import AskQuestion

QUESTIONS = {
    "questions": [
        {
            "header": "Format",
            "question": "How should the implementation be laid out?",
            "options": [
                {"label": "Jupyter notebook", "description": "each block explained, then coded"},
                {"label": ".py modules"},
            ],
        },
        {
            "header": "Scope",
            "question": "Which model first?",
            "options": [{"label": "Decoder-only"}, {"label": "Encoder-decoder"}],
        },
    ]
}


async def test_answers_come_back_and_the_turn_goes_on(tmp_path):
    """Seen live: plan mode's questions were written into the reply, to be answered by
    typing. With Ask the user picks each answer and the agent carries on in the same turn."""
    asked = []

    async def answer(questions):
        asked.append([q.header for q in questions])
        return ["Jupyter notebook", "my own: both, decoder first"]

    session = scripted_session(
        tmp_path / "w",
        [[("Ask", json.dumps(QUESTIONS))], "Thanks: planning a notebook, decoder first."],
        question_callback=answer,
        permission_mode="plan",
    )
    try:
        events = [event async for event in session.run("plan a transformer from the paper")]
        assert asked == [["Format", "Scope"]]
        result = next(m.content for m in session.history.messages if m.role == "tool")
        assert result == "The user answered:\nFormat: Jupyter notebook\nScope: my own: both, decoder first"
        assert session.history.messages[-1].content.startswith("Thanks")  # same turn
        assert events[-1] == Done()
    finally:
        session.close()


async def test_answering_in_chat_ends_the_turn(tmp_path):
    async def answer(questions):
        return None  # Esc

    session = scripted_session(
        tmp_path / "w", [[("Ask", json.dumps(QUESTIONS))], "(never asked)"], question_callback=answer
    )
    try:
        events = [event async for event in session.run("plan it")]
        assert events[-1] == Done(reason="awaiting_answer")
        assert all(m.content != "(never asked)" for m in session.history.messages)
    finally:
        session.close()


def test_ask_is_offered_only_where_someone_can_answer(tmp_path):
    from tinyharness.subagents.runner import MAIN_AGENT_ONLY

    async def answer(questions):
        return []

    for callback, offered in ((None, False), (answer, True)):
        session = scripted_session(tmp_path / str(offered), [], question_callback=callback)
        try:
            names = [tool.name for tool in session.tools.available(session.mode)]
            assert ("Ask" in names) == offered
        finally:
            session.close()
    assert "Ask" in MAIN_AGENT_ONLY  # a subagent has no one to ask


async def test_each_answer_stays_on_screen_as_one_line(monkeypatch):
    from tinyharness.cli import render
    from tinyharness.cli.theme import THEME

    picks = iter(["Jupyter notebook", None])

    async def answer_question(question, number, total):
        return next(picks)

    monkeypatch.setattr(render, "answer_question", answer_question)
    console = Console(file=io.StringIO(), width=100, theme=THEME, color_system=None)
    renderer = render.Renderer(console)
    questions = [AskQuestion.model_validate(q) for q in QUESTIONS["questions"]]
    assert await renderer.ask_questions(questions) is None  # Esc on the second
    out = console.file.getvalue()
    assert "✓ Format · Jupyter notebook" in out and "answering in chat" in out


async def test_the_picker_starts_on_the_recommended_answer(monkeypatch):
    """Enter takes the recommended answer; a number picks another; Esc answers in chat;
    the last row types your own."""
    monkeypatch.setattr("tinyharness.cli.widgets.TYPE_AHEAD_SECONDS", 0)
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from tinyharness.cli.widgets import answer_question

    question = AskQuestion.model_validate(QUESTIONS["questions"][0])
    for keys, expected in (
        ("\r", "Jupyter notebook"),
        ("2", ".py modules"),
        ("\x1b", None),
        ("3both\r", "both"),
    ):
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            pipe.send_text(keys)
            assert await answer_question(question, 1, 2) == expected, keys


async def test_keys_typed_as_a_question_appears_dont_answer_it():
    """Seen live: a message being typed when a question appeared went to the question,
    and its Enter picked an answer the user never chose."""
    import asyncio

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from tinyharness.cli.widgets import TYPE_AHEAD_SECONDS, answer_question

    question = AskQuestion.model_validate(QUESTIONS["questions"][0])
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text("Make it 2 lines.\r")  # meant for the message box
        answering = asyncio.create_task(answer_question(question, 1, 2))
        await asyncio.sleep(0.3)
        assert not answering.done()
        await asyncio.sleep(TYPE_AHEAD_SECONDS)
        pipe.send_text("2")  # now a deliberate choice
        assert await asyncio.wait_for(answering, 3) == ".py modules"


def test_the_ask_row_says_answered_not_the_answers_again():
    from tinyharness.cli.render import summarize_output

    assert summarize_output("Ask", "The user answered:\nFormat: Jupyter notebook") == "answered"


PROSE = (
    "Before I plan, a few questions:\n\n1. **Format**: notebook or modules?\n"
    "   - a) A notebook (recommended)\n   - b) .py modules\n2. **Scope**: decoder-only or the full model?"
)


async def test_questions_written_in_the_reply_are_read_for_the_picker(tmp_path):
    """Seen live: Ministral 8B asked in writing in 5 of 6 plan-mode runs and never called
    Ask. One extra call restates the questions as an Ask call, never shown or kept."""

    async def answer(questions):
        return []

    session = scripted_session(
        tmp_path / "w",
        [PROSE, [("Ask", json.dumps(QUESTIONS))]],
        question_callback=answer,
        permission_mode="plan",
    )
    try:
        [event async for event in session.run("plan a transformer")]
        before = list(session.history.messages)
        questions = await session.questions_in_reply()
        assert [q.header for q in questions] == ["Format", "Scope"]
        assert session.history.messages == before  # the extra call isn't kept
    finally:
        session.close()


async def test_no_extra_call_without_questions_or_without_ask(tmp_path):
    async def answer(questions):
        return []

    for reply, callback in (("Here's what I found: it's a small change.", answer), (PROSE, None)):
        session = scripted_session(tmp_path / str(callback is None), [reply], question_callback=callback)
        try:
            [event async for event in session.run("plan it")]
            calls_before = len(session.llm.stream.__self__.requests)
            assert await session.questions_in_reply() == []
            assert len(session.llm.stream.__self__.requests) == calls_before  # nothing asked
        finally:
            session.close()


async def test_the_cli_shows_the_picker_for_written_questions_and_sends_the_picks(tmp_path, monkeypatch):
    from tinyharness.cli import repl as repl_module
    from tinyharness.cli.render import Renderer
    from tinyharness.cli.theme import THEME

    async def unused(questions):
        return None

    session = scripted_session(
        tmp_path / "w",
        [PROSE, [("Ask", json.dumps(QUESTIONS))], "Planning a notebook, decoder first."],
        question_callback=unused,
        permission_mode="plan",
    )
    console = Console(file=io.StringIO(), width=100, theme=THEME, color_system=None)
    repl = repl_module.Repl(session, console, Renderer(console))
    picked = []

    async def ask_questions(questions):
        picked.append([q.header for q in questions])
        return ["Jupyter notebook", "Decoder-only"]

    monkeypatch.setattr(repl.renderer, "ask_questions", ask_questions)
    from types import SimpleNamespace

    # The input box as it runs (turns run under it), empty: nothing typed yet.
    repl._prompt = SimpleNamespace(
        app=SimpleNamespace(is_running=False), default_buffer=SimpleNamespace(text="")
    )
    try:
        await repl.handle("plan a transformer")
        await repl._turn
        await repl._offering
        await repl._turn  # the picks, sent as the next message
        assert picked == [["Format", "Scope"]]
        user = [m.content for m in session.history.messages if m.role == "user"][-1]
        assert user.endswith("Format: Jupyter notebook\nScope: Decoder-only")
        assert session.history.messages[-1].content == "Planning a notebook, decoder first."
    finally:
        session.close()


async def test_two_ask_calls_in_one_reply_ask_one_after_the_other(tmp_path):
    """Found in review: Ask is read-only, so two calls in one reply ran at the same time,
    and both pickers would want the keyboard."""
    import asyncio

    active, most = 0, 0

    async def answer(questions):
        nonlocal active, most
        active += 1
        most = max(most, active)
        await asyncio.sleep(0.05)
        active -= 1
        return [q.options[0].label for q in questions]

    one = {"questions": QUESTIONS["questions"][:1]}
    two = {"questions": QUESTIONS["questions"][1:]}
    session = scripted_session(
        tmp_path / "w",
        [[("Ask", json.dumps(one)), ("Ask", json.dumps(two))], "Thanks."],
        question_callback=answer,
    )
    try:
        [event async for event in session.run("plan it")]
        assert most == 1
        assert len([m for m in session.history.messages if m.role == "tool"]) == 2
    finally:
        session.close()
