import io

import pytest

from tinyharness import MissingAPIKeyError, Session, SessionOptions
from tinyharness.agent.history import Message


@pytest.fixture(autouse=True)
def outside_the_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # don't load the repo's own .env


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    options = SessionOptions(config_path=tmp_path / "none.toml", working_dir=tmp_path / "work")
    s = Session(options)
    yield s
    s.close()


def test_starts_on_the_provider_with_a_key(session):
    assert session.config.provider == "openai"
    assert session.llm.endpoint.startswith("https://api.openai.com/")


def test_switching_model_rebuilds_the_client_and_context_window(session):
    from tinyharness.llm.types import ModelInfo

    session.models.store("openai", [ModelInfo("gpt-6-luna", context=400_000)])
    session.set_model("gpt-6-luna")
    assert session.llm.model == "gpt-6-luna"
    assert session.context_window == session.compactor.context_window == 400_000


def test_unknown_context_windows_fall_back_to_the_default(session):
    from tinyharness.llm.providers import DEFAULT_CONTEXT

    session.set_model("some-new-model")
    assert session.context_window == DEFAULT_CONTEXT
    session.config.context_window_tokens = 50_000
    assert session.context_window == 50_000


def test_switching_to_a_provider_without_a_key_changes_nothing(session):
    with pytest.raises(MissingAPIKeyError) as err:
        session.set_model("anthropic:claude-sonnet-5")
    assert "ANTHROPIC_API_KEY" in str(err.value)
    assert (session.config.provider, session.llm.model) == ("openai", session.config.model)


def test_switching_provider_keeps_the_conversation(session, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gk")
    before = len(session.history.messages)
    session.set_provider("groq")
    assert session.config.provider == "groq"
    assert "groq.com" in session.llm.endpoint
    assert len(session.history.messages) == before


def test_missing_key_at_startup_names_the_variable(tmp_path):
    with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        Session(SessionOptions(config_path=tmp_path / "none.toml", working_dir=tmp_path / "w"))


def test_a_key_added_while_running_is_picked_up(session, tmp_path, monkeypatch):
    """Added to auth.json by hand, or by `tinyharness --setup` in another terminal."""
    import json

    from tinyharness.paths import auth_path

    monkeypatch.delenv("XAI_API_KEY", raising=False)
    auth_path().parent.mkdir(parents=True, exist_ok=True)
    auth_path().write_text(json.dumps({"xai": {"api_key": "xk-added-later"}}))
    session.set_provider("xai")
    assert session.config.provider == "xai"


async def test_checking_a_model_changes_nothing(session, monkeypatch):
    from tinyharness.llm.client import LLMClient, Ping

    asked = []

    async def fake_ping(self, model=None):
        asked.append((self.provider.name, model))
        return Ping(False, 404, "no such model")

    monkeypatch.setattr(LLMClient, "ping", fake_ping)
    before = (session.config.provider, session.config.model)
    check = await session.check_model("openai:gpt-image-x")
    assert check.model_unusable and asked == [("openai", "gpt-image-x")]
    assert (session.config.provider, session.config.model) == before


async def test_checking_a_provider_uses_its_default_model(session, monkeypatch):
    from tinyharness.llm.client import LLMClient, Ping

    monkeypatch.setenv("GROQ_API_KEY", "gk")
    asked = []

    async def fake_ping(self, model=None):
        asked.append(model)
        return Ping(True, 200, "ok")

    monkeypatch.setattr(LLMClient, "ping", fake_ping)
    await session.check_model("groq:")
    assert asked == [session.config.providers["groq"].default_model]


async def test_the_system_prompt_follows_a_model_switch(session):
    session.history.append(Message(role="system", content=session._system_prompt()))
    assert f"`{session.config.model}`" in session.history.messages[0].content
    session.set_model("gpt-6-luna")
    assert "`gpt-6-luna`" in session.history.messages[0].content


async def test_compact_summarizes_all_but_the_latest_exchange(session, monkeypatch):
    from tinyharness.agent.compaction import SUMMARY_MARK
    from tinyharness.agent.history import Message

    async def fake_one_shot(self, prompt, **kwargs):
        return "they asked for a game"

    monkeypatch.setattr(type(session.llm), "one_shot", fake_one_shot)
    assert not await session.compact()  # empty: nothing to do
    for text in ("make a game", "ok", "add an AI", "done", "thanks"):
        session.history.messages.append(
            Message(role="user" if text != "ok" and text != "done" else "assistant", content=text)
        )
    assert await session.compact()
    assert session.history.messages[0].content == f"{SUMMARY_MARK}\nthey asked for a game"
    assert [m.content for m in session.history.messages[1:]] == ["thanks"]


async def test_compact_keeps_the_original_when_the_summary_is_longer(session, monkeypatch):
    from tinyharness.agent.history import Message

    async def wordy(self, prompt, **kwargs):
        return "a very long summary " * 50

    monkeypatch.setattr(type(session.llm), "one_shot", wordy)
    original = [Message(role="user", content="hello world"), Message(role="assistant", content="Hi!")]
    original.append(Message(role="user", content="what model are you?"))
    session.history.messages = list(original)
    assert not await session.compact()
    assert session.history.messages == original


async def test_switching_models_keeps_the_context_reading(session, monkeypatch):
    """The footer shows the same conversation against the new model's window, not 0%."""
    from rich.console import Console

    from tinyharness.agent.history import Message
    from tinyharness.cli.commands import _switch
    from tinyharness.cli.render import Renderer
    from tinyharness.cli.repl import Repl
    from tinyharness.llm.client import Ping

    async def answers(spec):
        return Ping(ok=True, status=200, message="auth ok")

    monkeypatch.setattr(session, "check_model", answers)
    session.history.messages = [Message(role="user", content="word " * 5_000)]
    console = Console(file=io.StringIO())
    repl = Repl(session, console, Renderer(console))
    await _switch(repl, "openai:some-model")
    assert repl.context_used > 0
    assert repl.context_used == pytest.approx(
        session.compactor.used(session.history) / session.context_window
    )


def test_no_todo_list_by_default(session):
    """Current models track multi-step work without one; its upkeep costs steps and tokens."""
    assert "TodoWrite" not in session.tools.names()
    assert "TodoWrite" not in session._system_prompt()


@pytest.mark.parametrize("how", ["option", "env"])
def test_the_todo_list_can_be_turned_on(tmp_path, monkeypatch, how):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    if how == "env":
        monkeypatch.setenv("TINYHARNESS_TODO_LIST", "1")
    options = SessionOptions(
        config_path=tmp_path / "none.toml",
        working_dir=tmp_path / "work",
        todo_list=True if how == "option" else None,
    )
    s = Session(options)
    try:
        assert "TodoWrite" in s.tools.names()
        assert "# Planning" in s._system_prompt()
    finally:
        s.close()


async def test_a_session_closes_at_the_end_of_an_async_with(tmp_path, monkeypatch):
    from tinyharness import Session, SessionOptions

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    async with Session(SessionOptions(config_path=tmp_path / "none.toml", working_dir=tmp_path)) as s:
        scratch = s.scratch.dir
        assert scratch.exists()
    assert not scratch.exists()


async def test_every_turn_records_how_it_ended(session):
    """Failed turns used to leave no trace in the log, so a provider error, a stuck model
    and the user pressing Ctrl+C all looked the same afterwards."""
    import json

    from conftest import ScriptedLLM
    from tinyharness.llm.types import LLMError

    session.llm.stream = ScriptedLLM(["hello"]).stream
    [e async for e in session.run("hi")]

    async def failing(*a, **k):
        raise LLMError("LLM error 500: upstream exploded")
        yield

    session.llm.stream = failing
    with pytest.raises(LLMError):
        [e async for e in session.run("again")]
    session.transcript.close()
    records = [json.loads(line) for line in session.transcript.path.read_text().splitlines()]
    ends = [r for r in records if r["event"] == "turn_end"]
    assert [e["reason"] for e in ends] == ["stop", "error"]
    assert "upstream exploded" in ends[1]["error"]


def test_skills_are_offered_to_the_model_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    skills = tmp_path / "home" / "skills" / "frontend-design"
    skills.mkdir(parents=True)
    (skills / "SKILL.md").write_text(
        "---\nname: frontend-design\ndescription: "
        + "Design web pages well. " * 40
        + "\n---\nDo it like this."
    )
    monkeypatch.setenv("TINYHARNESS_HOME", str(tmp_path / "home"))
    on = Session(SessionOptions(config_path=tmp_path / "none.toml", working_dir=tmp_path / "work"))
    off = Session(
        SessionOptions(
            config_path=tmp_path / "none.toml", working_dir=tmp_path / "work", expose_skill_tool=False
        )
    )
    try:
        assert "Skill" in on.tools.names() and "Skill" not in off.tools.names()
        listing = on.tools.get("Skill").description
        assert (
            "frontend-design: Design web pages well." in listing and len(listing) < 1500
        )  # the description is cut short
    finally:
        on.close()
        off.close()


async def test_a_prompt_naming_a_skill_is_sent_as_typed(session):
    """A note telling the model to load any skill the user named was appended to every
    such prompt, even "don't use the claude-design-skill". The system prompt already says
    a skill the user asks for isn't optional."""
    from conftest import ScriptedLLM
    from tinyharness.skills import Skill

    session.skills.add(Skill("claude-design-skill", "Design.", "..."))
    session.llm.stream = ScriptedLLM(["ok"]).stream
    prompt = "Don't use the claude-design-skill for this one."
    [e async for e in session.run(prompt)]
    assert [m.content for m in session.history.messages if m.role == "user"] == [prompt]
    assert "a skill, tool or approach they name isn't optional" in session.history.messages[0].content
