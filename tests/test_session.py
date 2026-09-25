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
    user_env = tmp_path / "_user" / ".config" / "tinyharness" / ".env"
    user_env.parent.mkdir(parents=True, exist_ok=True)
    user_env.write_text("XAI_API_KEY=xk-added-later\n")
    session.set_provider("xai")
    assert session.config.provider == "xai"
    monkeypatch.delenv("XAI_API_KEY")


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
