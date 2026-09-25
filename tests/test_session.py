import pytest

from tinyharness import MissingAPIKeyError, Session, SessionOptions


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
    session.set_model("gpt-6-luna")
    assert session.llm.model == "gpt-6-luna"
    assert session.compactor.context_window == session.config.context_window


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
