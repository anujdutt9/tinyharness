"""First-run setup for an installed copy (from PyPI); a git checkout is left as it was."""

import os

import pytest
from rich.console import Console

from tinyharness.cli import app
from tinyharness.config import dotenv_paths, load_config, read_auth, recent_models
from tinyharness.llm.client import Ping
from tinyharness.paths import auth_path
from tinyharness.session import SessionOptions


@pytest.fixture
def installed(monkeypatch):
    """Behave like a copy installed from a package, not this git checkout."""
    monkeypatch.setattr("tinyharness.config.source_checkout", lambda: None)
    monkeypatch.setattr("tinyharness.cli.app.source_checkout", lambda: None)


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_a_checkout_reads_env_files_as_before(tmp_path):
    assert tmp_path / ".env" in dotenv_paths()


def test_an_installed_copy_reads_no_env_files(installed, tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-lying-around\n")
    assert dotenv_paths(working_dir=tmp_path) == []
    cfg = load_config(tmp_path / "none.toml")
    assert "OPENAI_API_KEY" not in os.environ
    assert not cfg.has_key("openai")


def test_a_provider_picked_from_credentials_is_named_accurately(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    cfg = load_config(tmp_path / "none.toml")
    assert cfg.credential_selected and cfg.auto_selected  # compatibility alias
    cfg.use_provider("gemini")
    assert not cfg.credential_selected and not cfg.auto_selected


@pytest.mark.parametrize(
    ("status", "message", "bad"),
    [
        (401, "Unauthorized", True),
        (400, "API key not valid. Please pass a valid API key.", True),  # Gemini
        (403, "Invalid API key", True),
        (400, "model not found", False),
        (None, "ConnectError", False),
        (500, "internal error", False),
    ],
)
def test_ping_recognizes_a_rejected_key(status, message, bad):
    assert Ping(ok=False, status=status, message=message).bad_key is bad


def _fake_setup(answer, asked):
    async def run_setup(console, config):
        asked.append(config.provider)
        return answer

    return run_setup


async def test_installed_first_run_asks_instead_of_using_a_key_it_found(installed, monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "g")  # e.g. exported for some other tool
    asked = []
    monkeypatch.setattr(app, "run_setup", _fake_setup("gemini", asked))
    assert await app.first_run_setup(SessionOptions(working_dir=tmp_path, use_last_model=True), Console())
    assert asked == ["gemini"]
    assert recent_models()[0].startswith("gemini:")  # remembered, so no setup next time

    asked.clear()
    assert await app.first_run_setup(SessionOptions(working_dir=tmp_path, use_last_model=True), Console())
    assert asked == []


async def test_installed_first_run_can_be_cancelled(installed, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "run_setup", _fake_setup(None, []))
    assert not await app.first_run_setup(SessionOptions(working_dir=tmp_path), Console())


async def test_choosing_a_provider_skips_setup(installed, monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    asked = []
    monkeypatch.setattr(app, "run_setup", _fake_setup("gemini", asked))
    assert await app.first_run_setup(SessionOptions(working_dir=tmp_path, provider="gemini"), Console())
    assert asked == []


async def test_a_checkout_never_runs_setup_unless_asked(monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(app, "run_setup", _fake_setup("gemini", asked))
    assert await app.first_run_setup(SessionOptions(working_dir=tmp_path), Console())
    assert asked == []
    assert await app.first_run_setup(SessionOptions(working_dir=tmp_path), Console(), force=True)
    assert asked == ["anthropic"]


# ---- adding a key from inside a session ----

ANTHROPIC_KEY = "sk-ant-" + "k" * 40


@pytest.fixture
def repl(tmp_path, monkeypatch):
    from tinyharness import Session
    from tinyharness.cli.render import Renderer
    from tinyharness.cli.repl import Repl
    from tinyharness.cli.theme import THEME

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "o" * 40)
    session = Session(
        SessionOptions(config_path=tmp_path / "none.toml", working_dir=tmp_path, provider="openai")
    )
    console = Console(width=120, record=True, theme=THEME)
    yield Repl(session, console, Renderer(console))
    session.close()


def providers_answer(monkeypatch, *verdicts):
    """Fake the provider's key check: each call gets the next verdict (the last repeats)."""
    seen = []

    async def ping(self, model=None):
        seen.append(self.api_key if hasattr(self, "api_key") else None)
        return verdicts[min(len(seen), len(verdicts)) - 1]

    monkeypatch.setattr("tinyharness.llm.client.LLMClient.ping", ping)
    return seen


def user_types(monkeypatch, *answers):
    """Fake what's pasted at the hidden prompt ("" is Esc)."""
    queue = list(answers)

    async def ask(prompt, **kw):
        assert kw.get("secret"), "keys must be asked for hidden"
        return queue.pop(0)

    monkeypatch.setattr("tinyharness.cli.setup.ask", ask)


def test_whoami_reports_the_resolved_key_source(repl):
    from tinyharness.cli.commands import whoami

    whoami(repl, "")
    text = repl.console.export_text()
    assert "key from" in text and "environment" in text


async def test_switching_to_a_provider_without_a_key_asks_for_it(repl, monkeypatch):
    from tinyharness.cli.commands import provider

    user_types(monkeypatch, ANTHROPIC_KEY)
    providers_answer(monkeypatch, Ping(ok=True, status=200, message="auth ok"))
    await provider(repl, "anthropic")
    assert repl.session.config.provider == "anthropic"
    assert read_auth() == {"anthropic": ANTHROPIC_KEY}
    if os.name != "nt":
        assert oct(auth_path().stat().st_mode & 0o777) == "0o600"
    text = repl.console.export_text()
    assert "Anthropic needs an API key" in text and "console.anthropic.com" in text
    assert "No API key for" not in text


async def test_skipping_the_key_keeps_the_current_model(repl, monkeypatch):
    from tinyharness.cli.commands import provider

    user_types(monkeypatch, "")
    await provider(repl, "anthropic")
    assert repl.session.config.provider == "openai"
    assert "Add one with /key anthropic" in repl.console.export_text()  # the fallback


async def test_a_rejected_key_is_asked_for_again(repl, monkeypatch):
    from tinyharness.cli.commands import provider

    user_types(monkeypatch, "sk-ant-wrong" + "x" * 30, ANTHROPIC_KEY)
    rejected = Ping(ok=False, status=401, message="invalid x-api-key")
    providers_answer(monkeypatch, rejected, Ping(ok=True, status=200, message="auth ok"))
    await provider(repl, "anthropic")
    assert repl.session.config.provider == "anthropic"
    assert "Anthropic rejected that key: invalid x-api-key" in repl.console.export_text()


async def test_saved_key_does_not_override_the_process_environment(repl, monkeypatch):
    from tinyharness.cli.commands import key

    new = "sk-" + "n" * 40
    user_types(monkeypatch, new)
    providers_answer(monkeypatch, Ping(ok=True, status=200, message="auth ok"))
    await key(repl, "openai")
    assert os.environ["OPENAI_API_KEY"] != new
    assert read_auth() == {"openai": new}
    assert repl.session.llm.api_key == os.environ["OPENAI_API_KEY"]
    text = repl.console.export_text()
    assert "uses a key from environment" in text
    assert "Saved OpenAI fallback; still using environment." in text
