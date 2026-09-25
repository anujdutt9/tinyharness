"""First-run setup for an installed copy (from PyPI); a git checkout is left as it was."""

import os

import pytest
from rich.console import Console

from tinyharness.cli import app
from tinyharness.config import dotenv_paths, load_config, parse_dotenv, recent_models, save_key, user_env_path
from tinyharness.llm.client import Ping
from tinyharness.llm.providers import load_catalog
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


def test_an_installed_copy_reads_only_the_key_file(installed, tmp_path):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-lying-around\n")
    assert dotenv_paths(working_dir=tmp_path) == [user_env_path()]
    cfg = load_config(tmp_path / "none.toml")
    assert "OPENAI_API_KEY" not in os.environ
    assert not cfg.has_key("openai")


def test_a_provider_picked_from_the_keys_found_is_marked_auto(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    cfg = load_config(tmp_path / "none.toml")
    assert cfg.auto_selected
    cfg.use_provider("gemini")
    assert not cfg.auto_selected


def test_save_key_fills_in_the_template_line(installed, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "old")  # so the test's value is undone afterwards
    path = save_key("MISTRAL_API_KEY", "new", load_catalog())
    assert parse_dotenv(path) == {"MISTRAL_API_KEY": "new"}
    assert path.read_text().count("MISTRAL_API_KEY") == 1  # replaced the commented-out line
    assert path.stat().st_mode & 0o777 == 0o600
    assert os.environ["MISTRAL_API_KEY"] == "new"


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
