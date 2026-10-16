"""First-run setup for an installed copy (from PyPI); a git checkout is left as it was."""

import os

import pytest
from rich.console import Console

from tinyharness.cli import app
from tinyharness.config import dotenv_paths, load_config, read_auth, recent_models
from tinyharness.llm.client import Ping
from tinyharness.llm.types import ModelInfo
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


async def test_choosing_an_undiscovered_local_provider_runs_setup(installed, monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(app, "run_setup", _fake_setup("ollama", asked))
    assert await app.first_run_setup(SessionOptions(working_dir=tmp_path, provider="ollama"), Console())
    assert asked == ["ollama"]


async def test_choosing_an_unconfigured_litellm_gateway_runs_setup(installed, monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(app, "run_setup", _fake_setup("litellm", asked))
    assert await app.first_run_setup(SessionOptions(working_dir=tmp_path, provider="litellm"), Console())
    assert asked == ["litellm"]


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
    assert "Anthropic · Claude is not connected" in text and "console.anthropic.com" in text
    assert "No API key for" not in text


async def test_model_picker_lists_only_the_active_providers_models(repl, monkeypatch):
    """Seen live: one recent OpenAI model looked like the whole catalog because
    `/model` first mixed recent choices from every connected provider."""
    from tinyharness.cli import commands
    from tinyharness.llm.types import ModelInfo

    seen = {}

    class Catalog:
        async def list_models(self):
            return [ModelInfo("gpt-one"), ModelInfo("gpt-two", "GPT Two", context=128_000)]

        async def ping(self, model=None):
            return Ping(ok=True, status=200, message="model ok")

    def client_for(provider):
        seen.setdefault("providers", []).append(provider)
        return Catalog()

    async def choose(title, items, current=None, **kwargs):
        seen["title"] = title
        seen["items"] = [(item.value, item.title, item.detail) for item in items]
        seen["current"] = current
        return "gpt-two"

    monkeypatch.setattr(repl.session, "client_for", client_for)
    monkeypatch.setattr(commands, "palette", choose)

    await commands.model(repl, "")

    assert seen["providers"] == ["openai", "openai"]  # list, then check the selected model
    assert seen["title"] == "Choose a model · OpenAI · GPT"
    assert seen["items"] == [
        ("gpt-one", "gpt-one", ""),
        ("gpt-two", "GPT Two", "128k context"),
    ]
    assert seen["current"] == "gpt-6-sol"
    assert repl.session.config.provider == "openai"
    assert repl.session.config.model == "gpt-two"


async def test_model_command_does_not_silently_change_provider(repl):
    from tinyharness.cli.commands import model

    await model(repl, "anthropic:claude-sonnet-5")

    assert repl.session.config.provider == "openai"
    output = repl.console.export_text()
    assert "/model only shows OpenAI · GPT models" in output
    assert "Switch to Anthropic · Claude with /provider first" in output


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
    assert "Anthropic · Claude rejected that key: invalid x-api-key" in repl.console.export_text()


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
    assert "Saved OpenAI · GPT fallback; still using environment." in text


async def test_openrouter_can_connect_in_the_browser_without_a_pasted_key(monkeypatch, tmp_path):
    from tinyharness.cli.setup import connect_provider
    from tinyharness.llm.providers import load_catalog

    answers = iter(["openrouter", "browser"])

    async def choose(*args, **kwargs):
        return next(answers)

    async def login():
        return "sk-or-v1-connected"

    monkeypatch.setattr("tinyharness.cli.setup.pick", choose)
    monkeypatch.setattr("tinyharness.cli.setup.openrouter_browser_login", login)
    providers_answer(monkeypatch, Ping(ok=True, status=200, message="auth ok"))
    providers = load_catalog()
    check = await connect_provider(Console(), providers["openrouter"], providers)

    assert check and check.ok
    assert read_auth() == {"openrouter": "sk-or-v1-connected"}
    assert providers["openrouter"].api_key == "sk-or-v1-connected"


async def test_chatgpt_plan_connects_without_an_openai_api_key(monkeypatch, tmp_path):
    from tinyharness.cli.setup import connect_provider
    from tinyharness.config import read_chatgpt_profile
    from tinyharness.llm.providers import load_catalog

    async def login(host_id, **kwargs):
        assert host_id.startswith("urn:uuid:")
        kwargs["on_registration"]("oaiapp_tiny")
        return {
            "client_id": "oaiapp_tiny",
            "ext_agent_host_id": host_id,
            "issuer": "https://auth.openai.com",
            "subject": "account-1",
            "email": "person@example.com",
            "access_token": "oauth-access",
            "refresh_token": "oauth-refresh",
            "expires_at": 2_000_000_000,
            "scopes": ["chatgpt.tokens.use.direct"],
        }

    async def fetch_models(self):
        return [ModelInfo("gpt-plan", "GPT Plan")]

    monkeypatch.setattr("tinyharness.cli.setup.chatgpt_browser_login", login)
    monkeypatch.setattr("tinyharness.llm.client.LLMClient.fetch_models", fetch_models)
    seen = providers_answer(monkeypatch, Ping(ok=True, status=200, message="auth ok"))
    providers = load_catalog()
    check = await connect_provider(Console(), providers["chatgpt"], providers)

    assert check and check.ok and seen == ["oauth-access"]
    assert providers["chatgpt"].default_model == "gpt-plan"
    assert read_chatgpt_profile()["email"] == "person@example.com"
    restored = load_config(tmp_path / "none.toml").providers["chatgpt"]
    assert restored.has_key and restored.default_model == "gpt-plan"


async def test_github_copilot_connects_a_subscription_and_saves_the_github_token(monkeypatch, tmp_path):
    from tinyharness.cli.setup import connect_provider
    from tinyharness.llm.providers import load_catalog

    async def login(**kwargs):
        return "ghu_copilot"

    async def fetch_models(self):
        return [ModelInfo("claude-sonnet", "Claude Sonnet", api="anthropic")]

    monkeypatch.setattr("tinyharness.cli.setup.github_copilot_login", login)
    monkeypatch.setattr("tinyharness.llm.client.LLMClient.fetch_models", fetch_models)
    providers_answer(monkeypatch, Ping(ok=True, status=200, message="auth ok"))
    providers = load_catalog()
    check = await connect_provider(Console(), providers["github_copilot"], providers)

    assert check and check.ok
    assert read_auth() == {"github_copilot": "ghu_copilot"}
    restored = load_config(tmp_path / "none.toml").providers["github_copilot"]
    assert restored.default_model == "claude-sonnet"


async def test_github_copilot_setup_shows_the_reported_plan(monkeypatch, tmp_path):
    from tinyharness.cli.setup import connect_provider
    from tinyharness.llm.providers import load_catalog

    async def login(**kwargs):
        return "ghu_copilot"

    async def fetch_models(self):
        self._copilot_sku = "free_educational_quota"
        return [
            ModelInfo(
                "auto",
                "Auto · GitHub chooses the best available model",
                api="chat",
                routed=True,
            )
        ]

    monkeypatch.setattr("tinyharness.cli.setup.github_copilot_login", login)
    monkeypatch.setattr("tinyharness.llm.client.LLMClient.fetch_models", fetch_models)
    providers_answer(monkeypatch, Ping(ok=True, status=200, message="auth ok"))
    console = Console(record=True)
    providers = load_catalog()
    check = await connect_provider(console, providers["github_copilot"], providers)

    assert check and check.ok
    assert providers["github_copilot"].default_model == "auto"
    assert "Plan: Copilot Student" in console.export_text()


async def test_github_copilot_can_choose_again_when_an_advertised_model_is_unavailable(monkeypatch, tmp_path):
    """Seen live: GitHub listed Haiku, then rejected it as an unsupported model."""
    from tinyharness.cli.setup import connect_provider
    from tinyharness.llm.providers import load_catalog

    async def login(**kwargs):
        return "ghu_copilot"

    async def fetch_models(self):
        return [
            ModelInfo("claude-haiku-4.5", "Claude Haiku", api="anthropic"),
            ModelInfo("gpt-4.1", "GPT-4.1", api="chat"),
        ]

    choices = iter(["claude-haiku-4.5", "gpt-4.1"])

    async def choose(*args, **kwargs):
        return next(choices)

    answers = iter(
        [
            Ping(ok=False, status=400, message="The requested model is not supported."),
            Ping(ok=True, status=200, message="auth ok"),
        ]
    )

    async def ping(self, model=None):
        return next(answers)

    monkeypatch.setattr("tinyharness.cli.setup.github_copilot_login", login)
    monkeypatch.setattr("tinyharness.cli.setup.pick", choose)
    monkeypatch.setattr("tinyharness.llm.client.LLMClient.fetch_models", fetch_models)
    monkeypatch.setattr("tinyharness.llm.client.LLMClient.ping", ping)

    providers = load_catalog()
    check = await connect_provider(Console(), providers["github_copilot"], providers)

    assert check and check.ok
    assert providers["github_copilot"].default_model == "gpt-4.1"
    assert read_auth() == {"github_copilot": "ghu_copilot"}


async def test_litellm_gateway_endpoint_key_and_live_model_are_remembered(monkeypatch, tmp_path):
    from tinyharness.cli.setup import connect_provider
    from tinyharness.llm.providers import load_catalog

    answers = iter(["https://litellm.example.com/v1", "company-key"])

    async def answer(*args, **kwargs):
        return next(answers)

    async def fetch_models(self):
        assert self.provider.base_url == "https://litellm.example.com/v1"
        assert self.api_key == "company-key"
        return [ModelInfo("team/claude")]

    async def choose(*args, **kwargs):
        return "custom"

    monkeypatch.setattr("tinyharness.cli.setup.ask", answer)
    monkeypatch.setattr("tinyharness.cli.setup.pick", choose)
    monkeypatch.setattr("tinyharness.llm.client.LLMClient.fetch_models", fetch_models)
    providers_answer(monkeypatch, Ping(ok=True, status=200, message="model ok"))
    providers = load_catalog()
    check = await connect_provider(Console(), providers["litellm"], providers)

    assert check and check.ok
    restored = load_config(tmp_path / "none.toml").providers["litellm"]
    assert (restored.base_url, restored.default_model, restored.api_key) == (
        "https://litellm.example.com/v1",
        "team/claude",
        "company-key",
    )


async def test_local_provider_discovers_and_checks_a_model(monkeypatch):
    from tinyharness.cli.setup import connect_provider
    from tinyharness.llm.providers import load_catalog

    async def fetch_models(self):
        return [ModelInfo("qwen3:8b"), ModelInfo("gemma3:4b", "Gemma 3 4B")]

    choices = iter(["current", "gemma3:4b"])

    async def choose(*args, **kwargs):
        return next(choices)

    monkeypatch.setattr("tinyharness.cli.setup.pick", choose)
    monkeypatch.setattr("tinyharness.llm.client.LLMClient.fetch_models", fetch_models)
    seen = providers_answer(monkeypatch, Ping(ok=True, status=200, message="model ok"))
    providers = load_catalog()
    info = providers["ollama"]
    check = await connect_provider(Console(), info, providers)

    assert check and check.ok
    assert info.default_model == info.small_model == "gemma3:4b"
    assert seen == [""]


async def test_remote_self_hosted_endpoint_and_optional_key_are_remembered(monkeypatch, tmp_path):
    from tinyharness.cli.setup import connect_provider
    from tinyharness.llm.providers import load_catalog

    attempts = []

    async def fetch_models(self):
        attempts.append((self.provider.base_url, self.api_key))
        if len(attempts) == 1:
            raise OSError("nothing is listening")
        return [ModelInfo("company/model")]

    answers = iter(["https://models.example.com/v1", "server-secret"])

    async def answer(*args, **kwargs):
        return next(answers)

    async def choose(*args, **kwargs):
        return "current"

    monkeypatch.setattr("tinyharness.cli.setup.ask", answer)
    monkeypatch.setattr("tinyharness.cli.setup.pick", choose)
    monkeypatch.setattr("tinyharness.llm.client.LLMClient.fetch_models", fetch_models)
    providers_answer(monkeypatch, Ping(ok=True, status=200, message="model ok"))
    providers = load_catalog()
    check = await connect_provider(Console(), providers["vllm"], providers)

    assert check and check.ok
    assert attempts == [
        ("http://127.0.0.1:8000/v1", ""),
        ("https://models.example.com/v1", "server-secret"),
    ]
    restored = load_config(tmp_path / "none.toml").providers["vllm"]
    assert (restored.base_url, restored.default_model) == (
        "https://models.example.com/v1",
        "company/model",
    )
    assert restored.api_key == "server-secret"


async def test_reconfiguring_an_endpoint_can_keep_its_saved_server_key(monkeypatch):
    from tinyharness.cli.setup import connect_provider
    from tinyharness.llm.providers import load_catalog

    providers = load_catalog()
    info = providers["vllm"]
    info.set_api_key("existing-secret", "auth.json")
    choices = iter(["custom", "keep"])

    async def choose(*args, **kwargs):
        return next(choices)

    async def answer(*args, **kwargs):
        return "https://new-vllm.example.com/v1"

    async def fetch_models(self):
        assert self.api_key == "existing-secret"
        return [ModelInfo("company/model")]

    monkeypatch.setattr("tinyharness.cli.setup.pick", choose)
    monkeypatch.setattr("tinyharness.cli.setup.ask", answer)
    monkeypatch.setattr("tinyharness.llm.client.LLMClient.fetch_models", fetch_models)
    providers_answer(monkeypatch, Ping(ok=True, status=200, message="model ok"))

    check = await connect_provider(Console(), info, providers)
    assert check and check.ok
    assert info.base_url == "https://new-vllm.example.com/v1"


async def test_reconfiguring_an_endpoint_can_remove_its_server_key(monkeypatch):
    from tinyharness.cli.setup import connect_provider
    from tinyharness.llm.providers import load_catalog

    providers = load_catalog()
    info = providers["vllm"]
    info.set_api_key("old-secret", "auth.json")
    choices = iter(["credential", "none"])

    async def choose(*args, **kwargs):
        return next(choices)

    async def fetch_models(self):
        assert self.api_key == ""
        return [ModelInfo("company/model")]

    monkeypatch.setattr("tinyharness.cli.setup.pick", choose)
    monkeypatch.setattr("tinyharness.llm.client.LLMClient.fetch_models", fetch_models)
    providers_answer(monkeypatch, Ping(ok=True, status=200, message="model ok"))

    check = await connect_provider(Console(), info, providers)
    assert check and check.ok
    assert info.api_key == "" and read_auth() == {}


def test_local_runtimes_are_builtin_but_not_silently_auto_selected(tmp_path):
    providers = load_config(tmp_path / "none.toml").providers
    for name in ("ollama", "lmstudio", "llamacpp", "vllm", "sglang"):
        assert providers[name].local and not providers[name].needs_key
    assert providers["litellm"].endpoint_setup and not providers["litellm"].local
    assert not providers["litellm"].has_key
    assert load_config(tmp_path / "none.toml").provider == "anthropic"


async def test_connect_command_is_available():
    from tinyharness.cli.commands import COMMANDS

    assert COMMANDS["/connect"].help == "connect or configure a model provider"
