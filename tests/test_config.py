import os

import pytest

from tinyharness.config import (
    Config,
    load_config,
    load_dotenv_files,
    parse_dotenv,
    remember_model,
    save_api_key,
)
from tinyharness.llm.providers import load_catalog
from tinyharness.permissions import PermissionMode


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # don't pick up the repo's own .env


def write_config(tmp_path, body: str = ""):
    path = tmp_path / "config.toml"
    path.write_text(f'working_dir = "{tmp_path}/work"\n{body}')
    return path


# ---- .env ----


def test_parse_dotenv(tmp_path):
    f = tmp_path / ".env"
    f.write_text(
        "# comment\nFOO=bar\nBAZ=\"quoted value\"\nQUX='single'\nexport EXPORTED=yes\nEMPTY=\nnoequals\n"
    )
    assert parse_dotenv(f) == {
        "FOO": "bar",
        "BAZ": "quoted value",
        "QUX": "single",
        "EXPORTED": "yes",
        "EMPTY": "",
    }


def test_real_env_wins_over_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("TH_TEST_KEY=fromfile\n")
    monkeypatch.setenv("TH_TEST_KEY", "fromenv")
    load_dotenv_files(tmp_path / ".env")
    assert os.environ["TH_TEST_KEY"] == "fromenv"


def test_dotenv_selects_provider(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-fromdotenv\n")
    cfg = load_config(write_config(tmp_path))
    assert os.environ["OPENAI_API_KEY"] == "sk-fromdotenv"
    assert cfg.provider == "openai"
    assert cfg.resolve_api_key() == "sk-fromdotenv"
    monkeypatch.delenv("OPENAI_API_KEY")


# ---- config file ----


def test_config_file_values(tmp_path):
    cfg = load_config(write_config(tmp_path, 'model = "m1"\npermission_mode = "plan"\n'))
    assert cfg.model == "m1"
    assert cfg.permission_mode is PermissionMode.PLAN
    assert cfg.working_dir == (tmp_path / "work").resolve()
    assert cfg.working_dir.is_dir()


def test_legacy_sandbox_root_key_still_works(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(f'sandbox_root = "{tmp_path}/legacy"\n')
    assert load_config(path).working_dir == (tmp_path / "legacy").resolve()


def test_invalid_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, 'permission_mode = "yolo"\n'))


def test_pasting_a_key_into_api_key_env_is_an_error(tmp_path):
    body = '[providers.gw]\nbase_url = "https://x/v1"\napi_key_env = "sk-or-v1-abc123"\n'
    with pytest.raises(ValueError, match="NAME of an environment variable"):
        load_config(write_config(tmp_path, body))


# ---- choosing a provider and model ----


def test_defaults_to_the_first_provider_with_a_key(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    cfg = load_config(write_config(tmp_path))
    assert cfg.provider == "gemini"
    assert cfg.model == cfg.provider_info.default_model


def test_with_no_keys_the_first_catalog_provider_is_chosen(tmp_path):
    cfg = load_config(write_config(tmp_path))
    assert cfg.provider == next(iter(load_catalog()))
    assert cfg.resolve_api_key() == ""


def test_provider_prefix_in_model_selects_the_provider(tmp_path):
    cfg = load_config(write_config(tmp_path, 'model = "openai:gpt-6-astra"\n'))
    assert (cfg.provider, cfg.model) == ("openai", "gpt-6-astra")


def test_only_known_provider_names_count_as_a_prefix():
    cfg = Config(provider="openrouter")
    cfg.select("meta-llama/llama-4:free")
    assert (cfg.provider, cfg.model) == ("openrouter", "meta-llama/llama-4:free")


def test_unknown_models_are_accepted_and_listed():
    cfg = Config(provider="openai")
    cfg.select("gpt-7-preview")
    assert cfg.model == "gpt-7-preview"
    assert "gpt-7-preview" in [m.id for m in cfg.provider_info.models]


def test_use_provider_switches_to_its_default_model():
    cfg = Config(provider="anthropic")
    cfg.use_provider("groq")
    assert cfg.base_url == load_catalog()["groq"].base_url
    assert cfg.model == load_catalog()["groq"].default_model
    with pytest.raises(ValueError):
        cfg.use_provider("nope")


def test_context_window_follows_the_model():
    cfg = Config(model="anthropic:claude-haiku-4-5")
    assert cfg.context_window == 200_000
    cfg.select("claude-sonnet-5")
    assert cfg.context_window == 1_000_000
    assert Config(model="anthropic:x", context_window_tokens=50_000).context_window == 50_000


def test_env_model_pins_active_model(tmp_path, monkeypatch):
    monkeypatch.setenv("TINYHARNESS_MODEL", "openai:pinned")
    cfg = load_config(write_config(tmp_path, 'model = "from-toml"\n'))
    assert (cfg.provider, cfg.model) == ("openai", "pinned")


def test_last_picked_model_is_used_unless_config_pins_one(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    remember_model("openai", "gpt-6-luna")
    assert load_config(write_config(tmp_path)).model == "gpt-6-luna"
    assert load_config(write_config(tmp_path, 'model = "openai:gpt-6-sol"\n')).model == "gpt-6-sol"
    assert load_config(write_config(tmp_path, 'provider = "openai"\n')).model == "gpt-6-luna"
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert load_config(write_config(tmp_path, 'provider = "groq"\n')).provider == "groq"


def test_last_picked_model_is_ignored_without_its_key(tmp_path):
    remember_model("openai", "gpt-6-luna")
    assert load_config(write_config(tmp_path)).model != "gpt-6-luna"


# ---- custom providers ----


def test_custom_gateway_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_KEY", "gk")
    body = (
        'provider = "gateway"\n[providers.gateway]\nbase_url = "https://llm.example.com/v1"\n'
        'api_key_env = "GATEWAY_KEY"\nmodels = ["m-large", { id = "m-small", context = 32000 }]\n'
    )
    cfg = load_config(write_config(tmp_path, body))
    assert (cfg.provider, cfg.model, cfg.base_url) == ("gateway", "m-large", "https://llm.example.com/v1")
    assert cfg.resolve_api_key() == "gk"
    cfg.select("m-small")
    assert cfg.context_window == 32_000


def test_config_extends_a_builtin_provider(tmp_path):
    body = '[providers.openai]\nmodels = ["o9"]\nsmall_model = "o9"\n'
    cfg = load_config(write_config(tmp_path, body))
    openai = cfg.providers["openai"]
    ids = [m.id for m in openai.models]
    assert ids[-1] == "o9" and "gpt-6-sol" in ids  # added, not replaced
    assert openai.small_model == "o9"
    assert openai.base_url == load_catalog()["openai"].base_url


# ---- saving keys ----


def test_save_api_key_adds_and_replaces(tmp_path, monkeypatch):
    env = tmp_path / "keys.env"
    env.write_text("# mine\nOTHER=1\nOPENAI_API_KEY=old\n")
    save_api_key("OPENAI_API_KEY", "new", env)
    save_api_key("GROQ_API_KEY", "g", env)
    assert parse_dotenv(env) == {"OTHER": "1", "OPENAI_API_KEY": "new", "GROQ_API_KEY": "g"}
    assert env.stat().st_mode & 0o777 == 0o600
    assert os.environ["GROQ_API_KEY"] == "g"
    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.delenv("GROQ_API_KEY")


def test_permission_rules_in_config(tmp_path):
    cfg = load_config(
        write_config(tmp_path, '[permissions]\nallow = ["Edit(src/*)"]\ndeny = ["Read(*.pem)"]\n')
    )
    assert cfg.permissions.allow == ["Edit(src/*)"]
    assert cfg.permissions.deny == ["Read(*.pem)"]


def test_typo_in_permissions_section_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, '[permissions]\nalow = ["Edit"]\n'))
