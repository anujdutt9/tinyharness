import os

import pytest

from tinyharness.config import (
    Config,
    forget_model,
    load_config,
    load_dotenv_files,
    parse_dotenv,
    recent_models,
    remember_model,
    write_env_template,
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


def test_dotenv_inline_comments(tmp_path):
    f = tmp_path / ".env"
    f.write_text('A=sk-123   # my key\nB="has # inside"\n')
    assert parse_dotenv(f) == {"A": "sk-123", "B": "has # inside"}


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


def test_use_provider_switches_to_its_default_model():
    cfg = Config(provider="anthropic")
    cfg.use_provider("groq")
    assert cfg.base_url == load_catalog()["groq"].base_url
    assert cfg.model == load_catalog()["groq"].default_model
    with pytest.raises(ValueError):
        cfg.use_provider("nope")


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


def test_recent_models_are_most_recent_first_and_can_be_forgotten():
    for spec in ("openai:a", "openai:b", "openai:a"):
        remember_model(*spec.split(":"))
    assert recent_models() == ["openai:a", "openai:b"]
    forget_model("openai:b")
    assert recent_models() == ["openai:a"]


# ---- custom providers ----


def test_custom_gateway_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("GATEWAY_KEY", "gk")
    body = (
        'provider = "gateway"\n[providers.gateway]\nbase_url = "https://llm.example.com/v1"\n'
        'api_key_env = "GATEWAY_KEY"\nmodels = ["m-large", { id = "m-small", context = 32000 }]\n'
    )
    cfg = load_config(write_config(tmp_path, body))
    assert (cfg.provider, cfg.model, cfg.base_url) == ("gateway", "m-large", "https://llm.example.com/v1")
    assert cfg.provider_info.api == "chat"  # the default format for custom endpoints
    assert cfg.resolve_api_key() == "gk"
    assert cfg.provider_info.models[1].context == 32_000


def test_config_adjusts_a_builtin_provider(tmp_path):
    cfg = load_config(write_config(tmp_path, '[providers.openai]\nsmall_model = "o9"\n'))
    openai = cfg.providers["openai"]
    assert openai.small_model == "o9"
    assert openai.base_url == load_catalog()["openai"].base_url  # the rest is untouched
    assert openai.api == "responses"


def test_every_builtin_provider_uses_a_known_api():
    from tinyharness.llm.adapters import ADAPTERS

    assert {p.api for p in load_catalog().values()} <= set(ADAPTERS)


# ---- key file ----


def test_env_template_lists_every_provider_and_is_private(tmp_path):
    path = write_env_template(load_catalog(), tmp_path / "keys.env")
    text = path.read_text()
    assert "# ANTHROPIC_API_KEY=" in text and "# OPENROUTER_API_KEY=" in text
    assert parse_dotenv(path) == {}  # everything commented out
    assert path.stat().st_mode & 0o777 == 0o600


def test_env_template_never_overwrites(tmp_path):
    path = tmp_path / "keys.env"
    path.write_text("OPENAI_API_KEY=mine\n")
    write_env_template(load_catalog(), path)
    assert path.read_text() == "OPENAI_API_KEY=mine\n"


def test_keys_in_the_user_key_file_are_found(tmp_path, monkeypatch):
    user_env = tmp_path / "_user" / ".config" / "tinyharness" / ".env"
    user_env.parent.mkdir(parents=True)
    user_env.write_text("GROQ_API_KEY=gk\n")
    cfg = load_config(write_config(tmp_path))
    assert cfg.provider == "groq"
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
