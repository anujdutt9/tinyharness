import json
import os

import pytest

from tinyharness.config import (
    Config,
    forget_model,
    load_config,
    load_dotenv_files,
    parse_dotenv,
    read_auth,
    recent_models,
    remember_model,
    save_key,
)
from tinyharness.llm.providers import load_catalog
from tinyharness.paths import auth_path, default_config_path, legacy_config_dir
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
    assert "OPENAI_API_KEY" not in os.environ
    assert cfg.provider == "openai"
    assert cfg.resolve_api_key() == "sk-fromdotenv"
    assert cfg.api_key_source == ".env"


def test_dotenv_beats_auth_json_for_development(no_keys, tmp_path):
    (tmp_path / ".env").write_text("GROQ_API_KEY=from-dotenv\n")
    write_auth({"groq": {"api_key": "from-auth"}})
    cfg = load_config(write_config(tmp_path, 'provider = "groq"\n'))
    assert cfg.resolve_api_key() == "from-dotenv"
    assert cfg.api_key_source == ".env"
    assert "GROQ_API_KEY" not in os.environ


# ---- config file ----


def test_config_file_values(tmp_path):
    cfg = load_config(write_config(tmp_path, 'model = "m1"\npermission_mode = "plan"\n'))
    assert cfg.model == "m1"
    assert cfg.permission_mode is PermissionMode.PLAN
    assert cfg.working_dir == (tmp_path / "work").resolve()
    assert cfg.working_dir.is_dir()


def test_sdk_key_belongs_only_to_the_selected_provider():
    cfg = Config(provider="openai", api_key="sdk-key")
    assert cfg.resolve_api_key() == "sdk-key"
    assert cfg.api_key_source == "SDK override"
    cfg.use_provider("anthropic")
    assert cfg.resolve_api_key() == ""
    assert not cfg.has_key("anthropic")
    cfg.use_provider("openai")
    assert cfg.resolve_api_key() == "sdk-key"


def test_background_model_roles_are_separate_and_keep_legacy_fallbacks(tmp_path):
    cfg = load_config(
        write_config(
            tmp_path,
            'provider = "openai"\n[providers.openai]\nsmall_model = "legacy-small"\n'
            '[models.roles]\nsafety_classifier = "guard"\nsearch_digest = "digest"\n'
            'compaction = "summarizer"\n',
        )
    )
    assert cfg.role_model("safety_classifier") == "guard"
    assert cfg.role_model("search_digest") == "digest"
    assert cfg.role_model("compaction") == "summarizer"
    assert cfg.small_model == "legacy-small"


def test_compaction_role_keeps_legacy_and_active_model_fallbacks(tmp_path):
    legacy = load_config(write_config(tmp_path, 'model = "lead"\ncompaction_model = "legacy"\n'))
    default = load_config(write_config(tmp_path, 'model = "lead"\n'))
    assert legacy.role_model("compaction") == "legacy"
    assert default.role_model("compaction") == "lead"


def test_background_roles_reject_cross_provider_specs(tmp_path):
    with pytest.raises(ValueError, match="must be a model ID for the active provider"):
        load_config(write_config(tmp_path, '[models.roles]\ncompaction = "openai:gpt-6-sol"\n'))


def test_required_sandbox_cannot_be_disabled(tmp_path):
    with pytest.raises(ValueError, match=r"needs sandbox\.enabled"):
        load_config(write_config(tmp_path, "[sandbox]\nenabled = false\nrequired = true\n"))


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
    assert cfg.credential_selected and cfg.auto_selected  # old name remains a read-only alias


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


def test_parallel_agent_limit_from_config_or_env(tmp_path, monkeypatch):
    assert load_config(write_config(tmp_path)).max_parallel_agents == 20
    assert load_config(write_config(tmp_path, "max_parallel_agents = 6\n")).max_parallel_agents == 6
    monkeypatch.setenv("TINYHARNESS_MAX_PARALLEL_AGENTS", "8")
    assert load_config(write_config(tmp_path, "max_parallel_agents = 6\n")).max_parallel_agents == 8


def test_last_picked_model_is_used_unless_config_pins_one(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    remember_model("openai", "gpt-6-luna")

    def load(body: str = "") -> Config:
        return load_config(write_config(tmp_path, body), use_last_model=True)

    assert load().model == "gpt-6-luna"
    assert load('model = "openai:gpt-6-sol"\n').model == "gpt-6-sol"
    assert load('provider = "openai"\n').model == "gpt-6-luna"
    monkeypatch.setenv("GROQ_API_KEY", "k")
    assert load('provider = "groq"\n').provider == "groq"


def test_sdk_ignores_the_last_picked_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    remember_model("openai", "gpt-6-luna")
    assert load_config(write_config(tmp_path)).model != "gpt-6-luna"


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


@pytest.fixture
def no_keys(monkeypatch):
    """No provider key set, and whatever a test sets undone afterwards."""
    for p in load_catalog().values():
        monkeypatch.delenv(p.api_key_env, raising=False)


def write_auth(data):
    auth_path().parent.mkdir(parents=True, exist_ok=True)
    auth_path().write_text(json.dumps(data))


def test_saved_keys_go_to_auth_json_by_provider(no_keys):
    providers = load_catalog()
    save_key("groq", "gk-1", providers)
    path = save_key("anthropic", "sk-ant-1", providers)
    save_key("anthropic", "sk-ant-2", providers)  # replaces; the other key stays
    assert path == auth_path()
    assert json.loads(path.read_text()) == {"anthropic": {"api_key": "sk-ant-2"}, "groq": {"api_key": "gk-1"}}
    assert path.stat().st_mode & 0o777 == 0o600
    assert providers["anthropic"].api_key == "sk-ant-2"  # usable right away
    assert providers["anthropic"].api_key_source == "auth.json"
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert [p.name for p in path.parent.glob(".auth.json.*")] == []  # no temporary file left


def test_keys_in_auth_json_are_found(no_keys, tmp_path):
    write_auth({"groq": {"api_key": "gk"}})
    cfg = load_config(write_config(tmp_path))
    assert cfg.provider == "groq" and cfg.resolve_api_key() == "gk"
    assert cfg.api_key_source == "auth.json"


def test_a_key_in_your_shell_beats_auth_json(no_keys, tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from-shell")
    write_auth({"groq": {"api_key": "from-file"}})
    cfg = load_config(write_config(tmp_path))
    assert cfg.resolve_api_key() == "from-shell"
    assert cfg.api_key_source == "environment"


def test_a_broken_auth_json_is_ignored_not_fatal(no_keys, tmp_path, caplog):
    write_auth({})
    auth_path().write_text("{not json")
    load_config(write_config(tmp_path))
    assert "ignoring" in caplog.text and "auth.json" in caplog.text


# ---- moving an earlier version's files in ----


def old_folder(config: str = "", env: str = ""):
    old = legacy_config_dir()
    old.mkdir(parents=True)
    if config:
        (old / "config.toml").write_text(config)
    if env:
        (old / ".env").write_text(env)
    return old


def test_the_old_config_folder_is_moved_into_one_place(no_keys):
    old = old_folder('theme = "light"\n', "# a comment\nGROQ_API_KEY=gk\nOPENAI_API_KEY=\n")
    cfg = load_config()
    assert default_config_path().read_text() == 'theme = "light"\n'
    assert read_auth() == {"groq": "gk"}
    assert not old.exists()  # nothing left behind
    assert cfg.theme == "light" and cfg.resolve_api_key() == "gk"


def test_what_cant_move_stays_where_it_was(no_keys, caplog):
    default_config_path().parent.mkdir(parents=True, exist_ok=True)
    default_config_path().write_text('theme = "dark"\n')
    old = old_folder('theme = "light"\n', "GROQ_API_KEY=gk\nSOME_TOOL_TOKEN=abc\n")
    load_config()
    assert default_config_path().read_text() == 'theme = "dark"\n'  # the new one isn't replaced
    assert (old / "config.toml").exists()
    assert read_auth() == {"groq": "gk"}
    assert parse_dotenv(old / ".env") == {"SOME_TOOL_TOKEN": "abc"}  # not a provider key: kept
    assert "SOME_TOOL_TOKEN" in caplog.text


def test_a_different_key_for_a_provider_already_in_auth_json_is_kept(no_keys):
    write_auth({"groq": {"api_key": "newer"}})
    old = old_folder(env="GROQ_API_KEY=older\n")
    load_config()
    assert read_auth() == {"groq": "newer"}  # auth.json wins...
    assert parse_dotenv(old / ".env") == {"GROQ_API_KEY": "older"}  # ...but the other isn't lost


def test_permission_rules_in_config(tmp_path):
    cfg = load_config(
        write_config(tmp_path, '[permissions]\nallow = ["Edit(src/*)"]\ndeny = ["Read(*.pem)"]\n')
    )
    assert cfg.permissions.allow == ["Edit(src/*)"]
    assert cfg.permissions.deny == ["Read(*.pem)"]


def test_typo_in_permissions_section_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, '[permissions]\nalow = ["Edit"]\n'))


def test_a_providers_tables_merge_with_the_built_in_ones():
    """Adding OpenRouter routing preferences used to replace the whole built-in extra_body,
    silently dropping prompt caching and cost reporting."""
    from tinyharness.llm.providers import load_catalog

    body = load_catalog({"openrouter": {"extra_body": {"provider": {"ignore": ["Novita"]}}}})[
        "openrouter"
    ].extra_body
    assert body["provider"] == {"ignore": ["Novita"]}
    assert body["usage"] == {"include": True} and "cache_control" in body


def test_a_key_set_for_this_run_beats_one_only_saved(no_keys, tmp_path, monkeypatch):
    """Seen live: with OPENROUTER_API_KEY in the run's .env and a stale Anthropic key in
    auth.json, Anthropic (first in the provider list) was picked and failed with a 401."""
    write_auth({"anthropic": {"api_key": "sk-ant-stale"}})
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fresh")
    assert load_config(write_config(tmp_path)).provider == "openrouter"
