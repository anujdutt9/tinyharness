"""User configuration: one TOML file, `~/.tinyharness/config.toml`.

Which model runs is decided in this order (first match wins):
  1. explicit overrides from the CLI (`--model`, `--provider`) or SessionOptions
  2. the TINYHARNESS_MODEL environment variable
  3. `model` in the config file
  4. the last model picked with /model (CLI only: `load_config(use_last_model=True)`,
     and only if it's from the configured `provider`, when one is set)
  5. the first provider (in catalog order) whose API key is set

A model can be named with its provider, `provider:model` (e.g. `openai:gpt-6-sol`),
which also switches provider. API keys never go in the config file. Each provider
reads its key from an environment variable (`ANTHROPIC_API_KEY`, ...), filled in this
order, first found wins:
  1. your shell
  2. ~/.tinyharness/auth.json, one entry per provider: {"anthropic": {"api_key": "..."}}.
     It's written for you (first-run setup, /key, or /provider and /model when a key is
     missing), only you can read it, and the agent's tools and sandbox can't.
  3. `.env` files: the working directory's, in a git checkout of tinyharness (development)

Everything of yours lives in ~/.tinyharness. Earlier versions kept config.toml and a
.env key file in ~/.config/tinyharness; they're moved on the first run (see
`migrate_legacy_config` and `migrate_legacy_keys`).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import secrets
import shutil
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from tinyharness.llm.providers import Provider, load_catalog
from tinyharness.llm.types import Effort
from tinyharness.llm.usage import Price
from tinyharness.paths import auth_path, default_config_path, home_dir, legacy_config_dir, source_checkout
from tinyharness.permissions.modes import PermissionMode
from tinyharness.sandbox.proxy import DEFAULT_ALLOWED_HOSTS

log = logging.getLogger(__name__)


class PermissionSettings(BaseModel):
    """Persistent rules, e.g. allow = ["Edit(src/*)"], deny = ["Read(*/secrets/*)"]."""

    model_config = ConfigDict(extra="forbid")

    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class SandboxSettings(BaseModel):
    """`[sandbox]`: how Bash commands are confined (see tinyharness/sandbox/)."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True  # False: commands run unsandboxed, and every one asks first
    allow_write: list[Path] = Field(default_factory=list)  # more writable dirs, e.g. a second repo
    allow_localhost: bool = False  # let commands use loopback (tests that start a local server)
    # Hosts commands may reach from inside the sandbox, through a local proxy: package
    # registries, GitHub and script CDNs by default. `*.example.com` covers subdomains.
    # [] turns the network off entirely. Anything else needs sandbox=false (it asks).
    allow_hosts: list[str] = Field(default_factory=lambda: list(DEFAULT_ALLOWED_HOSTS))


class ExtensionSettings(BaseModel):
    """`[extensions]`: which extensions to load, and a table of settings for each.

    [extensions]
    enabled = ["web", "agents"]
    [extensions.web]
    max_results = 5
    """

    model_config = ConfigDict(extra="allow")

    enabled: list[str] = Field(default_factory=list)

    def settings(self) -> dict[str, Any]:
        return dict(self.model_extra or {})


class Config(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    # --- model ---
    provider: str = ""  # "" = the first provider whose API key is set
    model: str = ""  # "" = the provider's default; "provider:model" also picks the provider
    api_key: str = Field(default="", repr=False)  # literal key (SDK use); normally from the env
    providers: dict[str, Provider] = Field(default_factory=load_catalog)
    # Reasoning effort: none | low | medium | high | max. None: the model's own default, and
    # medium for models whose reasoning is off unless asked for (Gemma 4, Qwen3 on OpenRouter).
    effort: Effort | None = None
    request_timeout: float = 1800.0  # seconds; slow reasoning models can take minutes per turn
    request_retries: int = 3  # on 429/5xx or a refused connection, before the reply starts
    # True when nothing named a provider or model and one was picked from the keys found
    # (an installed CLI then asks the user instead).
    auto_selected: bool = Field(default=False, exclude=True)

    # --- workspace & permissions ---
    working_dir: Path = Field(
        default_factory=Path.cwd,
        validate_default=True,
        validation_alias=AliasChoices("working_dir", "sandbox_root"),
    )
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    permissions: PermissionSettings = Field(default_factory=lambda: PermissionSettings())
    sandbox: SandboxSettings = Field(default_factory=lambda: SandboxSettings())
    extensions: ExtensionSettings = Field(default_factory=lambda: ExtensionSettings())
    scratch_max_age_hours: int = 24

    # --- terminal ---
    theme: Literal["system", "light", "dark"] = "system"  # read by the CLI at start (cli/theme.py)
    show_thinking: bool = False  # show the model's reasoning, where it shares it (ctrl+t toggles)

    # --- cost ---
    # Dollars per million tokens, by "provider:model" or model id, for providers that don't
    # report prices: pricing."anthropic:claude-sonnet-5" = { input = 3, output = 15, cached = 0.3 }
    pricing: dict[str, Price] = Field(default_factory=dict)

    # --- planning ---
    # Same as turning on the `todo` extension (the TodoWrite checklist); kept for older
    # configs. TINYHARNESS_TODO_LIST=1 turns it on too.
    todo_list: bool = False

    # List installed skills (name, when to use it, where its SKILL.md is) in the system prompt,
    # so the model can load one with Read when a task matches. False: skills run only by /name.
    skill_tool: bool = True

    # --- context management ---
    context_window_tokens: int | None = None  # None = what the provider reports for the model
    auto_compact_threshold: float = Field(default=0.90, gt=0, le=1)

    @field_validator("working_dir", mode="after")
    @classmethod
    def _resolve_dir(cls, v: Path) -> Path:
        return v.expanduser().resolve()

    def model_post_init(self, context: Any) -> None:
        provider, model = self.parse_model(self.model) if self.model else (self.provider, "")
        auto = not provider
        if auto:
            provider = next((n for n in self.providers if self.has_key(n)), next(iter(self.providers)))
        self.provider = provider
        self.select(f"{provider}:{model}")
        self.auto_selected = auto

    # -- the active provider and model ------------------------------------------------

    @property
    def provider_info(self) -> Provider:
        return self.providers[self.provider]

    @property
    def base_url(self) -> str:
        return self.provider_info.base_url

    @property
    def api_key_env(self) -> str:
        return self.provider_info.api_key_env

    @property
    def extra_headers(self) -> dict[str, str]:
        return self.provider_info.extra_headers

    @property
    def small_model(self) -> str:
        """Cheap model for background work: the auto-mode classifier and WebSearch digests."""
        return self.provider_info.small_model or self.model

    def price(self, provider: str, model: str) -> Price | None:
        """A price set in the config for this model, if any."""
        return self.pricing.get(f"{provider}:{model}") or self.pricing.get(model)

    def resolve_api_key(self) -> str:
        return self.api_key or self.provider_info.api_key

    def has_key(self, provider: str) -> bool:
        return self.providers[provider].has_key or (provider == self.provider and bool(self.api_key))

    # -- switching ----------------------------------------------------------------------

    def parse_model(self, spec: str) -> tuple[str, str]:
        """ "openai:gpt-6-sol" -> ("openai", "gpt-6-sol"); a bare id keeps the current provider.
        Only a known provider name counts as a prefix, so ids like "x:free" stay whole."""
        head, sep, tail = spec.partition(":")
        if sep and head in self.providers:
            return head, tail.strip()
        return self.provider, spec.strip()

    def select(self, spec: str) -> None:
        """Switch to a model, and to its provider if the spec names one."""
        provider, model = self.parse_model(spec)
        if provider not in self.providers:  # only possible for a bad `provider` in the config
            raise ValueError(f"unknown provider {provider!r}; choose one of: {', '.join(self.providers)}")
        self.provider = provider
        self.model = model or self.providers[provider].default_model
        self.auto_selected = False

    def use_provider(self, name: str) -> None:
        """Switch provider, using its default model."""
        if name not in self.providers:
            raise ValueError(f"unknown provider {name!r}; choose one of: {', '.join(self.providers)}")
        self.select(f"{name}:")


# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------


def parse_dotenv(path: Path) -> dict[str, str]:
    """Minimal .env parser: KEY=value, quoted values, `export` prefix, # comments
    (whole-line, or after an unquoted value)."""
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        key, sep, val = line.partition("=")
        if not sep or not key.strip():
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        else:
            val = val.split(" #", 1)[0].rstrip()  # inline comment
        out[key.strip()] = val
    return out


def load_dotenv_files(*paths: Path, override: bool = False) -> dict[str, str]:
    """Load .env files into os.environ. Later files win; real env wins unless override.
    Safe to call again (e.g. after the user adds a key): existing values are kept."""
    merged: dict[str, str] = {}
    for p in paths:
        if p.exists():
            merged.update(parse_dotenv(p))
    for k, v in merged.items():
        if override or k not in os.environ:
            os.environ[k] = v
    return merged


# ---------------------------------------------------------------------------
# Keys and the last-used model
# ---------------------------------------------------------------------------


def read_auth(path: Path | None = None) -> dict[str, str]:
    """{provider: key} from auth.json; {} if there's none (or it can't be read)."""
    keys: dict[str, str] = {}
    for name, entry in _read_auth_file(path).items():
        key = entry.get("api_key") if isinstance(entry, dict) else entry
        if isinstance(key, str) and key.strip():
            keys[name] = key.strip()
    return keys


def load_auth(providers: dict[str, Provider]) -> None:
    """Put auth.json's keys where providers read them. A key set in your shell wins."""
    for name, key in read_auth().items():
        if (provider := providers.get(name)) is not None and provider.needs_key:
            os.environ.setdefault(provider.api_key_env, key)


def save_key(provider: str, value: str, providers: dict[str, Provider]) -> Path:
    """Store a provider's key in auth.json (replacing its old one) and use it right away."""
    data = _read_auth_file()
    entry = data.get(provider)
    data[provider] = {**(entry if isinstance(entry, dict) else {}), "api_key": value}
    path = _write_auth_file(data)
    if providers[provider].needs_key:
        os.environ[providers[provider].api_key_env] = value
    return path


def _read_auth_file(path: Path | None = None) -> dict[str, Any]:
    path = path or auth_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log.warning("ignoring %s: %s", path, e)
        return {}
    if not isinstance(data, dict):
        log.warning("ignoring %s: expected an object of providers", path)
        return {}
    return data


def _write_auth_file(data: dict[str, Any]) -> Path:
    """Replace auth.json in one step (a crash can't leave half a file), readable only by you."""
    path = auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".auth.json.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
    path.chmod(0o600)
    return path


# ---------------------------------------------------------------------------
# Moving an earlier version's files into ~/.tinyharness
# ---------------------------------------------------------------------------


def migrate_legacy_config() -> None:
    """Move ~/.config/tinyharness/config.toml to ~/.tinyharness, unless one is already there."""
    old, new = legacy_config_dir() / "config.toml", default_config_path()
    if not old.is_file():
        return
    if new.exists():
        log.warning("using %s; %s is from an earlier version and is ignored (delete it when done)", new, old)
        return
    new.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(old, new)
    log.info("moved %s to %s", old, new)
    _remove_if_empty(legacy_config_dir())


def migrate_legacy_keys(providers: dict[str, Provider]) -> None:
    """Move provider keys from ~/.config/tinyharness/.env into auth.json. Lines that
    aren't a provider's key stay in the old file (with a warning); an emptied file goes."""
    old = legacy_config_dir() / ".env"
    if not old.is_file():
        return
    values = {k: v for k, v in parse_dotenv(old).items() if v}
    provider_for = {p.api_key_env: name for name, p in providers.items() if p.needs_key}
    stored = read_auth()
    moved = {
        provider_for[var]: key
        for var, key in values.items()
        if var in provider_for and provider_for[var] not in stored
    }
    if moved:
        data = _read_auth_file()
        for name, key in moved.items():
            data[name] = {"api_key": key}
        _write_auth_file(data)
        log.info("moved API keys for %s from %s to %s", ", ".join(sorted(moved)), old, auth_path())
    # Keep what isn't a key now in auth.json (another variable, or a different key for a
    # provider that already had one there: auth.json wins, but don't lose it).
    kept = {
        var: v
        for var, v in values.items()
        if not (var in provider_for and {**stored, **moved}.get(provider_for[var]) == v)
    }
    if kept:
        old.write_text("".join(f"{var}={v}\n" for var, v in kept.items()), encoding="utf-8")
        old.chmod(0o600)
        log.warning(
            "%s still has %s, which tinyharness doesn't move; set them in your shell, then delete the file",
            old,
            ", ".join(sorted(kept)),
        )
        return
    old.unlink()
    _remove_if_empty(legacy_config_dir())


def _remove_if_empty(directory: Path) -> None:
    with contextlib.suppress(OSError):  # not empty (or not ours): leave it
        (directory / ".DS_Store").unlink(missing_ok=True)
        directory.rmdir()


def dotenv_paths(working_dir: Path | None = None) -> list[Path]:
    """`.env` files keys are also read from, lowest priority first (the shell and
    auth.json beat them): only in a git checkout of tinyharness, for development."""
    if source_checkout() is None:
        return []
    paths = [Path.cwd() / ".env"]
    if working_dir is not None:
        paths.append(Path(working_dir).expanduser() / ".env")
    return list(dict.fromkeys(paths))


def _state_path() -> Path:
    return home_dir() / "state.json"


RECENT_MODELS = 5


def remember_model(provider: str, model: str) -> None:
    """Make the next session start with this model (unless the config pins one),
    and put it first in the recently used list."""
    path = _state_path()
    spec = f"{provider}:{model}"
    try:
        state = json.loads(path.read_text()) if path.exists() else {}
        state["last_model"] = spec
        recent = [s for s in state.get("recent_models", []) if s != spec]
        state["recent_models"] = [spec, *recent][:RECENT_MODELS]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2))
    except (OSError, ValueError) as e:
        log.debug("could not save the last model: %s", e)


def saved_theme() -> str:
    """The theme picked with /theme, if any."""
    return str(_state().get("theme", ""))


def save_theme(theme: str) -> None:
    state = _state()
    state["theme"] = theme
    try:
        _state_path().parent.mkdir(parents=True, exist_ok=True)
        _state_path().write_text(json.dumps(state, indent=2))
    except OSError as e:
        log.debug("could not save the theme: %s", e)


def note_launch() -> int:
    """Count interactive launches, and return the count (the banner is shorter after the first)."""
    state = _state()
    state["launches"] = int(state.get("launches", 0)) + 1
    try:
        _state_path().parent.mkdir(parents=True, exist_ok=True)
        _state_path().write_text(json.dumps(state, indent=2))
    except OSError as e:
        log.debug("could not save the launch count: %s", e)
    return int(state["launches"])


def forget_model(spec: str) -> None:
    """Drop a model from the recently used list (e.g. after it failed to respond)."""
    path = _state_path()
    state = _state()
    if spec not in state.get("recent_models", []):
        return
    state["recent_models"] = [s for s in state["recent_models"] if s != spec]
    if state.get("last_model") == spec:
        state.pop("last_model")
    try:
        path.write_text(json.dumps(state, indent=2))
    except OSError as e:
        log.debug("could not update the recent models: %s", e)


def _state() -> dict[str, Any]:
    try:
        data = json.loads(_state_path().read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _last_model() -> str:
    return str(_state().get("last_model", ""))


def recent_models() -> list[str]:
    """Recently used models as "provider:model", most recent first."""
    return [str(s) for s in _state().get("recent_models", [])]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def load_config(path: Path | None = None, *, use_last_model: bool = False) -> Config:
    """Read the config file and .env files. SDK code gets the same result every time;
    the CLI also passes `use_last_model` to resume with the model picked last."""
    if path is None:
        migrate_legacy_config()  # before reading it: it may be about to move here
    path = path or default_config_path()
    data = _read_toml(path)
    providers = load_catalog(data.pop("providers", None))

    migrate_legacy_keys(providers)
    load_auth(providers)  # before .env files: the first value found wins
    working_dir = data.get("working_dir") or data.get("sandbox_root")
    load_dotenv_files(*dotenv_paths(Path(working_dir) if working_dir else None))

    unknown = set(data) - set(Config.model_fields) - {"sandbox_root"}
    if unknown:
        log.warning("ignoring unknown config keys in %s: %s", path, ", ".join(sorted(unknown)))

    if todo_env := os.environ.get("TINYHARNESS_TODO_LIST", "").strip():
        data["todo_list"] = todo_env.lower() in ("1", "true", "yes", "on")
    if pinned := os.environ.get("TINYHARNESS_MODEL", "").strip():
        data["model"] = pinned
    elif use_last_model and "model" not in data and (last := _last_model()):
        name = last.partition(":")[0]
        if name in providers and providers[name].has_key and data.get("provider", name) == name:
            data["model"] = last

    cfg = Config.model_validate({**data, "providers": providers})
    cfg.working_dir.mkdir(parents=True, exist_ok=True)
    return cfg
