"""User configuration: one TOML file, `~/.config/tinyharness/config.toml`.

Which model runs is decided in this order (first match wins):
  1. explicit overrides from the CLI (`--model`, `--provider`) or SessionOptions
  2. the TINYHARNESS_MODEL environment variable
  3. `model` in the config file
  4. the last model picked with /model (CLI only: `load_config(use_last_model=True)`,
     and only if it's from the configured `provider`, when one is set)
  5. the first provider (in catalog order) whose API key is set

A model can be named with its provider, `provider:model` (e.g. `openai:gpt-6-sol`),
which also switches provider. API keys never go in the config file: they are read
from the environment, and `.env` files are loaded into it first (real env wins).
The user's own key file is ~/.config/tinyharness/.env.
"""

from __future__ import annotations

import json
import logging
import os
import tomllib
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from tinyharness.llm.providers import Provider, load_catalog
from tinyharness.llm.types import Effort
from tinyharness.paths import default_config_path, home_dir, source_checkout
from tinyharness.permissions.modes import PermissionMode

log = logging.getLogger(__name__)


class PermissionSettings(BaseModel):
    """Persistent rules, e.g. allow = ["Edit(src/*)"], deny = ["Read(*/secrets/*)"]."""

    model_config = ConfigDict(extra="forbid")

    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class Config(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="ignore")

    # --- model ---
    provider: str = ""  # "" = the first provider whose API key is set
    model: str = ""  # "" = the provider's default; "provider:model" also picks the provider
    api_key: str = Field(default="", repr=False)  # literal key (SDK use); normally from the env
    providers: dict[str, Provider] = Field(default_factory=load_catalog)
    effort: Effort | None = None  # reasoning effort: low | medium | high | max; None = the model's default
    request_timeout: float = 1800.0  # seconds; slow reasoning models can take minutes per turn
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
    scratch_max_age_hours: int = 24

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
        """Cheap model for background work, such as the auto-mode classifier."""
        return self.provider_info.small_model or self.model

    def resolve_api_key(self) -> str:
        return self.api_key or self.provider_info.api_key

    def has_key(self, provider: str) -> bool:
        return bool(self.providers[provider].api_key) or (provider == self.provider and bool(self.api_key))

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


def user_env_path() -> Path:
    """The user's key file: a .env next to the config file."""
    return default_config_path().parent / ".env"


def write_env_template(providers: dict[str, Provider], path: Path | None = None) -> Path:
    """Create the key file with a commented-out line per provider (never overwrites)."""
    path = path or user_env_path()
    if path.exists():
        return path
    lines = [
        "# tinyharness API keys. Uncomment a line and paste your key after the `=`.",
        "# One provider is enough. This file is only readable by you, and the agent",
        "# can't read it. Keys set in your shell take priority over this file.",
        "",
    ]
    for p in providers.values():
        if p.key_url:
            lines.append(f"# {p.label}: {p.key_url}")
        lines.extend([f"# {p.api_key_env}=", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def save_key(env_name: str, value: str, providers: dict[str, Provider]) -> Path:
    """Put a key in the user's key file (replacing any line for it) and use it right away."""
    path = write_env_template(providers)
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, raw in enumerate(lines):
        name = raw.strip().lstrip("#").strip().removeprefix("export ").partition("=")[0].strip()
        if name == env_name and "=" in raw:
            lines[i] = f"{env_name}={value}"
            break
    else:
        lines.append(f"{env_name}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)
    os.environ[env_name] = value
    return path


def dotenv_paths(config_path: Path | None = None, working_dir: Path | None = None) -> list[Path]:
    """Where keys are looked for, lowest priority first (the real environment beats all).
    An installed copy (not a git checkout) reads only the user's key file."""
    if source_checkout() is None:
        return [user_env_path()]
    config_path = config_path or default_config_path()
    paths = [user_env_path(), config_path.parent / ".env", Path.cwd() / ".env"]
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
    path = path or default_config_path()
    data = _read_toml(path)

    working_dir = data.get("working_dir") or data.get("sandbox_root")
    load_dotenv_files(*dotenv_paths(path, Path(working_dir) if working_dir else None))

    unknown = set(data) - set(Config.model_fields) - {"sandbox_root"}
    if unknown:
        log.warning("ignoring unknown config keys in %s: %s", path, ", ".join(sorted(unknown)))

    providers = load_catalog(data.pop("providers", None))
    if pinned := os.environ.get("TINYHARNESS_MODEL", "").strip():
        data["model"] = pinned
    elif use_last_model and "model" not in data and (last := _last_model()):
        name = last.partition(":")[0]
        if name in providers and providers[name].api_key and data.get("provider", name) == name:
            data["model"] = last

    cfg = Config.model_validate({**data, "providers": providers})
    cfg.working_dir.mkdir(parents=True, exist_ok=True)
    return cfg
