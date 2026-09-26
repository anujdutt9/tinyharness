"""Model providers: the built-in catalog and user-defined gateways.

The catalog lives in `providers.toml` next to this file and holds only how to
connect to each provider. Which models exist, and what they support, comes from
the provider itself (see models.py). A user's config can add a provider or adjust
a built-in one with a `[providers.<name>]` table:

    [providers.gateway]            # any OpenAI-compatible endpoint
    base_url = "https://llm.internal.example.com/v1"
    api_key_env = "GATEWAY_KEY"
    default_model = "claude-sonnet-5"

    [providers.openrouter]         # built-ins merge: this only changes one field
    small_model = "google/gemini-3.5-flash-lite"
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from tinyharness.llm.types import ModelInfo

CATALOG_PATH = Path(__file__).with_name("providers.toml")
DEFAULT_CONTEXT = 128_000  # when neither the provider nor the config says otherwise

Api = Literal["chat", "responses", "anthropic", "gemini"]


class Provider(BaseModel):
    name: str
    label: str = ""
    api: Api = "chat"  # wire format; see tinyharness/llm/adapters
    base_url: str
    api_key_env: str
    key_url: str = ""
    default_model: str = ""
    small_model: str = ""  # cheap model for background work (the auto-mode classifier)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)  # merged into every request
    effort_param: str = "reasoning_effort"  # chat API only: where the reasoning effort goes
    # The provider's own web search: true for APIs with a built-in search tool, a table
    # merged into the request for chat providers that switch it on another way, false: none.
    web_search: bool | dict[str, Any] = False
    models: list[ModelInfo] = Field(default_factory=list)  # for endpoints without a model list

    @field_validator("models", mode="before")
    @classmethod
    def _accept_plain_ids(cls, v: Any) -> Any:
        """`models = ["a", "b"]` is shorthand for `[{id = "a"}, {id = "b"}]`."""
        return [{"id": m} if isinstance(m, str) else m for m in v or []]

    @field_validator("api_key_env")
    @classmethod
    def _looks_like_a_variable_name(cls, v: str) -> str:
        if not v.replace("_", "").isalnum():
            raise ValueError(
                "api_key_env is the NAME of an environment variable (e.g. MY_API_KEY), "
                "not the key itself. Put the key in your environment or a .env file."
            )
        return v

    @model_validator(mode="after")
    def _fill_defaults(self) -> Provider:
        if not self.label:
            self.label = self.name
        if not self.default_model and self.models:
            self.default_model = self.models[0].id
        if not self.small_model:
            self.small_model = self.default_model
        return self

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "")


def load_catalog(overrides: dict[str, dict[str, Any]] | None = None) -> dict[str, Provider]:
    """Built-in providers merged with the user's `[providers.*]` tables.

    For a built-in, the user's fields replace the catalog's. A new name defines a
    new provider (it needs at least `base_url` and `api_key_env`).
    """
    with CATALOG_PATH.open("rb") as f:
        raw: dict[str, dict[str, Any]] = tomllib.load(f)
    for name, fields in (overrides or {}).items():
        raw.setdefault(name, {}).update(fields)
    return {name: Provider(name=name, **fields) for name, fields in raw.items()}
