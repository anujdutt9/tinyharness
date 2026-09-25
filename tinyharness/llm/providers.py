"""Model providers: the built-in catalog, user-defined gateways, and live model lists.

The catalog lives in `providers.toml` next to this file. A user's config can add a
provider or adjust a built-in one with a `[providers.<name>]` table:

    [providers.gateway]            # any OpenAI-compatible endpoint
    base_url = "https://llm.internal.example.com/v1"
    api_key_env = "GATEWAY_KEY"
    models = ["claude-sonnet-5", "gpt-6-sol"]

    [providers.openai]             # built-ins merge: this only adds a model
    models = ["o5-mini"]
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator, model_validator

CATALOG_PATH = Path(__file__).with_name("providers.toml")
DEFAULT_CONTEXT = 128_000


class ModelInfo(BaseModel):
    id: str
    label: str = ""
    note: str = ""
    context: int = DEFAULT_CONTEXT

    @property
    def display(self) -> str:
        return self.label or self.id


class Provider(BaseModel):
    name: str
    label: str = ""
    base_url: str
    api_key_env: str
    key_url: str = ""
    default_model: str = ""
    small_model: str = ""
    extra_headers: dict[str, str] = Field(default_factory=dict)
    models: list[ModelInfo] = Field(default_factory=list)

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

    def model(self, model_id: str) -> ModelInfo:
        """Catalog info for a model; unknown ids get defaults."""
        return next((m for m in self.models if m.id == model_id), ModelInfo(id=model_id))

    def add_model(self, model_id: str) -> None:
        if all(m.id != model_id for m in self.models):
            self.models.append(ModelInfo(id=model_id))


def load_catalog(overrides: dict[str, dict[str, Any]] | None = None) -> dict[str, Provider]:
    """Built-in providers merged with the user's `[providers.*]` tables.

    For a built-in, user fields replace the catalog's, except `models`, which are
    added to the built-in list. A new name defines a new provider (it needs at least
    `base_url` and `api_key_env`).
    """
    with CATALOG_PATH.open("rb") as f:
        raw: dict[str, dict[str, Any]] = tomllib.load(f)
    for name, fields in (overrides or {}).items():
        base = raw.setdefault(name, {})
        extra_models = fields.get("models", [])
        base.update({k: v for k, v in fields.items() if k != "models"})
        base["models"] = [*base.get("models", []), *extra_models]
    catalog = {name: Provider(name=name, **fields) for name, fields in raw.items()}
    for provider in catalog.values():  # user-added ids may repeat catalog ones; first wins
        unique: dict[str, ModelInfo] = {}
        for m in provider.models:
            unique.setdefault(m.id, m)
        provider.models = list(unique.values())
    return catalog


async def fetch_models(provider: Provider, api_key: str, timeout: float = 15.0) -> list[str]:
    """Every model id the provider serves, from its `GET /models` endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-api-key": api_key,  # Anthropic's /models reads this header instead
        **provider.extra_headers,
    }
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(f"{provider.base_url.rstrip('/')}/models", headers=headers)
    resp.raise_for_status()
    data = resp.json().get("data") or resp.json().get("models") or []
    ids = {str(item.get("id") or item.get("name", "")).removeprefix("models/") for item in data}
    return sorted(i for i in ids if i)
