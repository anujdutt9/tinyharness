"""Model lists, straight from each provider's own `/models` endpoint.

Nothing about individual models is hard-coded: context windows, output limits and
thinking support all come from what the provider reports (when it reports them).
Lists are cached on disk for a day, one file per provider.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tinyharness.llm.types import ModelInfo
from tinyharness.llm.usage import Price

log = logging.getLogger(__name__)

MAX_AGE_SECONDS = 24 * 3600
CACHE_VERSION = 6  # an older cache is refetched at once (6: routed models)

COPILOT_AUTO_ONLY_SKUS = frozenset({"free_limited_copilot", "free_educational_quota"})
COPILOT_PLAN_LABELS = {
    "free_limited_copilot": "Copilot Free",
    "free_educational_quota": "Copilot Student",
    "copilot_for_business_seat_quota": "Copilot Business",
    "copilot_enterprise_seat_quota": "Copilot Enterprise",
}


def github_copilot_plan(sku: str) -> str:
    """A user-facing plan family. GitHub's SKU names are hints, not a stable public API."""
    if sku in COPILOT_PLAN_LABELS:
        return COPILOT_PLAN_LABELS[sku]
    if "subscriber" in sku:
        return "Copilot paid plan"
    if "trial" in sku:
        return "Copilot trial"
    return ""


def parse_github_copilot_models(body: dict[str, Any], *, sku: str = "") -> list[ModelInfo]:
    """Read the models this account may pick, plus GitHub's Auto route."""
    # Auto is not a model id the API accepts: the client opens an Auto session, which
    # names the models to use and a token to send with them (see LLMClient._route).
    auto = ModelInfo(
        id="auto",
        label="Auto · GitHub chooses the best available model",
        reasoning=False,
        api="chat",
        routed=True,
    )
    if sku in COPILOT_AUTO_ONLY_SKUS:
        return [auto]
    models = [auto]
    for item in _copilot_items(body):
        policy = item.get("policy") or {}
        if item.get("model_picker_enabled") is not True or policy.get("state") == "disabled":
            continue
        if (model := _copilot_model(item)) is not None and model.id not in {m.id for m in models}:
            models.append(model)
    return models


def parse_chatgpt_models(body: dict[str, Any]) -> list[ModelInfo]:
    """Read the account-specific model picker returned for ChatGPT plan usage."""
    models = []
    for item in body.get("models") or []:
        if (
            not isinstance(item, dict)
            or item.get("visibility") != "list"
            or not isinstance(item.get("slug"), str)
        ):
            continue
        models.append(ModelInfo(id=item["slug"], label=str(item.get("display_name") or "")))
    return models


def github_copilot_catalog(body: dict[str, Any]) -> dict[str, ModelInfo]:
    """Every model GitHub can serve, including the ones only Auto routes to."""
    found = (_copilot_model(item) for item in _copilot_items(body))
    return {model.id: model for model in found if model is not None}


def _copilot_items(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item for item in body.get("data") or [] if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]


def _copilot_model(item: dict[str, Any]) -> ModelInfo | None:
    """One catalog entry, or None when it can't run an agent (no tools, no known endpoint)."""
    capabilities = item.get("capabilities") or {}
    supports = capabilities.get("supports") or {}
    limits = capabilities.get("limits") or {}
    if supports.get("tool_calls") is False:
        return None
    endpoints = item.get("supported_endpoints") or []
    api = next(
        (
            kind
            for path, kind in (
                ("/v1/messages", "anthropic"),
                ("/responses", "responses"),
                ("/chat/completions", "chat"),
            )
            if path in endpoints
        ),
        None,
    )
    if api is None:
        return None
    efforts = supports.get("reasoning_effort")
    return ModelInfo(
        id=item["id"],
        label=str(item.get("name") or ""),
        context=_positive_int(limits.get("max_context_window_tokens") or limits.get("max_prompt_tokens")),
        max_output=_positive_int(limits.get("max_output_tokens")),
        adaptive_thinking=supports.get("adaptive_thinking") or None,
        efforts=frozenset(map(str, efforts)) if isinstance(efforts, list) else None,
        reasoning=True if efforts or supports.get("adaptive_thinking") else None,
        images=supports.get("vision") if isinstance(supports.get("vision"), bool) else None,
        api=api,
    )


def _positive_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) and value > 0 else None


class ModelDirectory:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._memory: dict[str, tuple[float, dict[str, ModelInfo]]] = {}

    def _path(self, provider: str) -> Path:
        return self.cache_dir / f"{provider}.json"

    def _load(self, provider: str) -> tuple[float, dict[str, ModelInfo]]:
        if provider not in self._memory:
            fetched_at, models = 0.0, {}
            try:
                raw = json.loads(self._path(provider).read_text())
                fetched_at = float(raw["fetched_at"]) if raw.get("version") == CACHE_VERSION else 0.0
                models = {m["id"]: _from_json(m) for m in raw["models"]}
            except (OSError, ValueError, KeyError, TypeError):
                pass
            self._memory[provider] = (fetched_at, models)
        return self._memory[provider]

    def models(self, provider: str) -> list[ModelInfo]:
        return list(self._load(provider)[1].values())

    def info(self, provider: str, model: str) -> ModelInfo | None:
        return self._load(provider)[1].get(model)

    def is_fresh(self, provider: str) -> bool:
        return time.time() - self._load(provider)[0] < MAX_AGE_SECONDS

    def store(self, provider: str, models: list[ModelInfo]) -> None:
        now = time.time()
        self._memory[provider] = (now, {m.id: m for m in models})
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            body = {"version": CACHE_VERSION, "fetched_at": now, "models": [_to_json(m) for m in models]}
            self._path(provider).write_text(json.dumps(body))
        except OSError as e:
            log.debug("could not cache the %s model list: %s", provider, e)


def _to_json(m: ModelInfo) -> dict[str, Any]:
    d = asdict(m)
    d["efforts"] = sorted(m.efforts) if m.efforts is not None else None
    return d


def _from_json(d: dict[str, Any]) -> ModelInfo:
    efforts, price = d.get("efforts"), d.get("price")
    return ModelInfo(
        **{
            **d,
            "efforts": frozenset(efforts) if efforts is not None else None,
            "price": Price(**price) if isinstance(price, dict) else None,
        }
    )
