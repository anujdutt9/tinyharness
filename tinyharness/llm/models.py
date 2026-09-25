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

log = logging.getLogger(__name__)

MAX_AGE_SECONDS = 24 * 3600


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
                fetched_at = float(raw["fetched_at"])
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
            body = {"fetched_at": now, "models": [_to_json(m) for m in models]}
            self._path(provider).write_text(json.dumps(body))
        except OSError as e:
            log.debug("could not cache the %s model list: %s", provider, e)


def _to_json(m: ModelInfo) -> dict[str, Any]:
    d = asdict(m)
    d["efforts"] = sorted(m.efforts) if m.efforts is not None else None
    return d


def _from_json(d: dict[str, Any]) -> ModelInfo:
    efforts = d.get("efforts")
    return ModelInfo(**{**d, "efforts": frozenset(efforts) if efforts is not None else None})
