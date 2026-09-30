"""The interface every provider API adapter implements.

An adapter knows one wire format: how to turn a `Request` into an HTTP body, how
to turn the streamed events back into `StreamChunk`s, and how to read the
provider's model list. It never looks at model names; per-model behavior comes
only from what the provider's own model list reports (`ModelInfo`).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from tinyharness.llm.types import ModelInfo, Request, StreamChunk
from tinyharness.llm.usage import price_from_listing

if TYPE_CHECKING:
    from tinyharness.llm.providers import Provider


class StreamParser(ABC):
    """Turns one response's stream events into chunks. One instance per request."""

    @abstractmethod
    def feed(self, event: dict[str, Any]) -> list[StreamChunk]: ...

    def state(self) -> Any:
        """Data the provider needs back on the next request (e.g. signed reasoning), or None."""
        return None


class Adapter(ABC):
    name: ClassVar[str]
    request_path: ClassVar[str]  # appended to the provider's base_url
    models_path: ClassVar[str] = "/models"

    def __init__(self, provider: Provider) -> None:
        self.provider = provider

    def headers(self, api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"}

    @abstractmethod
    def payload(self, request: Request) -> dict[str, Any]: ...

    @abstractmethod
    def parser(self) -> StreamParser: ...

    def add_web_search(self, payload: dict[str, Any]) -> None:
        """Switch on the provider's own web search tool in a request body."""
        raise NotImplementedError

    @abstractmethod
    def ping_payload(self, model: str) -> dict[str, Any]:
        """The smallest valid non-streaming request, to check the key works."""

    def parse_models(self, body: dict[str, Any]) -> tuple[list[ModelInfo], dict[str, str] | None]:
        """One page of the model list, plus the query params for the next page (None: last page).

        The default reads the OpenAI-style `{"data": [{"id": ...}]}`, picking up a
        context size under any of the names providers use for it.
        """
        models = []
        for item in body.get("data") or []:
            top = item.get("top_provider") or {}
            models.append(
                ModelInfo(
                    id=str(item["id"]),
                    label=str(item.get("name") or item.get("display_name") or ""),
                    context=_first_int(
                        item, top, keys=("context_length", "context_window", "max_context_length")
                    ),
                    max_output=_first_int(item, top, keys=("max_output_tokens", "max_completion_tokens")),
                    price=price_from_listing(item["pricing"])
                    if isinstance(item.get("pricing"), dict)
                    else None,
                )
            )
        return models, None


def _first_int(*sources: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for source in sources:
        for key in keys:
            if isinstance(source.get(key), int) and source[key] > 0:
                return int(source[key])
    return None


def parse_arguments(arguments: str) -> dict[str, Any]:
    """Tool-call arguments as an object, for APIs that want JSON rather than a string."""
    try:
        value = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def tool_names(request: Request) -> dict[str, str]:
    """call id -> tool name, for APIs whose tool results must repeat the name."""
    return {call.id: call.name for m in request.messages for call in m.tool_calls}
