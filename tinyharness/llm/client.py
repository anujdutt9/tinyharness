"""The one place that talks HTTP to a model provider.

`LLMClient` pairs a provider (from providers.toml or the user's config) with the
adapter for its API format, and streams provider-neutral `StreamChunk`s. The
agent loop, compaction and classifier only ever see this interface.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from tinyharness.agent.history import Message, ReasoningState, ToolCall
from tinyharness.llm.adapters import ADAPTERS
from tinyharness.llm.models import ModelDirectory
from tinyharness.llm.types import (
    ContextOverflowError,
    Effort,
    LLMError,
    ModelInfo,
    Request,
    StreamChunk,
    ToolSpec,
)

if TYPE_CHECKING:
    from tinyharness.llm.providers import Provider

__all__ = [
    "CompletedTurn",
    "ContextOverflowError",
    "LLMClient",
    "LLMError",
    "StreamChunk",
    "ToolCallAccumulator",
]

log = logging.getLogger(__name__)

# How providers word "the input is too long for this model".
_OVERFLOW = re.compile(
    r"context_length_exceeded|prompt is too long|context (length|window)|maximum context"
    r"|too many (input )?tokens|input token count|exceeds? the (maximum|model'?s?)",
    re.IGNORECASE,
)


class ToolCallAccumulator:
    """Reassembles tool calls that arrive split across many stream deltas."""

    def __init__(self) -> None:
        self._by_index: dict[int, ToolCall] = {}

    def add(self, delta: dict[str, Any]) -> None:
        tc = self._by_index.setdefault(delta.get("index", 0), ToolCall(id="", name=""))
        if delta.get("id"):
            tc.id = delta["id"]
        fn = delta.get("function") or {}
        if fn.get("name"):
            tc.name += fn["name"]
        if fn.get("arguments"):
            tc.arguments += fn["arguments"]

    def calls(self) -> list[ToolCall]:
        return [self._by_index[i] for i in sorted(self._by_index)]


@dataclass
class CompletedTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None


class LLMClient:
    def __init__(
        self,
        provider: Provider,
        api_key: str,
        model: str,
        *,
        models: ModelDirectory | None = None,
        effort: Effort | None = None,
        timeout: float = 1800.0,
        transport: httpx.AsyncBaseTransport | None = None,  # tests replay recorded responses
    ) -> None:
        self.provider = provider
        self.adapter = ADAPTERS[provider.api](provider)
        self.api_key = api_key
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self.directory = models
        self.transport = transport
        self._tried_listing = False

    @property
    def endpoint(self) -> str:
        return self.provider.base_url.rstrip("/") + self.adapter.request_path

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            **self.adapter.headers(self.api_key),
            **self.provider.extra_headers,
        }

    # -- models -------------------------------------------------------------------

    def model_info(self, model: str | None = None) -> ModelInfo | None:
        """What we know about a model: from the config, else from the provider's cached list."""
        model = model or self.model
        configured = next((m for m in self.provider.models if m.id == model), None)
        listed = self.directory.info(self.provider.name, model) if self.directory else None
        return configured or listed

    async def list_models(self, *, refresh: bool = False) -> list[ModelInfo]:
        """The provider's model list (cached for a day), plus any models set in the config."""
        if self.directory is None:
            return list(self.provider.models)
        if refresh or not self.directory.is_fresh(self.provider.name):
            self.directory.store(self.provider.name, await self._fetch_models())
        configured = {m.id for m in self.provider.models}
        listed = [m for m in self.directory.models(self.provider.name) if m.id not in configured]
        return [*self.provider.models, *listed]

    async def _fetch_models(self) -> list[ModelInfo]:
        url = self.provider.base_url.rstrip("/") + self.adapter.models_path
        params: dict[str, str] | None = {}
        found: list[ModelInfo] = []
        async with httpx.AsyncClient(timeout=30, transport=self.transport) as client:
            while params is not None:
                resp = await client.get(url, headers=self._headers(), params=params)
                resp.raise_for_status()
                page, params = self.adapter.parse_models(resp.json())
                found.extend(page)
        return found

    async def _ensure_model_list(self) -> None:
        """Load the model list once per client, so requests can use what it says about the model."""
        if self._tried_listing or self.directory is None:
            return
        self._tried_listing = True
        if not self.directory.is_fresh(self.provider.name):
            try:
                await self.list_models(refresh=True)
            except Exception as e:  # listing is an optimization; requests work without it
                log.debug("could not list %s models: %s", self.provider.name, e)

    # -- requests -----------------------------------------------------------------

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream one reply. The last chunk carries `state` when the provider needs data back next turn."""
        await self._ensure_model_list()
        model = model or self.model
        request = Request(
            model=model,
            system="\n\n".join(m.content or "" for m in messages if m.role == "system"),
            messages=[m for m in messages if m.role != "system"],
            tools=list(tools or []),
            effort=self.effort,
            info=self.model_info(model),
            owner=f"{self.provider.name}:{model}",
        )
        payload = _merge(self.adapter.payload(request), self.provider.extra_body)
        parser = self.adapter.parser()
        log.debug(
            "POST %s model=%s messages=%d tools=%d", self.endpoint, model, len(messages), len(request.tools)
        )

        async with (
            httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client,
            client.stream("POST", self.endpoint, json=payload, headers=self._headers()) as resp,
        ):
            if resp.status_code >= 400:
                body = (await resp.aread()).decode(errors="replace")
                raise _http_error(resp.status_code, body, self.endpoint)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue  # blank lines, `event:` names, `:` keep-alive comments
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                try:
                    chunks = parser.feed(event)
                except LLMError as e:
                    raise (ContextOverflowError(str(e)) if _OVERFLOW.search(str(e)) else e) from None
                for chunk in chunks:
                    yield chunk

        if (state := parser.state()) is not None:
            yield StreamChunk(state=ReasoningState(request.owner, state))

    async def collect(
        self, messages: list[Message], tools: list[ToolSpec] | None = None, *, model: str | None = None
    ) -> CompletedTurn:
        """Non-streaming convenience: run `stream` to completion."""
        text: list[str] = []
        calls = ToolCallAccumulator()
        finish: str | None = None
        async for chunk in self.stream(messages, tools, model=model):
            text.append(chunk.text)
            if chunk.tool_call_delta:
                calls.add(chunk.tool_call_delta)
            finish = chunk.finish_reason or finish
        return CompletedTurn(text="".join(text), tool_calls=calls.calls(), finish_reason=finish)

    async def one_shot(self, prompt: str, *, system: str | None = None, model: str | None = None) -> str:
        messages = [Message(role="system", content=system)] if system else []
        messages.append(Message(role="user", content=prompt))
        return (await self.collect(messages, model=model)).text

    async def ping(self, model: str | None = None) -> Ping:
        """The cheapest possible request, to check the key and that the model can be used."""
        payload = self.adapter.ping_payload(model or self.model)
        try:
            async with httpx.AsyncClient(timeout=30.0, transport=self.transport) as client:
                resp = await client.post(self.endpoint, json=payload, headers=self._headers())
        except httpx.HTTPError as e:
            return Ping(ok=False, status=None, message=f"{type(e).__name__}: {e}")
        if resp.status_code == 200:
            return Ping(ok=True, status=200, message="auth ok")
        return Ping(ok=False, status=resp.status_code, message=_error_message(resp.text))


@dataclass(frozen=True)
class Ping:
    ok: bool
    status: int | None  # None: no answer (network error or timeout)
    message: str

    @property
    def model_unusable(self) -> bool:
        """The provider clearly said this model can't be used here (as opposed to a
        network hiccup, a server error, or the test request itself being off)."""
        if self.status == 404:
            return True
        return self.status in (400, 403, 422) and "model" in self.message.lower()


def _error_message(body: str) -> str:
    """The human-readable part of an error body, whatever the provider's JSON shape."""
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body.strip()[:300]
    error = data.get("error", data) if isinstance(data, dict) else data
    if isinstance(error, dict):
        return str(error.get("message") or json.dumps(error))[:300]
    return str(error)[:300]


def _http_error(status: int, body: str, endpoint: str) -> LLMError:
    if status in (401, 403):
        return LLMError(
            f"auth error {status} from {endpoint}; check that your API key is set and valid. Response: {body[:500]}"
        )
    if status in (400, 413, 422) and _OVERFLOW.search(body):
        return ContextOverflowError(f"the conversation is too long for this model: {body[:300]}")
    return LLMError(f"LLM error {status}: {body[:500]}")


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge a provider's `extra_body` into a request."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base
