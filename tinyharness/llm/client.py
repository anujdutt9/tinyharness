"""The one place that talks HTTP to a model provider.

`LLMClient` pairs a provider (from providers.toml or the user's config) with the
adapter for its API format, and streams provider-neutral `StreamChunk`s. The
agent loop, compaction and classifier only ever see this interface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx

from tinyharness.agent.history import Message, ReasoningState, ToolCall, well_formed, without_images
from tinyharness.llm import identity
from tinyharness.llm.adapters import ADAPTERS
from tinyharness.llm.calibration import CalibrationStore, ToolFormat, as_text, probe, tools_section
from tinyharness.llm.models import ModelDirectory
from tinyharness.llm.types import (
    DEFAULT_OPT_IN_EFFORT,
    ContextOverflowError,
    Effort,
    ImagesUnsupportedError,
    ImageTooLargeError,
    LLMError,
    ModelInfo,
    ProviderConnectionError,
    Request,
    StreamChunk,
    ToolSpec,
    normalize_finish,
)
from tinyharness.llm.usage import Usage, read_usage
from tinyharness.net import describe as describe_network_error

if TYPE_CHECKING:
    from tinyharness.llm.providers import Provider

__all__ = [
    "CompletedTurn",
    "ContextOverflowError",
    "ImageTooLargeError",
    "ImagesUnsupportedError",
    "LLMClient",
    "LLMError",
    "ProviderConnectionError",
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


# How providers word "this model takes no images" (vLLM serving a model text-only: "At most 0
# image(s) may be provided in one prompt").
_NO_IMAGES = re.compile(
    r"at most 0 images?\b|at most 0 image\(s\)|does(n't| not) support (image|vision)"
    r"|(image|vision) (input|inputs|content) (is |are )?not supported|not a multimodal model"
    r"|images? (is|are) not supported|model does not accept images?",
    re.IGNORECASE,
)


# How providers word "an image in this request is too large".
_IMAGE_TOO_LARGE = re.compile(
    r"image dimensions exceed|image (exceeds|is too large|too large)|exceeds? .{0,40}image"
    r"|image_too_large|image size",
    re.IGNORECASE,
)


# Answers worth asking again, before any of the reply has streamed: rate limits (429, which
# a gateway's admission control sends on purpose), overload (503, 529) and gateway errors.
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
# Failures before any reply arrived: the request can simply be sent again. (Not read
# timeouts: the server took the request and went quiet, and waiting again rarely helps.)
_RETRYABLE_TRANSPORT = (httpx.NetworkError, httpx.ConnectTimeout, httpx.RemoteProtocolError)
# How errors reported inside a stream word "busy, try again" (Anthropic's overloaded_error,
# OpenRouter's upstream errors), as opposed to a problem with the request itself.
# Only consulted for errors inside a stream; HTTP statuses are judged by RETRY_STATUSES.
# A status code counts only next to a word that says it is one: overflow messages quote
# token counts like "500 in the completion".
_TRANSIENT = re.compile(
    r"overloaded|rate.?limit|too many requests|temporar|service unavailable|timed? ?out"
    r"|internal (server )?error|try again|\b(error|status|code)\W{0,3}(429|5\d\d)\b",
    re.IGNORECASE,
)
MAX_RETRIES = 3
STALL_SECONDS = 300.0  # no bytes from the server for this long ends the request
RETRY_BASE_SECONDS = 1.0
RETRY_MAX_SECONDS = 60.0


def _retry_after(headers: httpx.Headers) -> str | None:
    """The server's requested wait in seconds, from `retry-after-ms` or `retry-after`."""
    if ms := headers.get("retry-after-ms"):
        try:
            return str(float(ms) / 1000)
        except ValueError:
            pass
    return headers.get("retry-after")


async def _sse_data(lines: AsyncIterator[str], other: list[str]) -> AsyncGenerator[str, None]:
    """The data of each server-sent event. An event's `data:` lines are joined (some servers
    pretty-print); servers that skip the blank line between events still work, because
    lines that don't join into JSON are taken one by one. Lines that aren't part of an
    event stream at all (an error page) are collected in `other`."""
    data: list[str] = []
    async for line in lines:
        if line.startswith("data:"):
            item = line[5:].strip()
            if not data and _whole(item):
                yield item  # complete on its own: don't hold it back until a blank line
            else:
                data.append(item)
        elif not line.strip():
            for item in _split_event(data):
                yield item
            data = []
        elif not line.startswith(("event:", "id:", "retry:", ":")) and len(other) < 20:
            other.append(line.strip())
    for item in _split_event(data):
        yield item


def _whole(item: str) -> bool:
    """A data line that is a whole event by itself: an object or list, or [DONE]."""
    if item == "[DONE]":
        return True
    if not item.startswith(("{", "[")):
        return False
    try:
        json.loads(item)
    except json.JSONDecodeError:
        return False
    return True


def _split_event(data: list[str]) -> list[str]:
    if len(data) > 1:
        joined = "\n".join(data)
        try:
            json.loads(joined)
            return [joined]
        except json.JSONDecodeError:
            pass
    return data


def retry_delay(retry_after: str | None, attempt: int) -> float:
    """Seconds to wait before attempt `attempt + 1`: the server's Retry-After when it gives
    one (in seconds), otherwise exponential backoff with jitter. Capped either way."""
    if retry_after:
        try:
            return min(RETRY_MAX_SECONDS, max(0.0, float(retry_after)))
        except ValueError:
            pass  # an HTTP date: fall back to backoff
    backoff = RETRY_BASE_SECONDS * 2**attempt
    return min(RETRY_MAX_SECONDS, backoff * random.uniform(0.5, 1.5))


class ToolCallAccumulator:
    """Reassembles tool calls that arrive split across many stream deltas."""

    def __init__(self) -> None:
        self._by_index: dict[int, ToolCall] = {}

    def add(self, delta: dict[str, Any]) -> None:
        index = delta.get("index")
        latest = max(self._by_index, default=0)
        if index is None:  # some OpenAI-compatible servers leave it out: a new id is a new call
            current = self._by_index.get(latest)
            new_call = current is not None and bool(delta.get("id")) and current.id not in ("", delta["id"])
            index = latest + 1 if new_call else latest
        tc = self._by_index.setdefault(index, ToolCall(id="", name=""))
        if delta.get("id"):
            tc.id = delta["id"]
        fn = delta.get("function") or {}
        if (name := fn.get("name")) and name != tc.name:  # some servers repeat it in every delta
            tc.name += name
        if fn.get("arguments"):
            tc.arguments += fn["arguments"]

    def calls(self) -> list[ToolCall]:
        calls = [self._by_index[i] for i in sorted(self._by_index)]
        for i, call in enumerate(calls):
            if not call.id:  # some OpenAI-compatible servers leave ids out; results need one
                call.id = f"call_{uuid.uuid4().hex[:12]}_{i}"
        return calls

    def get(self, index: int) -> ToolCall | None:
        """The call at `index` as assembled so far."""
        return self._by_index.get(index)


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
        retries: int = MAX_RETRIES,
        transport: httpx.AsyncBaseTransport | None = None,  # tests replay recorded responses
        tool_calls: str = "auto",  # "native", "text", or "auto": probe once per model (calibration.py)
        calibration: CalibrationStore | None = None,  # where probed formats are kept; None: native
    ) -> None:
        self.provider = provider
        self.adapter = ADAPTERS[provider.api](provider)
        self.api_key = api_key
        self.model = model
        self.effort = effort
        self.show_reasoning = False  # request readable reasoning where a provider makes it opt-in
        self.timeout = timeout
        self.retries = retries
        self.directory = models
        self.transport = transport
        self.tool_calls = tool_calls
        self.calibration = calibration
        self._formats: dict[str, ToolFormat] = {}  # probed this session, by model
        self.no_images: set[str] = set()  # models that refused an image this session
        self._tried_listing = False

    def _timeouts(self) -> httpx.Timeout:
        """`timeout` for connecting and sending; between pieces of a reply, no more than
        STALL_SECONDS: a connection that has gone silent that long is dead (servers that
        need longer to start a reply send keep-alive comments), and one stuck agent would
        otherwise hold up its whole team for the full timeout."""
        return httpx.Timeout(self.timeout, read=min(self.timeout, STALL_SECONDS))

    @property
    def endpoint(self) -> str:
        return self.provider.base_url.rstrip("/") + self.adapter.request_path

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            **self.adapter.headers(self.api_key),
            **self.provider.extra_headers,
        }
        if self.provider.send_ids and (who := identity.CURRENT.get()) is not None:
            headers.update(who.headers())
        return headers

    # -- models -------------------------------------------------------------------

    def takes_images(self, model: str | None = None) -> bool | None:
        """Whether `model` takes image input: False once it refused one this session, else
        what the config or the provider's model list says (None: not known)."""
        model = model or self.model
        if model in self.no_images:
            return False
        info = self.model_info(model)
        return info.images if info is not None else None

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
        web_search: bool = False,
        probing: bool = False,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream one reply. The last chunk carries `state` when the provider needs data back next turn."""
        await self._ensure_model_list()
        model = model or self.model
        owner = f"{self.provider.name}:{model}"
        if self.takes_images(model) is False:
            messages = without_images(messages)
        messages, _ = well_formed(messages, valid_id=re.compile(self.provider.tool_id_pattern), owner=owner)
        system = "\n\n".join(m.content or "" for m in messages if m.role == "system")
        conversation = [m for m in messages if m.role != "system"]
        tools = list(tools or [])
        if tools and not probing and await self.tool_format(model) == "text":
            # This model calls tools better in text: describe them, and send the history as text.
            system = "\n\n".join(filter(None, [system, tools_section(tools)]))
            conversation, tools = as_text(conversation), []
        request = Request(
            model=model,
            system=system,
            messages=conversation,
            tools=tools,
            effort=self._effort(model, agent_turn=bool(tools)),
            info=self.model_info(model),
            owner=owner,
            web_search=web_search,
            show_reasoning=self.show_reasoning,
            max_tokens=max_tokens,
        )
        payload = _merge(self.adapter.payload(request), self.provider.extra_body)
        log.debug(
            "POST %s model=%s messages=%d tools=%d", self.endpoint, model, len(messages), len(request.tools)
        )

        usage: Usage | None = None
        finished = False  # a stream that just stops (a proxy dropping it) isn't a reply
        unreadable = 0  # events that weren't valid JSON: skipped, but said so (text may be missing)
        started = False  # once a chunk has gone out, a failure can't be retried invisibly
        route = ""  # which upstream a router sent this to (OpenRouter names it in every chunk)
        other: list[str] = []  # lines that aren't server-sent events: an error page, say
        busy: LLMError | None = None  # a transient error reported inside the stream, to retry
        for attempt in range(self.retries + 1):
            parser = self.adapter.parser()
            try:
                async with (
                    httpx.AsyncClient(timeout=self._timeouts(), transport=self.transport) as client,
                    client.stream("POST", self.endpoint, json=payload, headers=self._headers()) as resp,
                ):
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode(errors="replace")
                        if resp.status_code in RETRY_STATUSES and attempt < self.retries:
                            delay = retry_delay(_retry_after(resp.headers), attempt)
                            log.warning(
                                "%s answered %d; retrying in %.1fs (%d of %d)",
                                self.endpoint, resp.status_code, delay, attempt + 1, self.retries,
                            )  # fmt: skip
                            await asyncio.sleep(delay)
                            continue
                        raise _http_error(resp.status_code, body, self.endpoint)
                    other.clear()
                    async for data in _sse_data(resp.aiter_lines(), other):
                        if data == "[DONE]":
                            finished = True
                            break
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            unreadable += 1
                            log.debug("unreadable stream event from %s: %.120s", self.endpoint, data)
                            continue
                        if isinstance(event.get("provider"), str):
                            route = event["provider"]
                        usage = read_usage(event, usage)
                        try:
                            chunks = parser.feed(event)
                        except LLMError as e:  # an error reported inside the stream
                            if started or attempt >= self.retries or not _TRANSIENT.search(str(e)):
                                raise _classify(e) from None
                            busy = e
                            break
                        for chunk in chunks:
                            if chunk.finish_reason is not None:
                                chunk.finish_detail = chunk.finish_reason
                                chunk.finish_reason = normalize_finish(chunk.finish_reason)
                                finished = True
                            started = True
                            yield chunk
                if busy is not None:  # "overloaded" before any output: ask again
                    delay = retry_delay(None, attempt)
                    log.warning("%s: %s; retrying in %.1fs", self.endpoint, busy, delay)
                    busy = None
                    await asyncio.sleep(delay)
                    continue
                break
            except httpx.TransportError as e:  # refused, reset, DNS, TLS, timeouts, dropped mid-reply
                if not started and attempt < self.retries and isinstance(e, _RETRYABLE_TRANSPORT):
                    delay = retry_delay(None, attempt)
                    log.warning("%s: %s; retrying in %.1fs", self.endpoint, type(e).__name__, delay)
                    await asyncio.sleep(delay)
                    continue
                failure = describe_network_error(e)
                raise ProviderConnectionError(
                    failure.message, host=failure.host, code=failure.code, slow_reply=failure.slow_reply
                ) from e

        if not finished and not started and other:  # a JSON or HTML body instead of a stream
            raise _classify(
                LLMError(f"{self.endpoint} didn't answer with an event stream: {' '.join(other)[:500]}")
            )
        if unreadable:
            log.warning(
                "%d stream event(s) from %s weren't valid JSON and were skipped; the reply may be "
                "missing some text (run with --debug to see them)",
                unreadable,
                self.endpoint,
            )
        if not finished:
            raise LLMError(
                f"incomplete response from {self.endpoint}: the stream ended without a finish event, "
                "so the reply was not added to the conversation. Try again; if it keeps happening, "
                "a proxy or network timeout may be cutting the connection."
            )
        if (state := parser.state()) is not None:
            yield StreamChunk(state=ReasoningState(request.owner, state))
        if usage is not None or route:
            yield StreamChunk(usage=usage, route=route)

    def _effort(self, model: str, *, agent_turn: bool) -> Effort | None:
        """The effort to request: the user's choice ("none" sends nothing); otherwise, on an
        agent's turn, a default for models whose reasoning is off unless asked for. Not for
        the harness's own quick calls (the permission classifier, summaries), which have
        no tools and would only get slower."""
        if self.effort is not None:
            return None if self.effort == "none" else self.effort
        info = self.model_info(model)
        return DEFAULT_OPT_IN_EFFORT if agent_turn and info is not None and info.reasoning else None

    async def collect(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        model: str | None = None,
        web_search: bool = False,
        max_tokens: int | None = None,
    ) -> CompletedTurn:
        """Non-streaming convenience: run `stream` to completion."""
        text: list[str] = []
        calls = ToolCallAccumulator()
        finish: str | None = None
        async for chunk in self.stream(
            messages, tools, model=model, web_search=web_search, max_tokens=max_tokens
        ):
            text.append(chunk.text)
            if chunk.tool_call_delta:
                calls.add(chunk.tool_call_delta)
            finish = chunk.finish_reason or finish
        return CompletedTurn(text="".join(text), tool_calls=calls.calls(), finish_reason=finish)

    async def tool_format(self, model: str | None = None) -> ToolFormat:
        """How `model` calls tools: from the config, the cache, or one probe request."""
        model = model or self.model
        if self.tool_calls in ("native", "text"):
            return self.tool_calls  # type: ignore[return-value]
        if self.calibration is None:
            return "native"
        if model in self._formats:
            return self._formats[model]
        owner = f"{self.provider.name}:{model}"
        fmt = self.calibration.get(owner)
        if fmt is None:
            try:
                fmt = await probe(
                    lambda messages, tools: self.stream(messages, tools, model=model, probing=True)
                )
            except Exception as e:  # can't tell: use the interface, and try again next session
                log.info("tool-call probe for %s failed: %s", owner, e)
                self._formats[model] = "native"
                return "native"
            self.calibration.put(owner, fmt)
            log.info("%s calls tools %s", owner, "through the API" if fmt == "native" else "in text")
        self._formats[model] = fmt
        return fmt

    async def one_shot(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        web_search: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        messages = [Message(role="system", content=system)] if system else []
        messages.append(Message(role="user", content=prompt))
        return (await self.collect(messages, model=model, web_search=web_search, max_tokens=max_tokens)).text

    @property
    def small_model(self) -> str:
        """This provider's cheap model, for background work (e.g. WebSearch digests)."""
        return self.provider.small_model or self.model

    async def one_shot_small(
        self,
        prompt: str,
        *,
        system: str | None = None,
        web_search: bool = False,
        model: str | None = None,
    ) -> str:
        """`one_shot` on a background model. If it refuses, the main model answers instead."""
        small = model or self.small_model
        if small != self.model:
            try:
                return await self.one_shot(prompt, system=system, model=small, web_search=web_search)
            except ProviderConnectionError:
                raise  # the network is down for both
            except LLMError:
                pass
        return await self.one_shot(prompt, system=system, web_search=web_search)

    @property
    def supports_web_search(self) -> bool:
        """Whether this provider has its own web search tool the model can use."""
        return bool(self.provider.web_search)

    async def ping(self, model: str | None = None) -> Ping:
        """The cheapest possible request, to check the key and that the model can be used."""
        payload = self.adapter.ping_payload(model or self.model)
        try:
            async with httpx.AsyncClient(timeout=30.0, transport=self.transport) as client:
                resp = await client.post(self.endpoint, json=payload, headers=self._headers())
        except httpx.TransportError as e:
            return Ping(ok=False, status=None, message=describe_network_error(e).message)
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

    @property
    def bad_key(self) -> bool:
        """The provider rejected the API key (Gemini says so with a 400)."""
        if self.status == 401:
            return True
        return self.status in (400, 403) and bool(_BAD_KEY.search(self.message))


_BAD_KEY = re.compile(r"api[ _-]?key|unauthori[sz]ed|authenticat|credential", re.IGNORECASE)


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
    if status in (400, 422) and _NO_IMAGES.search(body):
        return ImagesUnsupportedError(f"this model takes no images: {body[:300]}")
    if status in (400, 413, 422) and _IMAGE_TOO_LARGE.search(body):
        return ImageTooLargeError(f"an image is too large for this model: {body[:300]}")
    if status in (400, 413, 422) and _OVERFLOW.search(body):
        return ContextOverflowError(f"the conversation is too long for this model: {body[:300]}")
    return LLMError(f"LLM error {status}: {body[:500]}")


def _classify(e: LLMError) -> LLMError:
    """An error reported inside the stream, as the specific kind the loop can recover from."""
    text = str(e)
    if _NO_IMAGES.search(text):
        return ImagesUnsupportedError(text)
    if _IMAGE_TOO_LARGE.search(text):
        return ImageTooLargeError(text)
    return ContextOverflowError(text) if _OVERFLOW.search(text) else e


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge a provider's `extra_body` into a request."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base
