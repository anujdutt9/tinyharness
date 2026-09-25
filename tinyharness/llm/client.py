"""Minimal streaming client for OpenAI-compatible Chat Completions endpoints."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from tinyharness.agent.history import ToolCall

log = logging.getLogger(__name__)

_ANTHROPIC_TAGS = ("claude", "opus", "sonnet", "haiku")
_OPENAI_REASONER_TAGS = ("o1", "o3", "o4-mini", "o4mini", "gpt-5", "gpt-6")
_DEEPSEEK_REASONER_TAGS = ("deepseek-reasoner", "deepseek-r1")


def _apply_prompt_caching(payload: dict[str, Any]) -> None:
    """Mark the system message with `cache_control` so Anthropic-family backends
    cache the (large, stable) system + tools prefix. Others ignore the field.
    Mutates the payload in place; idempotent."""
    for m in payload.get("messages") or []:
        if m.get("role") != "system":
            continue
        content = m.get("content")
        if isinstance(content, list):
            if content and not any(isinstance(b, dict) and b.get("cache_control") for b in content):
                content[-1]["cache_control"] = {"type": "ephemeral"}
        elif isinstance(content, str) and content.strip():
            m["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        return  # only the first system message


def _apply_thinking(payload: dict[str, Any], model: str, budget: int) -> None:
    """Translate a token budget into each model family's reasoning parameter."""
    if budget <= 0:
        return
    m = (model or "").lower()
    if any(tag in m for tag in _ANTHROPIC_TAGS):
        payload["thinking"] = {"type": "enabled", "budget_tokens": int(budget)}
    elif any(tag in m for tag in _OPENAI_REASONER_TAGS + _DEEPSEEK_REASONER_TAGS):
        payload["reasoning_effort"] = "low" if budget <= 1024 else "medium" if budget <= 8192 else "high"


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    out = dict(headers)
    for k in out:
        if k.lower() in {"authorization", "x-api-key", "api-key"}:
            out[k] = out[k][:14] + "...REDACTED"
    return out


@dataclass
class StreamChunk:
    text: str = ""
    tool_call_delta: dict[str, Any] | None = None  # {index, id?, function: {name?, arguments?}}
    finish_reason: str | None = None


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


class LLMError(RuntimeError):
    pass


class LLMClient:
    """Streams chat completions with tool calling from any OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 1800.0,
        prompt_caching: bool = True,
        thinking_budget: int = 0,
    ) -> None:
        url = base_url.rstrip("/")
        self.endpoint = url if url.endswith("/chat/completions") else f"{url}/chat/completions"
        self.api_key = api_key
        self.model = model
        self.extra_headers = extra_headers or {}
        self.timeout = timeout
        self.prompt_caching = prompt_caching
        self.thinking_budget = max(0, thinking_budget)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }

    def _payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        model: str | None,
        temperature: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self.prompt_caching:
            _apply_prompt_caching(payload)
        _apply_thinking(payload, payload["model"], self.thinking_budget)
        return payload

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamChunk]:
        payload = self._payload(messages, tools, model, temperature)
        log.debug(
            "POST %s headers=%s messages=%d tools=%d",
            self.endpoint,
            _redact_headers(self._headers()),
            len(messages),
            len(tools or []),
        )

        async with (
            httpx.AsyncClient(timeout=self.timeout) as client,
            client.stream("POST", self.endpoint, json=payload, headers=self._headers()) as resp,
        ):
            if resp.status_code >= 400:
                body = (await resp.aread()).decode(errors="replace")[:500]
                if resp.status_code in (401, 403):
                    raise LLMError(
                        f"auth error {resp.status_code} from {self.endpoint}; "
                        f"check that your API key is set and valid. Response: {body}"
                    )
                raise LLMError(f"LLM error {resp.status_code}: {body}")

            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                finish = choices[0].get("finish_reason")
                if delta.get("content"):
                    yield StreamChunk(text=delta["content"], finish_reason=finish)
                for tcd in delta.get("tool_calls") or []:
                    yield StreamChunk(tool_call_delta=tcd, finish_reason=finish)
                if finish and not delta:
                    yield StreamChunk(finish_reason=finish)

    async def collect(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.7,
    ) -> CompletedTurn:
        """Non-streaming convenience: run `stream` to completion."""
        text: list[str] = []
        calls = ToolCallAccumulator()
        finish: str | None = None
        async for chunk in self.stream(messages, tools, model=model, temperature=temperature):
            text.append(chunk.text)
            if chunk.tool_call_delta:
                calls.add(chunk.tool_call_delta)
            finish = chunk.finish_reason or finish
        return CompletedTurn(text="".join(text), tool_calls=calls.calls(), finish_reason=finish)

    async def one_shot(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})
        turn = await self.collect(messages, model=model, temperature=temperature)
        return turn.text

    async def ping(self) -> tuple[bool, str]:
        """Cheapest possible authenticated request. Returns (ok, detail)."""
        payload = {"model": self.model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(self.endpoint, json=payload, headers=self._headers())
        except httpx.HTTPError as e:
            return False, f"{type(e).__name__}: {e}"
        if resp.status_code == 200:
            return True, "200 — auth ok"
        return False, f"{resp.status_code} — {resp.text[:300]}"
