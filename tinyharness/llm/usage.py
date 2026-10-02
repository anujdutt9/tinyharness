"""Token usage and cost, read from whatever each provider reports.

Every API puts usage somewhere in its stream: Anthropic in `message_start` and
`message_delta`, OpenAI Responses in `response.completed`, Chat Completions in the
last chunk (when asked), Gemini in `interaction.completed`. `read_usage` finds it in
any of them and normalizes it:

    input    input tokens billed at the full rate (cache writes included)
    cached   input tokens read from the prompt cache (billed much lower)
    output   output tokens, reasoning included

Cost comes from the provider when it reports one (OpenRouter does), else from a
price per million tokens: the provider's model list (OpenRouter's has one) or
`[pricing]` in your config. Nothing is hard-coded, so a cost is shown only when
it's actually known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Usage:
    input: int = 0
    cached: int = 0
    output: int = 0
    cost: float | None = None  # dollars, when known

    @property
    def tokens(self) -> int:
        return self.input + self.cached + self.output

    def __add__(self, other: Usage) -> Usage:
        cost = None if self.cost is None and other.cost is None else (self.cost or 0) + (other.cost or 0)
        return Usage(self.input + other.input, self.cached + other.cached, self.output + other.output, cost)


@dataclass(frozen=True)
class Price:
    """Dollars per million tokens."""

    input: float
    output: float
    cached: float | None = None  # None: same as input

    def cost(self, usage: Usage) -> float:
        cached = self.input if self.cached is None else self.cached
        return (usage.input * self.input + usage.cached * cached + usage.output * self.output) / 1e6


def read_usage(event: dict[str, Any], so_far: Usage | None) -> Usage | None:
    """Merge the usage in one stream event (if any) into what was seen so far."""
    raw = _find(event)
    if raw is None:
        return so_far
    usage = so_far or Usage()
    cache_read = _int(raw, "cache_read_input_tokens", "cached_tokens", "total_cached_tokens") or _int(
        raw.get("input_tokens_details") or raw.get("prompt_tokens_details") or {}, "cached_tokens"
    )
    cache_write = _int(raw, "cache_creation_input_tokens")
    total_in = _int(raw, "input_tokens", "prompt_tokens", "total_input_tokens")
    out = _int(raw, "output_tokens", "completion_tokens", "total_output_tokens")
    if "cache_read_input_tokens" in raw or "cache_creation_input_tokens" in raw:
        # Anthropic: input_tokens excludes the cache; everyone else's includes it.
        if total_in is not None:
            usage.input = total_in + (cache_write or 0)
        if cache_read is not None:
            usage.cached = cache_read
    else:
        if cache_read is not None:
            usage.cached = cache_read
        if total_in is not None:
            usage.input = max(total_in - usage.cached, 0)
    if out is not None:
        usage.output = out  # cumulative in every API that repeats it
    if isinstance(raw.get("cost"), int | float):
        usage.cost = float(raw["cost"])  # OpenRouter reports what it charged
    return usage


def _find(event: dict[str, Any]) -> dict[str, Any] | None:
    for container in (event, event.get("message"), event.get("response"), event.get("interaction")):
        if isinstance(container, dict) and isinstance(container.get("usage"), dict):
            return container["usage"]
    return None


def _int(raw: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if isinstance(raw.get(key), int):
            return int(raw[key])
    return None


def price_from_listing(pricing: dict[str, Any]) -> Price | None:
    """A model list's `pricing` (OpenRouter: dollars per token, as strings)."""
    try:
        per_token_in = float(pricing["prompt"])
        per_token_out = float(pricing["completion"])
    except (KeyError, TypeError, ValueError):
        return None
    cached = pricing.get("input_cache_read")
    try:
        cached_per_m = float(cached) * 1e6 if cached not in (None, "") else None
    except (TypeError, ValueError):
        cached_per_m = None
    return Price(per_token_in * 1e6, per_token_out * 1e6, cached_per_m)


class UsageMeter:
    """Running totals for a session, subagents included (they share the tool context).

    Every call sends the whole conversation again, so `total.input + total.cached` grows
    with each step and says little about the conversation's size (a 40-step turn over an
    80k context adds up to ~800k, most of it cheap cache reads). The size is the largest
    single call's context; what the model produced is `total.output`."""

    def __init__(self) -> None:
        self.total = Usage()
        self.calls = 0
        self.unpriced = 0  # calls whose cost isn't known
        self.contexts: list[int] = []  # each call's input (new + cached), in order
        self.last: Usage | None = None  # the latest call's usage, to see whether the cache is hitting
        self.cleared: tuple[Usage, int] | None = None  # (total, calls) at the last /clear

    def add(self, usage: Usage) -> None:
        self.total = self.total + usage
        self.calls += 1
        self.contexts.append(usage.input + usage.cached)
        self.last = usage
        if usage.cost is None:
            self.unpriced += 1

    def mark_clear(self) -> None:
        """The conversation was cleared. Totals stay (it's what the session spent);
        `since_clear` counts from here."""
        self.cleared = (Usage() + self.total, self.calls)

    def since_clear(self) -> tuple[Usage, int] | None:
        """(usage, calls) since the last clear, or None if there hasn't been one."""
        if self.cleared is None:
            return None
        before, calls = self.cleared
        now = self.total
        cost = None if now.cost is None else now.cost - (before.cost or 0)
        spent = Usage(now.input - before.input, now.cached - before.cached, now.output - before.output, cost)
        return spent, self.calls - calls

    def largest_context(self, since: int = 0) -> int:
        """The most tokens any one call has sent, counting calls from number `since`."""
        return max(self.contexts[since:], default=0)

    @property
    def cost(self) -> float | None:
        """Total dollars, or None if no call's cost is known."""
        return self.total.cost
