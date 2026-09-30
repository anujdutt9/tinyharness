"""Keeping the conversation inside the context window.

Three tiers, cheapest first, run before every model call:

  1. budget_cap    cap any single oversized message (e.g. a huge file read)
  2. microcompact  replace old, large tool results with a one-line stub
  3. summarize     when the whole history nears the window, have the model
                   summarize everything but the recent tail

Tiers 1 and 2 edit messages in place, exactly once, and mark them. Each
message's bytes therefore change at most once, which keeps the provider's
prompt-prefix cache valid on later turns.

Changing a message still invalidates the cached prefix from that message on, so
the next request pays to write the cache again. Tier 1 only touches messages
that have not been sent yet, so it costs nothing. Tier 2 does cost: stubbing one
result each time the tail moves on would mean a cache miss on nearly every turn.

While the cache is warm, clearing roughly breaks even on cost: the rewrite costs
about as much as dozens of later turns save. It also takes information away from
the model. So tier 2 clears only when it pays:

  - cache cold (the last call was more than `cache_ttl` ago): the whole prompt
    is written again anyway, so clearing is free and makes that write cheaper
  - cache warm: only once the history passes `clear_when_warm_at` of the window,
    to hold off the lossy summary, and only for at least `min_savings` tokens at
    once (the idea behind Anthropic's `clear_at_least` and opencode's prune)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from tinyharness.agent.history import History, Message
from tinyharness.llm.tokens import count_tokens
from tinyharness.prompts import load_prompt

if TYPE_CHECKING:
    from tinyharness.llm.client import LLMClient

TRUNCATED_MARK = "[truncated:"
STUB_MARK = "[trimmed:"
SUMMARY_MARK = "[prior conversation summary]"

# Results that are instructions or delegated work: expensive to redo, cheap to keep.
KEEP_TOOLS = frozenset({"Skill", "Agent", "Team"})


def _already_compacted(content: str) -> bool:
    return content.startswith((TRUNCATED_MARK, STUB_MARK))


class Compactor:
    def __init__(
        self,
        llm: LLMClient,
        *,
        context_window: int | Callable[[], int] = 200_000,
        threshold: float = 0.90,
        max_message_tokens: int = 16_000,
        stub_tool_results_over: int = 6_000,
        keep_tail_messages: int = 12,
        protect_tool_tokens: int = 40_000,
        min_savings: int = 20_000,
        clear_when_warm_at: float = 0.60,
        cache_ttl: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.llm = llm
        self._context_window = context_window
        self.threshold = threshold
        self.max_message_tokens = max_message_tokens
        self.stub_tool_results_over = stub_tool_results_over
        self.keep_tail = keep_tail_messages
        self.protect_tool_tokens = protect_tool_tokens
        self.min_savings = min_savings
        self.clear_when_warm_at = clear_when_warm_at
        self.cache_ttl = cache_ttl  # Anthropic's default; other providers keep theirs as long or longer
        self._clock = clock
        self._last_call: float | None = None

    @property
    def context_window(self) -> int:
        """May be a callable: the window is only known once the provider's model list is."""
        window = self._context_window
        return window() if callable(window) else window

    def mark_call(self) -> None:
        """Record a model call: the provider's cache for this prefix is fresh from now."""
        self._last_call = self._clock()

    @property
    def cache_cold(self) -> bool:
        """True if the prefix cache has likely expired, so rewriting history costs nothing extra.
        Before the first call (e.g. a resumed session) it is taken as cold."""
        return self._last_call is None or self._clock() - self._last_call > self.cache_ttl

    def budget_cap(self, history: History) -> int:
        """Truncate messages larger than `max_message_tokens`. Returns how many were capped."""
        capped = 0
        for m in history.messages:
            content = m.content or ""
            if not content or _already_compacted(content):
                continue
            tokens = count_tokens(content)
            if tokens <= self.max_message_tokens:
                continue
            keep_chars = self.max_message_tokens * 4  # ~4 chars per token
            m.content = f"{TRUNCATED_MARK} kept ~{self.max_message_tokens} of {tokens} tokens]\n\n{content[:keep_chars]}"
            m.state = None  # the provider's saved copy would bring the full text back
            capped += 1
        return capped

    def microcompact(self, history: History) -> int:
        """Stub old tool results, in one batch. Returns how many were stubbed.

        Protected: the last `keep_tail` messages, the most recent `protect_tool_tokens`
        of tool output, results of KEEP_TOOLS, and results under `stub_tool_results_over`.
        The rest is stubbed if the cache is cold, or if the history is past
        `clear_when_warm_at` of the window and at least `min_savings` tokens would go
        (see the module docstring)."""
        messages = history.messages
        if len(messages) <= self.keep_tail:
            return 0
        names = {c.id: c.name for m in messages for c in m.tool_calls}
        protect = self.protect_tool_tokens
        batch: list[tuple[Message, int]] = []
        for m in reversed(messages[: -self.keep_tail]):
            content = m.content or ""
            if (
                m.role != "tool"
                or _already_compacted(content)
                or names.get(m.tool_call_id or "") in KEEP_TOOLS
            ):
                continue
            tokens = count_tokens(content)
            if protect > 0:
                protect -= tokens
            elif tokens > self.stub_tool_results_over:
                batch.append((m, tokens))
        if not batch:
            return 0
        if not self.cache_cold and (
            sum(t for _, t in batch) < self.min_savings
            or self._used(history) < self.context_window * self.clear_when_warm_at
        ):
            return 0
        for m, tokens in batch:
            m.content = f"{STUB_MARK} {tokens} tokens of older tool output]"
        return len(batch)

    def needs_summary(self, history: History) -> bool:
        return self._used(history) >= self.context_window * self.threshold

    @staticmethod
    def _used(history: History) -> int:
        return sum(count_tokens(m.content or "") for m in history.messages)

    async def summarize(self, history: History, *, keep_tail: int | None = None) -> bool:
        """Replace everything before the recent tail with an LLM-written summary.
        `keep_tail` overrides how many recent messages stay (at least, back to a user message).

        The tail always starts at a user message, so an assistant tool call is
        never separated from its tool result (providers reject orphaned pairs).
        Returns False if there was nothing to summarize.
        """
        messages = history.messages
        system = [
            m for m in messages[:1] if m.role == "system" and not (m.content or "").startswith(SUMMARY_MARK)
        ]
        body = messages[len(system) :]

        cut = max(0, len(body) - (self.keep_tail if keep_tail is None else keep_tail))
        while cut > 0 and body[cut].role != "user":
            cut -= 1
        if cut == 0:
            return False

        transcript = "\n\n".join(f"[{m.role}] {(m.content or '')[:4000]}" for m in body[:cut])
        summary = await self.llm.one_shot(transcript, system=load_prompt("compaction"))
        history.messages = [
            *system,
            Message(role="system", content=f"{SUMMARY_MARK}\n{summary}"),
            *body[cut:],
        ]
        return True
