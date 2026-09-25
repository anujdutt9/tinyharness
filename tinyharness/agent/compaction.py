"""Keeping the conversation inside the context window.

Three tiers, cheapest first, run before every model call:

  1. budget_cap    cap any single oversized message (e.g. a huge file read)
  2. microcompact  replace old, large tool results with a one-line stub
  3. summarize     when the whole history nears the window, have the model
                   summarize everything but the recent tail

Tiers 1 and 2 edit messages in place, exactly once, and mark them. Each
message's bytes therefore change at most once, which keeps the provider's
prompt-prefix cache valid on later turns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tinyharness.agent.history import History, Message
from tinyharness.llm.tokens import count_tokens
from tinyharness.prompts import load_prompt

if TYPE_CHECKING:
    from tinyharness.llm.client import LLMClient

TRUNCATED_MARK = "[truncated:"
STUB_MARK = "[trimmed:"
SUMMARY_MARK = "[prior conversation summary]"


def _already_compacted(content: str) -> bool:
    return content.startswith((TRUNCATED_MARK, STUB_MARK))


class Compactor:
    def __init__(
        self,
        llm: LLMClient,
        *,
        context_window: int = 200_000,
        threshold: float = 0.90,
        max_message_tokens: int = 16_000,
        stub_tool_results_over: int = 6_000,
        keep_tail_messages: int = 12,
    ) -> None:
        self.llm = llm
        self.context_window = context_window
        self.threshold = threshold
        self.max_message_tokens = max_message_tokens
        self.stub_tool_results_over = stub_tool_results_over
        self.keep_tail = keep_tail_messages

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
            capped += 1
        return capped

    def microcompact(self, history: History) -> int:
        """Stub large tool results outside the recent tail. Returns how many were stubbed."""
        if len(history.messages) <= self.keep_tail:
            return 0
        stubbed = 0
        for m in history.messages[: -self.keep_tail]:
            content = m.content or ""
            if m.role != "tool" or _already_compacted(content):
                continue
            tokens = count_tokens(content)
            if tokens > self.stub_tool_results_over:
                m.content = f"{STUB_MARK} {tokens} tokens of older tool output]"
                stubbed += 1
        return stubbed

    def needs_summary(self, history: History) -> bool:
        total = sum(count_tokens(m.content or "") for m in history.messages)
        return total >= self.context_window * self.threshold

    async def summarize(self, history: History) -> bool:
        """Replace everything before the recent tail with an LLM-written summary.

        The tail always starts at a user message, so an assistant tool call is
        never separated from its tool result (providers reject orphaned pairs).
        Returns False if there was nothing to summarize.
        """
        messages = history.messages
        system = [
            m for m in messages[:1] if m.role == "system" and not (m.content or "").startswith(SUMMARY_MARK)
        ]
        body = messages[len(system) :]

        cut = max(0, len(body) - self.keep_tail)
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
