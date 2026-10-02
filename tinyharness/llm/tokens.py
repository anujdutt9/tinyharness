"""Token counting. Uses tiktoken when installed (`tinyharness[tokens]`), else ~4 chars/token."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinyharness.agent.history import Message

# What one image costs, roughly: a screenshot-sized picture is ~1,600 tokens on most APIs.
IMAGE_TOKENS = 1_600


def _load_counter() -> Callable[[str], int]:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(encoding.encode(text or ""))
    except Exception:  # not installed, or encoding download failed
        return lambda text: max(1, len(text or "") // 4)


count_tokens = _load_counter()


def estimate_tokens(messages: Iterable[Message]) -> int:
    """What these messages take up in the context: text, tool calls (a Write's content is
    in its arguments) and images. The one estimate everything uses (compaction, /context)."""
    total = 0
    for m in messages:
        total += 4 + count_tokens(m.content or "") + IMAGE_TOKENS * len(m.images)
        for call in m.tool_calls:
            total += count_tokens(call.name) + count_tokens(call.arguments)
    return total
