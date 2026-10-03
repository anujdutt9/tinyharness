"""Token counting. Uses tiktoken when installed (`tinyharness[tokens]`), else ~4 chars/token; a
model's own tokenizer when the config names one (`model_counter`)."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tinyharness.agent.history import Message

log = logging.getLogger(__name__)

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


@functools.cache
def model_counter(tokenizer: str | None) -> Callable[[str], int] | None:
    """A counter with a model's own tokenizer: `tokenizer` is a Hugging Face repo id (the
    model's, or any repo with its tokenizer.json), from a `tokenizer` setting in the config.
    None when unset, or when the `tokenizers` package or the download isn't available (the
    estimate is then calibrated against the provider's counts instead)."""
    if not tokenizer:
        return None
    try:
        from tokenizers import Tokenizer

        tok = _cached(Tokenizer, tokenizer) or Tokenizer.from_pretrained(tokenizer)
    except Exception as e:  # not installed, offline, no such repo
        log.warning("tokenizer %s unavailable (%s); estimating token counts", tokenizer, e)
        return None
    return lambda text: len(tok.encode(text or "", add_special_tokens=False).ids)


def _cached(tokenizer_class: Any, repo: str) -> Any:
    """The tokenizer from the local Hugging Face cache, without asking the Hub: once it has
    been downloaded, every session would otherwise check for a newer revision first (and
    print the Hub's warning about unauthenticated requests). None if it isn't cached yet."""
    try:
        from huggingface_hub import hf_hub_download

        return tokenizer_class.from_file(hf_hub_download(repo, "tokenizer.json", local_files_only=True))
    except Exception:  # not cached (or no huggingface_hub): download it
        return None


def estimate_tokens(messages: Iterable[Message], count: Callable[[str], int] | None = None) -> int:
    """What these messages take up in the context: text, tool calls (a Write's content is
    in its arguments), images, and the reasoning sent back with a reply. The one estimate
    everything uses (compaction, /context); `count`: a model's own tokenizer (`model_counter`)."""
    count = count or count_tokens
    total = 0
    for m in messages:
        total += 4 + count(m.content or "") + IMAGE_TOKENS * len(m.images)
        for call in m.tool_calls:
            total += count(call.name) + count(call.arguments)
        data = m.state.data if m.state is not None else None
        if isinstance(data, dict):  # chat APIs' reasoning text (reasoning, reasoning_content)
            total += sum(count(v) for v in data.values() if isinstance(v, str))
    return total
