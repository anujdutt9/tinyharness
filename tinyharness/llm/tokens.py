"""Token counting. Uses tiktoken when installed (`tinyharness[tokens]`), else ~4 chars/token."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


def _load_counter() -> Callable[[str], int]:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(encoding.encode(text or ""))
    except Exception:  # not installed, or encoding download failed
        return lambda text: max(1, len(text or "") // 4)


count_tokens = _load_counter()


def count_message_tokens(messages: list[dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        total += 4  # per-message framing overhead
        for v in m.values():
            if isinstance(v, str):
                total += count_tokens(v)
            elif v is not None:
                total += count_tokens(json.dumps(v, default=str))
    return total
