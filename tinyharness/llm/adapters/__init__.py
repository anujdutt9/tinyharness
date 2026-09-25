"""One adapter per provider API format. A provider picks its format with `api = ...`."""

from tinyharness.llm.adapters.anthropic import AnthropicAdapter
from tinyharness.llm.adapters.base import Adapter, StreamParser
from tinyharness.llm.adapters.chat import ChatAdapter
from tinyharness.llm.adapters.gemini import GeminiAdapter
from tinyharness.llm.adapters.responses import ResponsesAdapter

ADAPTERS: dict[str, type[Adapter]] = {
    a.name: a for a in (ChatAdapter, ResponsesAdapter, AnthropicAdapter, GeminiAdapter)
}

__all__ = ["ADAPTERS", "Adapter", "StreamParser"]
