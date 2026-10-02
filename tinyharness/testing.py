"""Testing with a scripted model instead of a real one: for your extensions and tools, and
for tinyharness's own tests.

    from tinyharness.events import ToolResult
    from tinyharness.testing import scripted_session

    async def test_hello(tmp_path):
        session = scripted_session(
            tmp_path, [[("hello", '{"name": "Ada"}')], "done"], extensions=[my_extension]
        )
        results = [e.output async for e in session.run("greet Ada") if isinstance(e, ToolResult)]
        assert results == ["Hello, Ada."]
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from tinyharness.agent.history import Message
from tinyharness.llm.types import StreamChunk
from tinyharness.session import Session, SessionOptions


class ScriptedLLM:
    """A model that replays a fixed list of turns. Each turn is a string (a text reply), a
    list of (tool name, arguments JSON) tool calls, or a list of raw StreamChunks. After
    the last turn it replies "done". `requests` holds the messages each call was sent."""

    def __init__(self, turns: list[Any]) -> None:
        self.turns = list(turns)
        self.requests: list[list[Message]] = []
        self.model = "fake"

    async def stream(
        self, messages: list[Message], tools: Any = None, **_: Any
    ) -> AsyncIterator[StreamChunk]:
        self.requests.append(list(messages))
        turn = self.turns.pop(0) if self.turns else "done"
        if isinstance(turn, str):
            yield StreamChunk(text=turn, finish_reason="stop")
            return
        if turn and isinstance(turn[0], StreamChunk):
            for chunk in turn:
                yield chunk
            return
        for i, (name, arguments) in enumerate(turn):
            yield StreamChunk(
                tool_call_delta={
                    "index": i,
                    "id": f"call_{i}",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        yield StreamChunk(finish_reason="tool_calls")

    async def one_shot(self, prompt: str, **_: Any) -> str:
        return "summary of prior conversation"


def scripted_session(
    working_dir: Path, turns: list[Any], *, extensions: list[Any] | None = None, **options: Any
) -> Session:
    """A session in `working_dir` whose model plays `turns`, with `extensions` (modules or
    setup functions) loaded. No API key or network is needed; tools really run."""
    working_dir.mkdir(parents=True, exist_ok=True)
    session = Session(
        SessionOptions(
            working_dir=working_dir,
            config_path=working_dir / ".no-config.toml",
            provider="openai",
            api_key="scripted",
            extensions=list(extensions or []),
            **options,
        )
    )
    session.llm.stream = ScriptedLLM(turns).stream  # type: ignore[method-assign,assignment]
    return session
