"""Conversation state in the OpenAI Chat Completions message shape."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = ""  # raw JSON string, exactly as the model produced it

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class Message:
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role="tool" replies

    def to_openai(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class History:
    messages: list[Message] = field(default_factory=list)

    def append(self, msg: Message) -> None:
        self.messages.append(msg)

    def add_tool_result(self, call_id: str, content: str) -> None:
        self.append(Message(role="tool", tool_call_id=call_id, content=content))

    def reset(self) -> None:
        self.messages.clear()

    def to_openai(self) -> list[dict[str, Any]]:
        return [m.to_openai() for m in self.messages]

    def repair(self) -> int:
        """Answer any tool calls left without a result (e.g. the user hit Ctrl-C
        mid-turn). Providers reject a history with dangling tool calls.
        Returns the number of placeholder results added."""
        answered = {m.tool_call_id for m in self.messages if m.role == "tool"}
        repaired: list[Message] = []
        added = 0
        for msg in self.messages:
            repaired.append(msg)
            for tc in msg.tool_calls:
                if tc.id not in answered:
                    repaired.append(Message(role="tool", tool_call_id=tc.id, content="(interrupted by user)"))
                    added += 1
        self.messages = repaired
        return added
