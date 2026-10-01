"""Conversation state, independent of any provider's API format.

Adapters (tinyharness/llm/adapters) translate these messages into each provider's
format. An assistant message can also carry `state`: data its provider requires
back on the next request (signed or encrypted reasoning). It is opaque here and
only ever sent back to the exact provider and model that produced it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]

INTERRUPTED = "(interrupted by user)"


def new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str = ""  # raw JSON string, exactly as the model produced it

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class Image:
    """An image for the model to look at: a screenshot the user attached, or a picture a
    tool read. `data` is base64; `name` says where it came from (shown to models that
    can't view images)."""

    media_type: str  # "image/png", "image/jpeg", "image/gif", "image/webp"
    data: str
    name: str = ""

    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.data}"


@dataclass
class ReasoningState:
    owner: str  # "provider:model" that produced it
    data: Any


@dataclass
class Message:
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role="tool" replies
    state: ReasoningState | None = None  # assistant only; see the module docstring
    id: str = field(default_factory=new_id)  # stable identity in the session log
    meta: dict[str, Any] = field(default_factory=dict)  # harness-only (e.g. the file a result came from)
    images: list[Image] = field(default_factory=list)  # user and tool messages: pictures to look at

    def to_dict(self) -> dict[str, Any]:
        """A plain dict (for token counting and transcripts); not any provider's format."""
        d: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            d["content"] = self.content
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        return d

    def to_record(self) -> dict[str, Any]:
        """Everything, losslessly, for the session log (see agent/sessionlog.py)."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in self.tool_calls
            ],
            "tool_call_id": self.tool_call_id,
            "state": {"owner": self.state.owner, "data": self.state.data} if self.state else None,
            "meta": self.meta,
            "images": [{"media_type": i.media_type, "data": i.data, "name": i.name} for i in self.images],
        }

    @classmethod
    def from_record(cls, r: dict[str, Any]) -> Message:
        state = r.get("state")
        return cls(
            role=r["role"],
            content=r.get("content"),
            tool_calls=[
                ToolCall(tc["id"], tc["name"], tc.get("arguments", "")) for tc in r.get("tool_calls", [])
            ],
            tool_call_id=r.get("tool_call_id"),
            state=ReasoningState(state["owner"], state["data"]) if state else None,
            id=r["id"],
            meta=r.get("meta") or {},
            images=[Image(i["media_type"], i["data"], i.get("name", "")) for i in r.get("images") or []],
        )


@dataclass
class History:
    messages: list[Message] = field(default_factory=list)

    def append(self, msg: Message) -> None:
        self.messages.append(msg)

    def add_tool_result(
        self,
        call_id: str,
        content: str,
        meta: dict[str, Any] | None = None,
        images: list[Image] | None = None,
    ) -> None:
        self.append(
            Message(
                role="tool", tool_call_id=call_id, content=content, meta=meta or {}, images=list(images or [])
            )
        )

    def reset(self) -> None:
        self.messages.clear()

    def to_dicts(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self.messages]

    def repair(self, placeholder: str = INTERRUPTED) -> int:
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
                    repaired.append(Message(role="tool", tool_call_id=tc.id, content=placeholder))
                    added += 1
        self.messages = repaired
        return added
