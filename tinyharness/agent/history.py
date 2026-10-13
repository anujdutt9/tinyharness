"""Conversation state, independent of any provider's API format.

Adapters (tinyharness/llm/adapters) translate these messages into each provider's
format. An assistant message can also carry `state`: data its provider requires
back on the next request (signed or encrypted reasoning). It is opaque here and
only ever sent back to the exact provider and model that produced it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import string
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
        """Make the stored history valid again after a turn was cut short (Ctrl-C mid tool
        call, a crash): see `well_formed`. Returns the number of placeholder results added."""
        self.messages, added = well_formed(self.messages, placeholder=placeholder)
        return added


# The result for a tool call that has none when a request is built (normally `repair`
# has already answered it; this is the backstop, so a request is never rejected for it).
MISSING_RESULT = "(no result recorded: it may or may not have run; check its effects before running it again)"


def without_images(messages: list[Message]) -> list[Message]:
    """The messages for a model that takes no images: each image replaced by a note saying so
    so the model can tell the user or check another way. Copies; the history keeps its
    images for a model that can see them."""
    out = []
    for m in messages:
        if not m.images:
            out.append(m)
            continue
        notes = [
            f"[ERROR: Cannot read {image.name or 'an image'} (this model does not support image input). "
            "Inform the user, or check another way.]"
            for image in m.images
        ]
        out.append(dataclasses.replace(m, images=[], content="\n".join(filter(None, [m.content, *notes]))))
    return out


def well_formed(
    messages: list[Message],
    *,
    placeholder: str = MISSING_RESULT,
    valid_id: re.Pattern[str] | None = None,
    owner: str = "",
) -> tuple[list[Message], int]:
    """`messages` in the shape every provider accepts, without changing `messages` itself:

    - each assistant message's tool calls are answered by the tool results right after
      it: a missing result gets `placeholder`, a result answering no call there is dropped
      (an id reused by a later turn can't be mistaken for an answer);
    - an assistant message with nothing in it (no text, no calls, no saved reasoning) is
      dropped: APIs reject empty turns;
    - with `valid_id`, tool-call ids the provider wouldn't accept (another provider's, after
      /model) are replaced, deterministically, so the prompt cache still matches. Not in
      replies whose saved reasoning goes back to `owner` as is: that holds the original
      ids, and the provider issued them itself.

    Returns the messages and how many placeholder results were added.
    """
    out: list[Message] = []
    pending: list[ToolCall] = []  # the latest assistant message's calls, not yet answered
    added = 0

    def answer_pending() -> None:
        nonlocal added
        for call in pending:
            out.append(Message(role="tool", tool_call_id=call.id, content=placeholder))
            added += 1
        pending.clear()

    for m in messages:
        if m.role == "tool":
            match = next((c for c in pending if c.id == m.tool_call_id), None)
            if match is not None:
                pending.remove(match)
                out.append(m)
            continue  # a result that answers nothing here
        answer_pending()
        if m.role == "assistant" and not (m.content or "").strip() and not m.tool_calls and m.state is None:
            continue
        out.append(m)
        if m.role == "assistant":
            pending = list(m.tool_calls)
    answer_pending()
    if valid_id is not None:
        echoed = {c.id for m in out if m.state is not None and m.state.owner == owner for c in m.tool_calls}
        out = [m if _echoed(m, echoed) else _with_valid_ids(m, valid_id) for m in out]
    return out, added


def _echoed(m: Message, ids: set[str]) -> bool:
    return bool(ids) and (m.tool_call_id in ids or any(c.id in ids for c in m.tool_calls))


def _with_valid_ids(m: Message, valid: re.Pattern[str]) -> Message:
    if m.tool_calls and not all(valid.fullmatch(c.id) for c in m.tool_calls):
        calls = [ToolCall(_valid_id(c.id, valid), c.name, c.arguments) for c in m.tool_calls]
        return dataclasses.replace(m, tool_calls=calls)
    if m.tool_call_id and not valid.fullmatch(m.tool_call_id):
        return dataclasses.replace(m, tool_call_id=_valid_id(m.tool_call_id, valid))
    return m


_ALPHANUMERIC = string.ascii_letters + string.digits


def _valid_id(call_id: str, valid: re.Pattern[str]) -> str:
    """A stand-in for `call_id` that `valid` accepts, the same every time."""
    if valid.fullmatch(call_id):
        return call_id
    digest = int.from_bytes(hashlib.sha256(call_id.encode()).digest())
    short = "".join(_ALPHANUMERIC[(digest >> (6 * i)) % 62] for i in range(9))  # Mistral: 9 alphanumerics
    for candidate in (f"call_{digest:x}"[:29], short):
        if valid.fullmatch(candidate):
            return candidate
    return call_id
