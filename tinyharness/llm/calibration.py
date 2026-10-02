"""How a model calls tools, found out once by trying.

Most models call tools through the API's tool-calling interface ("native"). Some smaller
or self-hosted ones do much better writing calls as text in their reply: with
natural-language or text-format tool calls, small and open models gained 24-43 points in
published comparisons, while frontier models gained little or nothing. Which one a model
needs isn't something a harness should guess from its name, so it's measured: the first
time a model is used, one small request offers it a single tool and sees how it calls it.
The answer is cached per provider and model (~/.tinyharness/cache/tool-calls.json), and
`tool_calls = "native" | "text"` in the config overrides it.

In text mode the tools are described in the system prompt, the model writes
`<call>{"name": ..., "arguments": {...}}</call>` (read by llm/textcalls.py), and earlier
calls and results in the history are sent as text too. The tag is our own on purpose:
seen live, several hosted servers stripped `<tool_call>` text out of replies even with
no tools in the request, leaving empty replies.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Literal

from tinyharness.agent.history import Message
from tinyharness.llm.textcalls import recover_calls
from tinyharness.llm.types import StreamChunk, ToolSpec

log = logging.getLogger(__name__)

ToolFormat = Literal["native", "text"]

PROBE_TOOL = ToolSpec(
    "get_time",
    "Get the current time in a timezone.",
    {
        "type": "object",
        "properties": {"timezone": {"type": "string", "description": "IANA timezone, e.g. UTC"}},
        "required": ["timezone"],
    },
)
PROBE_MESSAGES = [
    Message(role="system", content="You are checking that tool calls work. Call the tool you're asked to."),
    Message(role="user", content="What time is it in UTC? Call the get_time tool to find out."),
]


async def probe(stream: Callable[[list[Message], list[ToolSpec]], AsyncIterator[StreamChunk]]) -> ToolFormat:
    """One request offering one tool: native if the model calls it through the interface,
    text if it writes the call in its reply. Anything else (no call at all) keeps native."""
    text: list[str] = []
    async for chunk in stream(PROBE_MESSAGES, [PROBE_TOOL]):
        if chunk.tool_call_delta:
            name = (chunk.tool_call_delta.get("function") or {}).get("name")
            if name == PROBE_TOOL.name:
                return "native"
        if chunk.text:
            text.append(chunk.text)
    schemas = {PROBE_TOOL.name: PROBE_TOOL.parameters}
    _, calls = recover_calls("".join(text), schemas.get)
    return "text" if any(c.name == PROBE_TOOL.name for c in calls) else "native"


class CalibrationStore:
    """The probed format per "provider:model"."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def get(self, owner: str) -> ToolFormat | None:
        entry = self._load().get(owner)
        fmt = entry.get("tool_calls") if isinstance(entry, dict) else None
        return fmt if fmt in ("native", "text") else None

    def put(self, owner: str, fmt: ToolFormat) -> None:
        data = self._load()
        data[owner] = {"tool_calls": fmt, "checked": int(time.time())}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2, sort_keys=True))
        except OSError as e:
            log.warning("could not save tool-call calibration to %s: %s", self.path, e)

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}


# ---------------------------------------------------------------------- text protocol


def tools_section(tools: list[ToolSpec]) -> str:
    """The tools, described for a model that writes its calls as text."""
    lines = [
        "# Tools",
        "To call a tool, end your reply with one line per call, and write nothing after them:",
        '<call>{"name": "Read", "arguments": {"path": "src/app.py"}}</call>',
        'Each result comes back in the next message as <tool_result name="...">...</tool_result>.',
    ]
    for tool in tools:
        schema = json.dumps(tool.parameters, ensure_ascii=False, separators=(",", ":"))
        lines += ["", f"## {tool.name}", tool.description, f"Arguments (JSON schema): {schema}"]
    return "\n".join(lines)


def as_text(messages: list[Message]) -> list[Message]:
    """The history for a model that calls tools in text: its earlier calls written into its
    replies, and tool results as user messages; consecutive user messages merged."""
    names = {c.id: c.name for m in messages for c in m.tool_calls}
    out: list[Message] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            calls = [
                f'<call>{{"name": {json.dumps(c.name)}, "arguments": {c.arguments or "{}"}}}</call>'
                for c in m.tool_calls
            ]
            text = "\n".join(filter(None, [m.content or "", *calls]))
            out.append(Message(role="assistant", content=text))
        elif m.role == "tool":
            name = names.get(m.tool_call_id or "", "tool")
            body = f'<tool_result name="{name}">\n{m.content or ""}\n</tool_result>'
            _append_user(out, Message(role="user", content=body, images=list(m.images)))
        elif m.role == "user":
            _append_user(out, Message(role="user", content=m.content, images=list(m.images)))
        else:
            out.append(m)
    return out


def _append_user(out: list[Message], message: Message) -> None:
    if out and out[-1].role == "user":
        last = out[-1]
        out[-1] = Message(
            role="user",
            content="\n\n".join(filter(None, [last.content or "", message.content or ""])),
            images=[*last.images, *message.images],
        )
    else:
        out.append(message)
