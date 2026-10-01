"""Events streamed out of the agent loop.

Both the CLI and SDK consume the same stream:

    async for event in session.run("fix the failing test"):
        match event:
            case AssistantText(text=t): ...
            case ToolResult(name=n, output=o): ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class AssistantText:
    """A chunk of streamed model text."""

    text: str


@dataclass(frozen=True)
class ModelStuck:
    """The model's reply got stuck (repeating itself, or only blank output): it was stopped,
    and the model asked once to carry on."""

    reason: str


@dataclass(frozen=True)
class MessageDelivered:
    """A message sent while the agent worked (`Session.steer`) reached it: the agent sees
    it from its next step on."""

    text: str


@dataclass(frozen=True)
class Reasoning:
    """The model is reasoning before it answers (not every reply does). `text` is the next
    piece of its reasoning when the provider shares it (as is, or as a summary); empty
    when it only says that it's reasoning."""

    text: str = ""


@dataclass(frozen=True)
class ToolInputStreaming:
    """The model is writing a tool call: its arguments stream in before the call is
    complete (writing a whole file can take minutes). `chars` is how much has arrived;
    `target` is the path, URL or command once it's in. Sent when a call starts and then
    about once per KB; `ToolUse` follows when it's complete."""

    name: str
    chars: int
    target: str = ""


@dataclass(frozen=True)
class ToolUse:
    """The model asked to call a tool (before the permission check)."""

    name: str
    args: dict[str, Any]
    summary: str
    id: str = ""  # the call's id: its ToolResult has the same one


@dataclass(frozen=True)
class PermissionDecision:
    """Outcome of the permission gate for one tool call."""

    tool: str
    allowed: bool
    source: str  # which layer decided: mode | rule | hook | classifier | prompt | ...
    reason: str = ""


@dataclass(frozen=True)
class ToolsDispatched:
    """Approved tool calls are about to run, concurrently if `parallel`."""

    names: list[str]
    parallel: bool


@dataclass(frozen=True)
class ToolResult:
    """A call's result, sent as soon as the call finishes: the calls of a parallel batch
    report in the order they finish (match them to their ToolUse by `id`)."""

    name: str
    output: str
    is_error: bool = False
    id: str = ""


@dataclass(frozen=True)
class TodosUpdated:
    todos: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Compacted:
    """Context was reduced before the next model call."""

    kind: Literal["budget_cap", "microcompact", "summary"]
    count: int = 0


@dataclass(frozen=True)
class TokenUsage:
    """What one model call used, when the provider reports it. `cost` is in dollars,
    when known (reported by the provider, or from a price in the config or model list)."""

    model: str
    input: int
    cached: int
    output: int
    cost: float | None = None


SubagentStatus = Literal["started", "working", "done", "failed"]
# While "working": running a tool (`detail` is its summary), waiting for the model's next
# step, reasoning, or writing a tool call (`tool` is its name, `detail` its target).
SubagentPhase = Literal["tool", "waiting", "thinking", "writing"]


@dataclass(frozen=True)
class SubagentCall:
    """One of a subagent's tool calls, for showing what it has been doing."""

    summary: str  # e.g. "Bash(node .deck-verify/measure.mjs)"
    done: bool = False
    ok: bool = True
    result: str = ""  # the first line of what came back, shortened
    elapsed: float = 0.0  # seconds taken, or so far


@dataclass(frozen=True)
class SubagentUpdate:
    """Progress of a subagent or teammate while its Agent/Team call runs. These don't come
    through `session.run()` (the parent is waiting on the tool); they go to
    `SessionOptions.on_event` as they happen."""

    id: str  # unique per run
    agent: str  # the agent type, e.g. "explore"
    label: str  # what it was asked to do, shortened
    status: SubagentStatus
    detail: str = ""  # the tool it's running, or how it ended
    tools_used: int = 0
    elapsed: float = 0.0
    model: str = ""  # the model it runs on, e.g. "claude-haiku-4-5"
    phase: SubagentPhase = "tool"
    tool: str = ""  # the tool call being written, when phase is "writing"
    recent: tuple[SubagentCall, ...] = ()  # its last few tool calls, oldest first
    log: str = ""  # its transcript (JSONL), written as it goes


@dataclass(frozen=True)
class Done:
    reason: Literal["stop", "max_steps", "interrupted"] = "stop"


Event = (
    AssistantText
    | MessageDelivered
    | ModelStuck
    | Reasoning
    | ToolInputStreaming
    | ToolUse
    | PermissionDecision
    | ToolsDispatched
    | ToolResult
    | TodosUpdated
    | Compacted
    | TokenUsage
    | SubagentUpdate
    | Done
)
