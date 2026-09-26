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
class Reasoning:
    """The model started reasoning before answering (not every reply does)."""


@dataclass(frozen=True)
class ToolUse:
    """The model asked to call a tool (before the permission check)."""

    name: str
    args: dict[str, Any]
    summary: str


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
    name: str
    output: str
    is_error: bool = False


@dataclass(frozen=True)
class TodosUpdated:
    todos: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class Compacted:
    """Context was reduced before the next model call."""

    kind: Literal["budget_cap", "microcompact", "summary"]
    count: int = 0


SubagentStatus = Literal["started", "working", "done", "failed"]


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


@dataclass(frozen=True)
class Done:
    reason: Literal["stop", "max_steps", "interrupted"] = "stop"


Event = (
    AssistantText
    | Reasoning
    | ToolUse
    | PermissionDecision
    | ToolsDispatched
    | ToolResult
    | TodosUpdated
    | Compacted
    | SubagentUpdate
    | Done
)
