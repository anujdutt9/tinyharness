"""The Tool abstraction.

A tool is a class with:
  * `name` / `description` — what the model sees
  * `Input`                — a pydantic model; its JSON schema is sent to the model
                             and incoming arguments are validated against it
  * `run(args, ctx)`       — does the work and returns text for the model

Flags tell the harness how to treat it:
  * `read_only`        — no side effects: allowed in plan mode, safe to run in parallel
  * `needs_permission` — False for bookkeeping tools (e.g. TodoWrite) that skip the gate
  * `network`          — reaches other machines (WebFetch): asks per site, never free
  * `approval_boundary`— a contained capability that still needs an explicit decision
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

from tinyharness.agent.history import Image
from tinyharness.llm.types import ToolSpec
from tinyharness.llm.usage import UsageMeter

if TYPE_CHECKING:
    from collections.abc import Callable

    from tinyharness.config import Config
    from tinyharness.events import Event
    from tinyharness.workspace import Scratch, Workspace


LARGE_CHANGE = 20  # lines; removing more than this counts as destructive (auto mode asks)


@dataclass(frozen=True)
class ArtifactRef:
    """A file or URL the harness should present to the user without opening a host app."""

    target: str
    media_type: str = ""
    title: str = ""


@dataclass
class ToolOutput:
    """Return text, images, and user-facing artifacts from a tool.

    Attributes
    ----------
    text : str
        Text returned to the model.
    images : list[Image]
        Images returned to the model.
    artifacts : list[ArtifactRef]
        Files or URLs presented to the user by the harness UI, not sent to the model.
    """

    text: str
    images: list[Image] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)


class ToolError(Exception):
    """A tool failed in a way the model should see and can recover from."""


class ToolInput(BaseModel):
    """Validate the common behavior of tool input models."""

    # Lenient where intent is clear: a number given for a text field is that text (models
    # send `"content": 233`).
    model_config = ConfigDict(extra="ignore", coerce_numbers_to_str=True)
    # Fields holding file content, where an object or list means its JSON text (a .json
    # file's content). Only these: a list given for a Grep pattern is a mistake to report.
    content_fields: ClassVar[frozenset[str]] = frozenset()
    # Set by `Tool.pin_path`: the resolved path the permission gate approved.
    _approved_path: Path | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _structured_text(cls, data: Any) -> Any:
        if not isinstance(data, dict) or not cls.content_fields:
            return data
        fixed = dict(data)
        for key in cls.content_fields:
            if isinstance(fixed.get(key), dict | list):
                fixed[key] = json.dumps(fixed[key], indent=2, ensure_ascii=False)
        return fixed


InputT = TypeVar("InputT", bound=ToolInput)


@dataclass(frozen=True)
class ApprovalBoundary:
    """A contained capability whose first use needs an explicit decision."""

    key: str
    warning: str
    remember_label: str


@dataclass
class ToolContext:
    """Hold per-session state shared by tool calls."""

    workspace: Workspace
    config: Config
    scratch: Scratch | None = None
    read_paths: set[str] = field(default_factory=set)  # enforces read-before-write
    # Each file's content hash when the model last read or wrote it, so Edit can tell a
    # file that changed since (a command, a teammate, the user) from the one it remembers.
    seen_versions: dict[str, str] = field(default_factory=dict)
    todos: list[dict[str, str]] = field(default_factory=list)
    skills_prompt: str = ""  # the installed skills, as listed in the system prompt
    notify: Callable[[Event], None] | None = None  # side-channel events, e.g. subagent progress
    usage: UsageMeter = field(default_factory=UsageMeter)  # tokens and cost, subagents included
    # The user's latest message, so agents started on its behalf know what it's for.
    request: str = ""
    # Where content from other machines in this conversation came from ("WebFetch
    # docs.python.org"): once there is any, calls that can send data out need a person's
    # yes (permissions/gate.py). Shared with subagents; changed in place, never replaced.
    outside: list[str] = field(default_factory=list)
    # Trusted correlation metadata for durable tool records. Tools must not ask the
    # model to supply these: the session sets them itself.
    session_id: str = ""
    turn_id: str = ""


def note_outside(ctx: ToolContext, source: str) -> None:
    """Record that content from another machine (`source`) is now in the conversation."""
    if source not in ctx.outside:
        ctx.outside.append(source)


def file_version(path: Path) -> str | None:
    """A file's content hash, or None if it can't be read."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:20]
    except OSError:
        return None


def check_not_stale(ctx: ToolContext, path: Path) -> None:
    """Refuse to change a file that changed since this agent last read or wrote it (a
    command, another agent, the user): its copy of the file is out of date."""
    seen = ctx.seen_versions.get(str(path))
    if seen is not None and file_version(path) != seen:
        raise ToolError(
            f"{path} changed since you last read it (a command, another agent or the user "
            "changed it). Read it again, then make your change against what's there now."
        )


def mark_seen(ctx: ToolContext, path: Path) -> None:
    """The model has the file's current content now (it read or wrote it)."""
    ctx.read_paths.add(str(path))
    if (version := file_version(path)) is not None:
        ctx.seen_versions[str(path)] = version


class Tool(ABC, Generic[InputT]):
    """Define the interface implemented by every tool.

    Subclasses provide an input model and implement :meth:`run`. The remaining methods
    describe permissions, concurrency, and presentation to the harness.
    """

    name: ClassVar[str]
    description: str
    Input: type[InputT]
    read_only: ClassVar[bool] = False
    needs_permission: ClassVar[bool] = True
    network: ClassVar[bool] = False
    # What it saves is read by every later session as context (the agent's memory), so
    # instructions planted by outside content could outlive this conversation.
    persists: ClassVar[bool] = False

    @abstractmethod
    async def run(self, args: InputT, ctx: ToolContext) -> str | ToolOutput:
        """Execute the tool.

        Parameters
        ----------
        args : InputT
            Validated tool arguments.
        ctx : ToolContext
            State shared by tools in the session.

        Returns
        -------
        str or ToolOutput
            Content returned to the model.
        """
        ...

    def describe(self, args: InputT) -> str:
        """One-line label shown to the user, e.g. `Read(src/main.py)`."""
        path = self.target_path(args)
        return f"{self.name}({path})" if path else f"{self.name}()"

    def target_path(self, args: InputT) -> str | None:
        """The filesystem path this call touches, used for permission decisions."""
        path = getattr(args, "path", None)
        return str(path) if path else None

    def pin_path(self, args: InputT, path: Path) -> InputT:
        """`args` bound to the resolved `path` the permission gate approved (see
        `approved_path`), so the tool acts on exactly that path."""
        pinned = args.model_copy()
        pinned._approved_path = path
        return pinned

    def target_host(self, args: InputT) -> str | None:
        """The site a network tool contacts, used for permission decisions."""
        return None

    def approval_boundary(self, args: InputT) -> ApprovalBoundary | None:
        """A contained capability that still needs explicit user approval, if any."""
        return None

    def outside_source(self, args: InputT) -> str | None:
        """Where this call's result comes from, if it's another machine (and so could carry
        instructions the user didn't write); None for local results."""
        if not self.network:
            return None
        host = self.target_host(args)
        return f"{self.name} {host}" if host else self.name

    def risk(self, args: dict[str, Any], path: Path | None) -> str | None:
        """Why this call is destructive enough to always ask in auto mode, or None.
        Gets the raw arguments, since it runs before they're validated."""
        return None

    def contained(self, args: dict[str, Any]) -> bool:
        """Whether this call runs inside the OS sandbox (writes only to the project, no
        network), so modes that allow project edits may allow it without asking."""
        return False

    def can_send_out(self, args: dict[str, Any]) -> bool:
        """Whether this call can contact another machine with conversation data."""
        return self.network

    def escapes(self, args: dict[str, Any]) -> str | None:
        """Why this call crosses a hard boundary, or None. Such calls ask even in bypass
        mode: suppressing routine prompts never changes containment."""
        return None

    def rememberable_escape(self, args: dict[str, Any]) -> bool:
        """Whether a narrow boundary expansion may be remembered for its target.

        Most escapes remove containment and must be approved every time. A capability
        that remains sandbox-enforced, such as one exact network destination, can opt in.
        """
        return False

    def is_read_only(self, args: InputT) -> bool:
        """Whether this call has no side effects (safe to run alongside others)."""
        return self.read_only

    def writes(self, args: InputT) -> bool:
        """Whether this call can change files. Such calls from subagents running side by
        side take turns (agent/loop.py); everything else overlaps freely. By default, any
        call that isn't read-only; Bash says no for commands that only read."""
        return not self.is_read_only(args)

    def runs_alongside(self, args: InputT) -> bool:
        """Whether this call may run at the same time as the other calls in its batch. By
        default only read-only calls may. A tool that keeps its own writes safe can say yes:
        Agent and Team, whose subagents' writing calls take turns on a shared lock."""
        return self.is_read_only(args)

    def parse(self, raw: dict[str, Any]) -> InputT:
        """Validate raw tool arguments.

        Parameters
        ----------
        raw : dict[str, Any]
            Provider-decoded tool arguments.

        Returns
        -------
        InputT
            Validated input model.
        """
        return self.Input.model_validate(raw)

    def input_schema(self) -> dict[str, Any]:
        """Return the provider-ready JSON schema for this tool's input."""
        schema = self.Input.model_json_schema()
        defs = schema.pop("$defs", {})
        return _strip_titles(_inline_refs(schema, defs))

    def spec(self) -> ToolSpec:
        """The tool as the model sees it; each provider adapter formats this its own way."""
        return ToolSpec(self.name, self.description, self.input_schema())


def _inline_refs(schema: Any, defs: dict[str, Any]) -> Any:
    """Replace `{"$ref": "#/$defs/X"}` with X itself; not every provider resolves $ref."""
    if isinstance(schema, list):
        return [_inline_refs(v, defs) for v in schema]
    if not isinstance(schema, dict):
        return schema
    ref = schema.get("$ref", "")
    if ref.startswith("#/$defs/"):
        return _inline_refs(defs[ref.removeprefix("#/$defs/")], defs)
    return {k: _inline_refs(v, defs) for k, v in schema.items()}


def _strip_titles(schema: Any) -> Any:
    """Pydantic adds a `title` to every node; it costs tokens and tells the model nothing."""
    if isinstance(schema, list):
        return [_strip_titles(v) for v in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "title" and isinstance(value, str):
            continue
        if key == "properties" and isinstance(value, dict):
            # Keys here are field names — a field may legitimately be called "title".
            out[key] = {name: _strip_titles(prop) for name, prop in value.items()}
        else:
            out[key] = _strip_titles(value)
    return out


def approved_path(args: ToolInput, user_path: str, ctx: ToolContext) -> Path | None:
    """Where a call should act: the path the permission gate approved, if it went through
    one (resolving again could follow a symlink swapped in since), else `user_path`
    resolved now (tools called directly, without a gate)."""
    return args._approved_path or ctx.workspace.resolve(user_path)
