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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, PrivateAttr

from tinyharness.llm.types import ToolSpec
from tinyharness.llm.usage import UsageMeter

if TYPE_CHECKING:
    from collections.abc import Callable

    from tinyharness.config import Config
    from tinyharness.events import Event
    from tinyharness.workspace import Scratch, Workspace


LARGE_CHANGE = 20  # lines; removing more than this counts as destructive (auto mode asks)


class ToolError(Exception):
    """A tool failed in a way the model should see and can recover from."""


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # Set by `Tool.pin_path`: the resolved path the permission gate approved.
    _approved_path: Path | None = PrivateAttr(default=None)


InputT = TypeVar("InputT", bound=ToolInput)


@dataclass
class ToolContext:
    """Per-session state shared by all tool calls."""

    workspace: Workspace
    config: Config
    scratch: Scratch | None = None
    read_paths: set[str] = field(default_factory=set)  # enforces read-before-write
    todos: list[dict[str, str]] = field(default_factory=list)
    notify: Callable[[Event], None] | None = None  # side-channel events, e.g. subagent progress
    usage: UsageMeter = field(default_factory=UsageMeter)  # tokens and cost, subagents included


class Tool(ABC, Generic[InputT]):
    name: ClassVar[str]
    description: str
    Input: type[InputT]
    read_only: ClassVar[bool] = False
    needs_permission: ClassVar[bool] = True
    network: ClassVar[bool] = False

    @abstractmethod
    async def run(self, args: InputT, ctx: ToolContext) -> str: ...

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

    def risk(self, args: dict[str, Any], path: Path | None) -> str | None:
        """Why this call is destructive enough to always ask in auto mode, or None.
        Gets the raw arguments, since it runs before they're validated."""
        return None

    def contained(self, args: dict[str, Any]) -> bool:
        """Whether this call runs inside the OS sandbox (writes only to the project, no
        network), so modes that allow project edits may allow it without asking."""
        return False

    def escapes(self, args: dict[str, Any]) -> str | None:
        """Why this call runs outside the sandbox it would normally run in, or None.
        Such calls always ask: no rule, mode or classifier approves them (only
        bypassPermissions or a hook), and "don't ask again" isn't offered."""
        return None

    def is_read_only(self, args: InputT) -> bool:
        """Whether this call has no side effects (safe to run alongside others)."""
        return self.read_only

    def parse(self, raw: dict[str, Any]) -> InputT:
        return self.Input.model_validate(raw)

    def input_schema(self) -> dict[str, Any]:
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
