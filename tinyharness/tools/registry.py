from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tinyharness.permissions.modes import PermissionMode
from tinyharness.tools.base import Tool
from tinyharness.tools.edit import EditTool
from tinyharness.tools.glob import GlobTool
from tinyharness.tools.grep import GrepTool
from tinyharness.tools.open import OpenTool
from tinyharness.tools.read import ReadTool
from tinyharness.tools.write import WriteTool

AnyTool = Tool[Any]


class ToolRegistry:
    """The set of tools a session can offer to the model."""

    def __init__(self, tools: Iterable[AnyTool] = ()) -> None:
        self._tools: dict[str, AnyTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: AnyTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> AnyTool | None:
        return self._tools.get(name)

    def remove(self, name: str) -> None:
        self._tools.pop(name, None)

    def names(self) -> list[str]:
        return list(self._tools)

    def with_tools(self, extra: Iterable[AnyTool]) -> ToolRegistry:
        """A copy with more tools (e.g. a team's board for its members)."""
        return ToolRegistry([*self._tools.values(), *extra])

    def available(self, mode: PermissionMode, allowed: set[str] | None = None) -> list[AnyTool]:
        """Tools to offer the model: filtered by an allow-list and, in plan mode, to read-only tools."""
        return [
            tool
            for name, tool in self._tools.items()
            if (allowed is None or name in allowed) and (mode != PermissionMode.PLAN or tool.read_only)
        ]


def default_tools() -> list[AnyTool]:
    """The core tools that do not need session-owned dependencies.

    Bash is added by the session because it needs the active sandbox.
    """
    return [ReadTool(), WriteTool(), EditTool(), GlobTool(), GrepTool(), OpenTool()]


def default_registry() -> ToolRegistry:
    return ToolRegistry(default_tools())
