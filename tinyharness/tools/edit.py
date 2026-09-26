from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from tinyharness.tools.base import LARGE_CHANGE, Tool, ToolContext, ToolError, ToolInput


class EditInput(ToolInput):
    path: str = Field(description="Absolute path, or relative to the working directory")
    old_string: str = Field(description="Exact text to replace; must be unique unless replace_all")
    new_string: str = Field(description="Replacement text")
    replace_all: bool = Field(default=False, description="Replace every occurrence")


class EditTool(Tool[EditInput]):
    name = "Edit"
    description = (
        "Replace `old_string` with `new_string` in a file. `old_string` must match exactly "
        "and be unique unless replace_all=true; include surrounding lines to disambiguate. "
        "You must Read the file first in this session."
    )
    Input = EditInput

    def risk(self, args: dict[str, Any], path: Path | None) -> str | None:
        removed = len(str(args.get("old_string", "")).splitlines()) - len(
            str(args.get("new_string", "")).splitlines()
        )
        return f"deletes {removed} lines" if removed >= LARGE_CHANGE else None

    def describe(self, args: EditInput) -> str:
        removed, added = len(args.old_string.splitlines()), len(args.new_string.splitlines())
        suffix = " all" if args.replace_all else ""
        return f"Edit({args.path}, -{removed}/+{added} lines{suffix})"

    async def run(self, args: EditInput, ctx: ToolContext) -> str:
        path = ctx.workspace.resolve(args.path)
        if path is None or not path.is_file():
            raise ToolError(f"file not found: {path or args.path}")
        if str(path) not in ctx.read_paths:
            raise ToolError(f"Read {path} before editing it")
        if args.old_string == args.new_string:
            raise ToolError("old_string and new_string are identical")

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            raise ToolError(f"read failed: {e}") from e

        count = text.count(args.old_string)
        if count == 0:
            raise ToolError("old_string not found")
        if count > 1 and not args.replace_all:
            raise ToolError(
                f"old_string is not unique ({count} occurrences); "
                "pass replace_all=true or include more surrounding context"
            )

        replaced = count if args.replace_all else 1
        try:
            path.write_text(text.replace(args.old_string, args.new_string, replaced), encoding="utf-8")
        except OSError as e:
            raise ToolError(f"write failed: {e}") from e
        return f"edited {path} ({replaced} replacement{'s' if replaced != 1 else ''})"
