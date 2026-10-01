from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from tinyharness.tools.base import (
    LARGE_CHANGE,
    Tool,
    ToolContext,
    ToolError,
    ToolInput,
    approved_path,
    check_not_stale,
    mark_seen,
)
from tinyharness.workspace.files import write_text


class WriteInput(ToolInput):
    path: str = Field(description="Absolute path, or relative to the working directory")
    content: str = Field(description="Full file contents")


class WriteTool(Tool[WriteInput]):
    name = "Write"
    description = (
        "Create or overwrite a UTF-8 file. Parent directories are created as needed. "
        "To overwrite an existing file you must Read it first in this session."
    )
    Input = WriteInput

    def risk(self, args: dict[str, Any], path: Path | None) -> str | None:
        """Replacing most of an existing file is as good as deleting it."""
        if path is None or not path.is_file():
            return None
        try:
            before = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
        except OSError:
            return None
        after = len(str(args.get("content", "")).splitlines())
        if before >= LARGE_CHANGE and after < before / 2:
            return f"replaces a {before}-line file with {after} lines"
        return None

    def describe(self, args: WriteInput) -> str:
        return f"Write({args.path}, {len(args.content.splitlines())} lines)"

    async def run(self, args: WriteInput, ctx: ToolContext) -> str:
        path = approved_path(args, args.path, ctx)
        if path is None:
            raise ToolError(f"invalid path: {args.path}")
        if path.exists() and str(path) not in ctx.read_paths:
            raise ToolError(f"refusing to overwrite {path}: Read it first in this session before Write")
        if path.exists():
            check_not_stale(ctx, path)  # overwriting it would lose a change this agent never saw
        try:
            write_text(path, args.content, make_parents=True)  # all or nothing; see workspace/files.py
        except OSError as e:
            raise ToolError(f"write failed: {e.strerror or e}") from e
        mark_seen(ctx, path)
        return f"wrote {path} ({len(args.content):,} bytes)"
