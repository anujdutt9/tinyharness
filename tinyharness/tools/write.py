from __future__ import annotations

from pydantic import Field

from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput


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

    def describe(self, args: WriteInput) -> str:
        return f"Write({args.path}, {len(args.content.splitlines())} lines)"

    async def run(self, args: WriteInput, ctx: ToolContext) -> str:
        path = ctx.workspace.resolve(args.path)
        if path is None:
            raise ToolError(f"invalid path: {args.path}")
        if path.exists() and str(path) not in ctx.read_paths:
            raise ToolError(f"refusing to overwrite {path}: Read it first in this session before Write")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args.content, encoding="utf-8")
        except OSError as e:
            raise ToolError(f"write failed: {e}") from e
        ctx.read_paths.add(str(path))
        return f"wrote {path} ({len(args.content):,} bytes)"
