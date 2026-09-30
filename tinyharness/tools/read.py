from __future__ import annotations

from pathlib import Path

from pydantic import Field

from tinyharness.security import is_secret_path
from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput, approved_path
from tinyharness.workspace.files import read_text


class ReadInput(ToolInput):
    path: str = Field(description="Absolute path, or relative to the working directory")
    offset: int = Field(default=0, ge=0, description="0-indexed line to start at")
    limit: int = Field(default=2000, gt=0, description="Maximum number of lines to return")


def resolve_readable(args: ToolInput, path_arg: str, ctx: ToolContext) -> Path:
    """The path to read (the one the permission gate approved, see `approved_path`),
    refusing credential-shaped files."""
    path = approved_path(args, path_arg, ctx)
    if path is None:
        raise ToolError(f"invalid path: {path_arg}")
    if is_secret_path(path):
        raise ToolError(
            f"refusing to read secret-shaped path: {path}. "
            "Set TINYHARNESS_ALLOW_SECRET_READS=1 if you really mean it."
        )
    if not path.exists():
        raise ToolError(f"file not found: {path}")
    if not path.is_file():
        raise ToolError(f"not a file: {path}")
    return path


class ReadTool(Tool[ReadInput]):
    name = "Read"
    description = (
        "Read a UTF-8 text file. Returns numbered lines (`   42\\tcode`). "
        "Use offset/limit to page through large files."
    )
    Input = ReadInput
    read_only = True

    async def run(self, args: ReadInput, ctx: ToolContext) -> str:
        path = resolve_readable(args, args.path, ctx)
        try:
            # Not path.read_text(): that would follow a symlink swapped in since the check.
            lines = read_text(path, errors="replace").splitlines()
        except OSError as e:
            raise ToolError(f"read failed: {e.strerror or e}") from e

        ctx.read_paths.add(str(path))
        window = lines[args.offset : args.offset + args.limit]
        if not window:
            return "(empty file)"
        numbered = "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(window, start=args.offset + 1))
        remaining = len(lines) - args.offset - len(window)
        return numbered + (f"\n... ({remaining} more lines)" if remaining > 0 else "")
