from __future__ import annotations

from pathlib import Path

from pydantic import Field

from tinyharness.security import is_secret_path
from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput


class GlobInput(ToolInput):
    pattern: str = Field(description="Glob pattern, e.g. '**/*.py'. Use '**' to recurse.")
    path: str | None = Field(default=None, description="Directory to search (default: working directory)")
    limit: int = Field(default=200, gt=0, description="Maximum number of results")


def resolve_search_root(path_arg: str | None, ctx: ToolContext) -> Path:
    root = ctx.workspace.resolve(path_arg) if path_arg else ctx.workspace.root
    if root is None:
        raise ToolError(f"invalid path: {path_arg}")
    if not root.exists():
        raise ToolError(f"path not found: {root}")
    return root


def is_hidden(rel: Path) -> bool:
    return any(part.startswith(".") and part not in {".", ".."} for part in rel.parts)


class GlobTool(Tool[GlobInput]):
    name = "Glob"
    description = (
        "Find files by glob pattern, most recently modified first. "
        "Hidden directories and secret-shaped files are skipped."
    )
    Input = GlobInput
    read_only = True

    def describe(self, args: GlobInput) -> str:
        return f"Glob({args.pattern} in {args.path or '.'})"

    def target_path(self, args: GlobInput) -> str | None:
        return args.path or "."

    async def run(self, args: GlobInput, ctx: ToolContext) -> str:
        root = resolve_search_root(args.path, ctx)
        if not root.is_dir():
            raise ToolError(f"not a directory: {root}")
        try:
            matches = [
                p
                for p in root.glob(args.pattern)
                if not is_hidden(p.relative_to(root)) and not is_secret_path(p)
            ]
        except (ValueError, OSError) as e:
            raise ToolError(f"glob failed: {e}") from e

        if not matches:
            return f"(no matches for {args.pattern!r} in {root})"

        matches.sort(key=_mtime, reverse=True)
        shown = matches[: args.limit]
        header = f"{len(matches)} match(es)"
        if len(matches) > args.limit:
            header += f" (showing first {args.limit})"
        return header + ":\n" + "\n".join(_format_entry(p) for p in shown)


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _format_entry(p: Path) -> str:
    if p.is_dir():
        return f"{p}/"
    try:
        return f"{p}    ({p.stat().st_size:,} bytes)"
    except OSError:
        return str(p)
