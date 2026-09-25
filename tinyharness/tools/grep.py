"""Regex search over files. Pure Python, so there's no ripgrep dependency."""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import Field

from tinyharness.security import is_secret_path
from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput
from tinyharness.tools.glob import is_hidden, resolve_search_root

MAX_FILE_BYTES = 2_000_000
BINARY_PROBE_BYTES = 4096
SKIP_DIRS = frozenset({"node_modules", "__pycache__", "venv", "dist", "build"})


class GrepInput(ToolInput):
    pattern: str = Field(description="Python regular expression")
    path: str | None = Field(
        default=None, description="File or directory to search (default: working directory)"
    )
    include: str | None = Field(default=None, description="Only search files matching this glob, e.g. '*.py'")
    output_mode: Literal["content", "files_with_matches", "count"] = Field(
        default="files_with_matches",
        description="content: path:line:text per match; files_with_matches: paths only; count: matches per file",
    )
    case_insensitive: bool = False
    head_limit: int = Field(default=100, gt=0, description="Maximum lines/files returned")


class GrepTool(Tool[GrepInput]):
    name = "Grep"
    description = (
        "Search file contents with a regular expression. Skips binaries, hidden directories, "
        "common build/dependency directories and secret-shaped files."
    )
    Input = GrepInput
    read_only = True

    def describe(self, args: GrepInput) -> str:
        return f"Grep({args.pattern!r} in {args.path or '.'})"

    def target_path(self, args: GrepInput) -> str | None:
        return args.path or "."

    async def run(self, args: GrepInput, ctx: ToolContext) -> str:
        root = resolve_search_root(args.path, ctx)
        try:
            regex = re.compile(args.pattern, re.IGNORECASE if args.case_insensitive else 0)
        except re.error as e:
            raise ToolError(f"invalid regex: {e}") from e

        files = [root] if root.is_file() else list(_iter_searchable(root, args.include))
        if args.output_mode == "files_with_matches":
            return _files_with_matches(files, regex, args.head_limit, root)
        if args.output_mode == "count":
            return _count(files, regex, args.head_limit, root)
        return _content(files, regex, args.head_limit, root)


def _iter_searchable(root: Path, include: str | None) -> Iterator[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel_dirs = p.relative_to(root).parent
        if is_hidden(rel_dirs) or SKIP_DIRS.intersection(rel_dirs.parts):
            continue
        if include and not fnmatch.fnmatch(p.name, include):
            continue
        if is_secret_path(p) or _too_big_or_binary(p):
            continue
        yield p


def _too_big_or_binary(path: Path) -> bool:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return True
        with path.open("rb") as f:
            return b"\x00" in f.read(BINARY_PROBE_BYTES)
    except OSError:
        return True


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _files_with_matches(files: list[Path], regex: re.Pattern[str], limit: int, root: Path) -> str:
    hits: list[str] = []
    for p in files:
        text = _read(p)
        if text is not None and regex.search(text):
            hits.append(str(p))
            if len(hits) >= limit:
                break
    if not hits:
        return f"(no files matched {regex.pattern!r} under {root})"
    return f"{len(hits)} file(s) matched:\n" + "\n".join(hits)


def _count(files: list[Path], regex: re.Pattern[str], limit: int, root: Path) -> str:
    per_file: list[tuple[str, int]] = []
    for p in files:
        text = _read(p)
        if text is not None and (n := len(regex.findall(text))):
            per_file.append((str(p), n))
    if not per_file:
        return f"(0 matches for {regex.pattern!r} under {root})"
    per_file.sort(key=lambda item: -item[1])
    total = sum(n for _, n in per_file)
    lines = "\n".join(f"{n:>5}  {path}" for path, n in per_file[:limit])
    return f"{total} total match(es) across {len(per_file)} file(s):\n{lines}"


def _content(files: list[Path], regex: re.Pattern[str], limit: int, root: Path) -> str:
    out: list[str] = []
    for p in files:
        text = _read(p)
        if text is None:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                out.append(f"{p}:{lineno}:{line}")
                if len(out) >= limit:
                    return f"{len(out)} match(es) (head_limit reached):\n" + "\n".join(out)
    if not out:
        return f"(no matches for {regex.pattern!r} under {root})"
    return f"{len(out)} match(es):\n" + "\n".join(out)
