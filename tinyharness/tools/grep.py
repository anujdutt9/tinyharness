"""Regex search over files. Pure Python, so there's no ripgrep dependency."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import Field

from tinyharness.security import is_secret_path
from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput
from tinyharness.tools.glob import is_hidden, resolve_search_root
from tinyharness.workspace.files import read_bytes

MAX_FILE_BYTES = 2_000_000  # in a directory walk: bigger files are data or bundles, not code
# A file named directly is searched up to this size (it's read whole). Seen live: a 2.2 MB
# session log was refused and the model was told to page through it with Read instead.
MAX_NAMED_FILE_BYTES = 64_000_000
BINARY_PROBE_BYTES = 4096
SKIP_DIRS = frozenset({"node_modules", "__pycache__", "venv", "dist", "build", "target"})
MAX_LINE_CHARS = 300  # a match in a minified file would otherwise fill the context


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
        "common build/dependency directories and secret-shaped files. Use it instead of grep or rg "
        "in Bash."
    )
    Input = GrepInput
    read_only = True
    taken_from_json = True

    def describe(self, args: GrepInput) -> str:
        return f"Grep({args.pattern!r} in {args.path or '.'})"

    def target_path(self, args: GrepInput) -> str | None:
        return args.path or "."

    async def run(self, args: GrepInput, ctx: ToolContext) -> str:
        root = resolve_search_root(args.path, ctx, args)
        try:
            # MULTILINE: `^def main` means a line starting with it, in every output mode
            # (content mode matches line by line; the others search whole files).
            regex = re.compile(args.pattern, re.MULTILINE | (re.IGNORECASE if args.case_insensitive else 0))
        except re.error as e:
            raise ToolError(f"invalid regex: {e}") from e
        # A big tree takes a while: off the event loop, so other calls and Ctrl-C aren't held up.
        return await asyncio.to_thread(_search, root, regex, args)


def _search(root: Path, regex: re.Pattern[str], args: GrepInput) -> str:
    if root.is_file():
        # A file named directly gets the same checks a directory walk applies to each file.
        if is_secret_path(root):
            raise ToolError(
                f"refusing to search secret-shaped path: {root}. "
                "Set TINYHARNESS_ALLOW_SECRET_READS=1 if you really mean it."
            )
        _check_searchable(root)
        files, max_bytes = [root], MAX_NAMED_FILE_BYTES
    else:
        files, max_bytes = list(_iter_searchable(root, args.include)), MAX_FILE_BYTES
    if args.output_mode == "files_with_matches":
        return _files_with_matches(files, regex, args.head_limit, root, max_bytes)
    if args.output_mode == "count":
        return _count(files, regex, args.head_limit, root, max_bytes)
    return _content(files, regex, args.head_limit, root, max_bytes)


def _iter_searchable(root: Path, include: str | None) -> Iterator[Path]:
    """Files under `root` worth searching, in a stable order. Symlinks aren't followed,
    to files or directories (as ripgrep does by default): the secret-path check sees a
    link's own name, so `leak -> ~/.aws/credentials` would otherwise be read."""
    for dirpath, dirnames, filenames in os.walk(root):  # doesn't descend into linked dirs
        here = Path(dirpath)
        rel = here.relative_to(root)
        dirnames[:] = sorted(
            d for d in dirnames if not (is_hidden(rel / d) or d in SKIP_DIRS or (here / d).is_symlink())
        )
        for name in sorted(filenames):
            p = here / name
            if include and not fnmatch.fnmatch(name, include):
                continue
            if p.is_symlink() or is_secret_path(p):
                continue
            yield p


def _load(path: Path, max_bytes: int = MAX_FILE_BYTES) -> str | None:
    """The file's text, or None to skip it: unreadable, now a symlink, too big or binary.
    One read, without following symlinks, so the file checked is the file searched."""
    try:
        data = read_bytes(path, limit=max_bytes + 1)
    except OSError:
        return None
    if len(data) > max_bytes or b"\x00" in data[:BINARY_PROBE_BYTES]:
        return None
    return data.decode("utf-8", errors="replace")


def _check_searchable(path: Path) -> None:
    """The reason `_load` would skip this file, as an error."""
    try:
        data = read_bytes(path, limit=MAX_NAMED_FILE_BYTES + 1)
    except OSError as e:
        raise ToolError(f"can't read {path}: {e.strerror or e}") from None
    if len(data) > MAX_NAMED_FILE_BYTES:
        size = path.stat().st_size
        raise ToolError(
            f"file too large to search: {path} is {size / 1_000_000:.0f} MB; Grep reads a file whole, "
            f"up to {MAX_NAMED_FILE_BYTES // 1_000_000} MB. Search it with Bash instead, which streams "
            f"it: rg -n 'PATTERN' {path}"
        )
    if b"\x00" in data[:BINARY_PROBE_BYTES]:
        raise ToolError(f"not a text file: {path} looks binary; Grep only searches text")


def _files_with_matches(
    files: list[Path], regex: re.Pattern[str], limit: int, root: Path, max_bytes: int = MAX_FILE_BYTES
) -> str:
    hits: list[str] = []
    for p in files:
        text = _load(p, max_bytes)
        if text is not None and regex.search(text):
            hits.append(str(p))
            if len(hits) >= limit:
                break
    if not hits:
        return f"(no files matched {regex.pattern!r} under {root})"
    more = " (head_limit reached: more files may match)" if len(hits) >= limit else ""
    return f"{len(hits)} file(s) matched{more}:\n" + "\n".join(hits)


def _count(
    files: list[Path], regex: re.Pattern[str], limit: int, root: Path, max_bytes: int = MAX_FILE_BYTES
) -> str:
    per_file: list[tuple[str, int]] = []
    for p in files:
        text = _load(p, max_bytes)
        if text is not None and (n := len(regex.findall(text))):
            per_file.append((str(p), n))
    if not per_file:
        return f"(0 matches for {regex.pattern!r} under {root})"
    per_file.sort(key=lambda item: -item[1])
    total = sum(n for _, n in per_file)
    lines = "\n".join(f"{n:>5}  {path}" for path, n in per_file[:limit])
    return f"{total} total match(es) across {len(per_file)} file(s):\n{lines}"


def _content(
    files: list[Path], regex: re.Pattern[str], limit: int, root: Path, max_bytes: int = MAX_FILE_BYTES
) -> str:
    out: list[str] = []
    for p in files:
        text = _load(p, max_bytes)
        if text is None:
            continue
        for lineno, line in enumerate(text.split("\n"), start=1):  # numbered as Read and Edit number them
            if regex.search(line):
                if len(line) > MAX_LINE_CHARS:
                    line = f"{line[:MAX_LINE_CHARS]}… [{len(line) - MAX_LINE_CHARS} more chars]"
                out.append(f"{p}:{lineno}:{line}")
                if len(out) >= limit:
                    return f"{len(out)} match(es) (head_limit reached):\n" + "\n".join(out)
    if not out:
        return f"(no matches for {regex.pattern!r} under {root})"
    return f"{len(out)} match(es):\n" + "\n".join(out)
