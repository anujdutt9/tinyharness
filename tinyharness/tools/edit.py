from __future__ import annotations

import re
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
from tinyharness.workspace.files import read_text, write_text


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
        "Copy it from the file without Read's line-number prefixes. Keep it small: the lines "
        "that change plus enough around them to be unique. You must Read the file first in "
        "this session."
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
        path = approved_path(args, args.path, ctx)
        if path is None or not path.is_file():
            raise ToolError(f"file not found: {path or args.path}")
        if str(path) not in ctx.read_paths:
            raise ToolError(f"Read {path} before editing it")
        check_not_stale(ctx, path)
        if args.old_string == args.new_string:
            raise ToolError("old_string and new_string are identical")

        try:
            original = read_text(path)
        except OSError as e:
            raise ToolError(f"read failed: {e.strerror or e}") from e
        except UnicodeDecodeError:
            raise ToolError(f"{path} isn't UTF-8 text; Edit only changes UTF-8 files") from None

        # A file with Windows line endings throughout is edited as if it had \n ones (what
        # the model sees and writes), then written back with its own.
        crlf = "\r\n" in original and "\n" not in original.replace("\r\n", "")
        text = original.replace("\r\n", "\n") if crlf else original
        old, new = args.old_string, args.new_string
        if crlf:
            old, new = old.replace("\r\n", "\n"), new.replace("\r\n", "\n")
        if not old:
            if text:
                raise ToolError(
                    "old_string is empty. To add text, include the lines next to where it goes in "
                    "old_string; to replace the whole file, use Write"
                )
            spans, count, note = None, 1, ""  # filling an empty file
        else:
            spans, count, note, old, new = _locate(text, old, new)
        if count > 1 and not args.replace_all:
            raise ToolError(
                f"old_string is not unique ({count} occurrences); "
                "pass replace_all=true or include more surrounding context"
            )

        replaced = count if args.replace_all else 1
        first = (text.find(old) if old else 0) if spans is None else spans[0][0]
        if spans is None:
            new_text = text.replace(old, new, replaced) if old else new
        else:
            for start, end in reversed(spans[:replaced]):  # from the end, so offsets stay valid
                text = text[:start] + new + text[end:]
            new_text = text
        try:
            write_text(path, new_text.replace("\n", "\r\n") if crlf else new_text)  # all or nothing
        except OSError as e:
            raise ToolError(f"write failed: {e.strerror or e}") from e
        mark_seen(ctx, path)  # the model knows the change it just made
        more = f", {replaced - 1} more not shown" if replaced > 1 else ""
        return (
            f"edited {path} ({replaced} replacement{'s' if replaced != 1 else ''}{note}{more}). "
            f"The changed lines as they are now:\n{snippet(new_text, first, new)}"
        )


# Read's line numbers ("    42\t"), when a model copies them into old_string.
_LINE_NUMBER = re.compile(r"^ *\d+\t", re.MULTILINE)


def _locate(text: str, old: str, new: str) -> tuple[list[tuple[int, int]] | None, int, str, str, str]:
    """Find `old` in `text`: exactly, else with Read's line numbers taken off (and off
    `new` too, if it has them), else ignoring trailing whitespace. Returns (spans for a
    loose match or None for an exact one, how many matches, a note, old, new)."""
    if count := text.count(old):
        return None, count, "", old, new
    if (found := _without_line_numbers(text, old)) is not None:
        bare, at = found
        new_lines = [line for line in new.split("\n") if line]
        if new_lines and all(_LINE_NUMBER.match(line) for line in new_lines):
            new = _LINE_NUMBER.sub("", new)
        return [(at, at + len(bare))], 1, "; Read's line numbers were left out of old_string", bare, new
    if spans := loose_spans(text, old):
        # Models often get invisible whitespace wrong (a blank line written as spaces, a
        # trailing space dropped): same lines otherwise, so edit those lines. A span
        # covers whole lines without the newlines around them, so new_string loses the
        # ones old_string had there, or they'd be doubled.
        lead = len(old) - len(old.lstrip("\n"))
        trail = len(old) - len(old.rstrip("\n"))
        new = new[min(lead, len(new) - len(new.lstrip("\n"))) :]
        if trail:
            new = new[: len(new) - min(trail, len(new) - len(new.rstrip("\n")))]
        return spans, len(spans), "; matched ignoring trailing whitespace", old, new
    raise ToolError(f"old_string not found. {explain_mismatch(text, old)}")


def _without_line_numbers(text: str, old: str) -> tuple[str, int] | None:
    """`old` without Read's line-number prefixes, and where it is in `text`: only if every
    line has one and they are the real numbers of the lines it matches (so a TSV row
    starting "3\t" isn't mistaken for line 3). None otherwise."""
    body = old.rstrip("\n")
    numbers = [_LINE_NUMBER.match(line) for line in body.split("\n")]
    if not body or not all(numbers):
        return None
    lines = [int(m.group().strip()) for m in numbers if m]
    if lines != list(range(lines[0], lines[0] + len(lines))):
        return None
    bare = _LINE_NUMBER.sub("", body) + old[len(body) :]
    start = 0
    while (at := text.find(bare, start)) >= 0:
        if text.count("\n", 0, at) + 1 == lines[0] and (at == 0 or text[at - 1] == "\n"):
            return bare, at
        start = at + 1
    return None


SNIPPET_CONTEXT = 3  # unchanged lines shown before and after an edit
SNIPPET_MAX = 30  # lines of a long replacement shown (its start and end)


def snippet(text: str, start: int, inserted: str) -> str:
    """The edited region of `text` with a few lines around it, numbered like Read: so the
    model's next edit can copy from the file as it is now, without reading it all again."""
    lines = text.split("\n")
    first = text.count("\n", 0, start)
    last = first + inserted.count("\n")
    lo, hi = max(first - SNIPPET_CONTEXT, 0), min(last + SNIPPET_CONTEXT, len(lines) - 1)
    numbered = [f"{i + 1:>6}\t{lines[i]}" for i in range(lo, hi + 1)]
    if len(numbered) > SNIPPET_MAX:
        half = SNIPPET_MAX // 2
        numbered = [*numbered[:half], f"   …  ({len(numbered) - SNIPPET_MAX} lines)", *numbered[-half:]]
    return "\n".join(numbered)


def _lines_with_offsets(text: str) -> list[tuple[int, str]]:
    offsets, start = [], 0
    for line in text.split("\n"):
        offsets.append((start, line))
        start += len(line) + 1
    return offsets


def loose_spans(text: str, old: str) -> list[tuple[int, int]]:
    """Where `old` appears as whole lines once each line's trailing whitespace is ignored
    (spaces, tabs, a CR): (start, end) character offsets in `text`, non-overlapping.
    Leading whitespace still has to match: indentation is meaningful."""
    wanted = [line.rstrip() for line in old.strip("\n").split("\n")]
    if not any(wanted):
        return []
    lines = _lines_with_offsets(text)
    spans: list[tuple[int, int]] = []
    i = 0
    while i + len(wanted) <= len(lines):
        if all(lines[i + k][1].rstrip() == w for k, w in enumerate(wanted)):
            start, (last_start, last) = lines[i][0], lines[i + len(wanted) - 1]
            spans.append((start, last_start + len(last)))
            i += len(wanted)
        else:
            i += 1
    return spans


def explain_mismatch(text: str, old: str) -> str:
    """Where the closest copy of `old` stops matching the file, so the next attempt can
    fix that line instead of guessing (seen live: 11 tries at one edit, each failing on
    a blank line written as four spaces)."""
    wanted = old.strip("\n").split("\n")
    lines = [line for _, line in _lines_with_offsets(text)]
    first = wanted[0].strip()
    starts = [i for i, line in enumerate(lines) if first and first in line]
    if not starts:
        return (
            f"Its first line isn't in the file: {wanted[0].strip()[:120]!r}. Read the file "
            "again and copy the text exactly."
        )

    def matched(i: int) -> int:
        n = 0
        while n < len(wanted) and i + n < len(lines) and lines[i + n].rstrip() == wanted[n].rstrip():
            n += 1
        return n

    best = max(starts, key=matched)
    n = matched(best)
    if n == 0:  # the first line is in the file, but as part of a longer line
        n = 1
    if best + n >= len(lines):
        return f"The file ends at line {len(lines)}, before old_string does."
    return (
        f"The closest match starts at line {best + 1}; lines {best + 1}-{best + n} match, then "
        f"line {n + 1} of old_string differs from line {best + n + 1} of the file:\n"
        f"  old_string: {wanted[n][:200]!r}\n  file:       {lines[best + n][:200]!r}\n"
        "Copy that part of the file exactly (Read it again if needed), or use a smaller old_string."
    )
