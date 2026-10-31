from __future__ import annotations

import base64
import io
from pathlib import Path

from pydantic import Field

from tinyharness.agent.history import Image
from tinyharness.images import MAX_SIDE, fit
from tinyharness.security import is_secret_path
from tinyharness.tools.base import (
    Tool,
    ToolContext,
    ToolError,
    ToolInput,
    ToolOutput,
    approved_path,
    mark_seen,
)
from tinyharness.workspace.files import read_bytes

# How each image format starts (its "magic bytes"): trusted over the file's extension.
_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


def image_type(data: bytes) -> str | None:
    """The media type of an image the model can view, from its first bytes; None if not one."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return next((kind for magic, kind in _SIGNATURES if data.startswith(magic)), None)


def load_image(path: Path) -> Image:
    """An image file, for the model to look at (a screenshot you attach, say), shrunk to
    fit every provider's limits (see tinyharness.images). Raises ValueError if it isn't a
    PNG, JPEG, GIF or WebP, or can't be decoded."""
    data = path.read_bytes()
    media_type = image_type(data)
    if media_type is None:
        raise ValueError(f"{path.name} isn't a PNG, JPEG, GIF or WebP image")
    try:
        data, media_type = fit(data, media_type)
    except ValueError as e:
        raise ValueError(f"{path.name}: {e}") from None
    return Image(media_type, base64.b64encode(data).decode("ascii"), path.name)


MAX_BYTES = 50_000  # per Read (~12k tokens): more is shown as pages, with the offset to continue
MAX_LINE_CHARS = 2_000  # a minified line is cut, so one line can't fill the page


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


def pdf_lines(data: bytes, name: str, stop: int) -> tuple[list[str], int, int]:
    """A PDF's text as lines, with a `----- page N -----` line before each page.

    Pages are extracted only until there are more than `stop` lines, so reading the start
    of a long PDF doesn't wait for all of it. Returns (lines, pages read, pages in all).
    """
    from pypdf import PdfReader  # only when a PDF is read: it's slow to import

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        total = len(reader.pages)
    except Exception as e:
        raise ToolError(f"can't read {name}: not a readable PDF ({type(e).__name__}: {e})") from None
    lines: list[str] = []
    read = 0
    for read, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception as e:
            text = f"[couldn't extract this page: {type(e).__name__}]"
        lines.append(f"----- page {read} of {total} -----")
        lines.extend(text.split("\n") if text else ["(no text on this page: it may be a scanned image)"])
        if len(lines) > stop:
            break
    return lines, read, total


class ReadTool(Tool[ReadInput]):
    name = "Read"
    description = (
        "Read a UTF-8 text file; a PDF, as its text page by page; or an image (PNG, JPEG, GIF, "
        "WebP), which you then see: a screenshot, a chart, a rendered page. Text comes back as "
        "numbered lines (`    42\\tcode`); the numbers aren't part of the file. Up to 2000 lines "
        "or 50 KB per call: for more, continue with the offset it gives you."
    )
    Input = ReadInput
    read_only = True

    async def run(self, args: ReadInput, ctx: ToolContext) -> str | ToolOutput:
        path = resolve_readable(args, args.path, ctx)
        try:
            # Not path.read_text(): that would follow a symlink swapped in since the check.
            data = read_bytes(path)
        except OSError as e:
            raise ToolError(f"read failed: {e.strerror or e}") from e
        if (media_type := image_type(data)) is not None:
            try:
                shown, shown_type = fit(data, media_type)
            except ValueError as e:
                raise ToolError(f"can't view {path.name}: {e}") from None
            mark_seen(ctx, path)
            image = Image(shown_type, base64.b64encode(shown).decode("ascii"), path.name)
            note = "" if shown is data else f", shown to you at most {MAX_SIDE} px a side"
            return ToolOutput(
                f"{path.name}: {media_type} image, {len(data) // 1024:,} KB{note or ' (shown to you)'}",
                [image],
            )
        pages = None  # (pages read, pages in all), for a PDF
        if data.startswith(b"%PDF-"):  # its content, not its name, like images
            lines, *counts = pdf_lines(data, path.name, stop=args.offset + args.limit)
            pages = tuple(counts)
        elif b"\x00" in data[:4096]:  # an archive, a binary: text would be noise in the context
            raise ToolError(f"can't read {path.name}: it's a binary file (not text)")
        else:
            text = data.decode("utf-8", errors="replace")
            # Lines as Edit and Grep count them: split at \n only (not at form feeds and the
            # like), with a Windows \r dropped from the end.
            lines = [line.removesuffix("\r") for line in text.removesuffix("\n").split("\n")] if text else []

        mark_seen(ctx, path)
        if not lines:
            return "(empty file)"
        if args.offset >= len(lines):
            return f"(the file has {len(lines)} lines; offset {args.offset} is past the end)"
        page: list[str] = []
        size = 0
        for i, line in enumerate(lines[args.offset : args.offset + args.limit], start=args.offset + 1):
            if len(line) > MAX_LINE_CHARS:
                line = f"{line[:MAX_LINE_CHARS]}… [line cut: {len(line) - MAX_LINE_CHARS} more chars]"
            entry = f"{i:>6}\t{line}"
            entry_bytes = len(entry.encode())
            if page and size + entry_bytes > MAX_BYTES:
                break
            page.append(entry)
            size += entry_bytes + 1
        end = args.offset + len(page)
        if pages is not None and (end < len(lines) or pages[0] < pages[1]):
            # Its length isn't known until every page is extracted: say where it got to.
            marker = next((line for line in reversed(lines[:end]) if line.startswith("----- page ")), "")
            at = marker.removeprefix("----- ").removesuffix(" -----")
            page.append(
                f"... (showing lines {args.offset + 1}-{end}, up to {at}; continue with offset={end})"
            )
        elif end < len(lines):
            page.append(
                f"... (showing lines {args.offset + 1}-{end} of {len(lines)}; continue with offset={end})"
            )
        return "\n".join(page)
