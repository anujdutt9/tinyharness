from __future__ import annotations

import base64
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
        "Read a UTF-8 text file, or an image (PNG, JPEG, GIF, WebP), which you then see: a "
        "screenshot, a chart, a rendered page. Text comes back as numbered lines (`   42\\tcode`). "
        "Use offset/limit to page through large files."
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
        if b"\x00" in data[:4096]:  # an archive, a binary: text would be noise in the context
            kind = "a PDF: use the Pdf tool" if path.suffix.lower() == ".pdf" else "not text"
            raise ToolError(f"can't read {path.name}: it's a binary file ({kind})")
        lines = data.decode("utf-8", errors="replace").splitlines()

        mark_seen(ctx, path)
        window = lines[args.offset : args.offset + args.limit]
        if not window:
            return "(empty file)"
        numbered = "\n".join(f"{i:>6}\t{line}" for i, line in enumerate(window, start=args.offset + 1))
        remaining = len(lines) - args.offset - len(window)
        return numbered + (f"\n... ({remaining} more lines)" if remaining > 0 else "")
