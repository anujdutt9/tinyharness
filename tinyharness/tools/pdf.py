from __future__ import annotations

import io

from pydantic import Field
from pypdf import PdfReader

from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput, mark_seen
from tinyharness.tools.read import resolve_readable
from tinyharness.workspace.files import read_bytes


class PdfInput(ToolInput):
    path: str = Field(description="Absolute path, or relative to the working directory")
    page_start: int = Field(default=0, ge=0, description="0-indexed first page")
    page_end: int = Field(default=-1, description="Exclusive end page; -1 means the last page")
    max_chars: int = Field(default=200_000, gt=0, description="Cap on returned characters")


class PdfTool(Tool[PdfInput]):
    name = "Pdf"
    description = (
        "Extract text from a PDF, one section per page. Use this for .pdf files; "
        "Read only handles UTF-8 text."
    )
    Input = PdfInput
    read_only = True

    async def run(self, args: PdfInput, ctx: ToolContext) -> str:
        path = resolve_readable(args, args.path, ctx)
        if path.suffix.lower() != ".pdf":
            raise ToolError(f"not a PDF: {path.name}. Use Read for plain text files.")

        try:
            # From bytes read without following symlinks, not a path pypdf would open itself.
            reader = PdfReader(io.BytesIO(read_bytes(path)))
            if reader.is_encrypted:
                reader.decrypt("")
        except Exception as e:
            raise ToolError(f"failed to open PDF: {type(e).__name__}: {e}") from e
        if reader.is_encrypted:
            raise ToolError("PDF is encrypted and could not be opened with an empty password")

        n = len(reader.pages)
        start = args.page_start
        end = n if args.page_end < 0 or args.page_end > n else args.page_end
        if start >= end:
            raise ToolError(f"empty page range {start}..{end} (PDF has {n} pages)")

        mark_seen(ctx, path)
        out = [f"[PDF: {path.name}, {n} pages, extracting {start}..{end}]"]
        used = 0
        for i in range(start, end):
            try:
                text = (reader.pages[i].extract_text() or "").strip()
            except Exception as e:
                text = f"[error extracting page {i}: {type(e).__name__}: {e}]"
            chunk = f"\n\n----- page {i + 1} -----\n{text}"
            if used + len(chunk) > args.max_chars:
                out.append(f"\n\n[truncated at page {i + 1}: max_chars={args.max_chars} reached]")
                break
            out.append(chunk)
            used += len(chunk)
        return "".join(out)
