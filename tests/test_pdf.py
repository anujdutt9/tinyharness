import pytest
from pypdf import PdfWriter

from conftest import call
from tinyharness.tools.base import ToolError
from tinyharness.tools.pdf import PdfTool


def make_pdf(path, pages: int) -> None:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=300, height=300)
    with path.open("wb") as f:
        writer.write(f)


async def test_pdf_extracts_each_page(ctx, tmp_path):
    """Blank pages extract as empty text; the tool must still report every page."""
    make_pdf(tmp_path / "x.pdf", 2)
    out = await call(PdfTool(), {"path": "x.pdf"}, ctx)
    assert out.startswith("[PDF: x.pdf, 2 pages")
    assert "page 1" in out and "page 2" in out


async def test_pdf_page_range(ctx, tmp_path):
    make_pdf(tmp_path / "x.pdf", 3)
    out = await call(PdfTool(), {"path": "x.pdf", "page_start": 1, "page_end": 2}, ctx)
    assert "page 2" in out and "page 1" not in out and "page 3" not in out


async def test_pdf_missing_file(ctx):
    with pytest.raises(ToolError, match="not found"):
        await call(PdfTool(), {"path": "/nonexistent/file.pdf"}, ctx)


async def test_pdf_rejects_non_pdf(ctx, tmp_path):
    (tmp_path / "notes.md").write_text("not a pdf")
    with pytest.raises(ToolError, match="not a PDF"):
        await call(PdfTool(), {"path": "notes.md"}, ctx)
