"""Read PDFs page by page, with their text and images: the Pdf tool. Without it, the
model reads PDFs with Bash (pdftotext, pypdf)."""

from tinyharness.ext import Extension
from tinyharness.tools.pdf import PdfTool


def setup(th: Extension) -> None:
    th.add_tool(PdfTool())
