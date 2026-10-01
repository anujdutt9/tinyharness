"""Show you a file or URL in its app or the browser: the Open tool."""

from tinyharness.ext import Extension
from tinyharness.tools.open import OpenTool


def setup(th: Extension) -> None:
    th.add_tool(OpenTool())
    th.prompt_section("Showing things", "To show the user a file or URL, use the Open tool, not `open`.")
