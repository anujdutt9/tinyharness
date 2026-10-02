"""Search the web and read pages: WebSearch and WebFetch."""

from tinyharness.ext import Extension
from tinyharness.tools.web import WebFetchTool, WebSearchTool


def setup(th: Extension) -> None:
    session = th.session
    if session is not None:  # WebSearch digests results with the session's model
        th.add_tool(WebSearchTool(lambda: session.llm))
    th.add_tool(WebFetchTool())
    th.prompt_section(
        "Web",
        "Use WebSearch for anything current or that you don't know, and WebFetch to read a "
        "page in full. Cite the pages you rely on.",
    )
