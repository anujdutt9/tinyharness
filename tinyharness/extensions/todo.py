"""Durable task state for work with several steps: the TodoWrite tool."""

from tinyharness.ext import Extension
from tinyharness.tools.todo import TodoWriteTool


def setup(th: Extension) -> None:
    th.add_tool(TodoWriteTool())
    th.prompt_section(
        "Planning",
        "For work with several distinct steps, keep a short plan with TodoWrite; the user sees "
        "it. Skip it for simple work. The tool's description says how to keep it useful.",
    )
