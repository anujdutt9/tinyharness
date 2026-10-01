from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tinyharness.tools.base import Tool, ToolContext, ToolInput

Status = Literal["pending", "in_progress", "completed"]
_ICONS = {"pending": "○", "in_progress": "◐", "completed": "✔"}


class TodoItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    content: str = Field(min_length=1, description="Imperative form, e.g. 'Run tests'")
    active_form: str = Field(
        min_length=1, alias="activeForm", description="Present progressive, e.g. 'Running tests'"
    )
    status: Status


class TodoInput(ToolInput):
    todos: list[TodoItem] = Field(description="The complete list; replaces the previous one")


def format_todos(todos: list[dict[str, str]]) -> str:
    """Render a todo list as a compact status block."""
    if not todos:
        return "(todo list cleared)"
    counts = {s: sum(1 for t in todos if t["status"] == s) for s in _ICONS}
    summary = f"{counts['completed']}/{len(todos)} done"
    if counts["in_progress"]:
        summary += f", {counts['in_progress']} in progress"
    if counts["pending"]:
        summary += f", {counts['pending']} pending"
    lines = [
        f"  {_ICONS[t['status']]} {t['activeForm'] if t['status'] == 'in_progress' else t['content']}"
        for t in todos
    ]
    return "\n".join([summary, *lines])


class TodoWriteTool(Tool[TodoInput]):
    name = "TodoWrite"
    description = (
        "A short plan of the work, shown to the user. Pass the FULL list every time (it "
        "replaces the old one).\n\n"
        "Use it when the work has several distinct steps (not several tool calls for one "
        "step), or the user gives you several things to do or asks for a plan.\n\n"
        "Skip it when a plan adds nothing:\n"
        "- a question or explanation: 'what does this function do?'\n"
        "- a single change: 'add a docstring to parse()', 'rename this variable'\n"
        "- one command: 'run the tests and tell me what failed'\n"
        "- one artifact written in one go: 'write a script that renames these files'\n\n"
        "A good plan:\n"
        "- has a few coarse steps of about 5-7 words, each one you could check was done; no "
        "filler steps ('explore the code', 'plan the approach': the list is the plan)\n"
        "- matches how you'll work: writing a whole file in one call is one step\n\n"
        "Keeping it current: mark a step in_progress when you start it and completed when it's "
        "done; at most one in_progress. If one call finishes several steps, mark them all "
        "completed together. Change the plan when the work changes. Don't repeat the list in "
        "your reply: the user already sees it."
    )
    Input = TodoInput
    read_only = True
    needs_permission = False

    def describe(self, args: TodoInput) -> str:
        return f"TodoWrite({len(args.todos)} item(s))"

    async def run(self, args: TodoInput, ctx: ToolContext) -> str:
        ctx.todos = [t.model_dump(by_alias=True) for t in args.todos]
        return format_todos(ctx.todos)
