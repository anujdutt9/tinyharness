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
        "Plan and track multi-step work. Pass the FULL list every time (it replaces the old one). "
        "Keep exactly one item in_progress; mark items completed as soon as they are done."
    )
    Input = TodoInput
    read_only = True
    needs_permission = False

    def describe(self, args: TodoInput) -> str:
        return f"TodoWrite({len(args.todos)} item(s))"

    async def run(self, args: TodoInput, ctx: ToolContext) -> str:
        ctx.todos = [t.model_dump(by_alias=True) for t in args.todos]
        return format_todos(ctx.todos)
