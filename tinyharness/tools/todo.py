from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput

Status = Literal["pending", "in_progress", "blocked", "completed", "cancelled"]
_ICONS = {"pending": "○", "in_progress": "◐", "blocked": "!", "completed": "✔", "cancelled": "×"}


class TodoItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(
        default="",
        description="Stable task ID. Preserve an existing ID when updating the full list.",
    )
    content: str = Field(min_length=1, description="Imperative form, e.g. 'Run tests'")
    active_form: str = Field(
        min_length=1, alias="activeForm", description="Present progressive, e.g. 'Running tests'"
    )
    status: Status
    result: str = Field(
        default="",
        description="Brief evidence or outcome for a completed, blocked, or cancelled task",
    )


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
    if counts["blocked"]:
        summary += f", {counts['blocked']} blocked"
    if counts["cancelled"]:
        summary += f", {counts['cancelled']} cancelled"
    lines = [
        f"  {_ICONS[t['status']]} [{t['id']}] "
        f"{t['activeForm'] if t['status'] == 'in_progress' else t['content']}"
        + (f" — {t['result']}" if t.get("result") else "")
        for t in todos
    ]
    return "\n".join([summary, *lines])


def normalize_todo_ids(todos: list[dict[str, str]]) -> list[dict[str, str]]:
    """Give old or malformed saved tasks unique IDs before they re-enter the loop."""
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for saved in todos:
        task = dict(saved)
        task_id = str(task.get("id", "")).strip()
        if not task_id or task_id in seen:
            task_id = f"task-{uuid4().hex[:8]}"
        task["id"] = task_id
        seen.add(task_id)
        normalized.append(task)
    return normalized


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
        "Keeping it current: preserve each task's id; mark a step in_progress when you start it "
        "and completed when it's done; at most one in_progress. Use blocked when work cannot "
        "continue and cancelled when it no longer applies. Put brief evidence or the reason in "
        "result. If one call finishes several steps, update them together. Change the plan when "
        "the work changes. Don't repeat the list in your reply: the user already sees it."
    )
    Input = TodoInput
    read_only = True
    needs_permission = False

    def describe(self, args: TodoInput) -> str:
        return f"TodoWrite({len(args.todos)} item(s))"

    async def run(self, args: TodoInput, ctx: ToolContext) -> str:
        if sum(t.status == "in_progress" for t in args.todos) > 1:
            raise ToolError("At most one task can be in_progress; update the list and try again.")
        previous = {t.get("content", ""): t.get("id", "") for t in ctx.todos}
        todos: list[dict[str, str]] = []
        for item in args.todos:
            task = item.model_copy()
            task.id = task.id.strip() or previous.get(task.content, "") or f"task-{uuid4().hex[:8]}"
            todos.append(task.model_dump(by_alias=True, exclude_defaults=True))
        ids = [t["id"] for t in todos]
        if len(ids) != len(set(ids)):
            raise ToolError("Task IDs must be unique; preserve each task's own ID when updating the list.")
        ctx.todos = todos
        return format_todos(ctx.todos)
