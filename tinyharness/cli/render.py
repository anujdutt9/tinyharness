"""Turning the agent's event stream into terminal output.

Output is split in two:

    permanent   finished Markdown blocks, completed tool calls, todo lists.
                Printed once, scrolls up like normal terminal output.
    live        the Markdown block still streaming, tool calls still running,
                and the spinner. Redrawn in place at the bottom of the screen.

    ● Read  src/app.py
      ⎿ Read 120 lines
    ● Edit  src/app.py, -1/+4 lines
      ⎿ edited src/app.py (1 replacement)
    ⠼ Working… 4s · ctrl+c to interrupt

The spinner says "Working" while waiting on the model, and "Thinking" only once the
model reports that it is actually reasoning (many replies involve none).
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import dataclass

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

from tinyharness.cli import approval, markdown
from tinyharness.cli.theme import BRAND, ELBOW, split_summary, tool_line
from tinyharness.events import (
    AssistantText,
    Compacted,
    Done,
    Event,
    PermissionDecision,
    Reasoning,
    TodosUpdated,
    ToolResult,
    ToolsDispatched,
    ToolUse,
)
from tinyharness.permissions import Decision, PermissionRequest

WAITING = "Working"

# Decisions worth surfacing even though the user wasn't asked.
_NOTABLE_AUTO_ALLOW = {"rule", "hook", "classifier"}
_REJECTED = "The user rejected this tool call."

_COMPACTION_MESSAGES = {
    "budget_cap": "trimmed {n} oversized message(s)",
    "microcompact": "cleared {n} old tool result(s)",
    "summary": "summarized older messages to free up context",
}


def summarize_output(tool: str, output: str) -> str:
    """One line describing a tool result. The model still gets the full output."""
    text = output.strip()
    if not text:
        return "done"
    lines = text.splitlines()
    if tool == "Read":
        return f"Read {len(lines)} lines"
    if tool in ("Glob", "Grep"):  # they start with a summary: "3 match(es):" or "(no matches …)"
        return "no matches" if text.startswith("(") else lines[0].rstrip(":")
    first = lines[0]
    return first if len(first) <= 100 else first[:99] + "…"


def format_todos(todos: list[dict[str, str]]) -> Text:
    """The todo list as a checklist: done items struck through, the current one highlighted."""
    out = Text()
    if not todos:
        return Text("list cleared", style="muted")
    for i, todo in enumerate(todos):
        prefix = "    " if i else ""  # align under the ⎿ of the first line
        match todo["status"]:
            case "completed":
                out.append(f"{prefix}✔ ", style="ok").append(todo["content"], style="muted strike")
            case "in_progress":
                out.append(f"{prefix}◼ ", style="brand").append(todo["activeForm"], style="bold")
            case _:
                out.append(f"{prefix}◻ ", style="muted").append(todo["content"])
        out.append("\n")
    out.rstrip()
    return out


@dataclass
class _RunningTool:
    summary: str
    note: str = ""  # e.g. "allowed by rule", shown after the result

    @property
    def name(self) -> str:
        return split_summary(self.summary)[0]


class Renderer:
    def __init__(self, console: Console) -> None:
        self.console = console
        self.root: str | None = None  # "<working dir>/"; stripped so paths display relative
        self._live: Live | None = None
        self._spinner = Spinner("dots", style=BRAND)
        self._activity: str | None = None
        self._activity_since = 0.0
        self._streaming = ""  # the Markdown block still being written
        self._running: list[_RunningTool] = []
        self._tools_used = 0
        self._turn_started = 0.0

    # -- the live region ------------------------------------------------------

    def __rich__(self) -> RenderableType:
        parts: list[RenderableType] = []
        if self._streaming.strip():
            parts.append(markdown.render(self._streaming))
        parts.extend(tool_line(self._relative(t.summary), "subtle") for t in self._running)
        if self._activity:
            elapsed = int(time.monotonic() - self._activity_since)
            self._spinner.update(
                text=Text.assemble(
                    (f"{self._activity}…", BRAND), (f" {elapsed}s · ctrl+c to interrupt", "subtle")
                )
            )
            parts.append(self._spinner)
        return Group(*parts)

    def _start_live(self) -> None:
        if self._live is None:
            self._live = Live(self, console=self.console, refresh_per_second=12, transient=True)
            self._live.start()

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def wait(self, label: str) -> None:
        if label != self._activity:
            self._activity, self._activity_since = label, time.monotonic()

    def stop_waiting(self) -> None:
        self._activity = None

    # -- permanent output -----------------------------------------------------

    def _print(self, renderable: RenderableType) -> None:
        self.console.print(renderable, highlight=False)

    def _block(self, renderable: RenderableType) -> None:
        """Print a block with a blank line before it, so turns read as distinct steps."""
        self.console.print()
        self._print(renderable)

    def _flush_text(self, everything: bool) -> None:
        done, rest = (self._streaming, "") if everything else markdown.split_finished(self._streaming)
        if done.strip():
            self._block(markdown.render(done))
        self._streaming = rest

    def _relative(self, text: str) -> str:
        if not self.root:
            return text
        return text.replace(self.root, "").replace(self.root.rstrip("/"), ".")

    def _finish_tool(self, name: str, result: Text) -> None:
        tool = next((t for t in self._running if t.name == name), None)
        if tool is None:  # rejected before it was announced (unknown tool, bad arguments)
            tool = _RunningTool(name)
        else:
            self._running.remove(tool)
        bullet = "err" if result.style == "err" else "warn" if result.style == "warn" else "ok"
        line = Text("  ").append(f"{ELBOW} ", style="subtle").append(result)
        if tool.note:
            line.append(f" · {tool.note}", style="subtle")
        self._block(Group(tool_line(self._relative(tool.summary), bullet), line))

    # -- events ---------------------------------------------------------------

    async def render(self, events: AsyncGenerator[Event, None]) -> None:
        self._turn_started, self._tools_used = time.monotonic(), 0
        self.wait(WAITING)
        self._start_live()
        try:
            async with aclosing(events) as stream:
                async for event in stream:
                    self.handle(event)
        finally:
            self._flush_text(everything=True)
            self._running.clear()
            self.stop_waiting()
            self._stop_live()

    def handle(self, event: Event) -> None:
        match event:
            case Reasoning():
                if self._activity == WAITING:
                    self.wait("Thinking")
            case AssistantText(text=text):
                self.stop_waiting()
                self._streaming += text
                self._flush_text(everything=False)
            case ToolUse(summary=summary):
                self._flush_text(everything=True)
                self._running.append(_RunningTool(summary))
                self._tools_used += 1
            case PermissionDecision(allowed=True, source=source, reason=reason) if (
                source in _NOTABLE_AUTO_ALLOW and self._running
            ):
                self._running[-1].note = f"allowed by {source}" + (f": {reason}" if source != "rule" else "")
            case PermissionDecision():
                pass  # denials are shown with their result
            case ToolsDispatched(names=names, parallel=parallel):
                self.wait(f"Running {len(names)} tools" if parallel else f"Running {names[0]}")
            case ToolResult(name="TodoWrite", is_error=False):
                self._running = [t for t in self._running if t.name != "TodoWrite"]
            case ToolResult(name=name, output=output, is_error=is_error):
                output = self._relative(output)
                if output == _REJECTED:
                    result = Text("rejected", style="warn")
                elif is_error:
                    result = Text(output.strip().splitlines()[0] if output.strip() else "error", style="err")
                else:
                    result = Text(summarize_output(name, output), style="muted")
                self._finish_tool(name, result)
                if not self._running:
                    self.wait(WAITING)
            case TodosUpdated(todos=todos):
                self._block(
                    Group(
                        tool_line("Todos", "ok"),
                        Text("  ").append(f"{ELBOW} ", style="subtle").append(format_todos(todos)),
                    )
                )
                self.wait(WAITING)
            case Compacted(kind=kind, count=n):
                self._block(Text(f"↻ {_COMPACTION_MESSAGES[kind].format(n=n)}", style="subtle"))
            case Done(reason=reason):
                self._flush_text(everything=True)
                for tool in self._running:
                    self._block(
                        Group(
                            tool_line(self._relative(tool.summary), "subtle"),
                            Text("  ⎿ skipped", style="subtle"),
                        )
                    )
                self._running.clear()
                self.stop_waiting()
                self._done(reason)

    def _done(self, reason: str) -> None:
        if reason == "max_steps":
            self._block(
                Text("■ Stopped: reached the step limit. Say “continue” to keep going.", style="warn")
            )
        elif reason == "interrupted":
            self._block(Text("■ Stopped. What should it do instead?", style="warn"))
        elif self._tools_used:
            elapsed = time.monotonic() - self._turn_started
            calls = f"{self._tools_used} tool call{'s' if self._tools_used != 1 else ''}"
            self._block(Text(f"✻ Worked for {_duration(elapsed)} · {calls}", style="subtle"))

    # -- permission prompt ----------------------------------------------------

    async def ask_permission(self, request: PermissionRequest) -> Decision:
        self._flush_text(everything=True)
        self._stop_live()
        self.console.print()
        decision = await approval.ask(self.console, request, self._relative(request.summary))
        if decision.remember and request.remember_label.startswith("allow all edits"):
            self._print(
                Text("  ⏵⏵ accepting edits for the rest of the session (shift+tab to change)", "brand")
            )
        self._start_live()
        return decision


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"
