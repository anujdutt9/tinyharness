"""Turning the agent's event stream into terminal output.

Output is split in two:

    permanent   finished Markdown blocks, completed tool calls, todo lists.
                Printed once, scrolls up like normal terminal output.
    live        the Markdown block still streaming, tool calls still running,
                and the spinner. Redrawn in place at the bottom of the screen.

Each tool call is one row: a status glyph, the tool (colored by what it does:
looking, changing, running, the web, agents), what it was called on, and on the
right what came of it and how long it took. Subagents show under their call while they
run; once done they fold into its row (a failed one keeps a line of its own):

    ✓ Read    src/app.py                                         120 lines
    ✓ Edit    src/app.py                                           +4 −1
    ✗ Bash    npm run build                                 exit 1 · 2.3s
      ⎿ src/app.ts(12,5): error TS2322: Type 'string' is not assignable…
    ✓ Agent   reviewer: check the retry logic         haiku-4.5 · 3 tools · 3.4s
    ✓ Team    compare three laptops · 3 tasks          3 agents · 41 tools · 1m 2s

    ── 9s · 4 tools · 1 file +4 −1 · tests ✗ · 12.3k tokens · $0.03

The spinner says "Working" while waiting on the model, and "Thinking" only once the
model reports that it is actually reasoning (many replies involve none).
"""

from __future__ import annotations

import contextlib
import re
import time
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, aclosing
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.spinner import Spinner
from rich.style import Style
from rich.text import Text

from tinyharness.cli import approval, markdown
from tinyharness.cli.theme import (
    BRAND,
    DENIED,
    DONE,
    ELBOW,
    FAILED,
    MUTED,
    SPINNER,
    ellipsize,
    file_link,
    split_summary,
    tool_label,
    tool_line,
    tool_style,
)
from tinyharness.events import (
    AssistantText,
    Compacted,
    Done,
    Event,
    MessageDelivered,
    ModelStuck,
    PermissionDecision,
    Reasoning,
    SubagentUpdate,
    TodosUpdated,
    TokenUsage,
    ToolInputStreaming,
    ToolResult,
    ToolsDispatched,
    ToolUse,
)
from tinyharness.llm.usage import Usage, UsageMeter
from tinyharness.permissions import Decision, PermissionRequest

WAITING = "Working"
_WRITING_LABELS = ("Writing", "Editing", "Preparing")  # see Renderer._writing_label

_REJECTED = "The user rejected this tool call."

_COMPACTION_MESSAGES = {
    "budget_cap": "trimmed {n} oversized message(s)",
    "microcompact": "cleared {n} old tool result(s)",
    "summary": "summarized older messages to free up context",
}

_EDIT_STATS = re.compile(r", -(\d+)/\+(\d+) lines$")
_WRITE_STATS = re.compile(r", (\d+) lines$")
_EXIT_CODE = re.compile(r"\(exit code (\d+)\)\s*$")
_MATCH_HEADER = re.compile(r"^\d+ (total )?match\(es\)|^\d+ files?\b")
# Commands that run a test suite, so the turn's footer can say whether the tests passed.
_TEST_COMMAND = re.compile(
    r"\b(pytest|py\.test|tox|nox|jest|vitest|mocha|rspec|phpunit|ctest)\b"
    r"|\b(npm|pnpm|yarn|bun)\s+(run\s+)?test\b"
    r"|\b(go|cargo|mix|dotnet|swift|deno)\s+test\b"
    r"|\bmake\s+(test|check)\b"
    r"|\bpython3?\s+-m\s+(pytest|unittest)\b"
)
NAME_WIDTH = 6  # tool names are padded to this, so what follows lines up
RECENT_CALLS = 3  # a running subagent's latest tool calls, shown under it
PREVIEW_LINES = 3  # of a command's or search's output, under its row
# Tools whose output is worth a glance: what a command printed, what a search found. A
# Read, Edit or Write is summed up by its row (`212 lines`, `+4 −1`).
_PREVIEWED = frozenset({"Bash", "Grep", "Glob"})
# How a decision the user wasn't asked about is credited, under the call.
_ALLOWED_BY = {"classifier": "auto mode", "hook": "hook"}
LIVE_THOUGHT_LINES = 8  # while the model reasons, the latest lines of it (when shown)


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
    args: dict[str, Any] = field(default_factory=dict)
    note: str = ""  # who allowed it without asking (e.g. "auto mode · read-only query"), shown under it
    started: float = 0.0  # when it was dispatched (not when it was asked about)
    id: str = ""  # the call's id, to match its result (a parallel batch finishes in any order)
    summed_up: bool = False  # its row says it all (a passing test run): no output preview under it

    @property
    def name(self) -> str:
        return split_summary(self.summary)[0]


@dataclass
class _Turn:
    """What happened in one turn, for the footer."""

    started: float = 0.0
    tools: int = 0  # the main agent's calls
    agents: int = 0  # subagents and teammates that finished
    agent_tools: int = 0  # their calls
    files: dict[str, list[int]] = field(default_factory=dict)  # path -> [added, removed]
    tests_passed: bool | None = None  # the last test run's outcome
    usage_before: Usage = field(default_factory=Usage)
    calls_before: int = 0  # the meter's call count when the turn started


class Renderer:
    def __init__(self, console: Console) -> None:
        self.console = console
        self.root: str | None = None  # "<working dir>/"; stripped so paths display relative
        self.meter: UsageMeter | None = None  # the session's, for tokens and cost per turn
        self.last_output: tuple[str, str] | None = None  # (summary, full output) of the last tool
        self.show_thinking = False  # print the model's reasoning, or just how long it took
        self.last_thought = ""  # the latest reasoning, for ctrl+t to show after the fact
        self.last_thought_shown = False  # already on screen: ctrl+t won't print it again
        self._thought: list[str] = []  # reasoning so far in this model call
        self._thought_since: float | None = None
        # Embedded: the REPL keeps its input box up during a turn and draws the live region
        # itself (see `live_ansi`); rich's own Live display would fight it for the screen.
        self.embedded = False
        # Taking the terminal for a question (a permission prompt) while the input box is up.
        self.exclusive: Callable[[], AbstractAsyncContextManager[None]] | None = None
        self._live: Live | None = None
        self._spinner = Spinner("dots", style=BRAND)
        self._activity: str | None = None
        self._activity_since = 0.0
        self._last_activity: str | None = None  # survives a brief stop, so its clock does too
        self._activity_detail = ""  # e.g. how much of a file has been written so far
        self._streaming = ""  # the Markdown block still being written
        self._running: list[_RunningTool] = []
        self._turn = _Turn()
        self._agents: dict[str, SubagentUpdate] = {}  # running subagents, by run id
        self._agent_since: dict[str, float] = {}  # when each running subagent started (monotonic)
        self._finished_agents: list[SubagentUpdate] = []  # shown under their Agent/Team call
        self._last_was_row = False  # consecutive tool rows print as one block

    # -- the live region ------------------------------------------------------

    def __rich__(self) -> RenderableType:
        parts: list[RenderableType] = []
        if self._streaming.strip():
            parts.append(markdown.render(self._streaming))
        now = time.monotonic()
        frame = SPINNER[int(now * 10) % len(SPINNER)]
        for tool in self._running:
            elapsed = now - tool.started if tool.started else 0.0
            right = Text(_duration(elapsed), style="subtle") if elapsed >= 1 else Text()
            parts.append(self._row(Text(frame, style=BRAND), tool, right))
        agents = list(self._agents.values())
        for i, agent in enumerate(agents):
            since = self._agent_since.get(agent.id)
            parts.append(
                self._agent_line(
                    agent,
                    last=i == len(agents) - 1,
                    glyph=Text(frame, style=BRAND),
                    elapsed=now - since if since is not None else None,
                )
            )
            # What it has been doing: its last few calls (fewer each when several run).
            each = RECENT_CALLS if len(agents) == 1 else 1
            parts.extend(self._recent_calls(agent, each, frame, last=i == len(agents) - 1))
        if self.show_thinking and (thought := "".join(self._thought).strip()):
            tail = "\n".join(thought.splitlines()[-LIVE_THOUGHT_LINES:])
            parts.append(Padding(Text(tail, style="thinking"), (0, 0, 0, 2)))
        if self._activity:
            elapsed = int(now - self._activity_since)
            detail = f" · {self._activity_detail}" if self._activity_detail else ""
            self._spinner.update(
                text=Text.assemble(
                    (self._activity, BRAND),
                    (detail, "muted"),
                    ("…", BRAND),
                    (f" {elapsed}s · ctrl+c to interrupt", "subtle"),
                )
            )
            parts.append(self._spinner)
        return Group(*parts)

    def live_ansi(self, width: int) -> str:
        """The live region as terminal text, for the REPL to draw above its input box."""
        busy = self._streaming.strip() or self._running or self._agents or self._activity
        if not busy and self._thought_since is None:
            return ""
        with self.console.capture() as captured:
            self.console.print(self, width=width, highlight=False)
        return without_links(captured.get().rstrip("\n"))

    def _start_live(self) -> None:
        if self.embedded:
            return
        if self._live is None:
            self._live = Live(self, console=self.console, refresh_per_second=12, transient=True)
            self._live.start()

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def wait(self, label: str, detail: str = "") -> None:
        """Show `label` on the spinner. Its clock starts when the label changes, not when
        the same activity comes back after a pause (text streamed between pieces of a
        tool call used to restart it: 0s, 1s, 0s, 1s)."""
        if label != self._last_activity:
            self._activity_since = time.monotonic()
        self._activity = self._last_activity = label
        self._activity_detail = detail

    def stop_waiting(self) -> None:
        self._activity = None

    # -- permanent output -----------------------------------------------------

    def _print(self, renderable: RenderableType) -> None:
        self.console.print(renderable, highlight=False)

    def _block(self, renderable: RenderableType) -> None:
        """Print a block with a blank line before it, so turns read as distinct steps."""
        self.console.print()
        self._print(renderable)
        self._last_was_row = False

    def _rows(self, renderable: RenderableType) -> None:
        """Tool rows: a blank line before a run of them, none between them."""
        if not self._last_was_row:
            self.console.print()
        self._print(renderable)
        self._last_was_row = True

    def _flush_text(self, everything: bool) -> None:
        done, rest = (self._streaming, "") if everything else markdown.split_finished(self._streaming)
        if done.strip():
            self._block(markdown.render(done))
        self._streaming = rest

    def _relative(self, text: str) -> str:
        if not self.root:
            return text
        return text.replace(self.root, "").replace(self.root.rstrip("/"), ".")

    # -- rows -----------------------------------------------------------------

    def _recent_calls(self, agent: SubagentUpdate, count: int, frame: str, *, last: bool) -> list[Text]:
        """A running subagent's latest tool calls, under its line:

        └ ⠋ general-purpose  Verify deck rendering        opus-4.8 · 21 tools · 6m 28s
            … 18 earlier
            ✓ Edit    .deck-verify/measure.mjs                            +3 −1
            ✗ Bash    node .deck-verify/measure.mjs                exit code 1 · 4s
            ⠋ Bash    node .deck-verify/measure.mjs                             12s
        """
        prefix = "      " if last else "  │   "
        shown = agent.recent[-count:] if count else ()
        rows: list[Text] = []
        if (earlier := agent.tools_used - len(shown)) > 0 and shown:
            rows.append(Text(f"{prefix}… {earlier} earlier", style="subtle"))
        for call in shown:
            if not call.done:
                glyph = Text(frame, style=BRAND)
            else:
                glyph = Text(DONE, style="ok") if call.ok else Text(FAILED, style="err")
            right = Text()
            if call.done and call.result:
                result = " ".join(call.result.split())
                right.append(ellipsize(result, 40), style="muted" if call.ok else "err")
            if call.elapsed >= 1:
                right.append(" · " if right.cell_len else "", style="subtle").append(
                    _duration(call.elapsed), style="subtle"
                )
            rows.append(self._row(glyph, _RunningTool(call.summary), right, indent=prefix))
        return rows

    def _row(self, glyph: Text, tool: _RunningTool, right: Text, indent: str = "") -> Text:
        """`✓ Edit    src/app.py ·········· +4 −1`, fitted to the terminal's width: dot
        leaders take the eye from what the call was on to its result, like a notebook index."""
        name, label = tool.name, tool_label(tool.name)
        detail = self._detail(tool)
        width = self.console.width
        right = right[: max(width // 2, 10)]
        fixed = len(indent) + 2 + max(len(label), NAME_WIDTH) + 2
        room = width - fixed - (right.cell_len + 2 if right.cell_len else 0) - 1
        line = Text(indent).append_text(glyph).append(" ")
        line.append(label.ljust(NAME_WIDTH), style=tool_style(name))
        path = tool.args.get("path") if name in ("Read", "Edit", "Write", "Pdf", "Open") else None
        style = Style(color=MUTED) + (file_link(self._absolute(str(path))) if path else Style())
        line.append("  ").append(ellipsize(detail, max(room, 8)), style=style)
        if right.cell_len:
            gap = max(width - line.cell_len - right.cell_len - 1, 2)
            if gap >= 5:
                line.append(" ").append("·" * (gap - 2), style="subtle").append(" ")
            else:
                line.append(" " * gap)
            line.append_text(right)
        return line

    def _detail(self, tool: _RunningTool) -> str:
        """What the call was on: the summary's argument part, minus stats shown on the right."""
        detail = self._relative(split_summary(tool.summary)[1])
        if tool.name == "Edit":
            return _EDIT_STATS.sub("", detail)
        if tool.name == "Write":
            return _WRITE_STATS.sub("", detail)
        if tool.name == "Screenshot":  # deck.html#8 → deck.html · slide 8
            page, _, where = detail.partition("#")
            return f"{page} · slide {where}" if where.isdigit() else f"{page} · #{where}" if where else page
        if tool.name == "Bash" and "\n" in (command := str(tool.args.get("command", "")).strip()):
            # a heredoc or script: its first line, not the body spilling over several rows
            first, *rest = command.splitlines()
            prefix = "unsandboxed: " if detail.startswith("unsandboxed: ") else ""
            return f"{prefix}{self._relative(first)} (+{len(rest)} lines)"
        if "\n" in detail.strip():  # a subagent's call comes as its summary alone: same idea
            first, *rest = [line for line in detail.strip().splitlines() if line.strip()]
            return f"{first.strip()} (+{len(rest)} lines)"
        return detail

    def _absolute(self, path: str) -> str:
        return path if path.startswith(("/", "~")) or not self.root else f"{self.root}{path}"

    def _agent_line(
        self, update: SubagentUpdate, *, last: bool, glyph: Text, elapsed: float | None = None
    ) -> Text:
        """A subagent under its call: `└ ✓ reviewer  check the retry logic   haiku · 3 tools · 3s`.
        `elapsed` is live for a running one; updates only come when it does something."""
        right = Text()
        if update.model:
            right.append(_short_model(update.model), style="subtle").append(" · ", style="subtle")
        width = self.console.width
        if update.status in ("started", "working"):
            doing = _one_line(self._doing(update)) if update.status == "working" else "starting"
            right.append(ellipsize(doing, max(width // 4, 12)), style="muted").append(" · ", style="subtle")
        tools = update.tools_used
        took = update.elapsed if elapsed is None else elapsed
        right.append(f"{tools} tool{'s' if tools != 1 else ''} · {_duration(took)}", style="subtle")
        line = Text(f"  {'└' if last else '├'} ", style="subtle").append_text(glyph).append(" ")
        line.append(update.agent, style=tool_style("Agent")).append("  ")
        right = right[-max(width - line.cell_len - 12, 10) :]  # keep the end: tools and time
        room = width - line.cell_len - right.cell_len - 3
        line.append(ellipsize(" ".join(update.label.split()), max(room, 8)), style="muted")
        return line.append(" " * max(width - line.cell_len - right.cell_len - 1, 2)).append_text(right)

    def _running_for(self, result: ToolResult) -> _RunningTool | None:
        """The row a result belongs to: by call id (a parallel batch finishes in any
        order, and three WebSearches share a name), else the first by that name."""
        if result.id:
            return next((t for t in self._running if t.id == result.id), None)
        return next((t for t in self._running if t.name == result.name), None)

    def _finish_tool(self, result: ToolResult) -> None:
        tool = self._running_for(result)
        if tool is None:  # rejected before it was announced (unknown tool, bad arguments)
            tool = _RunningTool(result.name)
        else:
            self._running.remove(tool)
        glyph, right, under = self._outcome(tool, self._relative(result.output), result.is_error)
        children = self._children(tool) if tool.name in ("Agent", "Team") else []
        if children:  # finished agents fold into their call's row: what they used, not a line each
            right = _agents_summary(children, single=tool.name == "Agent")
            if any(child.status != "done" for child in children):
                glyph = Text(FAILED, style="err")
        elapsed = time.monotonic() - tool.started if tool.started else 0.0
        if elapsed >= 1:
            right.append(" · " if right.cell_len else "", style="subtle").append(
                _duration(elapsed), style="subtle"
            )
        lines: list[RenderableType] = [self._row(glyph, tool, right)]
        preview = (
            [] if tool.summed_up else self._preview(tool, self._relative(result.output), result.is_error)
        )
        if preview:
            under = None  # the preview already shows the line that would go here
        body = preview or ([under] if under else [])
        if tool.note:
            credit = Text(tool.note, style="subtle")
            credit.stylize("italic")
            body.append(credit)
        for i, line in enumerate(body):
            lines.append(Text("  ").append(f"{ELBOW} " if i == 0 else "  ", style="subtle").append_text(line))
        failed = [child for child in children if child.status != "done"]  # only these get a line
        for i, child in enumerate(failed):
            lines.append(self._agent_line(child, last=i == len(failed) - 1, glyph=Text(FAILED, style="err")))
        self._rows(Group(*lines))

    def _preview(self, tool: _RunningTool, output: str, is_error: bool) -> list[Text]:
        """The first few lines of a command's or search's output (the last few, when a
        command failed: that's where the error is), then how many more there are."""
        if tool.name not in _PREVIEWED or output == _REJECTED or output.startswith("denied by"):
            return []
        lines = [line.rstrip() for line in output.strip().splitlines()]
        if tool.name == "Bash":  # the harness's own notes; the row says exit code / timeout
            lines = [line for line in lines if not (line.startswith("(") and line.endswith(")"))]
        elif lines and _MATCH_HEADER.match(lines[0]):  # "3 match(es):" is on the row already
            lines = lines[1:]
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines or lines == ["(no output)"]:
            return []
        failed = is_error or bool(_EXIT_CODE.search(output)) or "(timed out after" in output
        shown = lines[-PREVIEW_LINES:] if failed else lines[:PREVIEW_LINES]
        room = max(self.console.width - 6, 20)
        style = "err" if failed else "muted"
        out = [Text(_cut(line.expandtabs(4), room), style=style) for line in shown]
        if (more := len(lines) - len(shown)) > 0:
            where = "above" if failed else "more"
            out.insert(
                0 if failed else len(out),
                Text(f"… {more} {where} line{'s' if more != 1 else ''}", style="subtle"),
            )
        return out

    def _children(self, tool: _RunningTool) -> list[SubagentUpdate]:
        """The finished subagents that belong to this Agent/Team call."""
        if tool.name == "Team":
            children, self._finished_agents = self._finished_agents, []
            return children
        label = " ".join(str(tool.args.get("description") or tool.args.get("prompt") or "").split())[:80]
        # The agent it ran: as asked for, or (left out) the default one its summary names.
        agent = tool.args.get("subagent_type") or split_summary(tool.summary)[1].partition(":")[0].strip()
        mine = [a for a in self._finished_agents if a.agent == agent and a.label == label]
        self._finished_agents = [a for a in self._finished_agents if a not in mine]
        return mine

    def _outcome(self, tool: _RunningTool, output: str, is_error: bool) -> tuple[Text, Text, Text | None]:
        """(status glyph, what came of it, an optional line underneath)."""
        name, text = tool.name, output.strip()
        if output == _REJECTED:
            return Text(DENIED, style="warn"), Text("rejected", style="warn"), None
        if is_error and text.startswith("denied by"):
            return Text(DENIED, style="warn"), Text(ellipsize(text, 60), style="warn"), None
        if is_error and name in ("Read", "Pdf") and "file not found" in text.splitlines()[0]:
            # Looking for a file that isn't there is routine (the model checks for a README,
            # an AGENTS.md): the model gets the error; the user just sees it wasn't there.
            return Text(FAILED, style="subtle"), Text("not found", style="muted"), None
        if is_error:
            first = text.splitlines()[0] if text else "error"
            return (
                Text(FAILED, style="err"),
                Text("failed", style="err"),
                Text(ellipsize(first, 200), style="err"),
            )
        ok = Text(DONE, style="ok")
        detail = split_summary(tool.summary)[1]
        if name == "Edit" and (m := _EDIT_STATS.search(detail)):
            removed, added = int(m.group(1)), int(m.group(2))
            self._count_edit(tool, added, removed)
            return ok, _diff_stat(added, removed), None
        if name == "Write" and (m := _WRITE_STATS.search(detail)):
            self._count_edit(tool, int(m.group(1)), 0)
            return ok, _diff_stat(int(m.group(1)), 0), None
        if name == "Bash":
            return self._bash_outcome(tool, text)
        if name == "Read":
            return ok, Text(f"{len(text.splitlines())} lines", style="muted"), None
        if name in ("Agent", "Team"):
            return ok, Text(), None
        if name == "Screenshot":  # the picture went to the model; where it's saved is no news
            return ok, Text("looked", style="muted"), None
        return ok, Text(ellipsize(_brief(name, text), 50), style="muted"), None

    def _bash_outcome(self, tool: _RunningTool, text: str) -> tuple[Text, Text, Text | None]:
        lines = [line for line in text.splitlines() if line.strip() and not line.startswith("(")]
        last = lines[-1].strip() if lines else ""
        is_test = bool(_TEST_COMMAND.search(str(tool.args.get("command", ""))))
        if "(timed out after" in text:
            if is_test:
                self._turn.tests_passed = False
            return Text(FAILED, style="err"), Text("timed out", style="warn"), None
        if m := _EXIT_CODE.search(text):
            if is_test:
                self._turn.tests_passed = False
            under = Text(ellipsize(last, 200), style="err") if last else None
            return Text(FAILED, style="err"), Text(f"exit {m.group(1)}", style="err"), under
        if is_test:
            self._turn.tests_passed = True
            if summary := _test_summary(text):  # the result on the row: `212 passed in 4.1s`
                tool.summed_up = True
                return Text(DONE, style="ok"), Text(summary, style="ok"), None
        if last:  # the output shows under the row
            return Text(DONE, style="ok"), Text(), None
        return Text(DONE, style="ok"), Text("no output", style="muted"), None

    def _count_edit(self, tool: _RunningTool, added: int, removed: int) -> None:
        path = str(tool.args.get("path") or self._detail(tool))
        counts = self._turn.files.setdefault(path, [0, 0])
        counts[0] += added
        counts[1] += removed

    # -- events ---------------------------------------------------------------

    async def render(self, events: AsyncGenerator[Event, None]) -> None:
        self._turn = _Turn(
            started=time.monotonic(),
            usage_before=self._usage(),
            calls_before=self.meter.calls if self.meter is not None else 0,
        )
        self.wait(WAITING)
        self._start_live()
        try:
            async with aclosing(events) as stream:
                async for event in stream:
                    self.handle(event)
        finally:
            self._flush_text(everything=True)
            self._running.clear()
            self._agents.clear()
            self._agent_since.clear()
            self.stop_waiting()
            self._stop_live()

    def handle(self, event: Event) -> None:
        match event:
            case Reasoning(text=text):
                if self._thought_since is None:
                    self._thought_since = time.monotonic()
                self._thought.append(text)
                if self._activity == WAITING:
                    self.wait("Thinking")
            case ModelStuck(reason=reason):
                self._finish_thought()
                self._streaming = ""  # the stuck reply's unprinted text: discarded, as it was
                self._block(
                    Text(
                        f"↻ The model got stuck: {reason}. Stopped it and asked it to carry on.", style="warn"
                    )
                )
                self.wait(WAITING)
            case MessageDelivered(text=text):
                self._flush_text(everything=True)
                shown = ellipsize(" ".join(text.split()), max(self.console.width - 30, 20))
                self._print(Text(f"  ↳ delivered to the agent: {shown}", style="subtle"))
                self._last_was_row = False
            case AssistantText(text=text):
                self._finish_thought()
                if text.strip():  # blank lines between tool-call pieces aren't an answer
                    self.stop_waiting()
                self._streaming += text
                self._flush_text(everything=False)
            case ToolInputStreaming(name=name, chars=chars, target=target):
                # Reasoning is over: the model is writing a tool call (a whole file, maybe),
                # which shows nothing until it's complete. Say so, not "Thinking".
                self._finish_thought()
                self._flush_text(everything=True)
                self.wait(
                    self._writing_label(name, target), f"{tokens(chars)} chars" if chars >= 1000 else ""
                )
            case ToolUse(summary=summary, args=args):
                self._finish_thought()
                if self._activity and self._activity.startswith(_WRITING_LABELS):
                    self.wait(WAITING)  # the call is complete: its row takes over
                self._flush_text(everything=True)
                self._running.append(_RunningTool(summary, args, id=event.id))
                self._turn.tools += 1
            case PermissionDecision(allowed=True, source=source, reason=reason) if (
                source in _ALLOWED_BY and self._running
            ):
                label = _ALLOWED_BY[source]
                self._running[-1].note = f"{label} · {reason}" if reason else label
            case PermissionDecision():
                pass  # denials are shown with their result
            case ToolsDispatched(names=names, parallel=parallel):
                now = time.monotonic()
                for tool in self._running:
                    if tool.name in names and not tool.started:
                        tool.started = now
                self.wait(f"Running {len(names)} tools" if parallel else f"Running {tool_label(names[0])}")
            case ToolResult(name="TodoWrite", is_error=False):
                self._running = [t for t in self._running if t.name != "TodoWrite"]
            case ToolResult() as result:
                running = self._running_for(result)
                self.last_output = (running.summary if running else result.name, result.output)
                self._finish_tool(result)
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
            case SubagentUpdate(id=run_id, status="started" | "working"):
                self._agents[run_id] = event
                self._agent_since.setdefault(run_id, time.monotonic() - event.elapsed)
            case SubagentUpdate(id=run_id):
                self._agents.pop(run_id, None)
                self._agent_since.pop(run_id, None)
                self._finished_agents.append(event)
                self._turn.agents += 1
                self._turn.agent_tools += event.tools_used
            case TokenUsage():
                pass  # counted by the session's meter, which includes subagents
            case Compacted(kind=kind, count=n):
                self._block(Text(f"↻ {_COMPACTION_MESSAGES[kind].format(n=n)}", style="subtle"))
            case Done(reason=reason):
                self._finish_thought()
                self._flush_text(everything=True)
                for tool in self._running:
                    self._rows(self._row(Text("·", style="subtle"), tool, Text("skipped", style="subtle")))
                self._running.clear()
                for i, left in enumerate(self._finished_agents):  # ones no call claimed
                    ok = left.status == "done"
                    mark = Text(DONE if ok else FAILED, style="ok" if ok else "err")
                    self._print(self._agent_line(left, last=i == len(self._finished_agents) - 1, glyph=mark))
                self._finished_agents.clear()
                self.stop_waiting()
                self._done(reason)

    def _finish_thought(self) -> None:
        """The model moved on from reasoning: show it, or just how long it took."""
        if self._thought_since is None:
            return
        elapsed = time.monotonic() - self._thought_since
        thought = "".join(self._thought).strip()
        self._thought, self._thought_since = [], None
        if thought:
            self.last_thought, self.last_thought_shown = thought, self.show_thinking
        if self.show_thinking and thought:
            self._block(
                Group(
                    Text("∴ Thinking", style="subtle"), Padding(Text(thought, style="thinking"), (0, 0, 0, 2))
                )
            )
        elif thought or elapsed >= 1:
            line = Text(f"∴ Thought for {_duration(elapsed)}", style="subtle")
            if thought:
                line.append(" · ctrl+t to show", style="subtle")
            self._block(line)

    def _doing(self, update: SubagentUpdate) -> str:
        match update.phase:
            case "thinking":
                return "Thinking"
            case "waiting":
                return WAITING
            case "writing":
                return self._writing_label(update.tool, update.detail)
            case _:
                name, args = split_summary(update.detail)
                shown = f"{tool_label(name)}({_one_line(args)})" if args else update.detail
                return self._relative(shown)

    def _writing_label(self, name: str, target: str) -> str:
        """`Writing adobe-deck.html`, `Editing src/app.py`, `Writing a command`, `Preparing WebFetch`."""
        shown = ellipsize(self._relative(target), 60) if target else ""
        match name:
            case "Write":
                return f"Writing {shown}" if shown else "Writing a file"
            case "Edit":
                return f"Editing {shown}" if shown else "Editing a file"
            case "Bash":
                return "Writing a command"
            case _:
                return f"Preparing {tool_label(name)}"

    def _usage(self) -> Usage:
        return Usage() + self.meter.total if self.meter is not None else Usage()

    def _done(self, reason: str) -> None:
        if reason == "max_steps":
            self._block(
                Text("■ Stopped: reached the step limit. Say “continue” to keep going.", style="warn")
            )
        elif reason == "interrupted":
            self._block(Text("■ Stopped. What should it do instead?", style="warn"))
        elif self._turn.tools:
            self._block(self.footer())

    def footer(self) -> Text:
        """`── 9s · 4 tools · 2 files +14 −3 · tests ✓ · 83k context · 2.1k out · $0.03`, or
        `128 tools (111 by 4 agents)` when agents did some of the work."""
        turn = self._turn
        line = Text("── ", style="subtle").append(_duration(time.monotonic() - turn.started), style="muted")
        dot = (" · ", "subtle")
        total = turn.tools + turn.agent_tools
        line.append(*dot).append(f"{total} tool{'s' if total != 1 else ''}", style="muted")
        if turn.agent_tools:
            n = turn.agents
            line.append(f" ({turn.agent_tools} by {n} agent{'s' if n != 1 else ''})", style="subtle")
        if turn.files:
            added = sum(a for a, _ in turn.files.values())
            removed = sum(r for _, r in turn.files.values())
            n = len(turn.files)
            line.append(*dot).append(f"{n} file{'s' if n != 1 else ''} ", style="muted")
            line.append_text(_diff_stat(added, removed))
        if turn.tests_passed is not None:
            line.append(*dot).append("tests ", style="muted")
            line.append(DONE if turn.tests_passed else FAILED, style="ok" if turn.tests_passed else "err")
        used = self._usage()
        spent = Usage(
            used.input - turn.usage_before.input,
            used.cached - turn.usage_before.cached,
            used.output - turn.usage_before.output,
        )
        # Not the sum of every call's input: each call re-sends the conversation, so that
        # grows with every step (see UsageMeter). The largest call, and what was produced.
        if self.meter is not None and (context := self.meter.largest_context(turn.calls_before)):
            line.append(*dot).append(f"{tokens(context)} context", style="subtle")
        if spent.output:
            line.append(*dot).append(f"{tokens(spent.output)} out", style="subtle")
        if used.cost is not None:
            cost = used.cost - (turn.usage_before.cost or 0)
            line.append(*dot).append(dollars(cost), style="muted")
        return line

    # -- permission prompt ----------------------------------------------------

    async def ask_permission(self, request: PermissionRequest) -> Decision:
        async with self.exclusive() if self.exclusive else contextlib.nullcontext():
            return await self._ask_permission(request)

    async def _ask_permission(self, request: PermissionRequest) -> Decision:
        self._flush_text(everything=True)
        self._stop_live()
        self.console.print()
        decision = await approval.ask(self.console, request, self._relative(request.summary))
        if decision.remember and request.remember_label.startswith("allow all edits"):
            self._print(
                Text("  ⏵⏵ accepting edits for the rest of the session (shift+tab to change)", "brand")
            )
        elif decision.remember and request.remember_label:  # the one answer worth keeping on screen
            label = request.remember_label
            self._print(Text(f"  ✓ {label[0].upper()}{label[1:]}", style="muted"))
        self._last_was_row = False
        self._start_live()
        return decision


_PASSED = re.compile(r"\b\d+ passed\b[^=\n]*")
_UNITTEST_RAN = re.compile(r"^Ran (\d+) tests? in [\d.]+s\s*\n\s*\nOK\b", re.MULTILINE)


def _test_summary(output: str) -> str:
    """A passing test run's result in a few words: pytest's and most runners' "N passed",
    or unittest's "Ran N tests ... OK"."""
    if m := _UNITTEST_RAN.search(output):
        return f"{m.group(1)} passed"
    found = _PASSED.findall(output)
    return found[-1].strip(" ,") if found else ""


def _one_line(text: str) -> str:
    """A status that must fit one row: a multi-line command or heredoc shows its first line
    and an ellipsis, not its body breaking the layout."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) <= 1:
        return " ".join(text.split())
    return f"{lines[0]} …"


def _cut(line: str, room: int) -> str:
    """Output cut at the end (a path is better cut in the middle; a line of output isn't)."""
    return line if len(line) <= room else line[: room - 1] + "…"


def _brief(name: str, text: str) -> str:
    """A tool's result in a few words, for the right-hand column."""
    first = summarize_output(name, text)
    if m := re.match(r"(\d+) total match\(es\) across (\d+) file", first):
        return f"{m.group(1)} matches in {m.group(2)} files"
    if m := re.match(r"(\d+) match\(es\)", first):
        return f"{m.group(1)} {'match' if m.group(1) == '1' else 'matches'}"
    return first.rpartition(" · ")[2]  # "https://… · 48,210 characters" → "48,210 characters"


def _agents_summary(children: list[SubagentUpdate], *, single: bool) -> Text:
    """`haiku-4.5 · 3 tools` for one agent; `4 agents · 147 tools` for a team."""
    tools = sum(child.tools_used for child in children)
    counted = f"{tools} tool{'s' if tools != 1 else ''}"
    if single and children[0].model:
        return Text(f"{_short_model(children[0].model)} · {counted}", style="subtle")
    n = len(children)
    return Text(f"{n} agent{'s' if n != 1 else ''} · {counted}", style="subtle")


_LINK = re.compile(r"\x1b\]8;[^\x07\x1b]*(?:\x07|\x1b\\)")


def without_links(ansi: str) -> str:
    """Drop clickable-link escapes (OSC 8), which prompt_toolkit would print as text."""
    return _LINK.sub("", ansi)


def _diff_stat(added: int, removed: int) -> Text:
    stat = Text()
    if added or not removed:
        stat.append(f"+{added}", style="diff.add")
    if removed:
        stat.append(" " if added else "").append(f"−{removed}", style="diff.del")
    return stat


def tokens(n: int) -> str:
    """950, 12.3k, 1.2M"""
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1000:.1f}k".replace(".0k", "k")
    return f"{n / 1_000_000:.1f}M".replace(".0M", "M")


def dollars(amount: float) -> str:
    return "<$0.01" if 0 < amount < 0.005 else f"${amount:.2f}"


def _duration(seconds: float) -> str:
    if seconds < 10:
        return f"{seconds:.1f}s".replace(".0s", "s")
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


def _short_model(model: str) -> str:
    """The model id without a gateway's vendor prefix: "anthropic/claude-haiku-4.5" → "claude-haiku-4.5"."""
    return model.rsplit("/", 1)[-1]
