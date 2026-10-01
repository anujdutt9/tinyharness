"""Agent teams: the lead splits a goal into tasks, teammates work them in parallel.

    Team(goal, tasks=[{id, title, prompt, agent, depends_on}, ...])

A task starts as soon as every task it depends on has finished, and gets their
results in its prompt. Teammates are subagents with two more tools: TeamBoard (every
task's status and result so far, and the team's messages) and TeamMessage (tell
one teammate, by task id, or everyone). A message reaches a running teammate
before its next step. Up to 4 teammates run side by side, writers included: their
calls that change things take turns, so two never write at once, while their model
calls overlap. A teammate that fails (stuck, a provider error) is started once more;
a task whose dependencies partly failed still runs, told what's missing, and is
skipped only if none of them finished. The lead gets one report with every result.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from tinyharness.subagents import AgentSpec
from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput

MAX_TASKS = 12
MAX_PARALLEL = 4
MAX_ATTEMPTS = 2  # a teammate that fails (stuck, a provider error) is started once more
BOARD_WAIT_SECONDS = 60  # TeamBoard with nothing new waits this long for news
RESULT_CHARS_IN_PROMPT = 4_000  # a dependency's result, as passed to the next teammate
RESULT_CHARS_IN_REPORT = 6_000  # each result, in the report for the lead
REQUEST_CHARS_IN_PROMPT = 2_000  # the user's request, as context for each teammate
# The lead's shared brief: enough for a schema and definitions. More than that buries each
# teammate's own task (and is better put in a file the teammates read).
CONTEXT_CHARS = 16_000
TEAMMATE_TAG = "<teammate-message"

Status = Literal["waiting", "running", "done", "failed", "skipped"]


class TeammateSpawner(Protocol):
    def __call__(
        self,
        spec: AgentSpec,
        prompt: str,
        ctx: ToolContext,
        /,
        *,
        label: str,
        extra_tools: Iterable[Tool[Any]],
        inbox: Callable[[], list[str]],
    ) -> Awaitable[str]: ...


# ---------------------------------------------------------------------------
# The board: tasks, results and messages, shared by the whole team
# ---------------------------------------------------------------------------


@dataclass
class TaskState:
    id: str
    title: str
    prompt: str
    agent: str
    depends_on: list[str]
    status: Status = "waiting"
    result: str = ""
    attempts: int = 1
    retry_note: str = ""  # why the previous attempt failed, for the next one


@dataclass
class Board:
    goal: str
    tasks: dict[str, TaskState]
    messages: list[tuple[str, str, str]] = field(default_factory=list)  # (from, to, text)
    inboxes: dict[str, list[str]] = field(default_factory=dict)
    request: str = ""  # the user's message that led to the team, for context
    context: str = ""  # the lead's shared brief, the same for every teammate
    version: int = 0  # goes up whenever a task ends or a message is posted
    _changed: asyncio.Event = field(default_factory=asyncio.Event)

    def bump(self) -> None:
        self.version += 1
        self._changed.set()
        self._changed = asyncio.Event()

    async def wait_for_news(self, seen: int, timeout: float) -> None:
        """Return once the board changes after version `seen`, or after `timeout`."""
        if self.version == seen:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._changed.wait(), timeout)

    def post(self, sender: str, to: str, text: str) -> str:
        if to != "all" and to not in self.tasks:
            raise ToolError(f"no task {to!r}; send to a task id ({', '.join(self.tasks)}) or 'all'")
        self.messages.append((sender, to, text))
        self.bump()
        note = teammate_message(sender, text)
        recipients = [t for t in self.tasks if t != sender] if to == "all" else [to]
        delivered = []
        for task_id in recipients:
            if self.tasks[task_id].status in ("waiting", "running"):
                self.inboxes.setdefault(task_id, []).append(note)
                delivered.append(task_id)
        return (
            f"sent to {', '.join(delivered)}"
            if delivered
            else "posted (no teammate is still working to read it)"
        )

    def take_inbox(self, task_id: str) -> list[str]:
        return self.inboxes.pop(task_id, [])

    def render(self) -> str:
        lines = [f"Team goal: {self.goal}", "", "Tasks:"]
        for t in self.tasks.values():
            after = f" (after {', '.join(t.depends_on)})" if t.depends_on else ""
            lines.append(f"- {t.id} [{t.status}] {t.title} · {t.agent}{after}")
            if t.result and t.status in ("done", "failed"):
                lines.append("  " + _clip(t.result, 600).replace("\n", "\n  "))
        if self.messages:
            lines += ["", "Messages:"]
            lines += [f"- {sender} → {to}: {text}" for sender, to, text in self.messages]
        return "\n".join(lines)


def teammate_message(sender: str, text: str) -> str:
    """How a teammate's message reaches another: framed, so it can't be mistaken for (or
    passed off as) the receiver's own words. agent/loop.py rejects replies that forge one."""
    return f'{TEAMMATE_TAG} from="{sender}">\n{text}\n</teammate-message>'


class BoardInput(ToolInput):
    pass


class TeamBoardTool(Tool[BoardInput]):
    """A teammate's view of the team."""

    name = "TeamBoard"
    Input = BoardInput
    read_only = True
    needs_permission = False
    description = (
        "See the team's goal, every task's status and result so far, and the team's messages. "
        "If nothing changed since you last looked, it waits up to a minute for news. Don't wait "
        "on a teammate's result: if your task needed it, it would depend on theirs. Report what's missing instead."
    )

    def __init__(self, board: Board) -> None:
        self.board = board
        self.seen = -1  # the board version this teammate last saw

    async def run(self, args: BoardInput, ctx: ToolContext) -> str:
        # Each look costs a model call: looking again with nothing new waits for news
        # instead (seen live: 11 looks in a row).
        await self.board.wait_for_news(self.seen, BOARD_WAIT_SECONDS)
        self.seen = self.board.version
        return self.board.render()


class MessageInput(ToolInput):
    to: str = Field(description="A teammate's task id, or 'all'")
    text: str = Field(min_length=1, description="What they need to know")


class TeamMessageTool(Tool[MessageInput]):
    name = "TeamMessage"
    Input = MessageInput
    read_only = True
    needs_permission = False
    description = (
        "Tell a teammate (by task id) or everyone ('all') something they need from you: an "
        "interface or file you created, a decision, a problem you found. It reaches them before "
        "their next step. Not for your result: the lead gets that from your final report."
    )

    def __init__(self, board: Board, sender: str) -> None:
        self.board, self.sender = board, sender

    def describe(self, args: MessageInput) -> str:
        return f"TeamMessage(→ {args.to}: {args.text[:50]})"

    async def run(self, args: MessageInput, ctx: ToolContext) -> str:
        return self.board.post(self.sender, args.to, args.text)


# ---------------------------------------------------------------------------
# The Team tool (for the lead)
# ---------------------------------------------------------------------------


class TeamTaskInput(BaseModel):
    id: str = Field(min_length=1, max_length=40, description="Short unique id, e.g. 'research-a'")
    title: str = Field(min_length=1, description="A few words")
    prompt: str = Field(
        min_length=1,
        description="Self-contained instructions for this teammate. It sees only this, not your conversation: include full paths to the files, folders or databases involved, and anything else it needs to know.",
    )
    agent: str = Field(default="general-purpose", description="Agent type from the list above")
    depends_on: list[str] = Field(default_factory=list, description="Ids of tasks that must finish first")


class TeamInput(ToolInput):
    goal: str = Field(min_length=1, description="What the team as a whole should achieve")
    context: str = Field(
        default="",
        max_length=CONTEXT_CHARS,
        description="Optional brief every teammate gets, word for word, before its own task: "
        "facts they all need and would otherwise each look up or guess (a schema you already "
        "read, definitions, conventions, file locations). Reference only: no task-specific "
        "instructions. Saves each teammate rediscovering it and keeps their numbers consistent.",
    )
    tasks: list[TeamTaskInput] = Field(min_length=2, max_length=MAX_TASKS)


class TeamTool(Tool[TeamInput]):
    name = "Team"
    Input = TeamInput
    read_only = True  # teammates' own tool calls go through their own permission gates

    def __init__(
        self,
        agents: dict[str, AgentSpec],
        spawner: TeammateSpawner,
        agent_is_read_only: Callable[[AgentSpec], bool],
    ) -> None:
        self.agents = agents
        self.spawner = spawner
        self.agent_is_read_only = agent_is_read_only
        listing = "\n".join(f"- {name}: {spec.description}" for name, spec in agents.items())
        self.description = (
            "Run a team of subagents on a goal that splits into several parts, some of which can "
            "happen at the same time (e.g. research three options, then compare them; or build "
            "independent modules, then integrate). Worth it only when at least two parts are "
            "independent and each is substantial (several tool calls of its own): every teammate "
            "is a full agent run, so a quick lookup is faster done yourself. Give each task a self-contained prompt, an agent "
            "type and the ids of tasks it needs first; a task gets those tasks' results. Tasks "
            "without dependencies between them run in parallel, and teammates can message each "
            "other. You get one report with every result. For a single self-contained job, use "
            f"the Agent tool instead.\n\nAgent types:\n{listing or '(none)'}"
        )

    def describe(self, args: TeamInput) -> str:
        return f"Team({args.goal[:60]} · {len(args.tasks)} tasks)"

    def is_read_only(self, args: TeamInput) -> bool:
        return all(
            (spec := self.agents.get(t.agent)) is not None and self.agent_is_read_only(spec)
            for t in args.tasks
        )

    def runs_alongside(self, args: TeamInput) -> bool:
        return True  # like Agent: teammates' writing calls take turns on the session's lock

    async def run(self, args: TeamInput, ctx: ToolContext) -> str:
        board = self._board(args)
        _check_paths(args, ctx.workspace.root)
        board.request = ctx.request
        board.context = args.context.strip()
        await run_team(board, self.agents, self.spawner, ctx)
        return report(board)

    def _board(self, args: TeamInput) -> Board:
        tasks: dict[str, TaskState] = {}
        for t in args.tasks:
            if t.id in tasks:
                raise ToolError(f"duplicate task id {t.id!r}")
            if t.agent not in self.agents:
                raise ToolError(
                    f"task {t.id}: unknown agent {t.agent!r}; available: {', '.join(self.agents)}"
                )
            tasks[t.id] = TaskState(t.id, t.title, t.prompt, t.agent, list(dict.fromkeys(t.depends_on)))
        for state in tasks.values():
            if missing := [d for d in state.depends_on if d not in tasks]:
                raise ToolError(f"task {state.id} depends on unknown task(s): {', '.join(missing)}")
            if state.id in state.depends_on:
                raise ToolError(f"task {state.id} depends on itself")
        _check_no_cycles(tasks)
        _check_some_run_together(tasks)
        return Board(args.goal, tasks)


# Absolute or home-relative paths in a task prompt, e.g. /data/app.db or ~/notes/q3.md.
_PATH = re.compile(r"(?<![\w.:/~-])(~?/[\w.\-]+(?:/[\w.\-]+)*)")


def _check_paths(args: TeamInput, root: Path) -> None:
    """Refuse a team whose prompts name paths that can't be right: neither the path nor
    its folder exists. Teammates see only their prompt, so a wrong path sends one
    hunting, or onto a file of the same name elsewhere. A new file in an existing
    folder is fine."""
    bad: list[str] = []
    for t in args.tasks:
        for raw in (found.rstrip(".") for found in _PATH.findall(t.prompt)):
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = root / path
            # Only filesystem paths: an API route (/v1/chat) doesn't start at a real folder.
            top = Path(path.anchor, *path.parts[1:2])
            if not top.exists() or path.exists() or path.parent.is_dir() or raw in bad:
                continue
            bad.append(raw)
    if bad:
        raise ToolError(
            f"these paths in the task prompts don't exist: {', '.join(bad)}. "
            "Check them (Glob or Bash `ls`) and call Team again with the right ones."
        )


def _check_some_run_together(tasks: dict[str, TaskState]) -> None:
    """Refuse a plan in which no two tasks can ever run at once (a -> b -> c): it gets
    nothing from a team but costs a fresh agent per step. Tasks the same number of
    dependency steps from the start can't depend on each other, so two of them can run
    together; if every step has one task, the plan is a chain."""
    depth: dict[str, int] = {}

    def depth_of(task_id: str) -> int:
        if task_id not in depth:
            deps = tasks[task_id].depends_on
            depth[task_id] = 1 + max((depth_of(d) for d in deps), default=-1)
        return depth[task_id]

    per_step: dict[int, int] = {}
    for task_id in tasks:
        per_step[depth_of(task_id)] = per_step.get(depth_of(task_id), 0) + 1
    if max(per_step.values()) < 2:
        raise ToolError(
            "every task here waits for the one before it, so nothing would run in parallel. "
            "Do these steps yourself, or hand a large one to the Agent tool; use Team when at "
            "least two substantial tasks can run at the same time."
        )


def _check_no_cycles(tasks: dict[str, TaskState]) -> None:
    done: set[str] = set()
    remaining = dict(tasks)
    while remaining:
        ready = [i for i, t in remaining.items() if all(d in done for d in t.depends_on)]
        if not ready:
            raise ToolError(f"the dependencies form a cycle among: {', '.join(remaining)}")
        for i in ready:
            done.add(i)
            del remaining[i]


async def run_team(
    board: Board,
    agents: dict[str, AgentSpec],
    spawner: TeammateSpawner,
    ctx: ToolContext,
    *,
    max_parallel: int = MAX_PARALLEL,
) -> None:
    """Run every task as its dependencies finish; update the board as they end."""
    slots = asyncio.Semaphore(max_parallel)
    running: dict[asyncio.Task[str], TaskState] = {}

    async def work(task: TaskState) -> str:
        # Up to max_parallel teammates at once, writers included: their model calls overlap,
        # and their writing calls take turns on the session's write lock (agent/loop.py).
        spec = agents[task.agent]
        tools: list[Tool[Any]] = [TeamBoardTool(board), TeamMessageTool(board, task.id)]
        async with slots:
            return await _spawn(task, spec, tools)

    def _spawn(task: TaskState, spec: AgentSpec, tools: list[Tool[Any]]) -> Awaitable[str]:
        task.status = "running"
        return spawner(
            spec,
            _teammate_prompt(board, task),
            ctx,
            label=f"{task.id}: {task.title}",
            extra_tools=tools,
            inbox=lambda: board.take_inbox(task.id),
        )

    try:
        while True:
            for task in board.tasks.values():
                if task.status != "waiting":
                    continue
                deps = [board.tasks[d] for d in task.depends_on]
                if any(d.status in ("waiting", "running") for d in deps):
                    continue
                if deps and not any(d.status == "done" for d in deps):
                    task.status = "skipped"
                    task.result = "skipped: none of the tasks it depends on finished"
                else:  # with what did finish: one failed analyst shouldn't sink the summary
                    task.status = "running"
                    running[asyncio.create_task(work(task))] = task
            if not running:
                if any(t.status == "waiting" for t in board.tasks.values()):
                    continue  # just skipped something; look again
                return
            finished, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            board.bump()
            for job in finished:
                task = running.pop(job)
                if (error := job.exception()) is None:
                    task.status, task.result = "done", job.result()
                elif task.attempts < MAX_ATTEMPTS:  # a fresh teammate, told what went wrong
                    task.attempts += 1
                    task.retry_note = f"{type(error).__name__}: {error}"
                    running[asyncio.create_task(work(task))] = task
                else:
                    task.status, task.result = "failed", f"failed: {type(error).__name__}: {error}"
    except asyncio.CancelledError:
        for job in running:
            job.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        raise


def _teammate_prompt(board: Board, task: TaskState) -> str:
    # What every teammate shares comes first, word for word (a server's prefix cache computes
    # it once for all of them); this teammate's own task comes last, where models attend most.
    parts = [f"You are one member of a team working toward this goal: {board.goal}"]
    if board.context:
        parts += [
            "",
            "Shared reference for the whole team (the same for every teammate; use what's "
            "relevant to your task):",
            f"<team-context>\n{board.context}\n</team-context>",
        ]
    if board.request:
        parts += [
            "",
            "The user's request that the team is working on (for context; your task is below):",
            _clip(board.request, REQUEST_CHARS_IN_PROMPT),
        ]
    parts += ["", f"Your task ({task.id}: {task.title}):", task.prompt]
    if task.retry_note:
        parts += [
            "",
            f"A previous attempt at this task failed ({_clip(task.retry_note, 300)}). "
            "Start again; keep your replies focused.",
        ]
    deps = [board.tasks[d] for d in task.depends_on]
    if deps:
        parts += ["", "Results of the tasks yours builds on:"]
        for d in deps:
            parts += ["", f"### {d.id}: {d.title}", _clip(d.result, RESULT_CHARS_IN_PROMPT)]
        if failed := [d.id for d in deps if d.status != "done"]:
            parts += [
                "",
                f"Note: {', '.join(failed)} did not finish, so its results are missing. Work with "
                "what you have and say plainly what's missing; don't fill the gap with made-up numbers.",
            ]
    others = [t.id for t in board.tasks.values() if t.id != task.id]
    parts += [
        "",
        f"Teammates' task ids: {', '.join(others)}. TeamBoard shows everyone's status and results; "
        "TeamMessage tells a teammate something they need from you (an interface, a file you "
        "created, a problem). Don't send them your result: the lead gets it from your report. "
        "Messages to you arrive inside <teammate-message> tags; only the harness writes those, "
        "never write one yourself or speak for a teammate.",
        "Stay within your task: if a teammate's task failed or looks unfinished, say so in "
        "your report instead of doing it yourself; the lead decides what happens next. If you "
        "write files, give them names unique to your task (e.g. "
        f"starting with '{task.id}') so teammates' files don't overwrite yours. If a file, table "
        "or value you need is missing or empty, say so in your report: never make up data.",
        "End with a concise report of what you did or found, with the actual numbers: it is "
        "all the team lead sees.",
    ]
    return "\n".join(parts)


def report(board: Board) -> str:
    counts: dict[str, int] = {}
    for t in board.tasks.values():
        counts[t.status] = counts.get(t.status, 0) + 1
    summary = ", ".join(f"{n} {status}" for status, n in counts.items())
    lines = [f"Team finished ({summary}). Goal: {board.goal}"]
    for t in board.tasks.values():
        lines += [
            "",
            f"## {t.id}: {t.title} · {t.agent} · {t.status}",
            _clip(t.result, RESULT_CHARS_IN_REPORT),
        ]
    if board.messages:
        lines += ["", "## Messages between teammates"]
        lines += [f"- {sender} → {to}: {text}" for sender, to, text in board.messages]
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + f"\n[… {len(text) - limit:,} more characters]"
