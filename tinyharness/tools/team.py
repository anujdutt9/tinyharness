"""Agent teams: the lead splits a goal into tasks, teammates work them in parallel.

    Team(goal, tasks=[{id, title, prompt, agent, depends_on}, ...])

A task starts as soon as every task it depends on is done, and gets their results
in its prompt. Teammates are subagents with two more tools: TeamBoard (every
task's status and result so far, and the team's messages) and TeamMessage (tell
one teammate, by task id, or everyone). A message reaches a running teammate
before its next step. Read-only teammates run side by side; teammates that can
write files take turns, so two never edit at once. If a task fails, the tasks
that depend on it are skipped. The lead gets one report with every result.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from tinyharness.subagents import AgentSpec
from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput

MAX_TASKS = 12
MAX_PARALLEL = 4
RESULT_CHARS_IN_PROMPT = 4_000  # a dependency's result, as passed to the next teammate
RESULT_CHARS_IN_REPORT = 6_000  # each result, in the report for the lead

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


@dataclass
class Board:
    goal: str
    tasks: dict[str, TaskState]
    messages: list[tuple[str, str, str]] = field(default_factory=list)  # (from, to, text)
    inboxes: dict[str, list[str]] = field(default_factory=dict)

    def post(self, sender: str, to: str, text: str) -> str:
        if to != "all" and to not in self.tasks:
            raise ToolError(f"no task {to!r}; send to a task id ({', '.join(self.tasks)}) or 'all'")
        self.messages.append((sender, to, text))
        note = f"[message from teammate {sender}] {text}"
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


class BoardInput(ToolInput):
    pass


class TeamBoardTool(Tool[BoardInput]):
    """A teammate's view of the team."""

    name = "TeamBoard"
    Input = BoardInput
    read_only = True
    needs_permission = False
    description = "See the team's goal, every task's status and result so far, and the team's messages."

    def __init__(self, board: Board) -> None:
        self.board = board

    async def run(self, args: BoardInput, ctx: ToolContext) -> str:
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
        "Tell a teammate (by task id) or everyone ('all') something they need: an interface or "
        "file you created, a decision, a problem you found. It reaches them before their next step."
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
    prompt: str = Field(min_length=1, description="Self-contained instructions for this teammate")
    agent: str = Field(default="general-purpose", description="Agent type from the list above")
    depends_on: list[str] = Field(default_factory=list, description="Ids of tasks that must finish first")


class TeamInput(ToolInput):
    goal: str = Field(min_length=1, description="What the team as a whole should achieve")
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
            "independent modules, then integrate). Give each task a self-contained prompt, an agent "
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

    async def run(self, args: TeamInput, ctx: ToolContext) -> str:
        board = self._board(args)
        await run_team(board, self.agents, self.spawner, self.agent_is_read_only, ctx)
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
        return Board(args.goal, tasks)


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
    agent_is_read_only: Callable[[AgentSpec], bool],
    ctx: ToolContext,
    *,
    max_parallel: int = MAX_PARALLEL,
) -> None:
    """Run every task as its dependencies finish; update the board as they end."""
    slots = asyncio.Semaphore(max_parallel)
    one_writer = asyncio.Lock()
    running: dict[asyncio.Task[str], TaskState] = {}

    async def work(task: TaskState) -> str:
        spec = agents[task.agent]
        tools: list[Tool[Any]] = [TeamBoardTool(board), TeamMessageTool(board, task.id)]
        async with slots:
            if agent_is_read_only(spec):
                return await _spawn(task, spec, tools)
            async with one_writer:
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
                if any(d.status in ("failed", "skipped") for d in deps):
                    task.status = "skipped"
                    task.result = "skipped: a task it depends on didn't finish"
                elif all(d.status == "done" for d in deps):
                    task.status = "running"
                    running[asyncio.create_task(work(task))] = task
            if not running:
                if any(t.status == "waiting" for t in board.tasks.values()):
                    continue  # just skipped something; look again
                return
            finished, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for job in finished:
                task = running.pop(job)
                if (error := job.exception()) is not None:
                    task.status, task.result = "failed", f"failed: {type(error).__name__}: {error}"
                else:
                    task.status, task.result = "done", job.result()
    except asyncio.CancelledError:
        for job in running:
            job.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        raise


def _teammate_prompt(board: Board, task: TaskState) -> str:
    parts = [
        f"You are one member of a team working toward this goal: {board.goal}",
        "",
        f"Your task ({task.id}: {task.title}):",
        task.prompt,
    ]
    deps = [board.tasks[d] for d in task.depends_on]
    if deps:
        parts += ["", "Results of the tasks yours builds on:"]
        for d in deps:
            parts += ["", f"### {d.id}: {d.title}", _clip(d.result, RESULT_CHARS_IN_PROMPT)]
    others = [t.id for t in board.tasks.values() if t.id != task.id]
    parts += [
        "",
        f"Teammates' task ids: {', '.join(others)}. TeamBoard shows everyone's status and results; "
        "TeamMessage tells a teammate something they need (an interface, a file you created, a "
        "problem). Messages to you appear in your conversation.",
        "Stay within your task. End with a concise report of what you did or found: it is all "
        "the team lead sees.",
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
