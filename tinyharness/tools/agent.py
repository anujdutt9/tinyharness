from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable, Callable

from pydantic import Field

from tinyharness.subagents import AgentSpec
from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput

DEFAULT_AGENT = "general-purpose"
# Subagents running at once across a session, as in Claude Code: the model decides how
# many to start, this is only the ceiling (config: max_parallel_agents).
MAX_PARALLEL_AGENTS = 20
Spawner = Callable[..., Awaitable[str]]  # (spec, prompt, ctx, *, label=...) -> final answer


class AgentSlots:
    """The session's ceiling on subagents running at once, shared by Agent calls and Team
    teammates. An Agent call past it fails right away, so the model can wait for the
    running ones (Claude Code does the same); a Team task waits for a free slot instead,
    since the team's plan is already laid out."""

    def __init__(self, limit: int = MAX_PARALLEL_AGENTS) -> None:
        self.limit = limit
        self.running = 0
        self._changed = asyncio.Condition()

    @contextlib.asynccontextmanager
    async def hold(self, *, wait: bool) -> AsyncIterator[None]:
        async with self._changed:
            if not wait and self.running >= self.limit:
                raise ToolError(
                    f"concurrent subagent limit reached ({self.limit} running): wait for some "
                    "to finish before starting more, or do this step yourself"
                )
            await self._changed.wait_for(lambda: self.running < self.limit)
            self.running += 1
        try:
            yield
        finally:
            async with self._changed:
                self.running -= 1
                self._changed.notify()


class AgentInput(ToolInput):
    # Optional: models often leave it out (other harnesses default it), and a required
    # field made those calls fail and be retried.
    subagent_type: str = Field(
        default="",
        description="Which agent, from the list in this tool's description (default: general-purpose)",
    )
    prompt: str = Field(
        min_length=1,
        description="Self-contained task description for the subagent. It sees only this, not your conversation: include full paths to the files, folders or databases involved, and anything else it needs to know. Brief it like a capable colleague new to the work: what it's for, how deep to go, and what to deliver (exact data, if another task builds on it).",
    )
    description: str = Field(default="", description="Short label for this run")


class AgentTool(Tool[AgentInput]):
    """Delegates a task to a subagent with its own, fresh context window.

    Only the subagent's final answer comes back, which keeps the parent's
    context small during broad searches or multi-step side quests.
    """

    name = "Agent"
    Input = AgentInput
    read_only = True  # the subagent's own tool calls go through its own permission gate

    def __init__(
        self,
        agents: dict[str, AgentSpec],
        spawner: Spawner,
        agent_is_read_only: Callable[[AgentSpec], bool] = lambda spec: False,
        slots: AgentSlots | None = None,
    ) -> None:
        self.agents = agents
        self.spawner = spawner
        self.agent_is_read_only = agent_is_read_only
        self.slots = slots or AgentSlots()
        listing = "\n".join(f"- {name}: {spec.description}" for name, spec in agents.items())
        self.description = (
            "Run a subagent in an isolated context window; only its final summary is returned. "
            "Use it for substantial, self-contained work (broad research, independent stages), "
            f"not for single file operations. Up to {self.slots.limit} subagents run at once."
            f"\n\nAvailable agents:\n{listing or '(none)'}"
        )

    def agent_name(self, args: AgentInput) -> str:
        """The agent to run: the one named, else general-purpose (or the first there is)."""
        if args.subagent_type:
            return args.subagent_type
        return DEFAULT_AGENT if DEFAULT_AGENT in self.agents else next(iter(self.agents), DEFAULT_AGENT)

    def is_read_only(self, args: AgentInput) -> bool:
        """Whether the agent can only look (no Write, Edit or Bash)."""
        spec = self.agents.get(self.agent_name(args))
        return spec is not None and self.agent_is_read_only(spec)

    def runs_alongside(self, args: AgentInput) -> bool:
        """Every agent can: their model calls overlap, and their writing calls take turns
        (the session's write lock), so two never change files at the same moment."""
        return True

    def describe(self, args: AgentInput) -> str:
        return f"Agent({self.agent_name(args)}: {args.description or args.prompt[:60]})"

    async def run(self, args: AgentInput, ctx: ToolContext) -> str:
        name = self.agent_name(args)
        spec = self.agents.get(name)
        if spec is None:
            raise ToolError(f"unknown subagent_type {name!r}; available: {', '.join(self.agents)}")
        async with self.slots.hold(wait=False):
            return await self.spawner(spec, args.prompt, ctx, label=args.description)
