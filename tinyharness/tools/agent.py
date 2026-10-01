from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import Field

from tinyharness.subagents import AgentSpec
from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput

DEFAULT_AGENT = "general-purpose"
Spawner = Callable[..., Awaitable[str]]  # (spec, prompt, ctx, *, label=...) -> final answer


class AgentInput(ToolInput):
    # Optional: models often leave it out (other harnesses default it), and a required
    # field made those calls fail and be retried.
    subagent_type: str = Field(
        default="",
        description="Which agent, from the list in this tool's description (default: general-purpose)",
    )
    prompt: str = Field(
        min_length=1,
        description="Self-contained task description for the subagent. It sees only this, not your conversation: include full paths to the files, folders or databases involved, and anything else it needs to know.",
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
    ) -> None:
        self.agents = agents
        self.spawner = spawner
        self.agent_is_read_only = agent_is_read_only
        listing = "\n".join(f"- {name}: {spec.description}" for name, spec in agents.items())
        self.description = (
            "Run a subagent in an isolated context window; only its final summary is returned. "
            "Use it for substantial, self-contained work (broad research, independent stages), "
            f"not for single file operations.\n\nAvailable agents:\n{listing or '(none)'}"
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
        return await self.spawner(spec, args.prompt, ctx, label=args.description)
