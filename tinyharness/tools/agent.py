from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import Field

from tinyharness.subagents import AgentSpec
from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput

Spawner = Callable[[AgentSpec, str, ToolContext], Awaitable[str]]


class AgentInput(ToolInput):
    subagent_type: str = Field(description="Agent name from the list above")
    prompt: str = Field(min_length=1, description="Self-contained task description for the subagent")
    description: str = Field(default="", description="Short label for this run")


class AgentTool(Tool[AgentInput]):
    """Delegates a task to a subagent with its own, fresh context window.

    Only the subagent's final answer comes back, which keeps the parent's
    context small during broad searches or multi-step side quests.
    """

    name = "Agent"
    Input = AgentInput
    read_only = True  # the subagent's own tool calls go through its own permission gate

    def __init__(self, agents: dict[str, AgentSpec], spawner: Spawner) -> None:
        self.agents = agents
        self.spawner = spawner
        listing = "\n".join(f"- {name}: {spec.description}" for name, spec in agents.items())
        self.description = (
            "Run a subagent in an isolated context window; only its final summary is returned. "
            "Use it for substantial, self-contained work (broad research, independent stages), "
            f"not for single file operations.\n\nAvailable agents:\n{listing or '(none)'}"
        )

    def describe(self, args: AgentInput) -> str:
        return f"Agent({args.subagent_type}: {args.description or args.prompt[:60]})"

    async def run(self, args: AgentInput, ctx: ToolContext) -> str:
        spec = self.agents.get(args.subagent_type)
        if spec is None:
            raise ToolError(
                f"unknown subagent_type {args.subagent_type!r}; available: {', '.join(self.agents)}"
            )
        return await self.spawner(spec, args.prompt, ctx)
