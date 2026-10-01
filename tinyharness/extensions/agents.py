"""Subagents and teams: the Agent tool (one job in a fresh context) and the Team tool (a
graph of tasks, independent ones in parallel, teammates messaging each other). Built only
on the public extension API (th.run_agent), like any other extension could be."""

from collections.abc import Callable, Iterable
from typing import Any

from tinyharness.ext import Extension
from tinyharness.subagents import AgentSpec
from tinyharness.tools.agent import AgentTool
from tinyharness.tools.base import Tool, ToolContext
from tinyharness.tools.team import TeamTool

GUIDANCE = """\
- Agent: one substantial, self-contained job that benefits from a fresh context (a broad
  search, an investigation, a review). Several Agent calls in one reply run in parallel.
- Team: a goal that splits into parts, some independent (research several options then
  compare; build separate pieces then integrate). Say which tasks depend on which;
  independent ones run in parallel and teammates can message each other. Put what they
  all need (a schema you read, definitions, conventions) in its `context`, once.

A subagent sees only the prompt you write, not this conversation: give it the full paths,
names and definitions it needs, and brief it as you would a capable colleague new to the
work: one job, what it's for, how deep to go, when it's done, and the form to deliver it
in (a short report, unless more is needed). When one agent's output feeds another, ask
for exact data, not only prose. Call tools directly for routine steps; a subagent costs a
full model run."""


def setup(th: Extension) -> None:
    session = th.session
    if session is None or not session.agents:
        return

    async def spawn(
        spec: AgentSpec,
        task: str,
        ctx: ToolContext,
        *,
        label: str = "",
        extra_tools: Iterable[Tool[Any]] = (),
        inbox: Callable[[], list[str]] | None = None,
    ) -> str:
        return await th.run_agent(
            task,
            agent=spec.name,
            model=spec.model,
            label=label,
            extra_tools=list(extra_tools),
            inbox=inbox,
            ctx=ctx,
        )

    def read_only(spec: AgentSpec) -> bool:
        """An agent that can't change anything, so several can safely run at once."""
        tools = session.tools
        names = tools.names() if spec.tools is None else spec.tools
        return all(tool.read_only for name in names if (tool := tools.get(name)) is not None)

    th.add_tool(AgentTool(session.agents, spawn, read_only))
    th.add_tool(TeamTool(session.agents, spawn, read_only))
    th.prompt_section("Subagents", GUIDANCE)
