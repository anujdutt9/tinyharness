from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from tinyharness.agent.compaction import Compactor
from tinyharness.agent.history import History, Message
from tinyharness.agent.loop import LoopDeps, run_agent_loop
from tinyharness.agent.transcript import Transcript
from tinyharness.events import AssistantText, SubagentStatus, SubagentUpdate, ToolResult, ToolUse
from tinyharness.subagents.specs import AgentSpec

if TYPE_CHECKING:
    from tinyharness.llm.client import LLMClient
    from tinyharness.permissions import PermissionGate
    from tinyharness.tools.base import Tool, ToolContext
    from tinyharness.tools.registry import ToolRegistry

# Tools that start more agents: a subagent never gets them (one level of delegation).
SPAWNING_TOOLS = frozenset({"Agent", "Team"})


async def run_subagent(
    spec: AgentSpec,
    task: str,
    *,
    ctx: ToolContext,
    llm: LLMClient,
    tools: ToolRegistry,
    gate: PermissionGate,
    max_steps: int = 25,
    label: str = "",
    extra_tools: Iterable[Tool[Any]] = (),
    inbox: Callable[[], list[str]] | None = None,
) -> str:
    """Run the normal agent loop with a fresh history and return only the final answer.

    The subagent shares the parent's workspace but has its own permission gate
    (so its mode applies) and logs to a separate "sidechain" transcript. Progress
    goes to `ctx.notify` as `SubagentUpdate`s. Teammates also get `extra_tools`
    (the team board) and an `inbox` of messages from other teammates.
    Subagents can't spawn further subagents.
    """
    history = History(
        [Message(role="system", content=spec.system_prompt or f"You are the {spec.name} subagent.")]
    )
    extra = list(extra_tools)
    registry = tools.with_tools(extra) if extra else tools
    wanted = set(spec.tools) | {t.name for t in extra} if spec.tools else set(registry.names())
    allowed = wanted - SPAWNING_TOOLS

    run_id = uuid.uuid4().hex[:8]
    started = time.monotonic()
    short = " ".join((label or task).split())[:80]
    used = 0

    def report(status: SubagentStatus, detail: str = "") -> None:
        if ctx.notify is not None:
            ctx.notify(
                SubagentUpdate(
                    run_id,
                    spec.name,
                    short,
                    status,
                    detail,
                    used,
                    time.monotonic() - started,
                )
            )

    with Transcript(ctx.workspace.root, f"{spec.name}-{run_id}", kind="sidechains") as transcript:
        transcript.write("session_start", agent=spec.name, task=task)
        deps = LoopDeps(
            llm=llm,
            tools=registry,
            gate=gate,
            compactor=Compactor(llm),
            transcript=transcript,
            ctx=ctx,
            allowed_tools=allowed,
            model=spec.model,
            max_steps=max_steps,
            inbox=inbox,
        )
        report("started")
        # Keep only the text of the final turn: earlier text is narration between tool calls.
        final: list[str] = []
        try:
            async for event in run_agent_loop(task, history, deps):
                if isinstance(event, ToolUse):
                    used += 1
                    report("working", event.summary)
                elif isinstance(event, ToolResult):
                    final.clear()
                elif isinstance(event, AssistantText):
                    final.append(event.text)
        except BaseException as e:
            report("failed", type(e).__name__ if not str(e) else str(e).splitlines()[0][:120])
            raise
        answer = "".join(final).strip()
        transcript.write("session_end", final=answer[:2000])
        report("done", f"{used} tool call{'s' if used != 1 else ''}")

    return answer or "(subagent finished without a summary)"
