from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from tinyharness.agent.compaction import Compactor
from tinyharness.agent.history import History, Message
from tinyharness.agent.loop import LoopDeps, run_agent_loop
from tinyharness.agent.transcript import Transcript
from tinyharness.events import AssistantText, ToolResult
from tinyharness.subagents.specs import AgentSpec

if TYPE_CHECKING:
    from tinyharness.llm.client import LLMClient
    from tinyharness.permissions import PermissionGate
    from tinyharness.tools.base import ToolContext
    from tinyharness.tools.registry import ToolRegistry


async def run_subagent(
    spec: AgentSpec,
    task: str,
    *,
    ctx: ToolContext,
    llm: LLMClient,
    tools: ToolRegistry,
    gate: PermissionGate,
    max_steps: int = 25,
) -> str:
    """Run the normal agent loop with a fresh history and return only the final answer.

    The subagent shares the parent's workspace but has its own permission gate
    (so its mode applies) and logs to a separate "sidechain" transcript.
    Subagents can't spawn further subagents.
    """
    history = History(
        [Message(role="system", content=spec.system_prompt or f"You are the {spec.name} subagent.")]
    )
    allowed = set(spec.tools or tools.names()) - {"Agent"}

    with Transcript(
        ctx.workspace.root, f"{spec.name}-{uuid.uuid4().hex[:8]}", kind="sidechains"
    ) as transcript:
        transcript.write("session_start", agent=spec.name, task=task)
        deps = LoopDeps(
            llm=llm,
            tools=tools,
            gate=gate,
            compactor=Compactor(llm),
            transcript=transcript,
            ctx=ctx,
            allowed_tools=allowed,
            model=spec.model,
            max_steps=max_steps,
        )
        # Keep only the text of the final turn: earlier text is narration between tool calls.
        final: list[str] = []
        async for event in run_agent_loop(task, history, deps):
            if isinstance(event, ToolResult):
                final.clear()
            elif isinstance(event, AssistantText):
                final.append(event.text)
        answer = "".join(final).strip()
        transcript.write("session_end", final=answer[:2000])

    return answer or "(subagent finished without a summary)"
