"""The agent loop.

    user prompt
       │
       ▼
    ┌─► compact context ─► call model (stream) ─► no tool calls? ──► Done
    │                                                │
    │                                                ▼
    │                          validate args ─► permission gate
    │                                                │
    └──── append results ◄── run tools (parallel if all read-only)

Everything the loop does is reported as an `Event`, so the CLI and SDK render
the same stream.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from tinyharness.agent.history import History, Message, ToolCall
from tinyharness.events import (
    AssistantText,
    Compacted,
    Done,
    Event,
    PermissionDecision,
    TodosUpdated,
    ToolResult,
    ToolsDispatched,
    ToolUse,
)
from tinyharness.llm.client import ToolCallAccumulator
from tinyharness.tools.base import Tool, ToolContext, ToolError

if TYPE_CHECKING:
    from tinyharness.agent.compaction import Compactor
    from tinyharness.agent.transcript import Transcript
    from tinyharness.hooks import HookRunner
    from tinyharness.llm.client import LLMClient
    from tinyharness.permissions import PermissionGate
    from tinyharness.tools.registry import ToolRegistry

TRANSCRIPT_RESULT_CHARS = 4000


@dataclass
class LoopDeps:
    """Everything the loop needs, passed in explicitly so it's easy to test and reuse."""

    llm: LLMClient
    tools: ToolRegistry
    gate: PermissionGate
    compactor: Compactor
    transcript: Transcript
    ctx: ToolContext
    hooks: HookRunner | None = None
    allowed_tools: set[str] | None = None
    model: str | None = None  # override the client's default model (subagents)
    max_steps: int = 50


@dataclass
class _ApprovedCall:
    call: ToolCall
    tool: Tool[Any]
    args: Any  # the tool's validated Input model


@dataclass
class _Batch:
    """Outcome of permission-checking one turn's tool calls."""

    approved: list[_ApprovedCall] = field(default_factory=list)
    interrupted: bool = False  # the user said "no": skip everything else in this turn
    feedback: str = ""  # ...and optionally told the agent what to do instead


@dataclass
class _Turn:
    text: list[str] = field(default_factory=list)
    calls: ToolCallAccumulator = field(default_factory=ToolCallAccumulator)


async def run_agent_loop(prompt: str, history: History, deps: LoopDeps) -> AsyncGenerator[Event, None]:
    await _fire_hook(deps, "UserPromptSubmit", {"prompt": prompt})
    deps.transcript.write("user", text=prompt)
    history.append(Message(role="user", content=prompt))

    for _ in range(deps.max_steps):
        async for event in _compact(history, deps):
            yield event

        # 1. Ask the model, streaming text to the caller as it arrives.
        available = {t.name: t for t in deps.tools.available(deps.gate.mode, deps.allowed_tools)}
        turn = _Turn()
        async for chunk in deps.llm.stream(
            history.to_openai(), [t.to_openai() for t in available.values()], model=deps.model
        ):
            if chunk.text:
                turn.text.append(chunk.text)
                yield AssistantText(chunk.text)
            if chunk.tool_call_delta:
                turn.calls.add(chunk.tool_call_delta)

        text, calls = "".join(turn.text), turn.calls.calls()
        deps.transcript.write("assistant", text=text, tool_calls=[c.to_openai() for c in calls])
        history.append(Message(role="assistant", content=text or None, tool_calls=calls))

        # 2. No tool calls means the model is finished.
        if not calls:
            await _fire_hook(deps, "Stop", {})
            yield Done()
            return

        # 3. Validate and permission-check each call, one at a time (prompts can't interleave).
        batch = _Batch()
        for call in calls:
            if batch.interrupted:
                history.add_tool_result(call.id, "(not run: the user interrupted)")
                continue
            async for event in _prepare_call(call, available, history, deps, batch):
                yield event

        # The user said "no". Skip the rest of the batch (it was planned without their
        # input). With feedback, it becomes their next message; otherwise stop.
        if batch.interrupted:
            for item in batch.approved:
                history.add_tool_result(item.call.id, "(not run: the user interrupted)")
            if not batch.feedback:
                yield Done(reason="interrupted")
                return
            deps.transcript.write("user", text=batch.feedback)
            history.append(Message(role="user", content=batch.feedback))
            continue

        approved = batch.approved
        if not approved:
            continue

        # 4. Run the approved calls. Read-only batches run concurrently.
        parallel = len(approved) > 1 and all(a.tool.read_only for a in approved)
        yield ToolsDispatched([a.tool.name for a in approved], parallel)
        if parallel:
            results = await asyncio.gather(*(_execute(a, deps) for a in approved))
        else:
            results = [await _execute(a, deps) for a in approved]

        # 5. Feed results back in the order the model asked for them.
        for item, (output, is_error) in zip(approved, results, strict=True):
            await _fire_hook(
                deps,
                "PostToolUse",
                {
                    "tool": item.tool.name,
                    "args": item.args.model_dump(),
                    "result": output[:TRANSCRIPT_RESULT_CHARS],
                },
            )
            deps.transcript.write("tool_result", tool=item.tool.name, result=output[:TRANSCRIPT_RESULT_CHARS])
            history.add_tool_result(item.call.id, output)
            yield ToolResult(item.tool.name, output, is_error)
            if item.tool.name == "TodoWrite" and not is_error:
                yield TodosUpdated(list(deps.ctx.todos))

    yield Done(reason="max_steps")


async def _compact(history: History, deps: LoopDeps) -> AsyncGenerator[Event, None]:
    compactor = deps.compactor
    if n := compactor.budget_cap(history):
        deps.transcript.write("compaction", kind="budget_cap", count=n)
        yield Compacted("budget_cap", n)
    if n := compactor.microcompact(history):
        deps.transcript.write("compaction", kind="microcompact", count=n)
        yield Compacted("microcompact", n)
    if compactor.needs_summary(history) and await compactor.summarize(history):
        deps.transcript.write("compaction", kind="summary")
        yield Compacted("summary")


async def _prepare_call(
    call: ToolCall,
    available: dict[str, Tool[Any]],
    history: History,
    deps: LoopDeps,
    batch: _Batch,
) -> AsyncGenerator[Event, None]:
    """Parse, validate and permission-check one call. Rejections are answered
    immediately (the model must get a result for every call); approvals are
    added to `batch`."""

    def reject(message: str) -> ToolResult:
        deps.transcript.write("tool_result", tool=call.name, error=message)
        history.add_tool_result(call.id, message)
        return ToolResult(call.name, message, is_error=True)

    tool = available.get(call.name)
    if tool is None:
        yield reject(f"error: unknown or unavailable tool {call.name!r}")
        return

    try:
        raw = json.loads(call.arguments or "{}")
        args = tool.parse(raw)
    except (json.JSONDecodeError, ValidationError) as e:
        yield reject(f"error: invalid arguments for {tool.name}: {e}")
        return

    summary = tool.describe(args)
    yield ToolUse(tool.name, raw, summary)
    deps.transcript.write("tool_use", tool=tool.name, args=raw)

    decision = await deps.gate.check(
        tool, raw, tool.target_path(args), summary=summary, history_tail=_tail(history)
    )
    deps.transcript.write(
        "permission", tool=tool.name, allow=decision.allow, source=decision.source, reason=decision.reason
    )
    yield PermissionDecision(tool.name, decision.allow, decision.source, decision.reason)

    if decision.allow:
        batch.approved.append(_ApprovedCall(call, tool, args))
    elif decision.interrupt:
        batch.interrupted, batch.feedback = True, decision.feedback
        yield reject("The user rejected this tool call.")
    else:
        yield reject(f"denied by {decision.source}: {decision.reason}")


async def _execute(item: _ApprovedCall, deps: LoopDeps) -> tuple[str, bool]:
    """Run one tool. Errors become text for the model rather than crashing the loop."""
    try:
        return await item.tool.run(item.args, deps.ctx), False
    except ToolError as e:
        return f"error: {e}", True
    except Exception as e:
        return f"error: {type(e).__name__}: {e}", True


async def _fire_hook(deps: LoopDeps, event: str, payload: dict[str, Any]) -> None:
    if deps.hooks is not None:
        await deps.hooks.fire(event, payload)


def _tail(history: History, messages: int = 6, chars: int = 500) -> str:
    """Recent context for the auto-mode classifier."""
    return "\n".join((m.content or "")[:chars] for m in history.messages[-messages:])
