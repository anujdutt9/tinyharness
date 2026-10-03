from __future__ import annotations

import asyncio
import dataclasses
import datetime
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tinyharness.agent.compaction import Compactor
from tinyharness.agent.history import History, Message
from tinyharness.agent.loop import STEP_LIMIT_RESULT, LoopDeps, run_agent_loop
from tinyharness.agent.prompts import environment, load_agent_md
from tinyharness.agent.transcript import Transcript
from tinyharness.events import (
    AssistantText,
    ModelStuck,
    Reasoning,
    SubagentCall,
    SubagentPhase,
    SubagentStatus,
    SubagentUpdate,
    ToolInputStreaming,
    ToolResult,
    ToolUse,
)
from tinyharness.llm.providers import DEFAULT_CONTEXT
from tinyharness.llm.tokens import model_counter
from tinyharness.llm.types import LLMError
from tinyharness.subagents.specs import AgentSpec

if TYPE_CHECKING:
    from tinyharness.llm.client import LLMClient
    from tinyharness.permissions import PermissionGate
    from tinyharness.tools.base import Tool, ToolContext
    from tinyharness.tools.registry import ToolRegistry

# Tools that start more agents: a subagent never gets them (one level of delegation).
SPAWNING_TOOLS = frozenset({"Agent", "Team"})
# The todo list is the main agent's plan, shown to the user; a subagent's TodoWrite would
# replace it. A subagent's job is one piece of that plan, so it doesn't get one. Memory is
# what the user and the main agent agree is worth keeping; a subagent sees too little of
# the conversation to judge that.
MAIN_AGENT_ONLY = frozenset({"TodoWrite", "Memory"})


RECENT_CALLS = 3  # a subagent's latest tool calls, shown under it while it runs
SUBAGENT_MAX_STEPS = 40  # model calls; a data-analysis teammate needs ~20-30 (schema, queries, retries)


@dataclass
class _Call:
    """A subagent's tool call, as it goes."""

    id: str
    summary: str
    started: float = field(default_factory=time.monotonic)
    done: bool = False
    ok: bool = True
    result: str = ""
    elapsed: float = 0.0

    def finish(self, output: str, *, ok: bool) -> None:
        self.done, self.ok, self.elapsed = True, ok, time.monotonic() - self.started
        lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
        # A failed command's outcome is its last line ("exit code 1"); anything else's, its first.
        line = (lines[-1] if not ok and lines else lines[0]) if lines else ""
        self.result = line[:100]

    def public(self) -> SubagentCall:
        elapsed = self.elapsed if self.done else time.monotonic() - self.started
        return SubagentCall(self.summary, self.done, self.ok, self.result, elapsed)


async def run_subagent(
    spec: AgentSpec,
    task: str,
    *,
    ctx: ToolContext,
    llm: LLMClient,
    tools: ToolRegistry,
    gate: PermissionGate,
    model: str | None = None,
    max_steps: int = SUBAGENT_MAX_STEPS,
    label: str = "",
    extra_tools: Iterable[Tool[Any]] = (),
    inbox: Callable[[], list[str]] | None = None,
    write_lock: asyncio.Lock | None = None,
) -> str:
    """Run the normal agent loop with a fresh history and return only the final answer.

    `llm` and `model` are what the subagent runs on (Session.llm_for resolves the
    spec's `model` setting, which may name another provider); None = the client's model.

    The subagent shares the parent's workspace but has its own permission gate
    (so its mode applies) and logs to a separate "sidechain" transcript. Progress
    goes to `ctx.notify` as `SubagentUpdate`s. Teammates also get `extra_tools`
    (the team board) and an `inbox` of messages from other teammates.
    Subagents can't spawn further subagents.
    """
    # Its own record of the files it has read: what's in another agent's context isn't in
    # this one's, so a teammate's edit must count as a change it hasn't seen. The rest of
    # the context (usage meter, todos, the user's request, progress events) stays shared.
    ctx = dataclasses.replace(ctx, read_paths=set(), seen_versions={})
    history = History([Message(role="system", content=subagent_system_prompt(spec, ctx))])
    extra = list(extra_tools)
    registry = tools.with_tools(extra) if extra else tools
    wanted = set(registry.names()) if spec.tools is None else set(spec.tools) | {t.name for t in extra}
    allowed = wanted - SPAWNING_TOOLS - MAIN_AGENT_ONLY

    run_id = uuid.uuid4().hex[:8]
    started = time.monotonic()
    short = " ".join((label or task).split())[:80]
    model_id = model or llm.model
    used = 0
    doing: tuple[SubagentPhase, str, str] | None = None  # last (phase, tool, detail) reported
    tool_calls: list[_Call] = []  # its tool calls so far; the last few go on screen
    log = ""

    def recent() -> tuple[SubagentCall, ...]:
        return tuple(call.public() for call in tool_calls[-RECENT_CALLS:])

    def report(
        status: SubagentStatus, detail: str = "", phase: SubagentPhase = "tool", tool: str = ""
    ) -> None:
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
                    model_id,
                    phase,
                    tool,
                    recent(),
                    log,
                )
            )

    def working(phase: SubagentPhase, detail: str = "", tool: str = "") -> None:
        """Report what it's doing now, when that changed (reasoning streams many events)."""
        nonlocal doing
        if doing != (phase, tool, detail):
            doing = (phase, tool, detail)
            report("working", detail, phase, tool)

    with Transcript(ctx.workspace.root, f"{spec.name}-{run_id}", kind="sidechains") as transcript:
        log = str(transcript.path)
        transcript.write("session_start", agent=spec.name, model=model_id, task=task)
        deps = LoopDeps(
            llm=llm,
            tools=registry,
            gate=gate,
            compactor=Compactor(
                llm,
                context_window=_context_window(llm, model),
                max_output=_max_output(llm, model),
                counter=lambda: model_counter(_tokenizer(llm, model)),
                reserved=ctx.config.compaction_reserved,
                preserve_recent_tokens=ctx.config.compaction_preserve_recent_tokens,
                threshold=ctx.config.auto_compact_threshold,
                # Unset: the subagent's own model, not the session's (its client may be the session's).
                model=ctx.config.compaction_model or model,
            ),
            transcript=transcript,
            ctx=ctx,
            allowed_tools=allowed,
            model=model,
            max_steps=max_steps,
            inbox=inbox,
            write_lock=write_lock,
        )
        report("started")
        # Text since the last tool result: what it was saying, if it fails (see SubagentFailed).
        final: list[str] = []
        try:
            async for event in run_agent_loop(task, history, deps):
                if isinstance(event, ToolUse):
                    used += 1
                    tool_calls.append(_Call(event.id, event.summary))
                    working("tool", event.summary)
                elif isinstance(event, ToolResult):
                    final.clear()
                    if call := next((c for c in tool_calls if c.id == event.id and not c.done), None):
                        call.finish(event.output, ok=not event.is_error)
                    doing = None  # a finished call is news, even if the phase didn't change
                    working("waiting")
                elif isinstance(event, Reasoning):
                    working("thinking")
                elif isinstance(event, ToolInputStreaming):
                    working("writing", event.target, event.name)
                elif isinstance(event, AssistantText):
                    final.append(event.text)
                elif isinstance(event, ModelStuck):
                    final.clear()  # that reply was discarded, so it isn't the answer either
        except BaseException as e:
            transcript.write("session_failed", error=f"{type(e).__name__}: {e}"[:500])
            report("failed", type(e).__name__ if not str(e) else str(e).splitlines()[0][:120])
            if isinstance(e, Exception):  # not a cancellation: its work so far isn't lost
                raise SubagentFailed(e, "".join(final), [c.summary for c in tool_calls], log) from e
            raise
        # The answer is its last reply as kept in the history: not every streamed piece since
        # the last tool call (a reply the harness asked it to redo would be glued on).
        # A reply that made tool calls is narration ("Let me read..."), not an answer, unless
        # it's the step-limit wrap-up, whose calls weren't run.
        last = next((m for m in reversed(history.messages) if m.role == "assistant"), None)
        final_reply = last is not None and (
            not last.tool_calls or history.messages[-1].content == STEP_LIMIT_RESULT
        )
        answer = (last.content or "").strip() if last is not None and final_reply else ""
        transcript.write("session_end", final=answer[:2000])
        report("done", f"{used} tool call{'s' if used != 1 else ''}")

    return answer or "(subagent finished without a summary)"


class SubagentFailed(LLMError):
    """A subagent stopped before finishing (stuck, a provider error). The message carries
    what it had done and said by then, so the caller can use the partial work or decide
    what to do instead of starting over blind."""

    def __init__(self, error: Exception, text: str, calls: list[str], log: str) -> None:
        parts = [f"{type(error).__name__}: {error}"]
        if calls:
            shown = calls[-8:]
            more = f" (the last {len(shown)} of {len(calls)})" if len(calls) > len(shown) else ""
            parts.append(f"Before it stopped it had made these tool calls{more}: " + "; ".join(shown))
        if text.strip():
            parts.append(f"Its last words: {text.strip()[-1500:]}")
        parts.append(f"Its full transcript: {log}")
        super().__init__("\n".join(parts))
        self.error = error


def subagent_system_prompt(spec: AgentSpec, ctx: ToolContext) -> str:
    """The agent's own prompt plus where it is working. Without this a subagent doesn't
    know its working directory or today's date: it guesses paths (and can end up
    querying an empty file of the same name) and misreads "last month"."""
    body = spec.system_prompt.strip() or f"You are the {spec.name} subagent."
    env = [
        "# Environment",
        f"Working directory: {ctx.workspace.root} (relative paths resolve here; "
        "files you create go here unless your task says otherwise).",
    ]
    if ctx.scratch is not None:
        env.append(f"Scratch directory for intermediate files: {ctx.scratch.dir}")
    env.append(f"Today's date: {datetime.date.today().isoformat()}")
    parts = [body, "\n".join(env), environment(ctx.workspace.root).replace("# Environment\n", "", 1)]
    if ctx.skills_prompt:  # subagents can load skills too: by reading SKILL.md
        parts.append(ctx.skills_prompt)
    # The project's own rules (how to run tests, which package manager) bind subagents too.
    if instructions := load_agent_md(ctx.workspace.root):
        parts.append(f"# Project instructions (AGENTS.md)\n\n{instructions}")
    return "\n\n".join(parts)


def _tokenizer(llm: Any, model: str | None) -> str | None:
    """The Hugging Face repo with the subagent's model's tokenizer, if the config names one."""
    info = llm.model_info(model) if hasattr(llm, "model_info") else None
    return info.tokenizer if info is not None else None


def _max_output(llm: Any, model: str | None) -> int | None:
    """The subagent's model's output limit, if the provider or the config says."""
    info = llm.model_info(model) if hasattr(llm, "model_info") else None
    return info.max_output if info is not None else None


def _context_window(llm: Any, model: str | None) -> int:
    """The subagent's model's window, as the provider reports it (a subagent may run on a
    smaller model than the session)."""
    info = llm.model_info(model) if hasattr(llm, "model_info") else None
    return info.context if info is not None and info.context else DEFAULT_CONTEXT
