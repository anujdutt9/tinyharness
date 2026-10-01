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
import contextlib
import functools
import json
import re
from collections.abc import AsyncGenerator, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from tinyharness.agent.history import History, Image, Message, ReasoningState, ToolCall
from tinyharness.agent.runaway import RunawayGuard
from tinyharness.agent.world import file_meta
from tinyharness.events import (
    AssistantText,
    Compacted,
    Done,
    Event,
    MessageDelivered,
    ModelStuck,
    PermissionDecision,
    Reasoning,
    TodosUpdated,
    TokenUsage,
    ToolInputStreaming,
    ToolResult,
    ToolsDispatched,
    ToolUse,
)
from tinyharness.images import fit_image
from tinyharness.llm.client import ContextOverflowError, ImageTooLargeError, ToolCallAccumulator
from tinyharness.llm.textcalls import looks_like_a_call, recover_calls, repair_arguments
from tinyharness.llm.types import LLMError, RepeatedFailureError, RunawayOutputError, StreamChunk
from tinyharness.llm.usage import Usage
from tinyharness.security import mask_secrets
from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolOutput
from tinyharness.tools.todo import format_todos

if TYPE_CHECKING:
    from tinyharness.agent.compaction import Compactor
    from tinyharness.agent.sessionlog import LogWriter
    from tinyharness.hooks import HookRunner
    from tinyharness.llm.client import LLMClient
    from tinyharness.permissions import PermissionGate
    from tinyharness.tools.registry import ToolRegistry

TRANSCRIPT_RESULT_CHARS = 4000
# Sent to the model after a reply of theirs got stuck and was stopped.
STUCK_NOTE = (
    "[tinyharness] Your last reply was stopped because {reason}, and was discarded. "
    "Carry on with the task from where you were; keep the reply focused."
)
# A reply with no text and no tool calls isn't an answer. Usually the server failed to parse
# the model's tool call and dropped it (seen with a hosted Qwen3-Coder: ~100 output tokens
# billed, nothing returned), so the same request is first sent again as it is; only then is
# the model told, in case it's the model.
SILENT_RETRIES = 2
EMPTY_NOTE = (
    "[tinyharness] Your last reply was empty: no text and no tool call. Carry on with the "
    "task: call the next tool, or if you are finished, give your final answer."
)
MAX_EMPTY_REPLIES = SILENT_RETRIES + 2  # in a row, before the turn is stopped
# Sent after a text reply that hit the output limit, so it can finish what it was saying.
CUT_OFF_NOTE = (
    "[tinyharness] Your last reply was cut off at the output token limit. Continue exactly "
    "where it stopped; don't repeat what you already wrote."
)
# The result for a tool call whose arguments were cut off by the output limit. It isn't
# run: what it would write is incomplete. Without this the model only hears "invalid JSON"
# and sends the same oversized call again.
CUT_OFF_CALL = (
    "error: not run: your reply hit the output token limit while writing this call, so its "
    "arguments are incomplete. Make it smaller: write a large file in parts (Write the first "
    "part, then add the rest with Edit), or change only what's needed with Edit."
)
# Sent when the step limit is reached, for one last reply without tools.
STEP_LIMIT_NOTE = (
    "[tinyharness] You have reached the step limit for this task, so no more tools will run. "
    "Reply now with your final answer: what you found or did, with the actual results, and "
    "what is left unfinished."
)
STEP_LIMIT_RESULT = "(not run: the step limit was reached)"
# A tool failing the same way again and again: the model is told to change approach after
# NOTE_AT failures in a row, and the turn stops at STOP_AT (seen live: 11 failing edits).
REPEAT_NOTE_AT = 2
REPEAT_STOP_AT = 4
_CHANGE_APPROACH = {
    "Edit": "Read the file again and copy the lines to replace exactly as they are now (your "
    "earlier edits changed it), or edit a smaller block, or rewrite the file with Write.",
    "Bash": "Read the error and fix the command, or try a different command or tool.",
}
RETRY_IMAGE_SIDE = 1568  # px: what images shrink to after a provider rejects one (Anthropic's sweet spot)
# Asked once a turn when it changed code and ran nothing since: finishing without checking
# is the most common way agents fail (Anthropic's long-running harness post; LangChain's
# pre-completion checklist). Docs and notes don't count as code.
UNVERIFIED_NOTE = (
    "[tinyharness] You changed {files} and haven't run anything since. Before you finish, run "
    "what checks it (the tests for it, a build, the linter, or rendering a page and looking) if "
    "there is something, and fix what fails. If nothing can check it, say so in your answer."
)
_NOT_CODE = (".md", ".txt", ".rst", ".csv", ".log")
MAX_STOP_BLOCKS = 2  # a Stop hook can keep a turn going this many times
# A reply ending in a tool call written as text that couldn't be read (seen live from Gemma 4:
# a made-up tool, a command that didn't parse). Answered this many times per turn.
MAX_UNREADABLE_CALLS = 2
UNREADABLE_CALL_NOTE = (
    "[tinyharness] Your last reply ends in what looks like a tool call written as text, and it "
    "couldn't be run: it names a tool that doesn't exist, or its arguments couldn't be read. "
    "Make the call through the tool-calling interface, with one of the available tools: {tools}."
)
# A teammate's answer is its last reply. When that reply came right after a teammate's
# message, it's usually an answer to the message ("waiting for your numbers"), not the
# report the lead needs (seen live: placeholders and invented numbers as "results").
REPORT_NOTE = (
    "[tinyharness] That reply ends your task, and your last reply is all the lead sees. "
    "Write your final report now: what you did and found, with the actual results. Don't "
    "wait for teammates; say what's missing instead."
)
TEAMMATE_NOTE = "<teammate-message"  # how the team board delivers a teammate's message
# Sent when a reply contains a made-up teammate message: models copy the framing they see
# and "report" other teammates' (invented) numbers, which the lead then trusts.
FORGED_NOTE = (
    "[tinyharness] Your last reply contained a <teammate-message>. Only the harness writes "
    "those; it was discarded. Reply in your own words, with only what you found yourself."
)


@dataclass
class LoopDeps:
    """Everything the loop needs, passed in explicitly so it's easy to test and reuse."""

    llm: LLMClient
    tools: ToolRegistry
    gate: PermissionGate
    compactor: Compactor
    transcript: LogWriter
    ctx: ToolContext
    hooks: HookRunner | None = None
    allowed_tools: set[str] | None = None
    model: str | None = None  # override the client's default model (subagents)
    max_steps: int = 50
    inbox: Callable[[], list[str]] | None = None  # messages that arrive mid-run (teammates)
    # Subagents running side by side share one: their calls that change things take turns,
    # so two never write at the same moment, while their model calls overlap freely.
    write_lock: asyncio.Lock | None = None


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
    reported: dict[int, tuple[int, str]] = field(default_factory=dict)  # index -> (KB, target) last sent
    state: ReasoningState | None = None
    usage: Usage | None = None
    finish_reason: str | None = None  # normalized: see llm/types.py FinishReason
    finish_detail: str = ""  # the provider's own word for it
    route: str = ""


async def run_agent_loop(
    prompt: str, history: History, deps: LoopDeps, images: Sequence[Image] = ()
) -> AsyncGenerator[Event, None]:
    await _fire_hook(deps, "UserPromptSubmit", {"prompt": prompt})
    deps.transcript.write("user", text=prompt, images=[image.name for image in images])
    history.append(Message(role="user", content=prompt, images=list(images)))

    stuck_once = False  # a reply got stuck this turn and the model was asked to carry on
    empty_replies = 0  # empty replies in a row
    failures = _Failures()
    unverified: list[str] = []  # code files changed since the last command that could check them
    asked_to_verify = asked_for_report = False
    answered_message = False  # its latest reply came after a teammate's message, with no tool call since
    stop_blocks = 0
    unreadable_calls = 0
    for _ in range(deps.max_steps):
        # Messages from teammates arrive between steps, as a user message.
        for note in deps.inbox() if deps.inbox is not None else []:
            # (The main session's inbox carries the user's own messages typed mid-turn.)
            answered_message = answered_message or note.startswith(TEAMMATE_NOTE)
            deps.transcript.write("user", text=note)
            history.append(Message(role="user", content=note))
            yield MessageDelivered(note)

        async for event in _compact(history, deps):
            yield event

        # 1. Ask the model, streaming text to the caller as it arrives. If the provider
        #    says the conversation no longer fits, summarize it and ask once more.
        available = {t.name: t for t in deps.tools.available(deps.gate.mode, deps.allowed_tools)}
        turn = _Turn()
        try:
            async for event in _ask_model(history, available, deps, turn):
                yield event
        except ContextOverflowError:
            if not await deps.compactor.summarize(history, notes=summary_notes(deps.ctx)):
                raise
            deps.transcript.write("compaction", kind="summary", reason="context overflow")
            yield Compacted("summary")
            turn = _Turn()
            async for event in _ask_model(history, available, deps, turn):
                yield event
        except RunawayOutputError as stuck:
            # The reply went nowhere (repeating itself, or blank): drop it, say why, and ask
            # once more. Stuck twice in a turn: stop, with a message the user can act on.
            tail = "".join(turn.text)[-300:]  # what it was repeating, to tell a real loop from a false alarm
            if stuck_once:
                deps.transcript.write("stuck", reason=str(stuck), tail=tail, final=True)
                raise _stuck_again(stuck) from None
            stuck_once = True
            deps.transcript.write("stuck", reason=str(stuck), tail=tail)
            yield ModelStuck(str(stuck))
            note = STUCK_NOTE.format(reason=stuck)
            deps.transcript.write("user", text=note)
            history.append(Message(role="user", content=note))
            turn = _Turn()
            try:
                async for event in _ask_model(history, available, deps, turn):
                    yield event
            except RunawayOutputError as again:
                raise _stuck_again(again) from None
        except ImageTooLargeError:
            # An image in the history is over this provider's limit, and would fail every
            # request from now on: make the images smaller (or drop them) and ask once more.
            if not (changed := _shrink_images(history)):
                raise
            deps.transcript.write("images_shrunk", count=changed)
            turn = _Turn()
            async for event in _ask_model(history, available, deps, turn):
                yield event

        text, calls = "".join(turn.text), turn.calls.calls()
        if not calls and text:
            # A call written as text (the server's tool-call parser failed on it) would
            # otherwise be taken for the final answer.
            text, calls = recover_calls(text, functools.partial(_schema, available))
            if calls:
                deps.transcript.write("recovered_calls", tools=[c.name for c in calls])
                # Saved reasoning would be sent back in place of these calls, without them.
                turn.state = None
            elif looks_like_a_call(text):
                # A call it meant to make but wrote in a form that can't be used: tell it,
                # rather than end the turn with the call as its answer.
                if turn.usage is not None:
                    yield _account(turn, deps)
                if unreadable_calls >= MAX_UNREADABLE_CALLS:
                    # Seen live (Gemma 4): the raw `<|tool_call>call:bash:ls -R.` was shown
                    # as the turn's answer once the notes ran out.
                    deps.transcript.write("stuck", reason="unreadable call", text=text[-300:])
                    raise _stuck_again(
                        RunawayOutputError("it kept writing tool calls as text that couldn't be run")
                    )
                unreadable_calls += 1
                note = UNREADABLE_CALL_NOTE.format(tools=", ".join(sorted(available)))
                deps.transcript.write("stuck", reason="unreadable call", text=text[-300:])
                history.append(Message(role="assistant", content=text))
                history.append(Message(role="user", content=note))
                continue
        if not calls and not text.strip():
            # Nothing to show or do. That isn't "finished": ask the model to carry on, and
            # stop the turn if it happens again straight away.
            if turn.usage is not None:
                yield _account(turn, deps)
            if turn.finish_reason in ("filtered", "refused"):  # asking again won't change that
                deps.transcript.write("stuck", reason="withheld", finish=turn.finish_detail, route=turn.route)
                raise LLMError(
                    f"the provider withheld the model's reply ({turn.finish_detail}). "
                    "Rephrase the request, or try another model with /model."
                )
            empty_replies += 1
            if empty_replies >= MAX_EMPTY_REPLIES:
                if turn.usage is not None and turn.usage.output:  # it wrote something; none arrived
                    raise RunawayOutputError(_dropped_replies(empty_replies, turn.route))
                raise _stuck_again(RunawayOutputError("its replies were empty"))
            deps.transcript.write(
                "stuck",
                reason="empty reply",
                attempt=empty_replies,
                finish=turn.finish_detail,
                route=turn.route,
                output_tokens=turn.usage.output if turn.usage else None,
            )
            if empty_replies > SILENT_RETRIES:
                yield ModelStuck("its reply was empty")
                deps.transcript.write("user", text=EMPTY_NOTE)
                history.append(Message(role="user", content=EMPTY_NOTE))
            continue
        empty_replies = 0
        if deps.inbox is not None and TEAMMATE_NOTE in text and not stuck_once:
            stuck_once = True
            if turn.usage is not None:
                yield _account(turn, deps)
            deps.transcript.write("stuck", reason="forged teammate message", text=text[:500])
            yield ModelStuck("it wrote a message as if from a teammate")
            deps.transcript.write("user", text=FORGED_NOTE)
            history.append(Message(role="user", content=FORGED_NOTE))
            continue
        deps.transcript.write(
            "assistant", text=text, tool_calls=[c.to_dict() for c in calls], finish=turn.finish_detail
        )
        history.append(Message(role="assistant", content=text or None, tool_calls=calls, state=turn.state))
        if turn.usage is not None:
            yield _account(turn, deps)
            deps.compactor.observe(history, turn.usage)

        # 2. No tool calls means the model is finished, unless its reply was cut off.
        if not calls and turn.finish_reason == "length":
            deps.transcript.write("user", text=CUT_OFF_NOTE)
            history.append(Message(role="user", content=CUT_OFF_NOTE))
            continue
        if calls:
            answered_message = False
        if not calls:
            if answered_message and not asked_for_report:
                asked_for_report = True
                deps.transcript.write("user", text=REPORT_NOTE)
                history.append(Message(role="user", content=REPORT_NOTE))
                continue
            if unverified and not asked_to_verify and "Bash" in available:
                asked_to_verify = True
                note = UNVERIFIED_NOTE.format(files=", ".join(unverified[:5]))
                deps.transcript.write("user", text=note)
                history.append(Message(role="user", content=note))
                continue
            stop = await _fire_hook(deps, "Stop", {})
            if stop and stop.get("decision") in ("block", "deny") and stop_blocks < MAX_STOP_BLOCKS:
                stop_blocks += 1
                note = f"[tinyharness] A Stop hook says the work isn't done yet:\n{stop.get('reason', '')}"
                deps.transcript.write("user", text=note)
                history.append(Message(role="user", content=note))
                continue
            yield Done()
            return

        # 3. Validate and permission-check each call, one at a time (prompts can't interleave).
        batch = _Batch()
        for call in calls:
            if batch.interrupted:
                history.add_tool_result(call.id, "(not run: the user interrupted)")
                continue
            cut_off = turn.finish_reason == "length"
            async for event in _prepare_call(call, available, history, deps, batch, failures, cut_off):
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
            if failures.stuck:  # every call in the batch was rejected, the same way again
                deps.transcript.write("stuck", reason="repeated tool failure", detail=failures.stuck)
                raise RepeatedFailureError(failures.stuck)
            continue

        # 4. Run the approved calls. Read-only batches run concurrently.
        parallel = len(approved) > 1 and all(a.tool.runs_alongside(a.args) for a in approved)
        yield ToolsDispatched([a.tool.name for a in approved], parallel)
        # 5. Report each result as its call finishes (a quick Read isn't held back by a slow
        #    search), with any credential values masked so they never enter the model's
        #    context. The model gets them in the order it asked for them, once all are in.
        finished: dict[int, tuple[str, dict[str, Any], list[Image]]] = {}
        try:
            async for i, (result, is_error) in _in_order_finished(approved, parallel, deps):
                item = approved[i]
                output = mask_secrets(result.text, _config_key(deps))
                after = await _fire_hook(
                    deps,
                    "PostToolUse",
                    {
                        "tool": item.tool.name,
                        "args": item.args.model_dump(),
                        "result": output[:TRANSCRIPT_RESULT_CHARS],
                    },
                )
                if after and after.get("reason"):
                    output += f"\n\n[hook] {after['reason']}"
                if not is_error:
                    _track_changes(unverified, item.tool.name, item.args)
                output += failures.note(item.tool.name, output, is_error)
                deps.transcript.write(
                    "tool_result", tool=item.tool.name, result=output[:TRANSCRIPT_RESULT_CHARS]
                )
                # Which file (and which version of it) this result came from, for resuming later.
                meta = {} if is_error else file_meta(item.tool, item.args, deps.ctx)
                finished[i] = (output, meta, result.images)
                yield ToolResult(item.tool.name, output, is_error, id=item.call.id)
                if item.tool.name == "TodoWrite" and not is_error:
                    yield TodosUpdated(list(deps.ctx.todos))
        finally:
            # In the order the model asked for them. If the turn was cut short (Ctrl-C), the
            # calls that did finish keep their real results: an Edit that was applied must
            # not be reported as "not run" and done again. The rest are answered by repair().
            for i, item in enumerate(approved):
                if i in finished:
                    history.add_tool_result(item.call.id, *finished[i])
        if failures.stuck:  # results are in the history first, so it stays valid
            deps.transcript.write("stuck", reason="repeated tool failure", detail=failures.stuck)
            raise RepeatedFailureError(failures.stuck)

    # Out of steps. One more reply, without running tools, so the work so far isn't lost:
    # a subagent's answer is its last text, and the caller sees nothing else.
    async for event in _wrap_up(history, deps):
        yield event
    yield Done(reason="max_steps")


async def _wrap_up(history: History, deps: LoopDeps) -> AsyncGenerator[Event, None]:
    deps.transcript.write("user", text=STEP_LIMIT_NOTE)
    history.append(Message(role="user", content=STEP_LIMIT_NOTE))
    # The tools are still described (some APIs require that once tool calls are in the
    # history); any it calls anyway are dropped.
    available = {t.name: t for t in deps.tools.available(deps.gate.mode, deps.allowed_tools)}
    turn = _Turn()
    try:
        async for event in _ask_model(history, available, deps, turn):
            yield event
    except (ContextOverflowError, RunawayOutputError) as e:
        deps.transcript.write("wrap_up_failed", reason=str(e))
        return
    text, calls = "".join(turn.text), turn.calls.calls()
    if turn.usage is not None:
        yield _account(turn, deps)
    if not text.strip() and not calls:
        return  # an empty assistant message would be rejected by the next request
    deps.transcript.write(
        "assistant", text=text, tool_calls=[c.to_dict() for c in calls], finish=turn.finish_detail
    )
    # Calls it made anyway are kept, each answered, so the history stays valid (saved
    # reasoning can contain them, and every provider wants a result for every call).
    history.append(Message(role="assistant", content=text or None, tool_calls=calls, state=turn.state))
    for call in calls:
        history.add_tool_result(call.id, STEP_LIMIT_RESULT)


class _Failures:
    """Each tool's latest run of failures with the same error, to catch a model retrying a
    call that can't work. A success with that tool resets it; other tools in between (a
    Read between two failing Edits) don't. Denials and rejections aren't counted: they're
    the user's or the rules' decision, not the model looping."""

    def __init__(self) -> None:
        self.runs: dict[str, tuple[str, int]] = {}  # tool -> (kind of error, failures in a row)
        self.stuck = ""  # why the turn must stop, once a run reaches REPEAT_STOP_AT

    def note(self, tool: str, output: str, is_error: bool) -> str:
        """Count this result; what to add to it for the model (usually nothing)."""
        if not is_error:
            self.runs.pop(tool, None)
            return ""
        if output.startswith(("denied by", "The user rejected")):
            return ""
        kind = _error_kind(output)
        previous, count = self.runs.get(tool, ("", 0))
        count = count + 1 if previous == kind else 1
        self.runs[tool] = (kind, count)
        if count >= REPEAT_STOP_AT:
            self.stuck = (
                f"{tool} failed {count} times in a row with the same error ({kind}), so the turn "
                "was stopped. Tell the agent what to do differently, or try a stronger model."
            )
            return ""
        if count >= REPEAT_NOTE_AT:
            hint = _CHANGE_APPROACH.get(
                tool, "Read the error, then change the arguments or use another tool."
            )
            return (
                f"\n\n[tinyharness] {tool} has now failed {count} times in a row with this error. "
                f"Don't repeat the same call: {hint}"
            )
        return ""


def _error_kind(output: str) -> str:
    """What kind of error this is, ignoring its details (line numbers, quoted text), so
    two attempts that fail for the same reason count as the same."""
    first = output.strip().splitlines()[0] if output.strip() else ""
    first = first.removeprefix("error:").strip()
    first = re.split(r"[.;]\s", first, maxsplit=1)[0]  # the reason, without the explanation after it
    first = re.sub(r"(['\"`]).*?\1", "…", first)
    return re.sub(r"\d+", "N", first)[:80]


def _find_tool(available: dict[str, Tool[Any]], name: str) -> Tool[Any] | None:
    """The tool a call names; models sometimes get the case wrong ("read")."""
    if name in available:
        return available[name]
    return next((t for n, t in available.items() if n.lower() == name.lower()), None)


def _schema(available: dict[str, Tool[Any]], name: str) -> dict[str, Any] | None:
    tool = _find_tool(available, name)
    return tool.input_schema() if tool is not None else None


def _arguments(call: ToolCall, *, repair: bool) -> dict[str, Any] | json.JSONDecodeError:
    """A call's arguments as an object. Known slips (a code fence, JSON encoded twice, two
    objects run together) are repaired, and the repaired JSON replaces the original in the
    history: some APIs reject a past call whose arguments aren't valid JSON."""
    try:
        raw = json.loads(call.arguments or "{}")
        if isinstance(raw, dict):
            return raw
        error = json.JSONDecodeError("not a JSON object", call.arguments, 0)
    except json.JSONDecodeError as e:
        error = e
    if repair and (fixed := repair_arguments(call.arguments)) is not None:
        call.arguments = fixed
        return dict(json.loads(fixed))
    return error


def _explain_invalid(tool: Tool[Any], error: json.JSONDecodeError | ValidationError) -> str:
    """What's wrong with a call's arguments, briefly: for the model to fix on its retry,
    and for the user's screen (pydantic's own message has types, inputs and a docs link)."""
    if isinstance(error, json.JSONDecodeError):
        return f"they aren't valid JSON ({error.msg} at character {error.pos})"
    problems = []
    for item in error.errors():
        where = ".".join(str(part) for part in item["loc"]) or "arguments"
        message = item["msg"]
        problems.append(f"{where}: {message[:1].lower()}{message[1:]}")
    required = tool.Input.model_json_schema().get("required") or []
    hint = f" (required: {', '.join(required)})" if required else ""
    return "; ".join(problems) + hint


async def _in_order_finished(
    approved: list[_ApprovedCall], parallel: bool, deps: LoopDeps
) -> AsyncGenerator[tuple[int, tuple[ToolOutput, bool]], None]:
    """(position in the batch, result) for each call, as each finishes: all at once for a
    parallel batch, one after another otherwise."""
    if not parallel:
        for i, item in enumerate(approved):
            yield i, await _execute(item, deps)
        return
    tasks = {asyncio.create_task(_execute(item, deps)): i for i, item in enumerate(approved)}
    pending = set(tasks)
    try:
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in sorted(done, key=tasks.__getitem__):
                yield tasks[task], task.result()
    finally:
        for task in pending:  # interrupted: don't leave calls running
            task.cancel()


_TARGET = re.compile(r'"(path|url|target|command)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _streaming(turn: _Turn, index: int) -> ToolInputStreaming | None:
    """Progress on a tool call still being written, when there's news: it started, it
    grew by another KB, or its target (path, URL, command) arrived."""
    call = turn.calls.get(index)
    if call is None or not call.name:
        return None
    target = ""
    if match := _TARGET.search(call.arguments):
        try:
            target = json.loads(f'"{match.group(2)}"')
        except ValueError:
            target = match.group(2)
    news = (len(call.arguments) // 1024, target)
    if turn.reported.get(index) == news:
        return None
    turn.reported[index] = news
    return ToolInputStreaming(call.name, len(call.arguments), target)


async def _ask_model(
    history: History, available: dict[str, Tool[Any]], deps: LoopDeps, turn: _Turn
) -> AsyncGenerator[Event, None]:
    tools = [t.spec() for t in available.values()]
    guard = RunawayGuard()
    async with contextlib.aclosing(deps.llm.stream(history.messages, tools, model=deps.model)) as stream:
        async for chunk in stream:
            if (why := guard.check(chunk)) is not None:
                raise RunawayOutputError(why)
            for event in _take(chunk, turn):
                yield event
    deps.compactor.mark_call()


def _dropped_replies(count: int, route: str) -> str:
    """Why a run of empty replies is the provider's, not the model's: output was billed but
    nothing came back (seen live: an OpenRouter upstream dropping Qwen3-Coder's tool calls)."""
    via = f" ({route})" if route else ""
    return (
        f"the provider{via} sent {count} empty replies in a row: the model produced output (it "
        "was billed) but none of it arrived, which usually means the provider's tool-call parser "
        "dropped it. Try again, or use another provider for this model"
        + (' (on OpenRouter: extra_body = { provider = { ignore = ["' + route + '"] } })' if route else "")
        + "."
    )


def _stuck_again(stuck: RunawayOutputError) -> RunawayOutputError:
    return RunawayOutputError(
        f"the model got stuck again ({stuck}), so the turn was stopped. "
        "Try again, or switch to a stronger model with /model."
    )


def _take(chunk: StreamChunk, turn: _Turn) -> list[Event]:
    """Add one streamed piece to the turn; the events to show for it."""
    events: list[Event] = []
    if chunk.reasoning:
        events.append(Reasoning())
    if chunk.reasoning_text:
        events.append(Reasoning(chunk.reasoning_text))
    if chunk.text:
        turn.text.append(chunk.text)
        events.append(AssistantText(chunk.text))
    if chunk.tool_call_delta:
        turn.calls.add(chunk.tool_call_delta)
        if (progress := _streaming(turn, chunk.tool_call_delta.get("index", 0))) is not None:
            events.append(progress)
    if chunk.finish_reason is not None:
        turn.finish_reason, turn.finish_detail = chunk.finish_reason, chunk.finish_detail
    if chunk.route:
        turn.route = chunk.route
    if chunk.state is not None:
        turn.state = chunk.state
    if chunk.usage is not None:
        turn.usage = chunk.usage
    return events


async def _compact(history: History, deps: LoopDeps) -> AsyncGenerator[Event, None]:
    compactor = deps.compactor
    if n := compactor.budget_cap(history):
        deps.transcript.write("compaction", kind="budget_cap", count=n)
        yield Compacted("budget_cap", n)
    if n := compactor.microcompact(history):
        deps.transcript.write("compaction", kind="microcompact", count=n)
        yield Compacted("microcompact", n)
    if compactor.needs_summary(history) and await compactor.summarize(history, notes=summary_notes(deps.ctx)):
        deps.transcript.write("compaction", kind="summary")
        yield Compacted("summary")


async def _prepare_call(
    call: ToolCall,
    available: dict[str, Tool[Any]],
    history: History,
    deps: LoopDeps,
    batch: _Batch,
    failures: _Failures | None = None,
    cut_off: bool = False,
) -> AsyncGenerator[Event, None]:
    """Parse, validate and permission-check one call. Rejections are answered
    immediately (the model must get a result for every call); approvals are
    added to `batch`. `cut_off`: the reply hit the output limit, so arguments that don't
    parse are incomplete rather than malformed."""

    def reject(message: str) -> ToolResult:
        if failures is not None and message.startswith("error:"):  # not a denial: a bad call
            message += failures.note(call.name, message, True)
        deps.transcript.write("tool_result", tool=call.name, error=message)
        history.add_tool_result(call.id, message)
        return ToolResult(call.name, message, is_error=True, id=call.id)

    tool = _find_tool(available, call.name)
    if tool is None:
        yield reject(
            f"error: unknown or unavailable tool {call.name!r}. Available: {', '.join(sorted(available))}"
        )
        return
    call.name = tool.name  # "read" -> "Read", so the history names the tool that ran

    raw = _arguments(call, repair=not cut_off)
    if isinstance(raw, json.JSONDecodeError):
        yield reject(
            CUT_OFF_CALL
            if cut_off
            else f"error: invalid arguments for {tool.name}: {_explain_invalid(tool, raw)}"
        )
        return
    try:
        args = tool.parse(raw)
    except ValidationError as e:
        yield reject(
            f"error: invalid arguments for {tool.name}: {_explain_invalid(tool, e)}. "
            f"Received: {call.arguments[:300]}"
        )
        return

    summary = tool.describe(args)
    yield ToolUse(tool.name, raw, summary, id=call.id)
    deps.transcript.write("tool_use", tool=tool.name, args=raw)

    decision = await deps.gate.check(
        tool,
        raw,
        tool.target_path(args),
        target_host=tool.target_host(args),
        summary=summary,
        history_tail=_tail(history),
    )
    deps.transcript.write(
        "permission", tool=tool.name, allow=decision.allow, source=decision.source, reason=decision.reason
    )
    yield PermissionDecision(tool.name, decision.allow, decision.source, decision.reason)

    if decision.allow:
        # Act on the path that was checked, not a fresh resolution of it (see Decision.path).
        pinned = tool.pin_path(args, decision.path) if decision.path is not None else args
        batch.approved.append(_ApprovedCall(call, tool, pinned))
    elif decision.interrupt:
        batch.interrupted, batch.feedback = True, decision.feedback
        yield reject("The user rejected this tool call.")
    else:
        yield reject(f"denied by {decision.source}: {decision.reason}")


async def _execute(item: _ApprovedCall, deps: LoopDeps) -> tuple[ToolOutput, bool]:
    """Run one tool. Errors become text for the model rather than crashing the loop."""
    turns: contextlib.AbstractAsyncContextManager[Any] = (
        deps.write_lock
        if deps.write_lock is not None and item.tool.writes(item.args)
        else contextlib.nullcontext()
    )
    try:
        async with turns:
            result = await item.tool.run(item.args, deps.ctx)
    except ToolError as e:
        return ToolOutput(f"error: {e}"), True
    except Exception as e:
        return ToolOutput(f"error: {type(e).__name__}: {e}"), True
    return (result if isinstance(result, ToolOutput) else ToolOutput(result)), False


def summary_notes(ctx: ToolContext) -> str:
    """Current state a summary must keep: the todo list as it is now."""
    return f"[current todo list]\n{format_todos(ctx.todos)}" if ctx.todos else ""


def _account(turn: _Turn, deps: LoopDeps) -> TokenUsage:
    """Price one model call (unless the provider already did), add it to the session's
    meter and report it."""
    assert turn.usage is not None
    usage = turn.usage
    model = deps.model or deps.llm.model
    provider = getattr(getattr(deps.llm, "provider", None), "name", "")
    if usage.cost is None:
        info = deps.llm.model_info(model) if hasattr(deps.llm, "model_info") else None
        price = deps.ctx.config.price(provider, model) or (info.price if info else None)
        if price is not None:
            usage.cost = price.cost(usage)
    deps.ctx.usage.add(usage)
    deps.transcript.write(
        "usage",
        model=model,
        input=usage.input,
        cached=usage.cached,
        output=usage.output,
        cost=usage.cost,
        **({"route": turn.route} if turn.route else {}),
    )
    return TokenUsage(model, usage.input, usage.cached, usage.output, usage.cost)


async def _fire_hook(deps: LoopDeps, event: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    return await deps.hooks.fire(event, payload) if deps.hooks is not None else None


def _track_changes(unverified: list[str], tool: str, args: Any) -> None:
    """Keep `unverified` as the code files changed since the last command that could check them."""
    if tool == "Bash":
        unverified.clear()
    elif tool in ("Edit", "Write"):
        path = str(getattr(args, "path", ""))
        if path and not path.lower().endswith(_NOT_CODE) and path not in unverified:
            unverified.append(path)


def _config_key(deps: LoopDeps) -> dict[str, str]:
    """A key passed directly to the SDK isn't in the environment; mask it too."""
    key = deps.ctx.config.api_key
    return {"api_key": key} if key else {}


def _shrink_images(history: History) -> int:
    """After a provider said an image is too large: shrink every image over RETRY_IMAGE_SIDE.
    If none was, the limit is something else, so drop them all (a note says which) rather
    than fail every request from now on. Returns how many images changed."""
    changed = 0
    for m in history.messages:
        for i, image in enumerate(m.images):
            try:
                smaller = fit_image(image, RETRY_IMAGE_SIDE)
            except ValueError:
                continue
            if smaller is not image:
                m.images[i] = smaller
                changed += 1
    if changed:
        return changed
    for m in history.messages:
        if m.images:
            names = ", ".join(image.name or "an image" for image in m.images)
            m.content = f"{m.content or ''}\n[removed: {names}; the model rejected it as too large]".strip()
            changed += len(m.images)
            m.images = []
    return changed


def _tail(history: History, items: int = 8, chars: int = 500) -> str:
    """Recent context for the auto-mode classifier: the user's messages and the agent's
    tool calls. Not tool output, the agent's own text or teammates' messages: a web page
    or file could say "the user already approved this", and the classifier would read it.
    (Claude Code's auto mode leaves them out for the same reason.)"""
    lines: list[str] = []
    for m in history.messages:
        if m.role == "user" and m.content and not m.content.startswith(TEAMMATE_NOTE):
            lines.append(f"[user] {m.content[:chars]}")
        elif m.role == "assistant":
            lines += [f"[agent called] {c.name}({c.arguments[:200]})" for c in m.tool_calls]
    return "\n".join(lines[-items:])
