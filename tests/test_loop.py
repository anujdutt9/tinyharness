"""Agent-loop behavior, driven by a scripted fake LLM."""

import asyncio
import time

from pydantic import Field

from conftest import ScriptedLLM, make_ctx
from tinyharness.agent.compaction import Compactor
from tinyharness.agent.history import History, Message
from tinyharness.agent.loop import LoopDeps, run_agent_loop
from tinyharness.agent.transcript import Transcript
from tinyharness.events import AssistantText, Done, PermissionDecision, ToolResult, ToolsDispatched
from tinyharness.permissions import Decision, PermissionGate, PermissionMode
from tinyharness.tools.base import Tool, ToolInput
from tinyharness.tools.registry import ToolRegistry
from tinyharness.workspace import Workspace


class SleepInput(ToolInput):
    seconds: float = Field(default=0.3)


def sleep_tool(name: str, read_only: bool) -> Tool:
    class SleepTool(Tool[SleepInput]):
        description = "sleep"
        Input = SleepInput

        async def run(self, args, ctx):
            await asyncio.sleep(args.seconds)
            return f"{self.name} done"

    SleepTool.name = name
    SleepTool.read_only = read_only
    return SleepTool()


async def run(tmp_path, llm, tools, mode=PermissionMode.BYPASS, callback=None):
    history = History([Message(role="system", content="test")])
    with Transcript(tmp_path, "test") as transcript:
        deps = LoopDeps(
            llm=llm,
            tools=ToolRegistry(tools),
            gate=PermissionGate(workspace=Workspace(tmp_path), mode=mode, callback=callback),
            compactor=Compactor(llm),
            transcript=transcript,
            ctx=make_ctx(tmp_path),
        )
        events = [e async for e in run_agent_loop("go", history, deps)]
    return events, history


async def test_plain_answer_ends_the_loop(tmp_path):
    events, history = await run(tmp_path, ScriptedLLM(["hello"]), [])
    assert events == [AssistantText("hello"), Done()]
    assert [m.role for m in history.messages] == ["system", "user", "assistant"]


async def test_read_only_tools_run_in_parallel(tmp_path):
    tools = [sleep_tool("A", True), sleep_tool("B", True), sleep_tool("C", True)]
    llm = ScriptedLLM([[("A", "{}"), ("B", "{}"), ("C", "{}")], "done"])
    start = time.monotonic()
    events, _ = await run(tmp_path, llm, tools)
    elapsed = time.monotonic() - start
    assert ToolsDispatched(["A", "B", "C"], parallel=True) in events
    assert elapsed < 0.7, f"3 × 0.3s tools should overlap; took {elapsed:.2f}s"


async def test_mutating_tools_run_sequentially(tmp_path):
    tools = [sleep_tool("Look", True), sleep_tool("Change", False)]
    llm = ScriptedLLM([[("Look", '{"seconds": 0.2}'), ("Change", '{"seconds": 0.2}')], "done"])
    start = time.monotonic()
    events, _ = await run(tmp_path, llm, tools)
    assert ToolsDispatched(["Look", "Change"], parallel=False) in events
    assert time.monotonic() - start >= 0.35


async def test_results_are_returned_in_call_order(tmp_path):
    tools = [sleep_tool("Slow", True), sleep_tool("Fast", True)]
    llm = ScriptedLLM([[("Slow", '{"seconds": 0.2}'), ("Fast", '{"seconds": 0.0}')], "done"])
    _, history = await run(tmp_path, llm, tools)
    tool_msgs = [m for m in history.messages if m.role == "tool"]
    assert [m.tool_call_id for m in tool_msgs] == ["call_0", "call_1"]


async def test_unknown_tool_and_bad_args_become_errors(tmp_path):
    llm = ScriptedLLM([[("Nope", "{}"), ("A", '{"seconds": "not a number"}')], "done"])
    events, history = await run(tmp_path, llm, [sleep_tool("A", True)])
    errors = [e for e in events if isinstance(e, ToolResult) and e.is_error]
    assert len(errors) == 2
    assert "unknown or unavailable tool" in errors[0].output
    assert "invalid arguments" in errors[1].output
    # Every tool call still got a reply, so the history stays valid for the next request.
    assert len([m for m in history.messages if m.role == "tool"]) == 2


async def test_plan_mode_hides_mutating_tools(tmp_path):
    llm = ScriptedLLM([[("Change", "{}")], "done"])
    events, _ = await run(tmp_path, llm, [sleep_tool("Change", False)], mode=PermissionMode.PLAN)
    result = next(e for e in events if isinstance(e, ToolResult))
    assert result.is_error and "unavailable" in result.output


async def test_denied_calls_are_reported_to_the_model(tmp_path):
    llm = ScriptedLLM([[("Change", "{}")], "ok"])
    events, history = await run(tmp_path, llm, [sleep_tool("Change", False)], mode=PermissionMode.DEFAULT)
    decision = next(e for e in events if isinstance(e, PermissionDecision))
    assert not decision.allowed
    assert history.messages[3].content.startswith("denied by prompt")


def test_history_repair_answers_dangling_tool_calls():
    from tinyharness.agent.history import ToolCall

    history = History(
        [
            Message(role="user", content="hi"),
            Message(role="assistant", tool_calls=[ToolCall("a", "Read"), ToolCall("b", "Read")]),
            Message(role="tool", tool_call_id="a", content="ok"),
        ]
    )
    assert history.repair() == 1
    assert {m.tool_call_id for m in history.messages if m.role == "tool"} == {"a", "b"}
    assert history.repair() == 0


async def test_saying_no_stops_the_turn(tmp_path):
    async def user(request):
        return Decision(False, "user stopped the turn", "user", interrupt=True)

    llm = ScriptedLLM([[("Change", "{}"), ("Change", "{}")], "should never be requested"])
    events, history = await run(
        tmp_path, llm, [sleep_tool("Change", False)], mode=PermissionMode.DEFAULT, callback=user
    )
    assert events[-1] == Done(reason="interrupted")
    assert len(llm.requests) == 1, "the model must not be called again after the user stops the turn"
    tool_msgs = [m for m in history.messages if m.role == "tool"]
    assert len(tool_msgs) == 2  # both calls answered, so the history stays valid


async def test_saying_no_with_feedback_redirects_the_agent(tmp_path):
    asked = []

    async def user(request):
        asked.append(request)
        return Decision(False, "user rejected", "user", interrupt=True, feedback="use notes.md instead")

    llm = ScriptedLLM([[("Change", "{}"), ("Change", "{}")], "ok, switching to notes.md"])
    events, history = await run(
        tmp_path, llm, [sleep_tool("Change", False)], mode=PermissionMode.DEFAULT, callback=user
    )
    assert len(asked) == 1, "the rest of the batch is skipped once the user says no"
    assert events[-1] == Done()
    assert [m.role for m in history.messages[-3:]] == ["tool", "user", "assistant"]
    assert history.messages[-2].content == "use notes.md instead"
