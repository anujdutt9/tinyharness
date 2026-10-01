"""A reply that streams on without going anywhere is stopped, and the model asked once more."""

import pytest

from conftest import ScriptedLLM, make_ctx
from tinyharness.agent.compaction import Compactor
from tinyharness.agent.history import History, Message
from tinyharness.agent.loop import LoopDeps, run_agent_loop
from tinyharness.agent.runaway import QUIET_SECONDS, RunawayGuard
from tinyharness.agent.transcript import Transcript
from tinyharness.events import AssistantText, ModelStuck
from tinyharness.llm.types import RunawayOutputError, StreamChunk
from tinyharness.permissions import PermissionGate, PermissionMode
from tinyharness.tools.registry import ToolRegistry
from tinyharness.workspace import Workspace


def feed(guard: RunawayGuard, chunks) -> str | None:
    for chunk in chunks:
        if (why := guard.check(chunk)) is not None:
            return why
    return None


def test_endless_blank_lines_are_caught():
    """What Qwen3.5-9B did: a tool call begun, then newline after newline."""
    assert (
        feed(RunawayGuard(), [StreamChunk(text="\n") for _ in range(10_000)])
        == "it kept repeating the same few characters"
    )


def test_the_same_words_over_and_over_are_caught():
    chunks = [StreamChunk(text="Let me check the database. ") for _ in range(1000)]
    assert feed(RunawayGuard(), chunks) == "it kept repeating the same output"


def test_a_long_real_reply_is_left_alone():
    """Code and tables repeat a little, but not one short piece end to end."""
    lines = [f"| {i} | customer-{i * 7919 % 1000} | ${i * 13.5:,.2f} |\n" for i in range(2000)]
    assert feed(RunawayGuard(), [StreamChunk(text=line) for line in lines]) is None


def test_a_long_tool_call_is_left_alone():
    """A whole file written in one call is many KB of arguments: progress, not a runaway."""
    deltas = [
        StreamChunk(tool_call_delta={"index": 0, "function": {"arguments": f"line {i}\\n"}})
        for i in range(5000)
    ]
    assert feed(RunawayGuard(), deltas) is None


def test_a_minute_of_only_whitespace_is_caught():
    now = [0.0]
    guard = RunawayGuard(clock=lambda: now[0])
    assert guard.check(StreamChunk(text="Working on it.")) is None
    for _ in range(int(QUIET_SECONDS) + 2):
        now[0] += 1
        why = guard.check(StreamChunk(text=" "))
    assert why == "it sent nothing but blank output for over 60 seconds"


def test_hidden_reasoning_counts_as_progress():
    now = [0.0]
    guard = RunawayGuard(clock=lambda: now[0])
    for _ in range(200):
        now[0] += 1
        assert guard.check(StreamChunk(reasoning=True)) is None


class StuckThenFine(ScriptedLLM):
    """Gets stuck on the first `stuck` replies, then answers."""

    def __init__(self, stuck: int) -> None:
        super().__init__(["Here is the plan."])
        self.stuck = stuck
        self.seen: list[list[Message]] = []

    async def stream(self, messages, tools=None, **kw):
        self.seen.append(list(messages))
        if self.stuck:
            self.stuck -= 1
            for _ in range(10_000):
                yield StreamChunk(text="\n")
            return
        async for chunk in super().stream(messages, tools, **kw):
            yield chunk


async def run(tmp_path, llm):
    history = History([Message(role="system", content="s")])
    with Transcript(tmp_path, "t") as transcript:
        deps = LoopDeps(
            llm=llm,
            tools=ToolRegistry([]),
            gate=PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.BYPASS),
            compactor=Compactor(llm),
            transcript=transcript,
            ctx=make_ctx(tmp_path),
        )
        return [e async for e in run_agent_loop("go", history, deps)], history


async def test_a_stuck_reply_is_stopped_and_the_model_asked_once_more(tmp_path):
    llm = StuckThenFine(stuck=1)
    events, history = await run(tmp_path, llm)
    assert ModelStuck("it kept repeating the same few characters") in events
    assert AssistantText("Here is the plan.") in events
    note = llm.seen[1][-1].content
    assert note.startswith("[tinyharness] Your last reply was stopped because it kept repeating")
    assert history.messages[-1].content == "Here is the plan."


async def test_stuck_twice_stops_the_turn_with_advice(tmp_path):
    with pytest.raises(RunawayOutputError, match=r"got stuck again .* switch to a stronger model"):
        await run(tmp_path, StuckThenFine(stuck=2))


def test_a_line_of_backticks_in_the_reasoning_is_caught_quickly():
    """Qwen3.5-9B's reasoning turned into ```````… at a few characters a second: waiting
    for 4,000 of them took minutes."""
    guard = RunawayGuard()
    sent, why = 0, None
    while why is None and sent < 1000:
        why = guard.check(StreamChunk(reasoning_text="`" * 5))
        sent += 5
    assert why == "it kept repeating the same few characters" and sent <= 600


def test_a_file_full_of_zeros_is_not_mistaken_for_a_stuck_reply():
    deltas = [
        StreamChunk(tool_call_delta={"index": 0, "function": {"arguments": "0, " * 100}}) for _ in range(10)
    ]
    assert feed(RunawayGuard(), deltas) is None
