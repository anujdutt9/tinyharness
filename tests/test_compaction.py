from conftest import ScriptedLLM
from tinyharness.agent.compaction import SUMMARY_MARK, Compactor
from tinyharness.agent.history import History, Message, ToolCall
from tinyharness.llm.tokens import count_tokens


def history(*messages: Message) -> History:
    return History(list(messages))


def user(text: str) -> Message:
    return Message(role="user", content=text)


def assistant(text: str = "", *calls: str, name: str = "Read") -> Message:
    return Message(role="assistant", content=text or None, tool_calls=[ToolCall(c, name) for c in calls])


def tool(call_id: str, text: str) -> Message:
    return Message(role="tool", tool_call_id=call_id, content=text)


def compactor(**kwargs) -> Compactor:
    return Compactor(ScriptedLLM([]), **kwargs)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


BIG = "lorem ipsum dolor sit amet " * 1500  # well over the per-result floor


def reads(n: int) -> list[Message]:
    """n Read calls, each with a BIG result, followed by a 4-message tail."""
    out: list[Message] = [user("go")]
    for i in range(n):
        out += [assistant("", f"r{i}"), tool(f"r{i}", BIG)]
    return out + [user(f"p{i}") for i in range(4)]


def stubbed(h: History) -> int:
    return sum((m.content or "").startswith("[trimmed:") for m in h.messages)


# ---- microcompact ----


def test_microcompact_stubs_old_large_tool_results():
    h = history(user("hi"), assistant("", "1"), tool("1", "x" * 20_000), *(user(f"p{i}") for i in range(4)))
    assert (
        compactor(stub_tool_results_over=100, keep_tail_messages=4, protect_tool_tokens=0).microcompact(h)
        == 1
    )
    assert h.messages[2].content.startswith("[trimmed:")


def test_microcompact_is_idempotent():
    """Each message changes at most once, so the provider's prompt cache stays valid."""
    h = history(tool("1", "x" * 20_000), *(user(f"p{i}") for i in range(4)))
    c = compactor(stub_tool_results_over=100, keep_tail_messages=4, protect_tool_tokens=0)
    assert [c.microcompact(h) for _ in range(3)] == [1, 0, 0]


def test_microcompact_keeps_everything_while_the_cache_is_warm_and_there_is_room():
    """Clearing a warm cache costs about what it saves, and loses information."""
    per = count_tokens(BIG)
    c = compactor(
        keep_tail_messages=4, protect_tool_tokens=0, min_savings=0, context_window=100 * per, clock=Clock()
    )
    c.mark_call()
    h = history(*reads(10))
    assert c.microcompact(h) == 0 and stubbed(h) == 0


def test_microcompact_clears_a_warm_cache_in_one_batch_once_the_window_fills():
    """Near the window, clearing holds off the summary; small clean-ups wait for a big one."""
    per = count_tokens(BIG)
    c = compactor(
        keep_tail_messages=4,
        protect_tool_tokens=0,
        min_savings=3 * per,
        context_window=4 * per,
        clock=Clock(),
    )
    c.mark_call()
    h = history(*reads(2))
    assert c.microcompact(h) == 0 and stubbed(h) == 0
    h = history(*reads(3))
    assert c.microcompact(h) == 3 and stubbed(h) == 3


def test_microcompact_clears_below_the_minimum_once_the_cache_is_cold():
    clock = Clock()
    c = compactor(keep_tail_messages=4, protect_tool_tokens=0, min_savings=10**9, cache_ttl=300, clock=clock)
    c.mark_call()
    h = history(*reads(1))
    assert c.microcompact(h) == 0
    clock.now += 301
    assert c.cache_cold
    assert c.microcompact(h) == 1


def test_microcompact_protects_the_most_recent_tool_output():
    per = count_tokens(BIG)
    h = history(*reads(4))
    assert compactor(keep_tail_messages=4, protect_tool_tokens=2 * per).microcompact(h) == 2
    results = [m for m in h.messages if m.role == "tool"]
    assert [r.content.startswith("[trimmed:") for r in results] == [True, True, False, False]


def test_microcompact_keeps_skill_and_agent_results():
    h = history(
        user("go"),
        assistant("", "s1", name="Skill"),
        tool("s1", BIG),
        assistant("", "a1", name="Agent"),
        tool("a1", BIG),
        *(user(f"p{i}") for i in range(4)),
    )
    assert compactor(keep_tail_messages=4, protect_tool_tokens=0).microcompact(h) == 0


# ---- budget cap ----


def test_budget_cap_truncates_huge_messages():
    h = history(user("y" * 50_000))
    assert compactor(max_message_tokens=1000).budget_cap(h) == 1
    assert h.messages[0].content.startswith("[truncated: kept ~1000 of")


def test_budget_cap_is_idempotent_and_skips_small_messages():
    h = history(user("y" * 50_000), user("short"))
    c = compactor(max_message_tokens=1000)
    assert [c.budget_cap(h) for _ in range(3)] == [1, 0, 0]
    assert h.messages[1].content == "short"


def test_budget_cap_leaves_stubs_alone():
    h = history(tool("1", "[trimmed: 8000 tokens of older tool output]"))
    assert compactor(max_message_tokens=10).budget_cap(h) == 0


def test_bytes_are_stable_across_turns():
    h = history(user("first"), tool("1", "z" * 60_000), *(user(f"p{i}") for i in range(4)))
    c = compactor(
        stub_tool_results_over=100, max_message_tokens=1000, keep_tail_messages=4, protect_tool_tokens=0
    )
    c.budget_cap(h)
    c.microcompact(h)
    snapshot = [m.content for m in h.messages]
    for _ in range(4):
        c.budget_cap(h)
        c.microcompact(h)
        assert [m.content for m in h.messages] == snapshot


# ---- summary ----


def test_needs_summary_above_threshold():
    h = history(*(user("x" * 1000) for _ in range(50)))
    assert compactor(context_window=200, threshold=0.5).needs_summary(h)
    assert not compactor(context_window=10_000_000).needs_summary(h)


async def test_summary_keeps_system_prompt_and_starts_tail_at_user():
    h = history(
        Message(role="system", content="you are a coding agent"),
        user("x" * 1000),
        assistant("", "tc1"),
        tool("tc1", "file content"),
        assistant("done"),
        user("next question"),
        assistant("", "tc2"),
        tool("tc2", "file content 2"),
        assistant("done 2"),
    )
    assert await compactor(keep_tail_messages=4).summarize(h)
    roles = [m.role for m in h.messages]
    assert h.messages[0].content == "you are a coding agent"
    assert h.messages[1].content.startswith(SUMMARY_MARK)
    assert roles[2] == "user", "tail must start at a user message so no tool call is orphaned"


async def test_summary_noop_when_nothing_to_cut():
    h = history(user("only message"))
    assert not await compactor(keep_tail_messages=4).summarize(h)


async def test_summary_can_keep_just_the_latest_exchange():
    h = history(
        Message(role="system", content="you are a coding agent"),
        user("first"),
        assistant("one"),
        user("second"),
        assistant("", "tc1"),
        tool("tc1", "file content"),
        assistant("two"),
    )
    assert await compactor(keep_tail_messages=12).summarize(h, keep_tail=1)
    assert [m.content for m in h.messages[2:]] == ["second", None, "file content", "two"]
    assert h.messages[1].content.startswith(SUMMARY_MARK)
