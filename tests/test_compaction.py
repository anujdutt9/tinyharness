from conftest import ScriptedLLM
from tinyharness.agent.compaction import SUMMARY_MARK, Compactor
from tinyharness.agent.history import History, Message, ToolCall


def history(*messages: Message) -> History:
    return History(list(messages))


def user(text: str) -> Message:
    return Message(role="user", content=text)


def assistant(text: str = "", *calls: str) -> Message:
    return Message(role="assistant", content=text or None, tool_calls=[ToolCall(c, "Read") for c in calls])


def tool(call_id: str, text: str) -> Message:
    return Message(role="tool", tool_call_id=call_id, content=text)


def compactor(**kwargs) -> Compactor:
    return Compactor(ScriptedLLM([]), **kwargs)


# ---- microcompact ----


def test_microcompact_stubs_old_large_tool_results():
    h = history(user("hi"), assistant("", "1"), tool("1", "x" * 20_000), *(user(f"p{i}") for i in range(4)))
    assert compactor(stub_tool_results_over=100, keep_tail_messages=4).microcompact(h) == 1
    assert h.messages[2].content.startswith("[trimmed:")


def test_microcompact_is_idempotent():
    """Each message changes at most once, so the provider's prompt cache stays valid."""
    h = history(tool("1", "x" * 20_000), *(user(f"p{i}") for i in range(4)))
    c = compactor(stub_tool_results_over=100, keep_tail_messages=4)
    assert [c.microcompact(h) for _ in range(3)] == [1, 0, 0]


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
    c = compactor(stub_tool_results_over=100, max_message_tokens=1000, keep_tail_messages=4)
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
