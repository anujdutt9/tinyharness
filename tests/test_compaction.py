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
    """A skill is loaded by reading its SKILL.md: those instructions stay, as an Agent's report does."""
    skill_read = Message(
        role="assistant", tool_calls=[ToolCall("s1", "Read", '{"path": "/home/skills/design/SKILL.md"}')]
    )
    h = history(
        user("go"),
        skill_read,
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


# ---- one long task (found by the audit: summarizing used to need a user message to cut at) ----


def one_long_task(n: int, result: str = BIG) -> History:
    """What an autonomous run looks like: one request, then only the agent's own steps."""
    out = [Message(role="system", content="sys"), user("build the report from data.db")]
    for i in range(n):
        out += [assistant("", f"c{i}"), tool(f"c{i}", result)]
    return history(*out)


async def test_one_long_task_can_be_summarized():
    h = one_long_task(40)
    llm = ScriptedLLM([])
    c = Compactor(llm, context_window=100_000, keep_tail_messages=12)
    assert await c.summarize(h)
    after = h.messages
    assert after[1].content.startswith(SUMMARY_MARK)
    # The kept part starts with the request, word for word, and never with a tool result.
    assert after[2].role == "user" and after[2].content.endswith("build the report from data.db")
    assert after[3].role == "assistant"
    calls = {c.id for m in after for c in m.tool_calls}
    assert calls == {m.tool_call_id for m in after if m.role == "tool"}


async def test_a_short_recent_exchange_is_kept_whole():
    h = history(
        Message(role="system", content="sys"),
        user("a"),
        assistant("one"),
        user("b"),
        assistant("", "t"),
        tool("t", "x"),
        assistant("two"),
    )
    assert await compactor().summarize(h, keep_tail=1)
    assert [m.content for m in h.messages[2:]] == ["b", None, "x", "two"]


def test_tool_call_arguments_count_toward_the_context():
    """30 Writes of ~120 KB each used to count as 63 tokens, so compaction never ran."""
    content = "x" * 120_000
    h = history(user("go"), *[assistant("", f"w{i}", name="Write") for i in range(3)])
    for m in h.messages[1:]:
        m.tool_calls[0].arguments = f'{{"path": "a.html", "content": "{content}"}}'
    assert compactor().used(h) > 3 * count_tokens(content)


def test_the_providers_count_is_the_baseline():
    from tinyharness.llm.usage import Usage

    h = history(user("hi"), assistant("hello"))
    c = compactor()
    c.observe(h, Usage(input=9_000, cached=1_000, output=10))
    assert c.used(h) == 10_010
    h.append(user("more"))
    assert 10_010 < c.used(h) < 10_030
    h.messages = h.messages[1:]  # rewritten (a summary, /clear): the count no longer applies
    assert c.used(h) < 100


async def test_microcompact_trims_old_tool_call_arguments():
    c = Compactor(ScriptedLLM([]), keep_tail_messages=2, protect_tool_tokens=0, cache_ttl=0, clock=Clock())
    write = assistant("", "w", name="Write")
    write.tool_calls[0].arguments = f'{{"path": "a.html", "content": "{BIG}"}}'
    write.state = object()  # saved reasoning repeats the arguments; it must go too
    h = history(user("go"), write, tool("w", "wrote a.html"), user("p1"), user("p2"))
    assert c.microcompact(h) == 1
    import json

    args = json.loads(h.messages[1].tool_calls[0].arguments)
    assert args["path"] == "a.html" and args["content"].startswith("[trimmed:")
    assert h.messages[1].state is None


async def test_the_summary_sees_tool_calls_files_todos_and_the_earlier_summary():
    seen = []

    class Recording(ScriptedLLM):
        async def one_shot(self, prompt, **_):
            seen.append(prompt)
            return "summary"

    h = history(
        Message(role="system", content="sys"),
        Message(role="system", content=f"{SUMMARY_MARK}\n" + "earlier fact " * 2000),
        user("fix the chart"),
        assistant("", "e1", name="Edit"),
        tool("e1", "edited"),
        user("thanks"),
        assistant("ok"),
    )
    h.messages[3].tool_calls[0].arguments = '{"path": "page.html", "old_string": "a", "new_string": "b"}'
    h.messages[4].meta = {"path": "/w/page.html", "sha": "1", "content": False}
    assert await Compactor(Recording([])).summarize(
        h, keep_tail=1, notes="[current todo list]\n  ○ Run tests"
    )
    prompt = seen[0]
    assert prompt.count("earlier fact") == 2000  # whole, not cut to 4,000 characters
    assert '[assistant called] Edit({"path": "page.html"' in prompt
    assert "[result of Edit] edited" in prompt and "/w/page.html" in prompt and "Run tests" in prompt


async def test_a_second_summary_still_keeps_the_request():
    h = one_long_task(40)
    c = Compactor(ScriptedLLM([]), context_window=100_000, keep_tail_messages=12)
    assert await c.summarize(h)
    for i in range(40, 60):
        h.messages += [assistant("", f"c{i}"), tool(f"c{i}", BIG)]
    assert await c.summarize(h)
    request = next(m for m in h.messages if m.role == "user")
    assert request.content.endswith("build the report from data.db")


def test_microcompact_leaves_arguments_it_cannot_shorten():
    import json

    c = Compactor(ScriptedLLM([]), keep_tail_messages=2, protect_tool_tokens=0, cache_ttl=0, clock=Clock())
    write = assistant("", "w", name="Write")
    write.tool_calls[0].arguments = json.dumps(
        {"path": "d.json", "content": {f"k{i}": "short value" for i in range(5000)}}
    )
    h = history(user("go"), write, tool("w", "ok"), user("p1"), user("p2"))
    assert c.microcompact(h) == 0 and c.microcompact(h) == 0


async def test_an_empty_summary_keeps_the_history():
    """A reasoning model can spend its whole output thinking and return no text; replacing the
    history with that would silently lose the agent's work."""

    class Thinker(ScriptedLLM):
        async def one_shot(self, prompt, **_):
            return ""

    h = history(Message(role="system", content="sys"), user("a " * 3000), assistant("b " * 3000), user("go on"), assistant("ok"))
    before = list(h.messages)
    assert not await Compactor(Thinker([])).summarize(h, keep_tail=1)
    assert h.messages == before


async def test_the_summary_runs_on_the_compaction_model_as_role_compaction():
    from tinyharness.llm import identity

    seen = {}

    class Client(ScriptedLLM):
        async def one_shot(self, prompt, *, model=None, **_):
            seen["role"], seen["model"] = identity.CURRENT.get().role, model
            return "summary"

    token = identity.CURRENT.set(identity.for_session("s1"))
    try:
        h = history(Message(role="system", content="sys"), user("a " * 3000), assistant("b " * 3000), user("go on"), assistant("ok"))
        assert await Compactor(Client([]), model="qwen-9b").summarize(h, keep_tail=1)
        assert identity.CURRENT.get().role == "lead"  # restored after the call
    finally:
        identity.CURRENT.reset(token)
    assert seen == {"role": "compaction", "model": "qwen-9b"}


# ---- when to summarize and how much to keep: from the model (opencode's rules) ----


def test_usable_is_the_window_less_the_models_output():
    assert compactor(context_window=200_000, max_output=8_192).usable == 200_000 - 8_192
    assert compactor(context_window=200_000, max_output=128_000).usable == 200_000 - 32_000  # at most 32K held back
    assert compactor(context_window=200_000).usable == 200_000 - 32_000  # unknown output: 32K
    assert compactor(context_window=65_536, max_output=lambda: 8_192).usable == 65_536 - 8_192  # read when asked
    assert compactor(context_window=65_536, reserved=17_000).usable == 65_536 - 17_000  # the config's own


def test_a_small_window_holds_back_at_most_half():
    """Model lists often give the output limit as the whole window; held back in full, every
    call would summarize."""
    assert compactor(context_window=32_768, max_output=32_768).usable == 16_384
    assert compactor(context_window=16_384).usable == 8_192  # unknown output


def test_summarizes_at_usable_unless_a_threshold_is_set():
    h = history(user("x"))
    c = compactor(context_window=100_000, max_output=20_000)
    c._baseline = (1, h.messages[0].id, 79_000)
    assert not c.needs_summary(h)
    c._baseline = (1, h.messages[0].id, 80_000)
    assert c.needs_summary(h)
    c.threshold = 0.9  # the old fraction still overrides
    assert not c.needs_summary(h)


def test_the_tail_budget_scales_with_the_window():
    assert compactor(context_window=65_536, max_output=32_768).tail_budget == int((65_536 - 32_000) * 0.25)
    assert compactor(context_window=8_000).tail_budget == 2_000  # floor
    assert compactor(context_window=1_000_000).tail_budget == 15_000  # cap
    assert compactor(preserve_recent_tokens=500).tail_budget == 500


def big_turns(n: int, words: int) -> History:
    """n user turns, each a Read whose result is `words` words."""
    out = [Message(role="system", content="sys")]
    for i in range(n):
        out += [user(f"q{i}"), assistant("", f"r{i}"), tool(f"r{i}", "word " * words), assistant(f"a{i}")]
    return history(*out)


async def test_the_kept_tail_is_a_token_budget_not_a_message_count():
    """Exp-1A: the last 12 messages were large tool results, so the summary saved ~2K tokens."""
    h = big_turns(12, 4_000)
    c = compactor(preserve_recent_tokens=10_000)
    assert await c.summarize(h)
    kept = [m for m in h.messages if not (m.content or "").startswith((SUMMARY_MARK, "sys"))]
    from tinyharness.llm.tokens import estimate_tokens

    assert estimate_tokens(kept) <= 10_000
    assert kept[0].role == "user"  # whole turns: this one fit
    calls = {c.id for m in kept for c in m.tool_calls}
    assert calls == {m.tool_call_id for m in kept if m.role == "tool"}  # no call without its result


async def test_a_turn_too_big_to_keep_whole_keeps_its_latest_steps():
    h = one_long_task(30, "word " * 2_000)
    assert await compactor(preserve_recent_tokens=6_000).summarize(h)  # each result ~2,500 tokens: two fit
    after = h.messages
    assert after[2].role == "user" and after[2].content.endswith("build the report from data.db")
    assert after[3].role == "assistant"  # never a tool result first
    assert sum(m.role == "tool" for m in after) == 2


async def test_when_not_even_the_last_step_fits_everything_is_summarized():
    h = one_long_task(3, "word " * 20_000)
    assert await compactor(preserve_recent_tokens=1_000).summarize(h)
    assert [m.role for m in h.messages] == ["system", "system", "user"]  # sys, summary, the request
    assert h.messages[2].content.endswith("build the report from data.db")


async def test_a_lone_request_is_not_summarized():
    assert not await compactor(preserve_recent_tokens=0).summarize(history(Message(role="system", content="s"), user("go")))


async def test_after_a_summary_it_waits_for_new_history_before_summarizing_again():
    """A small window where the system prompt, tools, summary and tail alone sit past the limit:
    summarizing again each step would only shrink the summary."""
    from tinyharness.llm.usage import Usage

    c = compactor(context_window=20_000, preserve_recent_tokens=2_000)  # usable 10,000
    h = big_turns(6, 1_500)
    assert c.needs_summary(h) and await c.summarize(h)
    c.observe(h, Usage(input=11_000, output=100))  # the provider: still over, after the summary
    assert not c.needs_summary(h)
    h.messages += [assistant("", "z"), tool("z", "word " * 2_500)]  # a tail's worth more
    assert c.needs_summary(h)


async def test_the_summary_request_is_built_like_opencodes():
    """The history in <conversation> tags with the instructions after it, and the system prompt
    saying not to continue: with the instructions only in the system prompt and the history
    ending in the agent's words, the 27B continued the conversation (a 37-token "summary")."""
    seen = {}

    class Client(ScriptedLLM):
        async def one_shot(self, prompt, *, system=None, **_):
            seen["prompt"], seen["system"] = prompt, system
            return "summary"

    h = history(
        Message(role="system", content="sys"),
        Message(role="system", content=f"{SUMMARY_MARK}\nrevenue fell 12% in July"),
        user("a " * 3000),
        assistant("I'll continue with the next query"),
        user("go on"),
        assistant("ok"),
    )
    assert await Compactor(Client([])).summarize(h, keep_tail=1)
    prompt = seen["prompt"]
    assert prompt.startswith("Here is the conversation so far:\n\n<conversation>\n")
    assert prompt.index("</conversation>") < prompt.index("<prior-summary>") < prompt.index("Summarize the conversation")
    assert "revenue fell 12% in July" in prompt.split("<prior-summary>")[1]
    assert "[earlier summary]" not in prompt  # the prior summary is its own block, not part of the history
    assert "Do not continue the conversation" in seen["system"]
