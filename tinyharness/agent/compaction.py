"""Keeping the conversation inside the context window.

Three tiers, cheapest first, run before every model call:

  1. budget_cap    cap any single oversized message (e.g. a huge file read)
  2. microcompact  replace old, large tool results with a one-line stub, and trim the
                   long strings in old tool calls (a Write's content is on disk already)
  3. summarize     when the whole history nears the window, have the model
                   summarize everything but the recent tail

How full the window is comes from the provider's own count for the last request,
plus an estimate for what was added since (`used`).

Tiers 1 and 2 edit messages in place, exactly once, and mark them. Each
message's bytes therefore change at most once, which keeps the provider's
prompt-prefix cache valid on later turns.

Changing a message still invalidates the cached prefix from that message on, so
the next request pays to write the cache again. Tier 1 only touches messages
that have not been sent yet, so it costs nothing. Tier 2 does cost: stubbing one
result each time the tail moves on would mean a cache miss on nearly every turn.

While the cache is warm, clearing roughly breaks even on cost: the rewrite costs
about as much as dozens of later turns save. It also takes information away from
the model. So tier 2 clears only when it pays:

  - cache cold (the last call was more than `cache_ttl` ago): the whole prompt
    is written again anyway, so clearing is free and makes that write cheaper
  - cache warm: only once the history passes `clear_when_warm_at` of the window,
    to hold off the lossy summary, and only for at least `min_savings` tokens at
    once (the idea behind Anthropic's `clear_at_least` and opencode's prune)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from tinyharness.agent.history import History, Message, ToolCall
from tinyharness.llm import identity
from tinyharness.llm.tokens import count_tokens, estimate_tokens
from tinyharness.prompts import load_prompt

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tinyharness.llm.client import LLMClient
    from tinyharness.llm.usage import Usage

TRUNCATED_MARK = "[truncated:"
STUB_MARK = "[trimmed:"
SUMMARY_MARK = "[prior conversation summary]"
# Starts the user message that repeats the request when a summary cuts into the middle of
# a task (the kept messages begin with the agent's work, and APIs want a user message first).
CONTINUING = "[continuing after a summary] The request being worked on:"
TAIL_SHARE = 0.3  # of the window: the most a summary keeps word for word to end at a user message
LONG_STRING = 500  # chars: tool-call strings longer than this are trimmed in old calls

# Results that are instructions or delegated work: expensive to redo, cheap to keep.
KEEP_TOOLS = frozenset({"Agent", "Team"})


def _kept(call: ToolCall | None) -> bool:
    """A result compaction leaves alone: delegated work, or a skill's instructions (a Read
    of a SKILL.md, which is how a skill is loaded)."""
    if call is None:
        return False
    return call.name in KEEP_TOOLS or (call.name == "Read" and '/SKILL.md"' in call.arguments)


def _already_compacted(content: str) -> bool:
    return content.startswith((TRUNCATED_MARK, STUB_MARK))


class Compactor:
    def __init__(
        self,
        llm: LLMClient,
        *,
        context_window: int | Callable[[], int] = 200_000,
        threshold: float = 0.90,
        max_message_tokens: int = 16_000,
        stub_tool_results_over: int = 6_000,
        keep_tail_messages: int = 12,
        protect_tool_tokens: int = 40_000,
        min_savings: int = 20_000,
        clear_when_warm_at: float = 0.60,
        cache_ttl: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        model: str | None = None,
    ) -> None:
        self.llm = llm
        self.model = model  # the model summaries are written on; None = the agent's own
        self._context_window = context_window
        self.threshold = threshold
        self.max_message_tokens = max_message_tokens
        self.stub_tool_results_over = stub_tool_results_over
        self.keep_tail = keep_tail_messages
        self.protect_tool_tokens = protect_tool_tokens
        self.min_savings = min_savings
        self.clear_when_warm_at = clear_when_warm_at
        self.cache_ttl = cache_ttl  # Anthropic's default; other providers keep theirs as long or longer
        self._clock = clock
        self._last_call: float | None = None
        # The provider's token count for the history up to a message: (message count, last
        # message's id, tokens). Dropped when history is rewritten in place.
        self._baseline: tuple[int, str, int] | None = None

    @property
    def context_window(self) -> int:
        """May be a callable: the window is only known once the provider's model list is."""
        window = self._context_window
        return window() if callable(window) else window

    def mark_call(self) -> None:
        """Record a model call: the provider's cache for this prefix is fresh from now."""
        self._last_call = self._clock()

    @property
    def cache_cold(self) -> bool:
        """True if the prefix cache has likely expired, so rewriting history costs nothing extra.
        Before the first call (e.g. a resumed session) it is taken as cold."""
        return self._last_call is None or self._clock() - self._last_call > self.cache_ttl

    def observe(self, history: History, usage: Usage) -> None:
        """Record the provider's own count after a model call: the prompt it read plus the
        reply, which is now the last message. More accurate than any estimate (it includes
        the system prompt, tool definitions and images, in the model's own tokenizer)."""
        tokens = usage.input + usage.cached + usage.output
        if tokens and history.messages:
            self._baseline = (len(history.messages), history.messages[-1].id, tokens)

    def used(self, history: History) -> int:
        """Tokens the history takes up in the context window."""
        messages = history.messages
        if self._baseline is not None:
            n, last_id, tokens = self._baseline
            if len(messages) >= n and messages[n - 1].id == last_id:
                return tokens + estimate_tokens(messages[n:])
        return estimate_tokens(messages)

    def budget_cap(self, history: History) -> int:
        """Truncate messages larger than `max_message_tokens`. Returns how many were capped."""
        capped = 0
        for m in history.messages:
            content = m.content or ""
            if not content or _already_compacted(content):
                continue
            tokens = count_tokens(content)
            if tokens <= self.max_message_tokens:
                continue
            keep_chars = self.max_message_tokens * 4  # ~4 chars per token
            m.content = f"{TRUNCATED_MARK} kept ~{self.max_message_tokens} of {tokens} tokens]\n\n{content[:keep_chars]}"
            m.state = None  # the provider's saved copy would bring the full text back
            capped += 1
        if capped:
            self._baseline = None
        return capped

    def microcompact(self, history: History) -> int:
        """Stub old tool results and trim old tool calls, in one batch. Returns how many
        messages changed.

        Protected: the last `keep_tail` messages, the most recent `protect_tool_tokens`
        of tool output and calls, KEEP_TOOLS, and anything under `stub_tool_results_over`.
        The rest is changed if the cache is cold, or if the history is past
        `clear_when_warm_at` of the window and at least `min_savings` tokens would go
        (see the module docstring)."""
        messages = history.messages
        if len(messages) <= self.keep_tail:
            return 0
        calls_by_id = {c.id: c for m in messages for c in m.tool_calls}
        protect = self.protect_tool_tokens
        batch: list[tuple[Message, int]] = []
        for m in reversed(messages[: -self.keep_tail]):
            if m.role == "tool":
                content = m.content or ""
                if _already_compacted(content) or _kept(calls_by_id.get(m.tool_call_id or "")):
                    continue
                tokens = count_tokens(content)
            elif m.role == "assistant" and m.tool_calls:
                # Only calls trimming would shorten: big arguments made of short strings can't
                # be, and counting them would rewrite the message on every pass.
                calls = [
                    c for c in m.tool_calls if not _kept(c) and _trim_arguments(c.arguments) != c.arguments
                ]
                tokens = sum(count_tokens(c.arguments) for c in calls)
            else:
                continue
            if protect > 0:
                protect -= tokens
            elif tokens > self.stub_tool_results_over:
                batch.append((m, tokens))
        if not batch:
            return 0
        if not self.cache_cold and (
            sum(t for _, t in batch) < self.min_savings
            or self.used(history) < self.context_window * self.clear_when_warm_at
        ):
            return 0
        for m, tokens in batch:
            if m.role == "tool":
                m.content = f"{STUB_MARK} {tokens} tokens of older tool output]"
            else:
                m.tool_calls = [
                    c if c.name in KEEP_TOOLS else ToolCall(c.id, c.name, _trim_arguments(c.arguments))
                    for c in m.tool_calls
                ]
                m.state = None  # the provider's saved copy has the full arguments
        self._baseline = None
        return len(batch)

    def needs_summary(self, history: History) -> bool:
        return self.used(history) >= self.context_window * self.threshold

    async def summarize(self, history: History, *, keep_tail: int | None = None, notes: str = "") -> bool:
        """Replace everything before the recent tail with an LLM-written summary.
        `keep_tail` overrides how many recent messages stay (more, if that would start the
        tail at a tool result). `notes`: current state worth keeping (the todo list).

        The tail can start mid-task, even inside one long agentic turn: never at a tool
        result, so a tool call is never separated from its result. When it doesn't start
        at a user message, the request being worked on is repeated, word for word, as the
        first message after the summary. Returns False if there was nothing to summarize.
        """
        messages = history.messages
        system = [
            m for m in messages[:1] if m.role == "system" and not (m.content or "").startswith(SUMMARY_MARK)
        ]
        body = messages[len(system) :]

        cut = self._cut(body, self.keep_tail if keep_tail is None else keep_tail)
        if cut == 0:
            return False
        head, tail = body[:cut], body[cut:]

        # On the configured compaction model (one model for every summary, no fallback), tagged
        # role "compaction" so a gateway can tell it from the agent's own turns.
        who = identity.CURRENT.get()
        token = identity.CURRENT.set(dataclasses.replace(who, role="compaction")) if who is not None else None
        try:
            prompt, instructions = self._transcript(head, notes), load_prompt("compaction")
            if self.model:
                summary = await self.llm.one_shot(prompt, system=instructions, model=self.model)
            else:
                summary = await self.llm.one_shot(prompt, system=instructions)
        finally:
            if token is not None:
                identity.CURRENT.reset(token)
        if not (summary or "").strip():
            # e.g. a reasoning model that spent its whole output thinking: dropping the history for an
            # empty summary would lose the agent's work, so keep it as it is.
            log.warning("compaction: the model returned an empty summary; history kept")
            return False
        kept = [*system, Message(role="system", content=f"{SUMMARY_MARK}\n{summary}")]
        if tail[0].role != "user":
            request = next((r for m in reversed(head) if (r := _request(m))), "")
            kept.append(Message(role="user", content=f"{CONTINUING}\n{request}"))
        history.messages = [*kept, *tail]
        self._baseline = None
        return True

    def _cut(self, body: list[Message], keep: int) -> int:
        """Where the kept tail starts. Preferably at a user message, so the whole latest
        exchange stays as it is; but if that leaves too much (one long task: the user's
        message is far back), at the latest point that isn't a tool result."""
        start = max(0, len(body) - keep)
        at_user = start
        while at_user > 0 and body[at_user].role != "user":
            at_user -= 1
        if at_user > 0 and estimate_tokens(body[at_user:]) <= self.context_window * TAIL_SHARE:
            return at_user
        while start > 0 and body[start].role == "tool":
            start -= 1
        return start

    def _transcript(self, head: list[Message], notes: str) -> str:
        """What the summarizer reads: every message, with the tool calls the agent made
        (what it ran and changed), shortened to fit in about half the window."""
        names = {c.id: c.name for m in head for c in m.tool_calls}
        changed = sorted(
            {
                m.meta["path"]
                for m in head
                if m.role == "tool" and m.meta.get("path") and not m.meta.get("content")
            }
        )
        budget = self.context_window * 2  # chars: ~half the window at ~4 chars per token
        for cap in (4000, 2000, 800, 300):
            parts: list[str] = []
            for m in head:
                content = m.content or ""
                if m.role == "system":  # an earlier summary: whole, so its facts don't erode
                    parts.append(f"[earlier summary]\n{content}")
                elif m.role == "tool":
                    parts.append(
                        f"[result of {names.get(m.tool_call_id or '', 'a tool')}] {_clip(content, cap)}"
                    )
                elif m.role == "assistant":
                    lines = [f"[assistant] {_clip(content, cap)}"] if content else []
                    lines += [f"[assistant called] {c.name}({_brief(c.arguments)})" for c in m.tool_calls]
                    parts.append("\n".join(lines))
                else:
                    parts.append(f"[user] {_clip(content, max(cap, 4000))}")
            if changed:
                parts.append("[files changed by Edit/Write in this part] " + ", ".join(changed))
            if notes:
                parts.append(notes)
            transcript = "\n\n".join(parts)
            if len(transcript) <= budget:
                break
        return transcript[-budget:]


def _request(m: Message) -> str:
    """The user's request in `m`, if it is one: a message they wrote, or the copy an earlier
    summary kept (then the only copy left). Not a harness note or a teammate's message."""
    content = m.content or ""
    if m.role != "user" or content.startswith(("[tinyharness]", "<teammate")):
        return ""
    return content.removeprefix(CONTINUING).strip()


def _clip(text: str, chars: int) -> str:
    return text if len(text) <= chars else f"{text[:chars]} …[{len(text) - chars} more chars]"


def _brief(arguments: str) -> str:
    """A tool call's arguments with long strings shortened: which file, not its content."""
    return _clip(_trim_arguments(arguments, keep=200, mark="chars"), 400)


def _trim_arguments(arguments: str, *, keep: int = 0, mark: str = "") -> str:
    """Arguments JSON with string values over LONG_STRING replaced by a stub (still valid
    JSON, so APIs that want an object get one)."""
    try:
        value = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return arguments[:LONG_STRING]
    if not isinstance(value, dict):
        return arguments

    def trim(v: object) -> object:
        if isinstance(v, str) and len(v) > LONG_STRING:
            return f"{v[:keep]}…[{len(v)} {mark}]" if mark else f"{STUB_MARK} {len(v)} chars]"
        if isinstance(v, list):
            return [trim(x) for x in v]
        if isinstance(v, dict):
            return {k: trim(x) for k, x in v.items()}
        return v

    return json.dumps({k: trim(v) for k, v in value.items()}, ensure_ascii=False)
