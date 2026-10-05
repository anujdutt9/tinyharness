"""Keeping the conversation inside the context window.

Three tiers, cheapest first, run before every model call:

  1. budget_cap    cap any single oversized message (e.g. a huge file read), keeping its
                   start and end
  2. microcompact  replace old, large tool results with a one-line stub, and trim the
                   long strings in old tool calls (a Write's content is on disk already)

Both save what they cut to a file in the session's scratch directory, named in the
message, so the model can Read it back instead of running the tool again.
  3. summarize     when the history leaves less room than the model's output needs,
                   have the model summarize everything but the recent tail

When to summarize and how much to keep follow opencode (session/overflow.ts,
session/compaction.ts), sized from the model, not fixed:

  - usable = window − reserve, where reserve is the model's output limit (at most
    OUTPUT_TOKEN_MAX), or `reserved` from the config. Summarize once the history
    reaches `usable`, so the reply always has its room. When the output limit is known,
    requests ask for exactly that reserve as their max_tokens (`reply_tokens`), so the
    trigger and the request agree.
  - The recent tail kept word for word is `preserve_recent_tokens`, by default a quarter
    of `usable` between TAIL_MIN and TAIL_MAX tokens: whole turns from the end, and part
    of the turn that doesn't fit (never starting at a tool result). After the provider
    says the prompt is too long, the tail is TAIL_MIN: the usual one has just failed.

The other limits scale with `usable` too, at most their 200K-window values. Fixed, they
shut tier 2 off on a 64K window (seen live): 40K of protected tool output and a 60%-of-
the-window warm threshold are both past the summary trigger, and SQL results of 1-2K
tokens are all under a 6K per-result floor, so every long run went straight to a summary.

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
  - cache warm: only once the history passes `clear_when_warm_at` of `usable`,
    to hold off the lossy summary, and only for at least `min_savings` tokens at
    once (the idea behind Anthropic's `clear_at_least` and opencode's prune)
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
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
# Legacy markers remain readable so sessions written by earlier tinyharness versions can
# still be compacted correctly. New checkpoints never repeat the original request.
CONTINUING = "[continuing after a summary] The request being worked on:"
RESUME = (
    "[This is the original request, for reference. The summary above says what is already done: continue "
    "from there with the next unfinished step, and don't repeat steps it reports as finished.]"
)
CHECKPOINT_CONTINUATION = """[continuing from a context checkpoint]
The checkpoint above is authoritative about execution state. Continue only from Active or Next Action. Do
not repeat anything in Completed, and do not recreate completed Agent or Team calls; reuse their recorded
results. If no work remains, report completion."""
# opencode's defaults (provider/transform.ts OUTPUT_TOKEN_MAX, session/compaction.ts)
OUTPUT_TOKEN_MAX = 32_000  # the most of the window held back for the reply
TAIL_SHARE = 0.25  # of `usable`: the recent tail kept word for word ...
TAIL_MIN, TAIL_MAX = 2_000, 15_000  # ... within these bounds
# Tier 1 and 2 limits, as a share of `usable`, at most these tokens (their old fixed values)
MESSAGE_SHARE, MESSAGE_MAX = 0.50, 16_000  # the largest single message (a 10K-token file read whole on 64K)
PROTECT_SHARE, PROTECT_MAX = 0.30, 40_000  # recent tool output tier 2 never clears
STUB_SHARE, STUB_MAX = 0.036, 6_000  # the smallest tool result worth stubbing
SAVINGS_SHARE, SAVINGS_MAX = 0.15, 20_000  # the least a warm-cache clearing must free
CLEAR_WHEN_WARM_AT = 0.75  # of `usable`: tier 2 may rewrite a warm cache from here on
# The summary's own output limit: a summary is a few thousand tokens, and a model given its
# whole output limit can spend all of it thinking and return nothing.
SUMMARY_MAX_TOKENS = 8_192
LONG_STRING = 500  # chars: tool-call strings longer than this are trimmed in old calls

# Calibration (when the model's tokenizer isn't available): the provider's count for the messages
# added between two requests over our estimate for them, as a moving average, within these bounds.
CALIBRATION_MIN_TOKENS = 200  # too few new tokens to learn from
CALIBRATION_BOUNDS = (0.5, 4.0)
CALIBRATION_WEIGHT = 0.3  # of each new sample

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
        max_output: int | Callable[[], int | None] | None = None,
        reserved: int | None = None,
        preserve_recent_tokens: int | None = None,
        threshold: float | None = None,
        max_message_tokens: int | None = None,
        stub_tool_results_over: int | None = None,
        keep_tail_messages: int = 12,
        protect_tool_tokens: int | None = None,
        min_savings: int | None = None,
        clear_when_warm_at: float = CLEAR_WHEN_WARM_AT,
        cache_ttl: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
        model: str | None = None,
        counter: Callable[[], Callable[[str], int] | None] | None = None,
    ) -> None:
        self.llm = llm
        self.model = model  # the model summaries are written on; None = the agent's own
        self._context_window = context_window
        self._max_output = max_output  # the model's output limit; None = unknown
        self.reserved = reserved  # tokens held back for the reply, overriding the model's output limit
        self.preserve_recent_tokens = preserve_recent_tokens
        self.threshold = threshold  # a fraction of the window instead of `usable` (old setting)
        self._max_message_tokens = max_message_tokens  # None: from `usable`
        self._stub_tool_results_over = stub_tool_results_over  # None: from `usable`
        self.keep_tail = keep_tail_messages
        self._protect_tool_tokens = protect_tool_tokens  # None: from `usable`
        self._min_savings = min_savings  # None: from `usable`
        self.clear_when_warm_at = clear_when_warm_at
        self.cache_ttl = cache_ttl  # Anthropic's default; other providers keep theirs as long or longer
        self._clock = clock
        self._last_call: float | None = None
        # The provider's token count for the history up to a message: (message count, last
        # message's id, tokens). Dropped when history is rewritten in place.
        self._baseline: tuple[int, str, int] | None = None
        # The prompt the provider counted on the first call after a summary: what a summary
        # can't remove (system prompt, tools, the summary, the kept tail). None: not known yet.
        self._floor: int | None = None
        self._floor_pending = False
        self._counter = counter  # the model's own tokenizer, when there is one
        # Calibration: our estimate × ratio ≈ the provider's count. `_last_request`: the message
        # count, last message id and the provider's prompt count of the previous request.
        self.ratio = 1.0
        self._calibrated = False
        self._last_request: tuple[int, str, int] | None = None

    @property
    def context_window(self) -> int:
        """May be a callable: the window is only known once the provider's model list is."""
        window = self._context_window
        return window() if callable(window) else window

    def _output_limit(self) -> int | None:
        return self._max_output() if callable(self._max_output) else self._max_output

    @property
    def reserve(self) -> int:
        """Tokens held back for the reply: `reserved`, or the model's output limit, at most
        OUTPUT_TOKEN_MAX. Unlike opencode, at most half the window: model lists often give
        the output limit as the whole window (or it is unknown on a 32K model), which would
        summarize on every call."""
        if self.reserved is not None:
            return self.reserved
        window = self.context_window
        return min(self._output_limit() or OUTPUT_TOKEN_MAX, OUTPUT_TOKEN_MAX, window // 2)

    @property
    def usable(self) -> int:
        """How much of the window the history may fill and still leave the reply its room
        (opencode's `usable`): the window less `reserve`."""
        return max(0, self.context_window - self.reserve)

    @property
    def reply_tokens(self) -> int | None:
        """The max_tokens to ask for: `reserve`, the room the trigger keeps. None when it is a
        guess (no output limit from the model list or the config): a guessed limit over the
        model's real one is an error on some APIs, so the provider's default is kept then.
        (Asking for the whole output limit, as before, can fail too: prompt + max_tokens over
        the window is an error on Anthropic's API once the history is near `usable`.)"""
        if self.reserved is None and not self._output_limit():
            return None
        return self.reserve or None

    @property
    def max_message_tokens(self) -> int:
        if self._max_message_tokens is not None:
            return self._max_message_tokens
        return min(MESSAGE_MAX, int(self.usable * MESSAGE_SHARE))

    @property
    def stub_tool_results_over(self) -> int:
        if self._stub_tool_results_over is not None:
            return self._stub_tool_results_over
        return min(STUB_MAX, int(self.usable * STUB_SHARE))

    @property
    def min_savings(self) -> int:
        if self._min_savings is not None:
            return self._min_savings
        return min(SAVINGS_MAX, int(self.usable * SAVINGS_SHARE))

    @property
    def protect_tool_tokens(self) -> int:
        if self._protect_tool_tokens is not None:
            return self._protect_tool_tokens
        return min(PROTECT_MAX, int(self.usable * PROTECT_SHARE))

    @property
    def tail_budget(self) -> int:
        """Tokens of recent history a summary keeps word for word."""
        if self.preserve_recent_tokens is not None:
            return self.preserve_recent_tokens
        return min(TAIL_MAX, max(TAIL_MIN, int(self.usable * TAIL_SHARE)))

    def mark_call(self) -> None:
        """Record a model call: the provider's cache for this prefix is fresh from now."""
        self._last_call = self._clock()

    @property
    def cache_cold(self) -> bool:
        """True if the prefix cache has likely expired, so rewriting history costs nothing extra.
        Before the first call (e.g. a resumed session) it is taken as cold."""
        return self._last_call is None or self._clock() - self._last_call > self.cache_ttl

    def estimate(self, messages: list[Message]) -> int:
        """Tokens these messages take up: with the model's tokenizer when there is one, scaled
        by the calibration against the provider's counts (which also covers what neither
        sees: how the chat template wraps each message, reasoning a provider doesn't resend)."""
        count = self._counter() if self._counter is not None else None
        return round(estimate_tokens(messages, count) * self.ratio)

    def _calibrate(self, history: History, prompt: int) -> None:
        """Learn the ratio from what was added between this request and the previous one: the
        provider counted both prompts, so the difference is exact for exactly those messages
        (the system prompt and tools, counted once in each, cancel out)."""
        messages = history.messages
        n = len(messages) - 1  # the request didn't have the reply just appended
        last = self._last_request
        if last is not None and n > last[0] and messages[last[0] - 1].id == last[1]:
            count = self._counter() if self._counter is not None else None
            ours = estimate_tokens(messages[last[0] : n], count)
            theirs = prompt - last[2]
            if ours >= CALIBRATION_MIN_TOKENS and theirs > 0:
                low, high = CALIBRATION_BOUNDS
                sample = min(high, max(low, theirs / ours))
                w = CALIBRATION_WEIGHT if self._calibrated else 1.0
                self.ratio, self._calibrated = (1 - w) * self.ratio + w * sample, True
        self._last_request = (n, messages[n - 1].id, prompt) if n > 0 else None

    def _rewritten(self) -> None:
        """History changed in place: the provider's counts no longer match it."""
        self._baseline = None
        self._last_request = None

    def observe(self, history: History, usage: Usage) -> None:
        """Record the provider's own count after a model call: the prompt it read plus the
        reply, which is now the last message. More accurate than any estimate (it includes
        the system prompt, tool definitions and images, in the model's own tokenizer)."""
        tokens = usage.input + usage.cached + usage.output
        if tokens and history.messages:
            self._baseline = (len(history.messages), history.messages[-1].id, tokens)
        if usage.input + usage.cached and history.messages:
            self._calibrate(history, usage.input + usage.cached)
        if self._floor_pending and usage.input + usage.cached:
            self._floor, self._floor_pending = usage.input + usage.cached, False

    def used(self, history: History) -> int:
        """Tokens the history takes up in the context window."""
        messages = history.messages
        if self._baseline is not None:
            n, last_id, tokens = self._baseline
            if len(messages) >= n and messages[n - 1].id == last_id:
                return tokens + self.estimate(messages[n:])
        return self.estimate(messages)

    def budget_cap(self, history: History, spill: Path | None = None) -> int:
        """Truncate messages larger than `max_message_tokens`, keeping the start and the end
        (a test run's or a build's verdict is in its last lines). The whole text goes to a
        file in `spill`, named in the message. Returns how many were capped."""
        capped = 0
        limit = self.max_message_tokens
        for m in history.messages:
            content = m.content or ""
            if not content or _already_compacted(content):
                continue
            tokens = count_tokens(content)
            if tokens <= limit:
                continue
            half = limit * 2  # chars for each end: ~4 chars per token
            where = _spilled(spill, m, content)
            cut = len(content) - 2 * half
            m.content = (
                f"{TRUNCATED_MARK} kept ~{limit} of {tokens} tokens, the start and the end{where}]\n\n"
                f"{content[:half]}\n\n…[{cut} chars cut]…\n\n{content[-half:]}"
            )
            m.state = None  # the provider's saved copy would bring the full text back
            capped += 1
        if capped:
            self._rewritten()
        return capped

    def microcompact(self, history: History, spill: Path | None = None) -> int:
        """Stub old tool results and trim old tool calls, in one batch. A stubbed result goes
        to a file in `spill`, named in the stub. Returns how many messages changed.

        Protected: the last `keep_tail` messages, the most recent `protect_tool_tokens`
        of tool output and calls, KEEP_TOOLS, and anything under `stub_tool_results_over`.
        The rest is changed if the cache is cold, or if the history is past
        `clear_when_warm_at` of `usable` and at least `min_savings` tokens would go
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
            or self.used(history) < self.usable * self.clear_when_warm_at
        ):
            return 0
        for m, tokens in batch:
            if m.role == "tool":
                where = _spilled(spill, m, m.content or "")
                m.content = f"{STUB_MARK} {tokens} tokens of older tool output{where}]"
            else:
                m.tool_calls = [
                    c if c.name in KEEP_TOOLS else ToolCall(c.id, c.name, _trim_arguments(c.arguments))
                    for c in m.tool_calls
                ]
                m.state = None  # the provider's saved copy has the full arguments
        self._rewritten()
        return len(batch)

    def needs_summary(self, history: History) -> bool:
        """Past the limit, and (after a summary) with at least a tail's worth of new history
        to fold in: when the system prompt, tools, summary and tail alone sit near the limit
        (a small window), summarizing again every step would only shrink the summary. A real
        overflow is still caught from the provider's error."""
        limit = self.context_window * self.threshold if self.threshold is not None else self.usable
        used = self.used(history)
        if used < limit:
            return False
        return self._floor is None or used - self._floor >= self.tail_budget

    async def summarize(
        self,
        history: History,
        *,
        keep_tail: int | None = None,
        tail_tokens: int | None = None,
        notes: str = "",
    ) -> bool:
        """Replace everything before the recent tail with an LLM-written summary.
        The tail is `tail_tokens` (default `tail_budget`) tokens of the latest turns;
        `keep_tail` keeps that many messages instead (more, if that would start the tail at a
        tool result). `notes`: current state worth keeping (the todo list).

        The tail can start mid-task, even inside one long agentic turn: never at a tool
        result, so a tool call is never separated from its result. When it doesn't start
        at a user message, a generic checkpoint continuation is inserted so APIs see a
        user message before the remaining assistant work. The original request stays in
        the summary instead of becoming a fresh instruction. Returns False if there was
        nothing to summarize.
        """
        messages = history.messages
        system = [
            m for m in messages[:1] if m.role == "system" and not (m.content or "").startswith(SUMMARY_MARK)
        ]
        body = messages[len(system) :]

        if keep_tail is not None:
            cut = self._cut(body, keep_tail)
        else:
            cut = self._cut_tokens(body, self.tail_budget if tail_tokens is None else tail_tokens)
        if cut == 0:
            return False
        head, tail = body[:cut], body[cut:]

        # On the configured compaction model (one model for every summary, no fallback), tagged
        # role "compaction" so a gateway can tell it from the agent's own turns.
        who = identity.CURRENT.get()
        token = identity.CURRENT.set(dataclasses.replace(who, role="compaction")) if who is not None else None
        try:
            prompt, instructions = self._prompt(head, notes), load_prompt("compaction_system")
            if self.model:
                summary = await self.llm.one_shot(
                    prompt, system=instructions, model=self.model, max_tokens=SUMMARY_MAX_TOKENS
                )
            else:
                summary = await self.llm.one_shot(prompt, system=instructions, max_tokens=SUMMARY_MAX_TOKENS)
        finally:
            if token is not None:
                identity.CURRENT.reset(token)
        if not (summary or "").strip():
            # e.g. a reasoning model that spent its whole output thinking: dropping the history for an
            # empty summary would lose the agent's work, so keep it as it is.
            log.warning("compaction: the model returned an empty summary; history kept")
            return False
        kept = [*system, Message(role="system", content=f"{SUMMARY_MARK}\n{summary}")]
        if not tail or tail[0].role != "user":
            kept.append(Message(role="user", content=CHECKPOINT_CONTINUATION))
        history.messages = [*kept, *tail]
        self._rewritten()
        self._floor, self._floor_pending = None, True
        return True

    def _summary_window(self) -> int:
        """The window of the model that writes the summary (`model` may be smaller than the agent's)."""
        info = self.llm.model_info(self.model) if self.model and hasattr(self.llm, "model_info") else None
        return info.context if info is not None and info.context else self.context_window

    def _cut_tokens(self, body: list[Message], budget: int) -> int:
        """Where the kept tail starts, for a tail of at most `budget` tokens (opencode's
        `select`): whole turns from the end (a turn starts at a user message), then the
        latest part of the turn that doesn't fit whole, never starting at a tool result.
        len(body): everything is summarized and a checkpoint continuation follows the summary,
        when not even the last step fits, or when everything does (as opencode: the
        provider's count, which an overflow reports, can exceed this estimate)."""
        if len(body) <= 1:
            return 0  # a lone request: nothing to summarize
        sizes = [self.estimate([m]) for m in body]
        after = [0] * (len(body) + 1)  # after[i]: tokens from message i to the end
        for i in range(len(body) - 1, -1, -1):
            after[i] = after[i + 1] + sizes[i]
        if after[0] <= budget:
            return len(body)
        starts = [i for i, m in enumerate(body) if m.role == "user"] or [0]
        keep = len(body)
        for start in reversed(starts):
            if after[start] <= budget:
                keep = start
                continue
            # Part of this turn: the earliest step that fits, after the turn's user message.
            return next(
                (i for i in range(start + 1, keep) if body[i].role != "tool" and after[i] <= budget), keep
            )
        return keep

    def _cut(self, body: list[Message], keep: int) -> int:
        """Where the kept tail starts, for the last `keep` messages. Preferably at a user
        message, so the whole latest exchange stays as it is; but if that leaves more than
        `tail_budget` (one long task: the user's message is far back), at the latest point
        that isn't a tool result."""
        start = max(0, len(body) - keep)
        at_user = start
        while at_user > 0 and body[at_user].role != "user":
            at_user -= 1
        if at_user > 0 and self.estimate(body[at_user:]) <= self.tail_budget:
            return at_user
        while start > 0 and body[start].role == "tool":
            start -= 1
        return start

    def _prompt(self, head: list[Message], notes: str) -> str:
        """The summarizer's request, built as opencode's (core/session/compaction.ts `buildPrompt`):
        the history inside <conversation> tags, an earlier summary in <prior-summary>, and the
        instructions after them. With the instructions only in the system prompt and the history
        ending in the agent's own words, a model continues the conversation instead (the 27B did,
        with thinking off)."""
        prior = "\n\n".join(
            (m.content or "").removeprefix(SUMMARY_MARK).strip() for m in head if m.role == "system"
        )
        parts = [
            "Here is the conversation so far:\n\n<conversation>\n"
            + self._transcript([m for m in head if m.role != "system"], notes)
            + "\n</conversation>"
        ]
        if prior:
            parts += [
                "Here is the summary of the conversation before the <conversation> above:\n\n"
                f"<prior-summary>\n{prior}\n</prior-summary>",
                load_prompt("compaction_update"),
            ]
        parts.append(load_prompt("compaction"))
        return "\n\n".join(parts)

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
        budget = self._summary_window() * 2  # chars: ~half the summary model's window at ~4 chars per token
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


def _spilled(directory: Path | None, m: Message, content: str) -> str:
    """Save `content` to a file in `directory` (the session's scratch space, which the model
    can Read without asking) and say where, for the message that replaces it. Empty if
    there is nowhere to put it."""
    if directory is None:
        return ""
    name = re.sub(r"[^\w-]", "_", m.tool_call_id or m.id)
    path = directory / f"compacted-{name}.txt"
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as e:
        log.warning("compaction: could not save the full text to %s: %s", path, e)
        return ""
    return f"; all of it is in {path}: Read or Grep that file if you need it"


def _request(m: Message) -> str:
    """Recover a request from an old repeated-request checkpoint.

    Kept for loading sessions produced before checkpoint continuations stopped copying the
    original request. Not a harness note or a teammate's message.
    """
    content = m.content or ""
    if m.role != "user" or content.startswith(("[tinyharness]", "<teammate")):
        return ""
    return content.removeprefix(CONTINUING).removesuffix(RESUME).strip()


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
