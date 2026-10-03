# Compaction that works for any model (plan for `agent-limits`)

Done on `agent-limits`: 1 (trigger from the model), 2 (token tail), 5 (summary prompt sized by the summary
model), plus a guard against summarizing every step when a small window's fixed overhead sits near the limit.
Item 4 is left to the server (the inference project's gateway sets the summary's `max_tokens` and thinking).

Written 2026-10-03, after Exp-1A in the inference project. Compared with OpenCode (`sst/opencode` @ `907b3bc`:
`packages/opencode/src/session/overflow.ts`, `session/compaction.ts`, `packages/llm/src/provider-error.ts`).

## What tinyharness already does the way OpenCode does

- Counts the provider's own usage, not an estimate (`Compactor.observe`, `used`).
- Catches a context-overflow error, summarizes and asks again (`agent/loop.py` `except ContextOverflowError`),
  detected by pattern (`llm/client.py` `_OVERFLOW`).
- Summarizes a plain-text transcript with clipped tool results and the earlier summary kept whole
  (`_transcript`); an empty summary keeps the history (`ef96cc9`).
- Subagents use the same threshold as the lead (`272dd8d`); summaries are tagged role `compaction` and run on one
  model (`90e9947`).

## The gaps, and the change for each

1. **Threshold is a fraction of the window; it should be the window minus the output.**
   Now: `needs_summary` is `used ≥ window × threshold` (0.90 default; 0.74 in the project config, picked by hand
   for 64K and our `max_tokens`). A fraction that is right for a 200K window is wrong for a 64K one with a 32K
   output. OpenCode: `usable = context − min(model output limit, 32,000)`, or `input limit − min(20,000, output)`
   when the model has a separate input limit; `compaction.reserved` overrides.
   Change: `needs_summary` is `used ≥ window − output_reserve`, with `output_reserve` per model
   (`models = [{ id, context, output }]`, default `min(output, 32_000)`, else 20,000) and an optional
   `compaction_reserved` in the config. Keep `auto_compact_threshold` as an optional override only.

2. **The kept tail is a message count; it should be a token budget.**
   Now: the last `keep_tail_messages = 12`, or back to a user message if that stays under `TAIL_SHARE` (0.3) of
   the window. Attempt 1: twelve messages of large tool results, so the summary saved ~2,300 tokens and the retry
   was still 369 over. OpenCode: 25% of the usable window, clamped to 2,000–15,000 tokens, walking back by whole
   turns and splitting a turn when one doesn't fit.
   Change: `_cut` walks back from the end summing `estimate_tokens` until `clamp(0.25 × usable, 2_000, 15_000)`;
   never starts the tail at a tool result (as now).

3. **On overflow the retry keeps the same tail.**
   Now: `summarize(history)` with the default tail, once; if that tail itself is too big, the retry fails.
   OpenCode: on overflow it summarizes everything before the last user request and replays that request after
   the summary.
   Change: on `ContextOverflowError`, summarize with the minimum tail (the current request only: `keep_tail=0`
   plus the `CONTINUING` copy of the request), then retry.

4. **The summary call doesn't set its own output size or thinking.**
   Now: `one_shot` sends no `max_tokens`; the project's gateway sets 4,096 and turns thinking off for role
   `compaction`. Against any other endpoint a reasoning model can think through its whole output and return
   nothing (Exp-1A attempt 2).
   Change: `one_shot(..., max_tokens=4_096)` for summaries (OpenCode's `SUMMARY_OUTPUT_TOKENS`), and the
   provider's no-thinking switch when it has one (`chat_template_kwargs.enable_thinking=false` for Qwen on
   vLLM, no `thinking` block on Anthropic, `reasoning_effort: low|minimal` on OpenAI).

5. **The summary prompt is sized from the agent's window, not the summary model's.**
   Now: `_transcript` keeps `context_window × 2` characters (half the window). With `compaction_model` set to a
   different model, that is the wrong window.
   Change: size it as `(summary model window − 4,096 − system prompt) × 4` characters.

6. **Tool-output protection is a fixed 40,000 tokens.**
   Now: `protect_tool_tokens = 40_000` is 61% of a 64K window, so microcompaction stubs almost nothing before
   the summary fires. (OpenCode's prune also protects 40K, but it is off by default and its models have 200K.)
   Change: `min(40_000, 0.3 × usable)`.

7. **Unknown window.** OpenCode turns compaction off when a model's context is 0 (unknown). tinyharness falls
   back to 200,000. Better: read `max_model_len` from `/v1/models` when the provider returns it (vLLM and the
   project's gateway do), else warn once that compaction is guessing.

## Tests to add

- 64K window, 32K output: compacts at 32K used, not at 0.90 × 64K.
- A tail of 12 messages × 5K tokens: the kept tail is ≤ 15K tokens.
- Overflow on a request whose default tail is too big: the retry succeeds with the minimum tail.
- Summary request carries `max_tokens = 4096` and the provider's no-thinking switch.
