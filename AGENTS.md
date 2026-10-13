# AGENTS.md

tinyharness: a small, readable LLM agent harness (CLI + Python SDK) for coding and
general agent work, on any major provider. Read by development agents and tinyharness
itself. Every line here should change what you do.

## Commands
- Tests: `uv run pytest -q` (all must pass; ~30s). Terminal tests are opt-in:
  `uv run pytest -q --terminal tests/test_terminal.py`.
- Lint and types: `uv run ruff check . && uv run ruff format --check . && uv run mypy`.
- Run it: `uv run tinyharness` (interactive) or `uv run tinyharness -p "..." --model provider:model`.
- Session logs, for debugging real runs: `~/.tinyharness/projects/<project>/sessions/*.jsonl`
  (subagents in `sidechains/`). `turn_end`, `stuck`, `recovered_calls` and the `finish`
  field on `assistant` records say what happened.

## Where things are
- `agent/loop.py`: the agent loop. `agent/history.py`: messages, and `well_formed`, the
  normalization every request goes through. `agent/compaction.py`: context management.
- `llm/client.py`: HTTP, streaming, retries, stop-reason normalization.
  `llm/adapters/`: one per wire format (chat, responses, anthropic, gemini).
  `llm/textcalls.py`: repairing malformed or text-written tool calls.
- `tools/`: one file per tool. The core offers Read, Write, Edit, Glob, Grep and Bash;
  the rest are turned on through extensions. `prompts/system.md`: the core system prompt.
- `ext.py`: the extension API (`setup(th)`; docs/extension-api.md). `extensions/`: the
  built-in ones (web, pdf, open, todo, agents). New features go here, not in the core.
- `sandbox/`: Seatbelt profile and `proxy.py`, the allowlist proxy that is a sandboxed
  command's only network. `permissions/`: the gate and modes.
- `subagents/`, `tools/team.py`: subagents and teams (the `agents` extension).
  `session.py`: ties it all together. `testing.py`: the scripted model for tests.

## Guarantees: don't break these
Each was a real failure; each has tests. A change that weakens one needs a reason.
1. Every tool call in the history gets exactly one result, right after its assistant
   message, whatever happens (Ctrl+C, errors, the step limit, compaction).
2. A reply's outcome comes from the normalized stop reason (`llm/types.py` FinishReason).
   Never run a tool call from a reply cut off at the output limit.
3. Earlier messages stay byte-identical unless compaction deliberately rewrites them,
   and the prefix (tools, then the system prompt with per-session facts last) is the
   same from call to call (prompt caching; people run this on their own vLLM servers).
4. Compaction always has a way out, including inside one long autonomous turn.
5. A tool error tells the model what to do next, not just what failed.
6. Adapters never look at model names; per-model behavior comes from the provider's
   model list or the provider config.
7. Provider flakiness (empty replies, dropped tool calls) is logged and handled
   generically, never with provider-specific workarounds.

## How to change things
- Fix the cause, from first principles; prefer removing a mechanism to adding one more.
  Before adding a special case, look for the existing mechanism it belongs in.
- Every fix gets a regression test named for the behavior; for a failure seen in real
  use, the docstring says what was seen ("Seen live: ...").
- Every user-visible change gets a line in `CHANGELOG.md`, written for users.
- Comments say why, in plain words, matching the surrounding density. No new
  dependencies without asking.
- Public classes and functions use concise numpydoc docstrings. Keep obvious private
  helpers to a one-line summary; add sections only when they convey useful information.
- Check established implementations and relevant standards before inventing a new
  approach; record the rationale that matters to tinyharness.
- Verify with a real model before calling an agent-behavior change done: the test suite
  uses scripted models and can't show how a real one reacts.

## Lessons
<!-- When corrected, add "When X, do Y". Newest first; delete what no longer applies. -->
- When you add an `except` around a block, check what else is raised inside it: a
  retry meant for stream errors once caught HTTP 4xx errors too and retried them.
- When a fix touches the loop, check every `continue` path for a way to loop forever or
  burn the step limit (notes the harness adds count as steps).
- When a model "keeps failing", read its session log before changing prompts: most
  "model" failures in the 2026-10 audit were the harness's.
