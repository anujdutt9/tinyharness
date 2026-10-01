# Features

Everything tinyharness does, in more detail than the README. For how the pieces fit together, see [How it works](../README.md#how-it-works).

## Reliability

Small and self-hosted models fail in ways the big ones rarely do, and providers fail in
ways no model can fix. tinyharness handles both in the harness, not with model-specific
special cases (see [AGENTS.md](../AGENTS.md#guarantees-dont-break-these) for the
guarantees and why each exists):

- Every reply has a clear outcome, normalized from each API's own stop reasons. A reply
  cut off at the output limit is continued; a tool call cut off mid-arguments isn't run,
  and the model is told to write in smaller pieces.
- The history sent to a provider is always well formed: every tool call answered,
  empty messages dropped, other providers' tool-call ids replaced after `/model`.
- Tool calls written as text (Hermes JSON, Qwen3-Coder XML) are recovered when a
  server's tool-call parser fails on them; malformed arguments are repaired.
- Compaction starts from the provider's own token count and can summarize inside one
  long task; the summary keeps the tool calls, files changed and the todo list.
- A model repeating a failing call is told to change approach, then stopped; one that
  changed code and ran nothing since is asked to verify it before finishing.
- The session log records each turn's outcome, the provider's stop reason and, on
  OpenRouter, which upstream served the request.

## Agent loop

Streaming, parallel dispatch of read-only tools, argument validation, Ctrl-C to interrupt a turn. Recovers from what small and self-hosted models do: empty replies and replies cut off at the output limit are continued, a runaway reply is stopped, and at the step limit the agent sums up instead of stopping mid-task. Requests that hit a rate limit or an overloaded server (429/5xx) are retried, honoring `Retry-After`

## Tools

`Read` `Write` `Edit` `Glob` `Grep` `Bash` `Pdf` `TodoWrite` (off unless `todo_list = true`) `WebSearch` `WebFetch` `Open` `Agent` `Team`; each one is a class plus a pydantic input model. `Read` shows the model images too, and image paths in your message (pasted screenshots) are attached to it. To look at a page it made, the model renders it with Playwright through Bash and Reads the PNG

## Permissions

Reads inside the project never ask; edits ask with a diff preview. Answers are Yes / Yes, don't ask again / No + feedback. Shift+Tab cycles modes (`default` → `acceptEdits` → `plan`); plus `auto` (LLM classifier) and `bypassPermissions`. Persistent `allow`/`deny` rules in config

## Sandbox

`Bash` runs in an OS sandbox (Seatbelt on macOS; on Linux there is no sandbox yet, so every command asks): writes only inside the project, no reading credential files, no network, and no handing work to other apps (`open`, AppleScript). Git hooks and `.tinyharness/` stay read-only, so nothing can plant code that runs later outside it (Write and Edit ask every time for those). A cloned repo's own hooks and plugins load only after you trust the folder, which is asked once, and only if it has any. Sandboxed commands run without asking in `acceptEdits`/`auto`. A command that needs the network (installing a package) can ask to run outside the sandbox, and that asks every time. `Open` shows you a document or page, and always asks

## Security

API keys never reach the model: tools refuse to read secret-shaped files (`.env`, SSH keys, …), configured key values are masked in every tool result, and a key pasted into the chat is caught before sending. Hooks get no keys in their environment, and keys are masked out of the input they're sent; session logs are private to you and masked the same way; symlinks are resolved before permission checks

## Context management

Three tiers: cap oversized messages → stub old tool output → summarize history. Old tool output is cleared only when it pays: once the cache has expired anyway, or near the window in batches of at least 20k tokens. The prompt cache isn't rebuilt every turn

## Skills

`SKILL.md` directories, loaded progressively; compatible with Claude Code skills

## Subagents and teams

Isolated context windows; built-in `explore` / `plan` / `general-purpose` / `researcher`, plus your own in Markdown, each on any model (`model: local:planner`) with its own step limit. `Team` runs a dependency graph of tasks, up to 4 teammates at once: independent tasks run in parallel, each teammate gets the results of the tasks it depends on, the user's request and its working directory, and teammates can message each other. Writing calls take turns on a shared lock; reads (including read-only shell commands) don't wait

## Hooks and plugins

Shell or Python hooks on 5 lifecycle events; plugins bundle skills, agents and hooks

## Providers

Each provider through its own current API: Anthropic Messages, OpenAI Responses, Gemini Interactions, and Chat Completions for the rest, plus any gateway. Reasoning carries across tool calls (signed/encrypted state is sent back unchanged). No per-model special cases: models, context windows and capabilities come from each provider's own model list

## Terminal UI

The input box stays up while the agent works: type to steer it (your message arrives before its next step), Ctrl+C to stop it. The model's reasoning is one line (`∴ Thought for 8s`) until `ctrl+t` or `/thinking` shows it in full. One row per tool call, colored by what it does (looking, changing, running, the web, agents), with its result on the right: `+14 −3`, `212 passed`, `exit 1` and the error under it. Subagents hang under their call. Each turn ends with a summary (time, files changed, whether the tests passed, tokens, cost), and a status bar shows the model, sandbox, git branch, context and cost. `ctrl+o` expands the last tool's output (or what running subagents are doing) above the input box and collapses it again, `/output` prints it in full; file paths are clickable; themes: `system` (the default: asks your terminal whether its background is light or dark), `light` or `dark`, set with `/theme` or `theme =` in the config; `NO_COLOR` is honored

## Two interfaces

Interactive CLI (streamed Markdown, live tool status, `/` autocomplete, pickers, mode-colored input box) and an async Python SDK, both consuming the same typed event stream

## Resume

Every conversation is saved losslessly as append-only JSONL in `~/.tinyharness/projects/<project>/`, never inside your repo (so it can't be committed) and where sandboxed commands can't write. `-c` continues the latest, `-r` picks one. On resume the harness checks what changed while you were away (files the agent read, git HEAD or branch, AGENT.md): stale file contents in the history are replaced by a stub, and the model is told what changed before your next message
