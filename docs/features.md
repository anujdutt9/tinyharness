# Features

Everything tinyharness does, in more detail than the README. For how the pieces fit together, see [How it works](../README.md#how-it-works).

## Reliability

Small and self-hosted models fail in ways the big ones rarely do, and providers fail in
ways no model can fix. tinyharness handles both in the harness, not with model-specific
special cases (see [AGENTS.md](../AGENTS.md#guarantees-dont-break-these) for the
guarantees and why each exists):

- How a model calls tools is measured, not guessed from its name: the first time a model
  is used, one small request offers it a tool. Models that call it through the API keep
  doing so; models that write the call in their reply get the tools in their prompt and
  write `<call>{...}</call>` (a tag no server-side parser strips). Cached per model in
  `~/.tinyharness/cache/tool-calls.json`; `tool_calls = "native" | "text"` overrides it.
- Every reply has a clear outcome, normalized from each API's own stop reasons. A reply
  cut off at the output limit is continued; a tool call cut off mid-arguments isn't run,
  and the model is told to write in smaller pieces.
- The history sent to a provider is always well formed: every tool call answered,
  empty messages dropped, other providers' tool-call ids replaced after `/model`.
- Tool calls written as text (Hermes JSON, Qwen3-Coder XML) are recovered when a
  server's tool-call parser fails on them; malformed arguments are repaired.
- Compaction starts from the provider's own token count and can summarize inside one
  long task; the summary keeps the tool calls, files changed and the todo list.
- A model repeating a failing call is told to change approach.
- The session log records each turn's outcome, the provider's stop reason and, on
  OpenRouter, which upstream served the request.

## Agent loop

Streaming, parallel dispatch of read-only tools, argument validation, Ctrl-C to interrupt a turn. Recovers from what small and self-hosted models do: empty replies and replies cut off at the output limit are continued, a runaway reply is stopped, and at the step limit the agent sums up instead of stopping mid-task. Requests that hit a rate limit or an overloaded server (429/5xx) are retried, honoring `Retry-After`

## Tools

Core: `Read` `Write` `Edit` `Glob` `Grep` `Bash`; each one is a class plus a pydantic input model. The rest are built-in extensions, off until you turn them on (`[extensions] enabled = [...]` in config, or `/extensions on <name>`): `web` (`WebSearch`, `WebFetch`), `pdf` (`Pdf`), `open` (`Open`), `todo` (`TodoWrite`; `todo_list = true` also turns it on), `agents` (`Agent`, `Team`), `memory` (`Memory`: notes about you and the project that every session starts with; `/memory` shows them). Your own tools are extensions too: [extension-api.md](extension-api.md). `Read` shows the model images too, and image paths in your message (pasted screenshots) are attached to it. To look at a page it made, the model renders it with Playwright through Bash and Reads the PNG

## Permissions

In the `default` mode, routine work doesn't ask: reads, edits inside the project and sandboxed commands run. Anything that leaves that boundary (a file in another folder, a command with `sandbox=false`, a network tool) asks, with a diff preview for edits. Answers are Yes / Yes, don't ask again / No + feedback. Shift+Tab cycles modes (`default` → `auto` → `plan` → `ask`): `auto` lets an LLM classifier decide the unclear cases, `plan` is read-only, `ask` asks before every edit and command; `bypassPermissions` never asks. (`acceptEdits`, the earlier name, means `default`.) Persistent `allow`/`deny` rules in config

**Outside content.** Prompt injection needs three things together: private data, content
someone else wrote, and a way to send data out ("the lethal trifecta"; Meta's "Rule of
Two"). tinyharness tracks the second: once a fetched page, search results or a command's
download is in the conversation, any call that could send data to another machine needs a
person's yes, whatever the mode. Network tools lose the automatic public-page allowance,
and leaving the sandbox says why it's asking. Sites you approved (config rules, "don't ask
again", an extension's approved needs) stay allowed, sandboxed commands are unaffected
(they reach only the allowlist and can't read credentials), and `/clear` starts over.
`/sandbox` shows where the outside content came from.

## Sandbox

`Bash` runs in an OS sandbox (Seatbelt on macOS; on Linux there is no sandbox yet, so every command asks): writes only inside the project, no reading credential files, network only to an allowlist of hosts (package registries, GitHub, CDNs) through a local proxy, and no handing work to other apps (`open`, AppleScript). Git hooks and `.tinyharness/` stay read-only, so nothing can plant code that runs later outside it (Write and Edit ask every time for those). A cloned repo's own hooks and plugins load only after you trust the folder, which is asked once, and only if it has any. Sandboxed commands run without asking in `default` and `auto`. A command that needs another host, or to write elsewhere, can ask to run outside the sandbox, and that asks every time. `/sandbox` shows the limits; `[sandbox] allow_hosts` changes the hosts. `Open` shows you a document or page, and always asks

## Security

API keys never reach the model: tools refuse to read secret-shaped files (`.env`, SSH keys, …), configured key values are masked in every tool result, and a key pasted into the chat is caught before sending. Hooks get no keys in their environment, and keys are masked out of the input they're sent; session logs are private to you and masked the same way; symlinks are resolved before permission checks

## Context management

Three tiers: cap oversized messages → stub old tool output → summarize history. Old tool output is cleared only when it pays: once the cache has expired anyway, or near the window in batches of at least 20k tokens. The prompt cache isn't rebuilt every turn

## Skills

`SKILL.md` directories, loaded progressively; compatible with Claude Code skills

## Subagents and teams

Isolated context windows; built-in `explore` / `plan` / `general-purpose` / `researcher`, plus your own in Markdown, each on any model (`model: local:planner`) with its own step limit. `Team` runs a dependency graph of tasks whose size the model chooses (3-5 teammates suit most goals). Up to `max_parallel_agents` subagents (default 20, or `TINYHARNESS_MAX_PARALLEL_AGENTS`) run at once across the session, Agent calls and teammates together; past that an Agent call fails so the model can wait, and Team tasks wait their turn: independent tasks run in parallel, each teammate gets the results of the tasks it depends on, the user's request and its working directory, and teammates can message each other. Writing calls take turns on a shared lock; reads (including read-only shell commands) don't wait

## Hooks and plugins

Shell or Python hooks on 5 lifecycle events; plugins bundle skills, agents and hooks

## Providers

Each provider through its own current API: Anthropic Messages, OpenAI Responses, Gemini Interactions, and Chat Completions for the rest, plus any gateway. Reasoning carries across tool calls (signed/encrypted state is sent back unchanged). No per-model special cases: models, context windows and capabilities come from each provider's own model list

## Terminal UI

The input box stays up while the agent works: type to steer it (your message arrives before its next step), Ctrl+C to stop it. `/btw <question>` asks a side question at any time, even mid-turn: it's answered from the conversation so far, without tools, in one call that is mostly read from the cache, and neither question nor answer is added to the conversation. The answer shows above the input box until Esc (or your next message) closes it, leaving nothing in the scrollback; `/output` prints it in full. When a turn ends, the empty input box suggests your likely next message (Tab takes it to edit or send); `prompt_suggestions = false` turns that off. The model's reasoning is one line (`∴ Thought for 8s`) until `ctrl+t` or `/thinking` shows it in full. One row per tool call, colored by what it does (looking, changing, running, the web, agents), with its result on the right: `+14 −3`, `212 passed`, `exit 1` and the error under it. Subagents hang under their call. Each turn ends with a summary (time, files changed, whether the tests passed, tokens, cost), and a status bar shows the model, sandbox, git branch, context and cost. `ctrl+o` expands the last tool's output (or what running subagents are doing) above the input box and collapses it again, `/output` prints it in full; file paths are clickable; themes: `system` (the default: asks your terminal whether its background is light or dark), `light` or `dark`, set with `/theme` or `theme =` in the config; `NO_COLOR` is honored

## Two interfaces

Interactive CLI (streamed Markdown, live tool status, `/` autocomplete, pickers, mode-colored input box) and an async Python SDK, both consuming the same typed event stream

## Resume

Every conversation is saved losslessly as append-only JSONL in `~/.tinyharness/projects/<project>/`, never inside your repo (so it can't be committed) and where sandboxed commands can't write. `-c` continues the latest, `-r` picks one. On resume the harness checks what changed while you were away (files the agent read, git HEAD or branch, AGENT.md): stale file contents in the history are replaced by a stub, and the model is told what changed before your next message
