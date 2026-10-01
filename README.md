# tinyharness

A small, readable coding-agent harness in Python: the loop that turns an LLM into an
agent that reads, searches and edits your code, with permissions, skills, subagents,
hooks and context management. The agent core is about 6.5k lines (12k with the terminal UI) with no agent framework
underneath, so you can read the whole thing in an afternoon.

```text
$ tinyharness
╭──────────────────────────────────────────────────╮
│  ◆ tinyharness  v0.1.0                           │
│  a small, hackable coding agent                  │
│                                                  │
│  model    Claude Sonnet 5  via Anthropic         │
│  project  ~/code/myapp                           │
│  mode     default · asks before edits            │
╰──────────────────────────────────────────────────╯

❯ add a --verbose flag to cli.py

✓ Read    cli.py                                           112 lines
✓ Bash    uv run pytest -q                     48 passed in 1.2s · 1.4s

╭─ Edit  cli.py, -1/+3 lines ──────────────────────╮
│ - parser.add_argument("--debug")                 │
│ + parser.add_argument("--debug")                 │
│ + parser.add_argument("--verbose", "-v")         │
╰──────────────────────────────────────────────────╯
 Make this edit?
 ❯ 1. Yes
   2. Yes, allow all edits in this project for this session
   3. No, and tell it what to do instead  esc

✓ Edit    cli.py                                               +3 −1
✓ Bash    uv run pytest -q                     49 passed in 1.2s · 1.3s

Added `--verbose` (alias `-v`), with a test.

✻ 9s · 4 tools · 1 file +3 −1 · tests ✓ · 14.2k tokens · $0.03
┌──────────────────────────────────────────────────────────────────┐
│❯ Ask anything, or type / for commands                            │
└──────────────────────────────────────────────────────────────────┘
  ⏵⏵ accept edits · shift+tab    Sonnet 5 · sandbox · ⎇ main ✱ · 12% context · $0.03
```

## Features

| | |
|---|---|
| **Agent loop** | Streaming, parallel dispatch of read-only tools, argument validation, Ctrl-C to interrupt a turn. Recovers from what small and self-hosted models do: empty replies and replies cut off at the output limit are continued, a runaway reply is stopped, and at the step limit the agent sums up instead of stopping mid-task. Requests that hit a rate limit or an overloaded server (429/5xx) are retried, honoring `Retry-After` |
| **Tools** | `Read` `Write` `Edit` `Glob` `Grep` `Bash` `Pdf` `Screenshot` `TodoWrite` (off unless `todo_list = true`) `WebSearch` `WebFetch` `Open` `Agent` `Team`; each one is a class plus a pydantic input model. `Read` shows the model images too, and image paths in your message (pasted screenshots) are attached to it. `Screenshot` renders a local page in a sandboxed headless browser (Playwright's chrome-headless-shell) so the model can look at its work |
| **Permissions** | Reads inside the project never ask; edits ask with a diff preview. Answers are Yes / Yes, don't ask again / No + feedback. Shift+Tab cycles modes (`default` → `acceptEdits` → `plan`); plus `auto` (LLM classifier) and `bypassPermissions`. Persistent `allow`/`deny` rules in config |
| **Sandbox** | `Bash` runs in an OS sandbox (Seatbelt on macOS; on Linux there is no sandbox yet, so every command asks): writes only inside the project, no reading credential files, no network, and no handing work to other apps (`open`, AppleScript). Git hooks and `.tinyharness/` stay read-only, so nothing can plant code that runs later outside it (Write and Edit ask every time for those). A cloned repo's own hooks and plugins load only after you trust the folder, which is asked once, and only if it has any. Sandboxed commands run without asking in `acceptEdits`/`auto`. A command that needs the network (installing a package) can ask to run outside the sandbox, and that asks every time. `Open` shows you a document or page, and always asks |
| **Security** | API keys never reach the model: tools refuse to read secret-shaped files (`.env`, SSH keys, …), configured key values are masked in every tool result, and a key pasted into the chat is caught before sending. Hooks get no keys in their environment, and keys are masked out of the input they're sent; session logs are private to you and masked the same way; symlinks are resolved before permission checks |
| **Context management** | Three tiers: cap oversized messages → stub old tool output → summarize history. Old tool output is cleared only when it pays: once the cache has expired anyway, or near the window in batches of at least 20k tokens. The prompt cache isn't rebuilt every turn |
| **Skills** | `SKILL.md` directories, loaded progressively; compatible with Claude Code skills |
| **Subagents and teams** | Isolated context windows; built-in `explore` / `plan` / `general-purpose` / `researcher`, plus your own in Markdown, each on any model (`model: local:planner`) with its own step limit. `Team` runs a dependency graph of tasks, up to 4 teammates at once: independent tasks run in parallel, each teammate gets the results of the tasks it depends on, the user's request and its working directory, and teammates can message each other. Writing calls take turns on a shared lock; reads (including read-only shell commands) don't wait |
| **Hooks and plugins** | Shell or Python hooks on 5 lifecycle events; plugins bundle skills, agents and hooks |
| **Providers** | Each provider through its own current API: Anthropic Messages, OpenAI Responses, Gemini Interactions, and Chat Completions for the rest, plus any gateway. Reasoning carries across tool calls (signed/encrypted state is sent back unchanged). No per-model special cases: models, context windows and capabilities come from each provider's own model list |
| **Terminal UI** | The input box stays up while the agent works: type to steer it (your message arrives before its next step), Ctrl+C to stop it. The model's reasoning is one line (`∴ Thought for 8s`) until `ctrl+t` or `/thinking` shows it in full. One row per tool call, colored by what it does (looking, changing, running, the web, agents), with its result on the right: `+14 −3`, `212 passed`, `exit 1` and the error under it. Subagents hang under their call. Each turn ends with a summary (time, files changed, whether the tests passed, tokens, cost), and a status bar shows the model, sandbox, git branch, context and cost. `ctrl+o` expands the last tool's output (or what running subagents are doing) above the input box and collapses it again, `/output` prints it in full; file paths are clickable; themes: `system` (the default: asks your terminal whether its background is light or dark), `light` or `dark`, set with `/theme` or `theme =` in the config; `NO_COLOR` is honored |
| **Two interfaces** | Interactive CLI (streamed Markdown, live tool status, `/` autocomplete, pickers, mode-colored input box) and an async Python SDK, both consuming the same typed event stream |
| **Resume** | Every conversation is saved losslessly as append-only JSONL in `~/.tinyharness/projects/<project>/`, never inside your repo (so it can't be committed) and where sandboxed commands can't write. `-c` continues the latest, `-r` picks one. On resume the harness checks what changed while you were away (files the agent read, git HEAD or branch, AGENT.md): stale file contents in the history are replaced by a stub, and the model is told what changed before your next message |

## Install

Requires Python 3.11+. Install the command with [uv](https://docs.astral.sh/uv/) (or
[pipx](https://pipx.pypa.io)), which keeps it in its own environment:

```bash
uv tool install tinyharness        # or: pipx install tinyharness
```

Then run `tinyharness` from inside the project you want to work on. The first run
asks which provider to use and for its API key (paste it when asked); one key for
any supported provider is enough. `uv tool upgrade tinyharness`
updates it.

To use the SDK in your own code, add it as a dependency:

```bash
uv add tinyharness                 # or: pip install tinyharness
```

To hack on tinyharness itself:

```bash
git clone https://github.com/anujdutt9/tinyharness && cd tinyharness
uv sync
uv run tinyharness
```

Useful flags: `-c` (continue the latest conversation), `-r` (pick one to resume), `-p "prompt"` (run one prompt and exit), `--mode acceptEdits`,
`--model <name>` (or `provider:model`), `--provider openai`, `--cwd <dir>`, `--config <file>`,
`--setup` (choose a provider and add its key), `--expose-skills`, `--import-skills`, `--debug`.

In the REPL, type `/` to autocomplete commands, or `/help` to list them all. The
main ones:

| Command | |
|---|---|
| `/model` | pick a model from any provider you have a key for, or search a provider's full list |
| `/provider` | switch provider |
| `/effort` | set how hard the model reasons (`low` … `max`), the same way for every model |
| `/mode` | change the permission mode (or press Shift+Tab) |
| `/skills`, `/<skill> [args]` | list or run skills |
| `/context`, `/clear` | show context usage, or start a fresh conversation |
| `/whoami`, `/ping` | check the provider and API key |
| `/key [provider]` | add or replace a provider's API key (also offered by `/provider` and `/model` when one is missing) |

## SDK

```python
import asyncio
from tinyharness import AssistantText, Session, SessionOptions, ToolUse


async def main():
    async with Session(SessionOptions(permission_mode="acceptEdits")) as session:
        async for event in session.run("write fib.py with a fib(n) function"):
            match event:
                case AssistantText(text=text):
                    print(text, end="")
                case ToolUse(summary=summary):
                    print(f"\n● {summary}")


asyncio.run(main())
```

For headless runs, pass `permission_callback=` (an async function taking a `PermissionRequest`
and returning a `Decision`) to decide what would otherwise prompt. A request with a `warning`
(such as a command that asks to run outside the sandbox) comes to the callback every time.

Subagents can be defined in code as well as in Markdown files; a name that's already
taken replaces that agent. Their progress arrives as `SubagentUpdate` events on `on_event=`:

```python
from tinyharness import AgentSpec, SessionOptions

reviewer = AgentSpec(
    name="reviewer",
    description="Reviews a diff for bugs.",  # the model reads this to decide when to use it
    system_prompt="You are a meticulous code reviewer...",
    tools=["Read", "Grep", "Glob", "Bash"],  # leave out for every tool the session has
    permission_mode="plan",  # optional limit: never more permissive than the session
    model="anthropic:claude-haiku-4-5",  # or "small", "inherit" (the default), or a model id
    max_steps=30,  # model calls before it must report back (default 40)
)
options = SessionOptions(agents=[reviewer], on_event=print)  # tools=[...] adds your own Tool classes
```

Conversations are saved as they go and can be resumed from code too:

```python
from tinyharness import Session, SessionOptions, list_sessions

for info in list_sessions():  # newest first: id, title, updated, messages
    print(info.id, info.title)
session = Session(SessionOptions(resume="latest"))  # or resume=info.id
print(session.resume_report.summary())  # e.g. "12 messages · 1 file changed since"
```

`SessionOptions(store=...)` takes any `SessionStore` (three methods: `writer`, `load`,
`sessions`) to keep conversations somewhere other than files in the project.

A subagent's `model` is fixed by whoever defines the agent, and can come from any
provider you have a key for: a GPT session can hand review to a Claude agent.
Tools passed in `tools=` are available to subagents too.
See [`examples/`](https://github.com/anujdutt9/tinyharness/tree/main/examples) for a one-shot query, a multi-turn session, resuming a conversation, a custom tool,
a permission policy with hooks, skills, and subagents on different models.

## How it works

```
 CLI (cli/) ──┐
              ├──► Session (session.py) ──► agent loop (agent/loop.py) ──► LLM (llm/client.py ─ llm/adapters/)
 SDK ─────────┘        wires components          │                           retries, model lists
                                                 ├─ compaction (agent/compaction.py)
                                                 ├─ runaway / empty-reply recovery (agent/runaway.py)
                                                 ├─ permission gate (permissions/gate.py) ── hooks
                                                 └─ tools (tools/*.py) ── workspace, sandbox
                                                      │
                                                      └─ Agent / Team ──► subagents (subagents/runner.py)
                                                                          the same loop, fresh history,
                                                                          any provider:model
```

Suggested reading order:

| # | File | What you'll learn |
|---|---|---|
| 1 | [`agent/loop.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/agent/loop.py) | The core loop: stream the model, validate calls, gate them, run them, repeat |
| 2 | [`tools/base.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/tools/base.py), [`tools/read.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/tools/read.py) | How tools are defined and exposed to the model |
| 3 | [`permissions/gate.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/permissions/gate.py) | The layered permission check |
| 4 | [`agent/compaction.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/agent/compaction.py) | Staying inside the context window without breaking the prompt cache |
| 5 | [`session.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/session.py) | Wiring it all together; state that persists between prompts |
| 6 | [`llm/adapters/`](https://github.com/anujdutt9/tinyharness/tree/main/tinyharness/llm/adapters) | One small adapter per provider API; the rest of the code never sees a provider's format |
| 7 | [`subagents/runner.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/subagents/runner.py) | Subagents are the same loop with a fresh history |
| 8 | [`skills/skill.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/skills/skill.py), [`hooks/hooks.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/hooks/hooks.py) | Extensibility |
| 9 | [`cli/`](https://github.com/anujdutt9/tinyharness/tree/main/tinyharness/cli) | Rendering events (`render.py`), the REPL, slash commands; colors live in `theme.py` |

All prompts sent to the model (the system prompt, the compaction summary, the safety classifier and the built-in subagents) are Markdown files in [`tinyharness/prompts/`](https://github.com/anujdutt9/tinyharness/tree/main/tinyharness/prompts). Tool descriptions stay in each tool's class, next to the input schema they describe.

## Configuration

tinyharness needs an API key for at least one provider. The first run asks for one
(with a link to where to get it), checks it with the provider and keeps it in
`~/.tinyharness/auth.json`, one entry per provider, readable only by you and never by
the agent's tools. Switching to a provider with no key (`/provider`, `/model`) asks
for its key the same way, and `/key` adds or replaces one any time. A key set in your
shell (`export ANTHROPIC_API_KEY=...`) takes priority. After that, everything is
optional: `/model` switches models (and remembers your choice), and one file,
`~/.tinyharness/config.toml`, holds any defaults you want.

Everything tinyharness keeps for you is in one folder, `~/.tinyharness/`:
`config.toml`, `auth.json`, your `skills/`, `agents/`, `plugins/` and `hooks.toml`, and
its own state and session logs. (Earlier versions used `~/.config/tinyharness/`; its
files are moved on the first run.) See
[`examples/config.toml`](https://github.com/anujdutt9/tinyharness/blob/main/examples/config.toml). API keys never go in the config file.

**Providers.** Each one is used through its own current API:

| API format | Providers |
|---|---|
| Anthropic Messages | Anthropic |
| OpenAI Responses | OpenAI, xAI |
| Gemini Interactions | Google Gemini |
| OpenAI Chat Completions | OpenRouter, DeepSeek, Mistral, Groq, and custom gateways |

[`llm/providers.toml`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/llm/providers.toml) holds only how to connect to
each one. Models are listed live from the provider (cached for a day), and whatever
the provider reports about a model (context window, output limit, reasoning support)
is used as-is; nothing is special-cased by model name. When a provider doesn't say
(OpenAI doesn't publish context windows), a default applies, and if a request is
ever rejected as too long, the history is summarized and the request retried.

A gateway is one table in the config file:

```toml
model = "gateway:claude-sonnet-5"

[providers.gateway]
base_url = "https://llm.internal.example.com/v1"
api_key_env = "GATEWAY_API_KEY"
api = "chat"   # or "responses", "anthropic", "gemini"
```

**Self-hosted models (vLLM, Ollama, llama.cpp, SGLang).** Any OpenAI-compatible
server is a provider; leave out `api_key_env` when it needs no key. tinyharness reads the
served models and their context windows from `/v1/models` (vLLM's `max_model_len`), so
compaction triggers at the server's real limit. Agents can run on different servers:

```toml
model = "local:planner"

[providers.local]
label = "vLLM on my GPU box"
base_url = "http://127.0.0.1:8000/v1"
# api_key_env = "VLLM_API_KEY"         # only if the server was started with --api-key
extra_body = { temperature = 0.2 }      # merged into every request
```

For tool use, start vLLM with `--enable-auto-tool-choice --tool-call-parser <parser>`
(e.g. `hermes` for Qwen2.5/Qwen3 instruct models, `qwen3_coder` for Qwen3-Coder). An agent
file can then pick a model with `model: local:worker`.

| Location | Contents |
|---|---|
| `~/.tinyharness/` | your skills, agents, plugins, `hooks.toml`, `AGENT.md`; `projects/<project>/` holds each project's conversation logs and scratch files |
| `<project>/.tinyharness/` | project skills, agents, plugins and hooks: only what you put there, to commit and share. tinyharness never writes into your project |
| `<project>/AGENT.md` (or `CLAUDE.md`) | project instructions added to the system prompt |

## Extending

- [docs/skills.md](https://github.com/anujdutt9/tinyharness/blob/main/docs/skills.md): writing and sharing skills
- [docs/extending.md](https://github.com/anujdutt9/tinyharness/blob/main/docs/extending.md): custom tools, hooks, subagents and plugins

## Development

```bash
uv sync                   # installs dev tools too
uv run pytest             # tests use a scripted fake LLM; no API key needed
uv run pytest --terminal  # also the screen-layout tests, in an emulated terminal (slower)
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv build && uv run python scripts/check_dist.py   # build and smoke-test the package
```

Releasing: bump `__version__` in `tinyharness/__init__.py`, add a
[CHANGELOG](https://github.com/anujdutt9/tinyharness/blob/main/CHANGELOG.md) entry, then push a
matching tag (`git tag v0.2.0 && git push --tags`). The release workflow tests, builds
and publishes to PyPI.

## License

MIT
