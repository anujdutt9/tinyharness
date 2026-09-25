# tinyharness

A small, readable coding-agent harness in Python: the loop that turns an LLM into an
agent that reads, searches and edits your code, with permissions, skills, subagents,
hooks and context management. It's about 3.8k lines with no agent framework
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

● Read  cli.py
  ⎿ Read 112 lines

╭─ Edit  cli.py, -1/+3 lines ──────────────────────╮
│ - parser.add_argument("--debug")                 │
│ + parser.add_argument("--debug")                 │
│ + parser.add_argument("--verbose", "-v")         │
╰──────────────────────────────────────────────────╯
 Make this edit?
 ❯ 1. Yes
   2. Yes, allow all edits in this project for this session
   3. No, and tell it what to do instead  esc

● Edit  cli.py, -1/+3 lines
  ⎿ edited cli.py (1 replacement)

Added `--verbose` (alias `-v`).

✻ Worked for 9s · 2 tool calls
┌──────────────────────────────────────────────────┐
│❯ Ask anything, or type / for commands            │
└──────────────────────────────────────────────────┘
  ⏵⏵ accept edits · shift+tab to switch     12% context
```

## Features

| | |
|---|---|
| **Agent loop** | Streaming, parallel dispatch of read-only tools, argument validation, step limit, Ctrl-C to interrupt a turn |
| **Tools** | `Read` `Write` `Edit` `Glob` `Grep` `Pdf` `TodoWrite` `Agent`; each one is a class plus a pydantic input model |
| **Permissions** | Reads inside the project never ask; edits ask with a diff preview. Answers are Yes / Yes, don't ask again / No + feedback. Shift+Tab cycles modes (`default` → `acceptEdits` → `plan`); plus `auto` (LLM classifier) and `bypassPermissions`. Persistent `allow`/`deny` rules in config |
| **Security** | API keys never reach the model: tools refuse to read secret-shaped files (`.env`, SSH keys, …), configured key values are masked in every tool result, and a key pasted into the chat is caught before sending. Hook processes never see keys; symlinks are resolved before permission checks |
| **Context management** | Three tiers: cap oversized messages → stub old tool output → summarize history. Each message changes at most once, so provider prompt caches stay valid |
| **Skills** | `SKILL.md` directories, loaded progressively; compatible with Claude Code skills |
| **Subagents** | Isolated context windows; built-in `explore` / `plan` / `general-purpose`, plus your own in Markdown |
| **Hooks and plugins** | Shell or Python hooks on 5 lifecycle events; plugins bundle skills, agents and hooks |
| **Providers** | Each provider through its own current API: Anthropic Messages, OpenAI Responses, Gemini Interactions, and Chat Completions for the rest, plus any gateway. Reasoning carries across tool calls (signed/encrypted state is sent back unchanged). No per-model special cases: models, context windows and capabilities come from each provider's own model list |
| **Two interfaces** | Interactive CLI (streamed Markdown, live tool status, `/` autocomplete, pickers, mode-colored input box) and an async Python SDK, both consuming the same typed event stream |
| **Transcripts** | Every session is logged as append-only JSONL in `.tinyharness/sessions/` |

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/<you>/tinyharness && cd tinyharness
uv sync
cp .env.example .env        # add at least one API key
uv run tinyharness          # run it from inside the project you want to work on
```

Useful flags: `-p "prompt"` (run one prompt and exit), `--mode acceptEdits`,
`--model <name>` (or `provider:model`), `--provider openai`, `--cwd <dir>`, `--debug`.

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

## SDK

```python
import asyncio
from tinyharness import AssistantText, Session, SessionOptions, ToolUse


async def main():
    session = Session(SessionOptions(permission_mode="acceptEdits"))
    async for event in session.run("write fib.py with a fib(n) function"):
        match event:
            case AssistantText(text=text):
                print(text, end="")
            case ToolUse(summary=summary):
                print(f"\n● {summary}")
    session.close()


asyncio.run(main())
```

For headless runs, pass `permission_callback=` (an async function taking a `PermissionRequest`
and returning a `Decision`) to decide what would otherwise prompt.
See [`examples/`](examples) for a one-shot query, a multi-turn session, a custom tool,
a permission policy with hooks, and skills.

## How it works

```
 CLI (cli/) ──┐
              ├──► Session (session.py) ──► agent loop (agent/loop.py) ──► LLM (llm/client.py ─ llm/adapters/)
 SDK ─────────┘        wires components          │
                                                 ├─ compaction (agent/compaction.py)
                                                 ├─ permission gate (permissions/gate.py) ── hooks
                                                 └─ tools (tools/*.py) ── workspace (workspace/)
```

Suggested reading order:

| # | File | What you'll learn |
|---|---|---|
| 1 | [`agent/loop.py`](tinyharness/agent/loop.py) | The core loop: stream the model, validate calls, gate them, run them, repeat |
| 2 | [`tools/base.py`](tinyharness/tools/base.py), [`tools/read.py`](tinyharness/tools/read.py) | How tools are defined and exposed to the model |
| 3 | [`permissions/gate.py`](tinyharness/permissions/gate.py) | The layered permission check |
| 4 | [`agent/compaction.py`](tinyharness/agent/compaction.py) | Staying inside the context window without breaking the prompt cache |
| 5 | [`session.py`](tinyharness/session.py) | Wiring it all together; state that persists between prompts |
| 6 | [`llm/adapters/`](tinyharness/llm/adapters) | One small adapter per provider API; the rest of the code never sees a provider's format |
| 7 | [`subagents/runner.py`](tinyharness/subagents/runner.py) | Subagents are the same loop with a fresh history |
| 8 | [`skills/skill.py`](tinyharness/skills/skill.py), [`hooks/hooks.py`](tinyharness/hooks/hooks.py) | Extensibility |
| 9 | [`cli/`](tinyharness/cli) | Rendering events (`render.py`), the REPL, slash commands; colors live in `theme.py` |

All prompts sent to the model (the system prompt, the compaction summary, the safety classifier and the built-in subagents) are Markdown files in [`tinyharness/prompts/`](tinyharness/prompts). Tool descriptions stay in each tool's class, next to the input schema they describe.

## Configuration

tinyharness needs an API key for at least one provider. Run it once with no key and it
shows where to get one and creates `~/.config/tinyharness/.env` with a commented-out
line per provider: uncomment one and add your key (or `export` it in your shell).
Keys added while it's running are picked up by `/provider` and `/model`. After that,
everything is optional: `/model` switches models (and remembers your choice), and
one file, `~/.config/tinyharness/config.toml`, holds any defaults you want. See
[`examples/config.toml`](examples/config.toml). API keys never go in the config file.

**Providers.** Each one is used through its own current API:

| API format | Providers |
|---|---|
| Anthropic Messages | Anthropic |
| OpenAI Responses | OpenAI, xAI |
| Gemini Interactions | Google Gemini |
| OpenAI Chat Completions | OpenRouter, DeepSeek, Mistral, Groq, and custom gateways |

[`llm/providers.toml`](tinyharness/llm/providers.toml) holds only how to connect to
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

| Location | Contents |
|---|---|
| `~/.tinyharness/` | your skills, agents, plugins, `hooks.toml`, `AGENT.md` |
| `<project>/.tinyharness/` | project skills, agents, plugins and hooks, plus session transcripts and scratch files |
| `<project>/AGENT.md` (or `CLAUDE.md`) | project instructions added to the system prompt |

## Extending

- [docs/skills.md](docs/skills.md): writing and sharing skills
- [docs/extending.md](docs/extending.md): custom tools, hooks, subagents and plugins

## Development

```bash
uv sync                   # installs dev tools too
uv run pytest             # tests use a scripted fake LLM; no API key needed
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

## License

MIT
