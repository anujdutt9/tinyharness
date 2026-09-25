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
| **Security** | Tools refuse to read secret-shaped files (`.env`, SSH keys, …), hook processes never see API keys, symlinks are resolved before permission checks |
| **Context management** | Three tiers: cap oversized messages → stub old tool output → summarize history. Each message changes at most once, so provider prompt caches stay valid |
| **Skills** | `SKILL.md` directories, loaded progressively; compatible with Claude Code skills |
| **Subagents** | Isolated context windows; built-in `explore` / `plan` / `general-purpose`, plus your own in Markdown |
| **Hooks and plugins** | Shell or Python hooks on 5 lifecycle events; plugins bundle skills, agents and hooks |
| **Providers** | 8 built-in providers plus any OpenAI-compatible gateway; `/model` lists every model you have a key for and can search a provider's full catalog; first-run setup asks for a key |
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
| `/provider`, `/login` | switch provider, or add an API key |
| `/think` | set the reasoning budget |
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
              ├──► Session (session.py) ──► agent loop (agent/loop.py) ──► LLM (llm/client.py)
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
| 6 | [`subagents/runner.py`](tinyharness/subagents/runner.py) | Subagents are the same loop with a fresh history |
| 7 | [`skills/skill.py`](tinyharness/skills/skill.py), [`hooks/hooks.py`](tinyharness/hooks/hooks.py) | Extensibility |
| 8 | [`cli/`](tinyharness/cli) | Rendering events (`render.py`), the REPL, slash commands; colors live in `theme.py` |

All prompts sent to the model (the system prompt, the compaction summary, the safety classifier and the built-in subagents) are Markdown files in [`tinyharness/prompts/`](tinyharness/prompts). Tool descriptions stay in each tool's class, next to the input schema they describe.

## Configuration

The first time you run `tinyharness` with no API key, it asks you to pick a provider
and paste a key, and saves it to `~/.config/tinyharness/.env`. After that,
everything is optional: `/model` switches models (and remembers your choice), and
one file, `~/.config/tinyharness/config.toml`, holds any defaults you want. See
[`examples/config.toml`](examples/config.toml). API keys never go in the config file.

**Providers.** Built in: Anthropic, OpenAI, OpenRouter, Google Gemini, xAI, DeepSeek,
Mistral and Groq, each with a short list of recommended models
([`llm/providers.toml`](tinyharness/llm/providers.toml)). Any OpenAI-compatible
gateway can be added in the config file:

```toml
model = "gateway:claude-sonnet-5"

[providers.gateway]
base_url = "https://llm.internal.example.com/v1"
api_key_env = "GATEWAY_API_KEY"
models = ["claude-sonnet-5", "gpt-6-sol"]
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
