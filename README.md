<p align="center">
  <img src="docs/assets/banner.png" alt="tinyharness: an agent harness you can read in an afternoon. Scout, the tinyharness pony, wears a yellow work harness with a notebook pouch." width="880">
</p>

<p align="center">
  <a href="https://pypi.org/project/tinyharness-lm/"><img src="https://img.shields.io/pypi/v/tinyharness-lm?color=1f5f99" alt="PyPI"></a>
  <a href="https://pypi.org/project/tinyharness-lm/"><img src="https://img.shields.io/pypi/pyversions/tinyharness-lm?color=1f5f99" alt="Python versions"></a>
  <a href="https://github.com/anujdutt9/tinyharness/actions/workflows/ci.yml"><img src="https://github.com/anujdutt9/tinyharness/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/anujdutt9/tinyharness/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-1f5f99" alt="Apache 2.0 license"></a>
</p>

**tinyharness** is the loop that turns an LLM into an agent: it reads, searches and edits
your code, runs commands in a sandbox, asks before it changes things, and keeps long
sessions inside the context window. It works as a coding agent in your terminal and as
a Python SDK for building your own. The agent core is about 9k lines of code with no
framework underneath, so you can read the whole thing in an afternoon.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/session-dark.svg">
  <img src="docs/assets/session-light.svg" alt="A tinyharness session: Scout greets you beside the model, project and sandbox; the agent runs the failing tests, reads invoice.py, fixes the discount bug with a one-line edit, and the tests pass.">
</picture>

## Why tinyharness

- **Readable.** About 9k lines of code for the agent core (12k with the terminal UI),
  commented for people learning how agents work. Start with [`agent/loop.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/agent/loop.py).
- **Any model, even yours.** Anthropic, OpenAI API, ChatGPT plans, Gemini, xAI, OpenRouter, GitHub Copilot,
  DeepSeek, Mistral and Groq, plus LiteLLM and any OpenAI-compatible server: vLLM,
  Ollama, llama.cpp, SGLang.
- **Robust with small models.** Cut-off replies, empty replies, tool calls written as
  text, malformed arguments and repeated failures are handled by the harness, so mid-size
  open models get work done too. [How](https://github.com/anujdutt9/tinyharness/blob/main/docs/features.md#reliability).
- **Safe without the friction.** Commands run with filesystem and network isolation;
  routine contained work doesn't ask, while a newly required exact host can be approved
  without dropping containment. Downloading new executable code asks once; cached tools
  then run normally, and bare Python installs use a project `.venv`. API keys never reach
  the model. Once outside content is
  in the conversation, anything that could send data out asks a person first.
- **Small and fast.** Seven core tools and a short prompt (about 3k tokens before your
  instructions) that stays the same from call to call, so the provider's cache hits.
- **Extensible.** One Python file with `setup(th)` adds tools, prompt sections, commands
  and hooks. Web search, PDFs, subagents and parallel teams are built-in extensions you
  turn on.

## Quickstart

Requires Python 3.11+. Install the command with [uv](https://docs.astral.sh/uv/) (or
[pipx](https://pipx.pypa.io)):

```bash
uv tool install tinyharness-lm     # or: pipx install tinyharness-lm
cd your-project && tinyharness
```

The first run asks where your models come from and offers the connection methods that
provider supports. Eligible Plus and Pro users can choose ChatGPT plan and Continue with
ChatGPT without creating an API key; OpenAI API remains a separate choice. OpenRouter
connects in the browser, and GitHub Copilot uses GitHub's device sign-in with the models
included in that subscription. Ollama, LM Studio, llama.cpp,
vLLM and SGLang are discovered on their
standard local endpoints. If one is remote, uses another port or requires a server key,
setup asks for those details and remembers them. LiteLLM setup similarly accepts a local
or company-hosted proxy URL and optional gateway key. The same guided screen can replace
an endpoint or its key later with `/connect`; editing a configuration file is not required.
`uv tool upgrade tinyharness-lm` updates it.

In the session, type `/` to autocomplete commands. The ones you'll use most:

| Command | |
|---|---|
| `/provider` | switch the active model provider |
| `/model` | pick from the active provider's models |
| `/connect [provider]` | connect a provider in the browser or with an API key |
| `/mode` | change the permission mode (or press Shift+Tab) |
| `/effort` | how hard the model reasons, `low` … `max`, the same for every model |
| `/skills`, `/<skill>` | find or run skills |
| `/extensions` | what's on and what it costs per request; `/extensions on web` |
| `/sandbox` | what commands can write, read and reach |
| `/btw <question>` | a quick side question, even while the agent works; not added to the conversation |
| `/sessions`, `/context`, `/clear` | switch conversations, inspect context, or start fresh |
| `/key [provider]` | manually add or replace an API key saved in `auth.json` |

Useful flags: `-c` (continue the latest conversation), `-r` (pick one), `-p "prompt"`
(run one prompt and exit), `--model provider:model`, `--mode ask`, `--cwd <dir>`.

## Use it from Python

```bash
uv add tinyharness-lm              # or: pip install tinyharness-lm
```

```python
import asyncio
from tinyharness import AssistantText, Session, SessionOptions, ToolUse


async def main():
    async with Session(SessionOptions()) as session:
        async for event in session.run("write fib.py with a fib(n) function"):
            match event:
                case AssistantText(text=text):
                    print(text, end="")
                case ToolUse(summary=summary):
                    print(f"\n● {summary}")


asyncio.run(main())
```

The CLI and the SDK consume the same typed event stream. For headless runs, pass
`permission_callback=` to decide what would otherwise prompt. Extensions can be passed in
code (`SessionOptions(extensions=[...])`) and tested against a scripted model
(`tinyharness.testing`). Subagents can be defined in
code (`AgentSpec`) or Markdown, each on any provider's model, and conversations can be
resumed from code (`SessionOptions(resume="latest")`). Post-turn evaluators can be passed
with `SessionOptions(evaluators=[...], evaluator_timeout=30.0)`; their bounded results are
appended to the versioned trace without changing the completed conversation. See [`examples/`](https://github.com/anujdutt9/tinyharness/tree/main/examples) for a
one-shot query, a multi-turn session, resuming, a custom tool, a permission policy with
hooks, skills, subagents on different models, an extension, and a post-turn evaluator.

## What's inside

| | |
|---|---|
| **Agent loop** | Streaming, parallel read-only tools, argument validation and repair, recovery from cut-off and empty replies, retries with backoff |
| **Tools** | Core: `Read` `Write` `Edit` `Glob` `Grep` `Bash` `Open`. Built-in extensions you turn on: `web` (WebSearch, WebFetch), `pdf`, `todo`, `agents` (Agent, Team) |
| **Extensions** | One Python file with `setup(th)` adds tools, prompt sections, slash commands and event handlers ([docs/extension-api.md](docs/extension-api.md)); `/extensions on|off` |
| **Permissions** | Routine work in the project and the sandbox doesn't ask; anything beyond it asks, with a diff for edits. Modes `default`, `auto` (an LLM classifier for unclear cases), `plan`, `ask` (every edit and command) and `bypassPermissions`; persistent allow/deny rules |
| **Sandbox** | Native Seatbelt on macOS; optional Sandbox Runtime on Linux, Windows and as a fallback. Commands have narrow writes, blocked credential reads (SRT covers the home and project neighborhoods) and allowlisted network access; `/sandbox` shows it |
| **Context** | Cap oversized messages, stub old tool output, then summarize, without rebuilding the prompt cache every turn |
| **Subagents and teams** | Isolated contexts on any model; `Team` runs a dependency graph of tasks; the model picks the team's size, up to 20 subagents at once (`max_parallel_agents`) |
| **Skills, hooks, plugins** | Portable `SKILL.md` skills, shell or Python hooks on 5 events, and plugins that bundle them |
| **Resume & traces** | Lossless, schema-versioned JSONL logs outside your repo with correlated main-loop model/tool decisions and latency; on resume, the agent is told what changed while you were away |

Each of these in detail: [docs/features.md](https://github.com/anujdutt9/tinyharness/blob/main/docs/features.md).

## How it works

```
 CLI (cli/) ──┐
              ├──► Session (session.py) ──► agent loop (agent/loop.py) ──► LLM (llm/client.py ─ llm/adapters/)
 SDK ─────────┘        wires components          │                           retries, model lists
                                                 ├─ compaction (agent/compaction.py)
                                                 ├─ runaway / empty-reply recovery (agent/runaway.py)
                                                 ├─ permission gate (permissions/gate.py) ── hooks
                                                 └─ tools (tools/*.py) ── workspace, sandbox + allowlist proxy
                                                      │
                                                      └─ extensions (ext.py, extensions/) add tools,
                                                         prompt sections, commands and hooks; the agents
                                                         extension's Agent / Team ──► subagents
                                                         (subagents/runner.py): the same loop, fresh
                                                         history, any provider:model
```

Suggested reading order:

| # | File | What you'll learn |
|---|---|---|
| 1 | [`agent/loop.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/agent/loop.py) | The core loop: stream the model, validate calls, gate them, run them, repeat |
| 2 | [`tools/base.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/tools/base.py), [`tools/read.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/tools/read.py) | How tools are defined and exposed to the model |
| 3 | [`permissions/gate.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/permissions/gate.py) | The layered permission check |
| 4 | [`agent/compaction.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/agent/compaction.py) | Staying inside the context window without breaking the prompt cache |
| 5 | [`agent/history.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/agent/history.py) | The conversation, and the normalization every request goes through |
| 6 | [`session.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/session.py) | Wiring it all together; state that persists between prompts |
| 7 | [`llm/adapters/`](https://github.com/anujdutt9/tinyharness/tree/main/tinyharness/llm/adapters) | One small adapter per provider API; the rest of the code never sees a provider's format |
| 8 | [`ext.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/ext.py), [`extensions/`](https://github.com/anujdutt9/tinyharness/tree/main/tinyharness/extensions) | How a feature plugs in: the built-in extensions use the same API as yours |
| 9 | [`subagents/runner.py`](https://github.com/anujdutt9/tinyharness/blob/main/tinyharness/subagents/runner.py) | Subagents are the same loop with a fresh history |
| 10 | [`cli/`](https://github.com/anujdutt9/tinyharness/tree/main/tinyharness/cli) | Rendering events (`render.py`), the REPL, slash commands; colors in `theme.py`, Scout in `scout.py` |

Every prompt sent to the model is a Markdown file in [`tinyharness/prompts/`](https://github.com/anujdutt9/tinyharness/tree/main/tinyharness/prompts).

## Docs

- [docs/features.md](https://github.com/anujdutt9/tinyharness/blob/main/docs/features.md): everything above, in detail
- [docs/configuration.md](https://github.com/anujdutt9/tinyharness/blob/main/docs/configuration.md): provider connections, gateways, self-hosted models, where files live
- [docs/extending.md](https://github.com/anujdutt9/tinyharness/blob/main/docs/extending.md): extensions, custom tools, hooks, subagents and plugins
- [docs/extension-api.md](https://github.com/anujdutt9/tinyharness/blob/main/docs/extension-api.md): the extension API in full
- [docs/skills.md](https://github.com/anujdutt9/tinyharness/blob/main/docs/skills.md): writing and sharing skills
- [CHANGELOG.md](https://github.com/anujdutt9/tinyharness/blob/main/CHANGELOG.md) and [AGENTS.md](https://github.com/anujdutt9/tinyharness/blob/main/AGENTS.md) (how to work on tinyharness, for people and agents)

## Development

```bash
git clone https://github.com/anujdutt9/tinyharness && cd tinyharness
uv sync                   # installs dev tools too
uv run tinyharness
uv run pytest             # tests use a scripted fake LLM; no API key needed
uv run pytest --terminal  # also the screen-layout tests, in an emulated terminal (slower)
uv run ruff check . && uv run ruff format --check . && uv run mypy
uv build && uv run python scripts/check_dist.py   # build and smoke-test the package
```

Releasing: bump `__version__` in `tinyharness/version.py`, add a
[CHANGELOG](https://github.com/anujdutt9/tinyharness/blob/main/CHANGELOG.md) entry, then push a matching tag (`git tag v0.2.0 && git push --tags`).
The release workflow tests, builds and publishes to PyPI.

## License

[Apache License 2.0](LICENSE).
