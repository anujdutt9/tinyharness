# Extending tinyharness

The main way to add anything is an **extension**: one Python file with a `setup(th)`
function that adds tools, prompt sections, slash commands and event handlers. The
built-in features (web, pdf, todo, agents) are extensions too. The full contract is
in [extension-api.md](extension-api.md); this page shows the common cases.

Besides extensions there are hooks, subagents, plugins and skills (see
[skills.md](skills.md)). Your own live in `~/.tinyharness/`; a project's live in
`<project>/.tinyharness/` (its hooks, plugins and extensions load only once you trust
the folder).

## Extensions

```python
# ~/.tinyharness/extensions/wordcount.py
"""Counts words in text files."""

from tinyharness.ext import Extension
from tinyharness.tools.base import ToolContext, ToolError


def setup(th: Extension) -> None:
    @th.tool(read_only=True)  # read-only: never asks, may run in parallel, allowed in plan mode
    async def word_count(path: str, ctx: ToolContext) -> str:
        """Count the words in a text file.

        path: the file to count
        """
        file = ctx.workspace.resolve(path)
        if file is None or not file.is_file():
            raise ToolError(f"not a file: {path}")  # the model sees this and can recover
        return f"{len(file.read_text().split())} words"
```

Turn it on in `~/.tinyharness/config.toml`, or for one session with
`/extensions on wordcount`:

```toml
[extensions]
enabled = ["wordcount"]
```

The docstring and type hints become the tool's description and schema; a parameter
named `ctx` gets the tool context. `/extensions` lists what's loaded and how many tokens
each one adds to every request. From the SDK, pass extensions in code:
`SessionOptions(extensions=[wordcount])`, a module or a setup function.

## Tools as classes

For full control (custom permission checks, a label for the user, images in the result),
write a `Tool` subclass and add it with `th.add_tool(...)` in an extension, or pass it in
`SessionOptions(tools=[...])`:

```python
from pydantic import Field
from tinyharness import Tool, ToolContext, ToolError, ToolInput


class WordCountInput(ToolInput):
    path: str = Field(description="File to count")


class WordCount(Tool[WordCountInput]):
    name = "WordCount"
    description = "Count the words in a text file."
    Input = WordCountInput
    read_only = True

    async def run(self, args: WordCountInput, ctx: ToolContext) -> str:
        path = ctx.workspace.resolve(args.path)
        if path is None or not path.is_file():
            raise ToolError(f"not a file: {args.path}")
        return f"{len(path.read_text().split())} words"


SessionOptions(tools=[WordCount()])
```

| Attribute | Effect |
|---|---|
| `read_only = True` | shown in plan mode; batches of read-only calls run concurrently |
| `needs_permission = False` | skips the permission gate (only for tools with no side effects) |
| `target_path(args)` | the path the gate checks; defaults to `args.path` |
| `describe(args)` | the one-line label shown to the user |

## Hooks

Hooks run at lifecycle events. A `PreToolUse` hook can allow or deny a tool call,
and it runs before the permission mode is applied.

| Event | Payload | What a result does |
|---|---|---|
| `SessionStart` | `session_id` | nothing |
| `UserPromptSubmit` | `prompt` | nothing |
| `PreToolUse` | `tool`, `args` | `deny` stops the call; the reason goes to the model |
| `PostToolUse` | `tool`, `args`, `result` | a `reason` is added to the tool's result |
| `Stop` | — | `block` (or `deny`) keeps the agent going, with the reason as its next instruction (twice per turn at most) |

**Shell hooks** go in `hooks.toml`. The payload arrives as JSON on stdin. Print
`{"decision": "deny", "reason": "..."}` to block the call. A non-zero exit code also
denies, with what the command printed as the reason. Hook processes never see
environment variables that look like credentials, and credentials are masked out of
the JSON they receive. A hook gets 10 seconds unless it sets `timeout`.

```toml
[[hook]]
event = "PreToolUse"
matcher = { tool = "Write" }
command = "./scripts/no-generated-files.sh"

# The turn can't end while the tests fail: their output goes back to the agent.
[[hook]]
event = "Stop"
command = "uv run pytest -q -x"
timeout = 300
```

An extension reacts to the same events with `@th.on("tool_call")`, `"tool_result"`,
`"user_prompt"` and `"turn_end"` (see [extension-api.md](extension-api.md#events)).
**Python hooks** can also be passed through the SDK:

```python
def guard(event, payload):
    if payload["tool"] == "Write" and payload["args"]["path"].endswith(".lock"):
        return {"decision": "deny", "reason": "lockfiles are generated"}


SessionOptions(hooks=[Hook(event="PreToolUse", type="python", callable=guard)])
```

## Post-turn evaluators

SDK hosts can attach observational evaluators. They run after the agent loop has completed
but before the final `Done` event reaches the SDK consumer. Each receives a deep-copied
message snapshot and writes its return value to the session JSONL as an `evaluation` event.
Each evaluator has a 30-second default timeout; a failure or timeout is logged and never
changes the task's answer or history. Set a shorter bound with `evaluator_timeout`.

```python
async def tests_passed(ctx):
    return {"passed": (ctx.working_dir / ".tests-passed").exists()}


SessionOptions(evaluators=[tests_passed], evaluator_timeout=10.0)
```

## Subagents

Subagents and teams come with the `agents` extension (`/extensions on agents`). The
`Agent` tool runs a task in a fresh context and returns only the final answer.
The model uses it on its own when a job is big enough. It comes with four built-in
agents: `explore`, `plan` and `researcher` (read-only; `researcher` searches the web when the `web` extension is on)
and `general-purpose`. Several agents started in one reply run in parallel, writers
included: their calls that change things (Bash, Edit, Write) take turns, so two never
write at once, while their model calls overlap. To add your own, put a Markdown file in
`agents/`:

```markdown
---
name: reviewer
description: Reviews a change for bugs and missing tests.
tools: [Read, Grep, Glob]
permission_mode: default   # optional: never more permissive than this
---
You are a meticulous reviewer. Report concrete bugs with file:line references.
```

Subagents run in the session's permission mode, and follow it when you switch (even
mid-run). An agent's `permission_mode` is a limit: it can make the agent stricter than
the session, never more permissive. A subagent's permission prompts say which agent is
asking. Subagents can't start further subagents. Their transcripts go to
`~/.tinyharness/projects/<project>/sidechains/`, and their progress is shown live (SDK: `on_event`).

**Teams.** For a goal that splits into parts, the model can call `Team` with a list of
tasks, each with an agent type and the tasks it depends on. A task starts when its
dependencies are done and gets their results; independent tasks run in parallel (up
to 4 at a time), and teammates' calls that change things take turns. Teammates share a board
(`TeamBoard`) and can message each other (`TeamMessage`); if a task fails, the tasks
that depend on it are skipped. The lead gets one report with every result.

## Plugins

A plugin bundles skills, agents and hooks into one directory you can share:

```
~/.tinyharness/plugins/my-kit/
  plugin.toml          name = "my-kit"
  skills/<name>/SKILL.md
  agents/<name>.md
  hooks.toml
```

## Providers

A provider that speaks one of the four supported API formats is just config (see
[`examples/config.toml`](../examples/config.toml)):

```toml
[providers.gateway]
api = "chat"            # "chat" | "responses" | "anthropic" | "gemini"
base_url = "https://llm.internal.example.com/v1"
api_key_env = "GATEWAY_API_KEY"
default_model = "claude-sonnet-5"
```

A new API format is a new adapter in [`tinyharness/llm/adapters/`](../tinyharness/llm/adapters):
subclass `Adapter`, implement `payload()` (a provider-neutral `Request` to your API's
body), `parser()` (your stream events to `StreamChunk`s) and `ping_payload()`, and
override `parse_models()` if the model list isn't OpenAI-shaped. Register it in
`adapters/__init__.py`. Two rules keep every model treated the same:

- Never branch on model names. Per-model behavior comes only from the provider's
  model list (`ModelInfo`), e.g. its context window or supported reasoning efforts.
- If the API returns data it needs back next turn (signed or encrypted reasoning),
  return it from `StreamParser.state()` and send it back unchanged when
  `request.echo(message)` returns it. The client only hands it back to the same
  provider and model.
