# Extending tinyharness

There are five extension points: tools, hooks, subagents, plugins and skills
(see [skills.md](skills.md)). User-global extensions live in `~/.tinyharness/`.
Project ones live in `<project>/.tinyharness/`.

## Tools

A tool has three parts: a pydantic model for its arguments, a description for the
model, and an async `run`. The argument model is turned into the JSON schema sent to
the LLM, and incoming arguments are checked against it before `run` is called.

```python
from pydantic import Field
from tinyharness import Tool, ToolContext, ToolError, ToolInput


class WordCountInput(ToolInput):
    path: str = Field(description="File to count")


class WordCount(Tool[WordCountInput]):
    name = "WordCount"
    description = "Count the words in a text file."
    Input = WordCountInput
    read_only = True  # allowed in plan mode; may run in parallel with other reads

    async def run(self, args: WordCountInput, ctx: ToolContext) -> str:
        path = ctx.workspace.resolve(args.path)
        if path is None or not path.is_file():
            raise ToolError(f"not a file: {args.path}")  # the model sees this and can recover
        return f"{len(path.read_text().split())} words"


session.tools.register(WordCount())
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

**Python hooks** are passed through the SDK:

```python
def guard(event, payload):
    if payload["tool"] == "Write" and payload["args"]["path"].endswith(".lock"):
        return {"decision": "deny", "reason": "lockfiles are generated"}


SessionOptions(hooks=[Hook(event="PreToolUse", type="python", callable=guard)])
```

## Subagents

The `Agent` tool runs a task in a fresh context and returns only the final answer.
The model uses it on its own when a job is big enough. It comes with four built-in
agents: `explore`, `plan` and `researcher` (read-only; `researcher` searches the web)
and `general-purpose`. Several agents started in one reply run in parallel, writers
included: their calls that change things (Bash, Edit, Write) take turns, so two never
write at once, while their model calls overlap. To add your own, put a Markdown file in
`agents/`:

```markdown
---
name: reviewer
description: Reviews a change for bugs and missing tests.
tools: [Read, Grep, Glob]
permission_mode: acceptEdits   # optional: never more permissive than this
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
