# Extension API (draft for review)

Status: approved 1 Oct 2026; v1 implemented in `tinyharness/ext.py`. The design behind it, with the
benchmark and sources, is in the redesign proposal; this page is the contract.

## Why

tinyharness sends a 9.1k-token prefix on every model call, against pi's 2.8k, because
every feature is always on: 14 tools and a 2.3k-token system prompt. Most sessions use
six of those tools. So the core keeps only what every session needs, and everything else
becomes an extension you turn on: one Python file with a `setup()` function.

Goals:

- A small core: agent loop, session, sandbox and permission gate, CLI, and six tools
  (Read, Write, Edit, Bash, Grep, Glob).
- One way to add anything: a tool, a prompt section, a slash command, a reaction to an
  event. The built-in features use the same API, so it's proven on real features.
- Security holds when other people's code is loaded: extensions say what they need, the
  gate and sandbox apply to their tools, and project extensions need folder trust.
- The prompt prefix stays byte-stable for the whole session, so prompt caches (Anthropic,
  OpenAI, OpenRouter, vLLM) keep hitting.

Not goals (for v1): running extensions in their own processes, an extension marketplace,
changing tools or prompt sections in the middle of a session.

## A complete extension

```python
# ~/.tinyharness/extensions/hello.py
"""Greets people. (The module docstring is the description shown in /extensions.)"""

from tinyharness.ext import Extension

NEEDS: list[str] = []  # what it needs beyond the sandbox; see "Capabilities"


def setup(th: Extension) -> None:
    @th.tool(read_only=True)
    async def hello(name: str) -> str:
        """Greet someone by name.

        name: who to greet
        """
        return f"Hello, {name}!"

    th.prompt_section("hello", "Use the hello tool to greet people by name.")

    @th.on("tool_call")
    async def no_force_push(call, ctx):
        if call.tool == "Bash" and "push --force" in call.input.get("command", ""):
            return ctx.block("force-push is off in this project")

    @th.command("hello", help="say hello")
    async def hello_command(arg: str, ctx) -> None:
        ctx.say(f"Hello, {arg or 'world'}!")
```

Turn it on in `~/.tinyharness/config.toml` (settings for one extension go in its own table):

```toml
[extensions]
enabled = ["web", "agents", "hello"]

[extensions.hello]
greeting = "Hi"
```

Or for this session only: `/extensions on hello`.

## Where extensions come from

An entry in `enabled` names a module. It's looked up in this order (first match wins):

| Source | Example | Trust |
|---|---|---|
| Built-in | `tinyharness/extensions/web.py` | always loaded |
| Your folder | `~/.tinyharness/extensions/hello.py` or `hello/__init__.py` | always loaded |
| The project | `<project>/.tinyharness/extensions/x.py` | only in a trusted folder (`trust.py`), fingerprinted like plugins |
| An installed package | entry point group `tinyharness.extensions`, e.g. `pip install tinyharness-jira` | loaded because you installed it and named it |

A project can also turn extensions on in `<project>/.tinyharness/config.toml`. Only names
from the first three sources may be turned on that way; a project can't turn on a package
it didn't ship.

One extension that fails to import or raises in `setup()` is skipped with a warning; the
session still starts.

## The `th` object

Everything an extension does goes through the `Extension` object passed to `setup()`.

### Tools

```python
@th.tool(name=None, read_only=False, network=False, writes=None)
async def fn(arg: type, ...) -> str | ToolOutput: ...

th.add_tool(MyTool())   # an existing Tool subclass (tools/base.py), unchanged
```

- The decorator builds the tool from the function: its name (or `name=`), the
  docstring's first paragraph as the description, the type hints and defaults as the
  schema (via pydantic), and `arg: description` lines in the docstring as field
  descriptions.
- `read_only=True`: no side effects. It never asks, may run alongside other calls, and is
  allowed in plan mode.
- `network=True`: it reaches other machines. It follows the network rules below.
- Returning `str` gives text; `ToolOutput` (already in `tools/base.py`) can carry images.
  Raising `ToolError` gives the model an error result.
- `th.add_tool` takes any `Tool` subclass, so today's tools move into extensions without
  being rewritten.

A tool with the same name as a core tool replaces it, and `/extensions` shows the
replacement.

### Prompt sections

```python
th.prompt_section(name: str, text: str)
```

Text added to the system prompt, under `# <name>`. Sections are placed after the core
prompt, in the order extensions appear in `extensions`, and in registration order within
one extension. Short text belongs in the tool's description; a section is for guidance
that spans tools (how to brief a teammate, say).

### Events

```python
@th.on(event)
async def handler(payload, ctx) -> None | Result: ...
```

| Event | Payload | A handler can |
|---|---|---|
| `session_start` | session info | prepare state |
| `user_prompt` | `{"prompt": text}` | add context for the model: `return ctx.note(text)` |
| `tool_call` | a `Call` (`.tool`, `.input`) | stop it: `return ctx.block(reason)` |
| `tool_result` | `{"tool", "args", "result"}` | add a note for the model: `return ctx.note(text)` |
| `turn_end` | `{}` | keep the turn going: `return ctx.block(instruction)` (at most twice per turn) |

Returning `None` lets things proceed. (Changing a call's input and a `session_end` event
are left for later; nothing needs them yet.)

Handlers run in the order extensions were loaded, and the first block wins. These are the
same events as today's hooks: `hooks.toml` stays, and each shell hook becomes a handler
on this event bus (`PreToolUse` = `tool_call`, `PostToolUse` = `tool_result`,
`Stop` = `turn_end`, `UserPromptSubmit` = `user_prompt`, `SessionStart` =
`session_start`). There is one mechanism, with two ways to configure it.

### Slash commands

```python
@th.command(name, help="", usage="")
async def handler(arg: str, ctx) -> None: ...
```

They appear in `/help` and the `/` menu under the extension's name. `ctx.say(text)` prints
to the user; `await ctx.send(text)` sends a message to the model as if the user typed it.

### Running a subagent

```python
answer = await th.run_agent(
    task: str,
    *,
    agent: str = "general-purpose",   # an agent spec from agents/*.md (its tools and prompt)
    model: str | None = None,         # default: the agent's, else the session's model
    label: str = "",
    extra_tools: list[Tool] | None = None,
    inbox: Callable[[], list[str]] | None = None,
) -> str
```

A child agent loop with its own history, under the same gate and sandbox, with progress
shown in the UI. This is the core primitive the `agents` extension builds Agent and Team
on (`subagents/runner.py` today), and anyone else can build their own orchestration on it.

### Everything else

- `th.remove_tool(name)`: take a core tool away (an extension that replaces how something is done).
- `th.config`: this extension's table from config (`[extensions.hello]`), as a dict.
- `th.session`: read-only facts: working directory, model, provider, mode.
- `th.state_dir`: a folder for this extension's files (`~/.tinyharness/extensions-data/<name>/`).
- `th.log`: a logger named for the extension.

## Rules

1. **Frozen at session start.** Tools and prompt sections can only be registered in
   `setup()`. After that the tool list and system prompt don't change on their own, so the
   cached prefix stays byte-identical. `/extensions on|off <name>` switches one right away
   and says the next request rebuilds the prompt cache once.
2. **Deterministic prefix.** Core prompt, then extension sections; core tools, then
   extension tools, in config order. Schemas are serialized with sorted keys.
3. **Same gate for everyone.** Extension tools go through the permission gate exactly like
   core tools: `read_only` never asks, writes follow the mode, network follows the
   network rules. An extension can't mark itself as exempt.
4. **Disposal.** Everything `setup()` registered is removed when the extension is turned off.
5. **No silent prompt growth.** `/extensions` lists each one with the tokens it adds to
   every request (tool schemas plus sections), so the cost of turning one on is visible.

## Capabilities

`NEEDS` declares what an extension needs beyond the sandbox:

```python
NEEDS = ["network:api.github.com", "write:~/notes"]
```

- **Consent.** The first time an extension with `NEEDS` is loaded, you're shown the list
  once and asked to allow it. The answer is stored with a fingerprint of the file, as
  folder trust is, so a changed extension asks again.
- **Network.** `network:<host>` adds hosts to the sandbox's allowlist for this extension's
  tools. A `network=True` tool reaching any other host asks, as WebFetch does today.
- **Writes.** `write:<path>` lets this extension's tools write there without asking.
- **Commands.** `th.run(cmd)` runs a shell command in the same sandbox as Bash, with this
  extension's `NEEDS` applied. Extensions should use it rather than `subprocess`.

Limit, stated plainly: v1 doesn't sandbox an extension's own Python. Python running in the
harness process has the harness's rights, as in pi, opencode and Claude Code. `NEEDS` is
enforced for its tools' permission decisions and for commands run through `th.run`.
Running each extension in a sandboxed subprocess (as Cline does) is the v2 path, and the
API above doesn't change for it.

## What moves where

| Today | After |
|---|---|
| Read, Write, Edit, Bash, Grep, Glob | core |
| WebSearch, WebFetch | `web` extension |
| Pdf | `pdf` extension (Bash with `pdftotext` or pypdf covers most uses without it) |
| Open | `open` extension |
| TodoWrite (and `todo_list = true`) | `todo` extension |
| Agent, Team, TeamBoard, TeamMessage, agents/*.md | `agents` extension, built on `th.run_agent` |
| Auto-mode classifier | `classifier` extension, run only on escapes |
| Skill tool | removed: skills are listed in the prompt and read with Read |
| `hooks.toml`, Python hooks in the SDK | unchanged for users; run on the event bus |
| Plugins (skills + agents + hooks bundles) | unchanged; a plugin may also contain `extension.py` |
| `SessionOptions.tools` (SDK) | unchanged; it's a programmatic extension |

By default `enabled = []`: a new user gets the lean core. `tinyharness --setup` offers the
common set (web, agents, todo) as one choice.

## Skills

Unchanged for you: `/claude-design-skill [args]` still puts the skill's instructions in
your message, as it does today.

What changes is how the model finds a skill on its own. The core prompt lists each skill:

```
# Skills
Read a skill's SKILL.md and follow it when the task matches. Its other files are
next to it; read them only when a step needs them.
- claude-design-skill: Design artifacts in HTML/CSS… (~/.tinyharness/skills/claude-design-skill/SKILL.md)
```

Read already allows the skill folders (they're read roots of the workspace), so no tool is
needed. This saves the Skill tool's schema (about 200 tokens); the listing still costs
about 50 tokens per installed skill, as it does now.

## The core prompt

After the move, the core system prompt holds only what every session needs, about 500
tokens: who it is, the sandbox in two sentences, "verify your work", answer style, the
environment, and the skills listing. Guidance for a feature moves into that feature's
extension, so it's present exactly when the feature is (how to brief a teammate goes with
`agents`; citing pages goes with `web`).

## Testing an extension

```python
from tinyharness.testing import load_extension, ScriptedLLM

async def test_hello(tmp_path):
    session = load_extension("hello", working_dir=tmp_path,
                             llm=ScriptedLLM([[("hello", '{"name": "Ada"}')], "done"]))
    events = await session.run_all("greet Ada")
    assert "Hello, Ada!" in events.tool_results()
```

`ScriptedLLM` is the test double already in `tests/conftest.py`, moved into the package.

## How we'll know it worked

The benchmark from the redesign proposal, rerun with three additions: a research task
(needs `web`), a team task (needs `agents`), and a Gemma 4 column. Accepted when:

- the prefix with no extensions is under 3k tokens;
- cost is within 1.2× of pi on the five original tasks;
- success is no lower than today on every task, with the needed extension turned on;
- the built-in extensions pass their existing tests unchanged, through `th.add_tool`.

## Decided

1. **Mid-session switching** is allowed (`/extensions on|off`); `/extensions` shows each
   one's tokens per request, and switching says the cache is rebuilt once.
2. **Tool search** is not in v1; revisit when someone has more than about ten extension
   tools on.
3. **Names** stay short: `th`, `NEEDS`.
