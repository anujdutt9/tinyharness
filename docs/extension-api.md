# Extension API

An extension is one Python file with a `setup(th)` function. It can add tools, sections
of the system prompt, slash commands and handlers for agent events. The built-in features
(web, pdf, open, todo, agents) are extensions written against this same API, in
[`tinyharness/extensions/`](../tinyharness/extensions); the API itself is
[`tinyharness/ext.py`](../tinyharness/ext.py).

The core stays small: the agent loop, sessions, the sandbox and permission gate, the CLI,
and six tools (Read, Write, Edit, Glob, Grep, Bash). Everything else is something you turn
on, so a session doesn't pay for tools it never uses: every tool definition and prompt
section is sent with every request.

## A complete extension

```python
# ~/.tinyharness/extensions/hello.py
"""Greets people. (The module docstring is the description shown in /extensions.)"""

from tinyharness.ext import Extension


def setup(th: Extension) -> None:
    @th.tool(read_only=True)
    async def hello(name: str) -> str:
        """Greet someone by name.

        name: who to greet
        """
        return f"{th.config.get('greeting', 'Hello')}, {name}."

    th.prompt_section("Greetings", "Use the hello tool to greet people by name.")

    @th.on("tool_call")
    async def no_force_push(call, ctx):
        if call.tool == "Bash" and "push --force" in call.input.get("command", ""):
            return ctx.block("force-push is off in this project")

    @th.command("hello", help="say hello")
    async def hello_command(arg: str, ctx) -> None:
        ctx.say(f"Hello, {arg or 'world'}!")
```

Turn it on in `~/.tinyharness/config.toml`. Settings for one extension go in its own
table and reach it as `th.config`:

```toml
[extensions]
enabled = ["web", "agents", "hello"]

[extensions.hello]
greeting = "Hi"
```

Or for the current session only: `/extensions on hello`. `/extensions` lists what's loaded,
what failed to load and why, and how many tokens each extension adds to every request.

## Where extensions come from

An entry in `enabled` names an extension, looked up in this order (first match wins):

| Source | Example | Loaded |
|---|---|---|
| Built-in | `tinyharness/extensions/web.py` | always |
| Your folder | `~/.tinyharness/extensions/hello.py` or `hello/__init__.py` | always |
| The project | `<project>/.tinyharness/extensions/x.py` | only in a folder you've trusted (it's code that runs outside the sandbox) |
| An installed package | entry point group `tinyharness.extensions`, e.g. `pip install tinyharness-jira` | when you name it |

From the SDK, pass extensions in code: `SessionOptions(extensions=[module_or_setup_function])`.
One named both in config and in code loads once.

An extension that fails to import or raises in `setup()` is skipped and reported in
`/extensions`; the session still starts.

## The `th` object

Everything an extension does goes through the `Extension` passed to `setup()`.

### Tools

```python
@th.tool(name=None, read_only=False, network=False)
async def fn(arg: type, ..., ctx: ToolContext) -> str | ToolOutput: ...

th.add_tool(MyTool())   # a Tool subclass (tools/base.py), for full control
th.remove_tool("Glob")  # take a core tool away
```

- The decorator builds a tool from an async function: its name (or `name=`), the first
  paragraph of its docstring as the description, its type hints and defaults as the
  schema, and `arg: description` lines in the docstring as the field descriptions. A
  parameter named `ctx` receives the `ToolContext` (workspace, scratch folder, usage).
- `read_only=True`: no side effects. It never asks, may run alongside other read-only
  calls, and is offered in plan mode.
- `network=True`: it reaches other machines, and the permission gate treats it like
  WebFetch.
- Return text, or a `ToolOutput` to include images. Raise `ToolError` to give the model an
  error it can act on.
- A tool with the same name as a core tool replaces it.

### Prompt sections

```python
th.prompt_section(name: str, text: str)
```

Text added to the system prompt under `# <name>`, after the core prompt, in the order the
extensions were turned on. Guidance that belongs to one tool goes in its description; a
section is for guidance about a feature as a whole (how to brief a teammate, say), and it
is sent only while the extension is on.

### Events

```python
@th.on(event)
async def handler(payload, ctx): ...
```

| Event | Payload | A handler can |
|---|---|---|
| `session_start` | `{"session_id", "resumed"}` | prepare state |
| `user_prompt` | `{"prompt"}` | add context for the model: `return ctx.note(text)` |
| `tool_call` | a `Call` (`.tool`, `.input`) | stop the call: `return ctx.block(reason)`; the reason goes to the model |
| `tool_result` | `{"tool", "args", "result"}` | add a note to the result: `return ctx.note(text)` |
| `turn_end` | `{}` | keep the turn going: `return ctx.block(instruction)` (twice per turn at most) |

Returning `None` lets things proceed; the first block wins. These are the events
`hooks.toml` uses (`SessionStart`, `UserPromptSubmit`, `PreToolUse`, `PostToolUse`,
`Stop`), on the same bus, so a shell hook and an extension handler see the same things.

### Slash commands

```python
@th.command(name, help="", usage="")
async def handler(arg: str, ctx) -> None: ...
```

They appear in `/help` and the `/` menu. `ctx.say(text)` prints to the user;
`await ctx.send(text)` sends a message to the agent as if the user had typed it;
`ctx.session` is the session.

### Running a subagent

```python
answer = await th.run_agent(
    task,
    agent="general-purpose",  # an agent definition: built-in, agents/*.md, or AgentSpec
    model=None,  # default: the agent's model, else the session's
    label="",
    extra_tools=None,
    inbox=None,  # a function returning messages to deliver between steps
)
```

Runs a subagent to completion, with its own history, under the session's permission gate
and sandbox, and returns its answer. The `agents` extension builds the Agent and Team tools
on this; you can build your own orchestration the same way.

### Everything else

- `th.config`: this extension's table from config (`[extensions.<name>]`), as a dict.
- `th.session`: the session (working directory, model, mode, tools).
- `th.state_dir`: a folder for the extension's own files,
  `~/.tinyharness/extensions-data/<name>/`.
- `th.log`: a logger named for the extension.

## Rules

1. **Registered at session start.** Tools and prompt sections are registered in `setup()`
   and then stay the same, so the provider's cached prefix keeps hitting. `/extensions on`
   or `off` switches one immediately, and the next request rebuilds the cache once.
2. **A stable order.** Core prompt, then extension sections; core tools, then extension
   tools; extensions in the order they were turned on.
3. **One permission gate.** Extension tools go through the same checks as core tools:
   `read_only` tools never ask, other tools follow the permission mode, network tools ask
   per site. An extension can't exempt its tools.
4. **Visible cost.** `/extensions` shows the tokens each extension adds to every request
   (its tool definitions and prompt sections).

## Declaring what an extension needs

An extension can declare what it needs beyond the sandbox, and `/extensions` shows it:

```python
NEEDS = ["network:api.github.com", "write:~/notes"]
```

Today this is a declaration for you to read; it isn't enforced. Planned: asking once
before loading an extension with needs (remembered with a fingerprint of the file, like
folder trust), adding its hosts to the sandbox's allowlist for its own tools, and a
`th.run(cmd)` helper that runs commands in the sandbox with those needs applied.

What no extension system in this design does: sandbox an extension's own Python code.
It runs in the harness's process with the harness's rights, as in pi, opencode and
Claude Code. Only load extensions you trust, as you would any package.

## Built-in extensions

| Extension | Adds | Without it |
|---|---|---|
| `web` | WebSearch, WebFetch, and when to cite pages | no web access beyond the sandbox's allowlist |
| `pdf` | Pdf: pages, text and images of a PDF | the model reads PDFs with Bash (`pdftotext`, pypdf) |
| `open` | Open: shows you a file or URL in its app | the model tells you the path |
| `todo` | TodoWrite: a live checklist for multi-step work | `todo_list = true` turns it on too |
| `agents` | Agent and Team, and how to brief a subagent | no subagents |

None is on by default. Hooks (`hooks.toml`), plugins, skills and `SessionOptions(tools=...)`
work as before, alongside extensions.

## Skills

Skills don't need an extension or a tool. The core prompt lists each installed skill (its
name, the start of its description and where its `SKILL.md` is), and the model loads one
by reading that file when a task matches. `/<skill-name> [args]` runs one yourself. The
list costs about 80 tokens per skill; `skill_tool = false` leaves skills to `/<name>` only.

## Testing an extension

`tinyharness.testing` runs a real session against a scripted model: no API key, no
network, and the tools really run.

```python
from tinyharness.events import ToolResult
from tinyharness.testing import scripted_session

import hello  # your extension module


async def test_hello(tmp_path):
    session = scripted_session(tmp_path, [[("hello", '{"name": "Ada"}')], "done"], extensions=[hello])
    results = [e.output async for e in session.run("greet Ada") if isinstance(e, ToolResult)]
    assert results == ["Hello, Ada."]
```

Each turn is a string (a text reply) or a list of `(tool name, arguments JSON)` calls.
