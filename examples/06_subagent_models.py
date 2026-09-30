"""Choosing a subagent's model. Whoever defines the agent picks it, in its AgentSpec or
Markdown file; the model doesn't choose at run time, so cost and behavior stay
predictable. `model` accepts:

    "inherit" (or leave it out)     the session's model
    "small"                         the session provider's cheap model: its `small_model`
                                    in tinyharness/llm/providers.toml (anthropic →
                                    claude-haiku-4-5, openai → gpt-6-luna, ...), which you
                                    can change under [providers.<name>] in config.toml
    "anthropic:claude-haiku-4-5"    any provider you have a key for (a GPT session can
                                    hand work to a Claude agent, and the other way round)
    "gpt-6-sol"                     a bare model id, from the session's provider

Here three agents each use a different form, and all three work on the same file:

    reviewer        model="inherit"                     (AgentSpec, below)
    line-counter    model: small                        (Markdown file, below)
    second-opinion  model="anthropic:claude-haiku-4-5"  (AgentSpec; override with argv[1])

uv run python examples/06_subagent_models.py
uv run python examples/06_subagent_models.py openai:gpt-6-sol
"""

import asyncio
import sys
import tempfile
from pathlib import Path

from tinyharness import (
    AgentSpec,
    AssistantText,
    Event,
    MissingAPIKeyError,
    Session,
    SessionOptions,
    SubagentUpdate,
    ToolUse,
)

# An agent can also be a Markdown file, in ~/.tinyharness/agents/ or
# <project>/.tinyharness/agents/. The same `model` values work in its frontmatter.
#
# How `model: small` resolves, step by step (values for an OpenRouter session):
#
#   1. The file is parsed into AgentSpec(name="line-counter", model="small").
#      "small" is a keyword here, not a model name.
#   2. When the main model calls the Agent tool, the session calls
#      Session.llm_for("small").
#   3. llm_for keeps the session's own client (same provider, same API key) and
#      asks the config for Config.small_model.
#   4. Config.small_model returns the session provider's `small_model` from
#      tinyharness/llm/providers.toml, or the session's model if it has none:
#
#          [openrouter]
#          default_model = "anthropic/claude-sonnet-5"   # what the session runs on
#          small_model   = "anthropic/claude-haiku-4.5"  # what "small" means
#
#   5. The subagent runs on that client with model="anthropic/claude-haiku-4.5".
#
# So `small` means "the cheap model of whatever provider the session is on". The
# same file gives claude-haiku-4-5 on an anthropic session, gpt-6-luna on openai,
# gemini-3.5-flash-lite on gemini, and never needs another API key. To change it:
#
#     # ~/.tinyharness/config.toml
#     [providers.openrouter]
#     small_model = "google/gemini-3.5-flash-lite"
LINE_COUNTER_MD = """\
---
name: line-counter
description: Counts the lines in files. Use for any question about file length.
tools: [Read, Glob]
model: small
---
You count lines in files. Reply with one line per file: `<path>: <n> lines`.
"""

REVIEW_TOOLS = ["Read", "Grep", "Glob"]


def on_event(event: Event) -> None:
    """Subagent progress arrives here while the parent waits on its Agent call."""
    if isinstance(event, SubagentUpdate) and event.status in ("started", "done", "failed"):
        print(f"\n  ↳ [{event.agent} on {event.model}] {event.status} {event.detail}".rstrip())


async def main() -> None:
    other_provider_model = sys.argv[1] if len(sys.argv) > 1 else "anthropic:claude-haiku-4-5"

    work_dir = Path(tempfile.mkdtemp(prefix="tinyharness-"))
    (work_dir / "fib.py").write_text(
        "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n - 1) + fib(n - 2)\n"
    )
    agents_dir = work_dir / ".tinyharness" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "line-counter.md").write_text(LINE_COUNTER_MD)

    agents = [
        AgentSpec(
            name="reviewer",
            description="Reviews code for bugs and performance problems. Read-only.",
            system_prompt="You are a terse code reviewer. List concrete problems, most severe first.",
            tools=REVIEW_TOOLS,
            model="inherit",  # the session's model
        ),
        AgentSpec(
            name="second-opinion",
            description="An independent review from a different model. Read-only.",
            system_prompt="You give a second, independent code review. Be brief.",
            tools=REVIEW_TOOLS,
            model=other_provider_model,  # a model from another provider
        ),
    ]

    session = Session(SessionOptions(working_dir=work_dir, agents=agents, on_event=on_event))
    try:
        # Show what each agent runs on. Session.llm_for resolves the `model` setting,
        # the same way the Agent tool does when it starts one.
        print(f"session: {session.config.provider}:{session.config.model}")
        for name in ("reviewer", "line-counter", "second-opinion"):
            spec = session.agents[name]
            try:
                llm, model = session.llm_for(spec.model)
            except MissingAPIKeyError as e:
                print(f"  {name}: {e}\n  (falling back to the session's model)")
                spec.model = "inherit"
                llm, model = session.llm_for(spec.model)
            resolved = f"{llm.provider.name}:{model or llm.model}"
            print(f"  {name:<15} model={spec.model:<28} → {resolved}")
        print()

        prompt = (
            "For fib.py: use the line-counter agent to count its lines, and get reviews "
            "from both the reviewer and the second-opinion agents. Then compare the reviews."
        )
        async for event in session.run(prompt):
            match event:
                case AssistantText(text=text):
                    print(text, end="", flush=True)
                case ToolUse(summary=summary):
                    print(f"\n● {summary}")
        print()
    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(main())
