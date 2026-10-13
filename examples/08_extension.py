"""An extension: one setup(th) function that adds a tool, a prompt section and a guard.

The same function works as a file in ~/.tinyharness/extensions/ (turned on with
`[extensions] enabled = ["notes"]` in config.toml or `/extensions on notes`); here it's
passed in code. See docs/extension-api.md.

uv run python examples/08_extension.py
"""

import asyncio
import json
import tempfile
from pathlib import Path

from tinyharness import AssistantText, Session, SessionOptions, ToolResult, ToolUse
from tinyharness.ext import Extension


def notes(th: Extension) -> None:
    """Searchable project notes kept in notes.json."""

    @th.tool(read_only=True)
    async def find_notes(topic: str) -> str:
        """Find the project's notes about a topic.

        Parameters
        ----------
        topic : str
            Word to find, such as ``"deploy"``.
        """
        store = Path(th.session.workspace.root if th.session else ".") / "notes.json"
        found = [n for n in json.loads(store.read_text()) if topic.lower() in n.lower()]
        return "\n".join(f"- {n}" for n in found) or f"no notes about {topic!r}"

    th.prompt_section("Notes", "This project keeps its decisions in notes; search them with find_notes.")

    @th.on("tool_call")
    async def keep_notes_safe(call, ctx):
        if call.tool == "Bash" and "notes.json" in call.input.get("command", ""):
            return ctx.block("notes.json is edited by people, not commands")


async def main() -> None:
    work_dir = Path(tempfile.mkdtemp(prefix="tinyharness-"))
    (work_dir / "notes.json").write_text(
        json.dumps(
            [
                "Deploy on Tuesdays only; Fridays are frozen.",
                "The deploy script needs DEPLOY_ENV=prod.",
                "Use uv, not pip, for Python dependencies.",
            ]
        )
    )
    async with Session(SessionOptions(working_dir=work_dir, extensions=[notes])) as session:
        ext = session.extensions[0]
        print(
            f"extension {ext.name!r}: {ext.description} (+{session.extension_tokens(ext)} tokens per request)\n"
        )
        async for event in session.run("When can we deploy, and what does the deploy script need?"):
            match event:
                case AssistantText(text=text):
                    print(text, end="", flush=True)
                case ToolUse(summary=summary):
                    print(f"\n● {summary}")
                case ToolResult(output=output):
                    print("  ⎿ " + output.replace("\n", "\n    "))
        print()


if __name__ == "__main__":
    asyncio.run(main())
