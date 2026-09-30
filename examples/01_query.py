"""The smallest possible program: one prompt, streamed events.

uv run python examples/01_query.py "create hello.py that prints hi"
"""

import asyncio
import sys
import tempfile

from tinyharness import AssistantText, Done, SessionOptions, ToolResult, ToolUse, query


async def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Create hello.py that prints 'hi', then run it."
    options = SessionOptions(
        working_dir=tempfile.mkdtemp(prefix="tinyharness-"),
        # Edits inside working_dir, and commands in the sandbox, run without asking.
        # Anything else is denied: a one-shot script has no one to ask.
        permission_mode="acceptEdits",
    )

    async for event in query(prompt, options):
        match event:
            case AssistantText(text=text):
                print(text, end="", flush=True)
            case ToolUse(summary=summary):
                print(f"\n● {summary}")
            case ToolResult(output=output, is_error=is_error):
                print(f"  {'✗' if is_error else '⎿'} {output.splitlines()[0] if output else ''}")
            case Done():
                print(f"\n[done] files are in {options.working_dir}")


if __name__ == "__main__":
    asyncio.run(main())
