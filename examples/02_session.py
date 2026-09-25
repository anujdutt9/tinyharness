"""A multi-turn conversation: the Session keeps history and permissions between prompts.

uv run python examples/02_session.py
"""

import asyncio
import tempfile

from tinyharness import AssistantText, Session, SessionOptions, ToolUse


async def ask(session: Session, prompt: str) -> None:
    print(f"\n» {prompt}")
    async for event in session.run(prompt):
        match event:
            case AssistantText(text=text):
                print(text, end="", flush=True)
            case ToolUse(summary=summary):
                print(f"\n● {summary}")
    print()


async def main() -> None:
    session = Session(
        SessionOptions(working_dir=tempfile.mkdtemp(prefix="tinyharness-"), permission_mode="acceptEdits")
    )
    try:
        await ask(session, "Write fib.py with a function fib(n) that returns the nth Fibonacci number.")
        await ask(session, "Now add a __main__ block that prints fib(10).")  # remembers fib.py
        await ask(session, "What did we just build? One sentence.")
    finally:
        session.close()


if __name__ == "__main__":
    asyncio.run(main())
