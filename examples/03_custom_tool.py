"""Adding your own tool: a pydantic Input model, a description, and a `run` method.

uv run python examples/03_custom_tool.py
"""

import asyncio
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field

from tinyharness import (
    AssistantText,
    Session,
    SessionOptions,
    Tool,
    ToolContext,
    ToolError,
    ToolInput,
    ToolUse,
)


class ClockInput(ToolInput):
    timezone: str = Field(default="UTC", description="IANA timezone name, e.g. 'Asia/Tokyo'")


class ClockTool(Tool[ClockInput]):
    name = "Clock"
    description = "Get the current date and time in a timezone."
    Input = ClockInput
    read_only = True  # no side effects: allowed in plan mode, runs in parallel with other reads
    needs_permission = False  # harmless, so skip the permission prompt

    def describe(self, args: ClockInput) -> str:
        return f"Clock({args.timezone})"

    async def run(self, args: ClockInput, ctx: ToolContext) -> str:
        try:
            now = datetime.now(ZoneInfo(args.timezone))
        except ZoneInfoNotFoundError:
            # The model sees this message as the tool result and can retry with a valid name.
            raise ToolError(f"unknown timezone {args.timezone!r}") from None
        return now.strftime("%A %Y-%m-%d %H:%M:%S %Z")


async def main() -> None:
    session = Session(
        SessionOptions(working_dir=tempfile.mkdtemp(prefix="tinyharness-"), tools=[ClockTool()])
    )
    try:
        async for event in session.run("What time is it in Tokyo and in New York right now?"):
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
