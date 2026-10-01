"""Running a skill from code. Uses the example skills bundled in examples/skills/.

uv run python examples/05_skills.py "Ada"
"""

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

from tinyharness import AssistantText, SessionOptions, ToolUse, invoke_skill, load_skills


async def main() -> None:
    # Point TINYHARNESS_HOME at a temp dir containing the example skills.
    home = Path(tempfile.mkdtemp(prefix="tinyharness-home-"))
    shutil.copytree(Path(__file__).parent / "skills", home / "skills")
    os.environ["TINYHARNESS_HOME"] = str(home)

    work_dir = Path(tempfile.mkdtemp(prefix="tinyharness-"))
    for skill in load_skills(work_dir):
        print(f"skill /{skill.name}: {skill.description}")

    name = sys.argv[1] if len(sys.argv) > 1 else "Ada"
    options = SessionOptions(working_dir=work_dir, permission_mode="default")
    async for event in invoke_skill("hello-world", args=name, options=options):
        match event:
            case AssistantText(text=text):
                print(text, end="", flush=True)
            case ToolUse(summary=summary):
                print(f"\n● {summary}")

    greeting = work_dir / "greeting.txt"
    print(f"\n\n{greeting}: {greeting.read_text().strip() if greeting.exists() else '(not written)'}")


if __name__ == "__main__":
    asyncio.run(main())
