"""Resuming a conversation from where it was saved, in a later run of the program.

Every conversation is saved as it goes (in ~/.tinyharness/projects/<project>/, never in
the project itself), so a program can exit and a later one can carry on. Run these one
after another; each is a separate process:

    uv run python examples/07_resume.py start         # begin a task, then exit
    uv run python examples/07_resume.py list          # the saved conversations
    uv run python examples/07_resume.py resume        # continue the latest one
    uv run python examples/07_resume.py resume <id>   # ...or a specific one

Resuming rebuilds the conversation exactly: the model remembers the task, what it read
and its todo list. Things may have changed in between, so the harness checks: try
editing notes.md in the demo folder before `resume`, and the model is told the file
changed (and its old copy of the file is marked stale) before your next message.

In the CLI, `tinyharness -c` continues the latest conversation and `-r` picks one.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

from tinyharness import AssistantText, Session, SessionOptions, ToolUse, list_sessions

# One folder for every run, so they share the saved conversations.
DEMO = Path(tempfile.gettempdir()) / "tinyharness-resume-demo"
NOTES = "Trip plan\n- Book flights to Lisbon\n- Find a hotel near Alfama\n- Plan day trip to Sintra\n"


async def ask(session: Session, prompt: str) -> None:
    print(f"\n» {prompt}")
    async for event in session.run(prompt):
        match event:
            case AssistantText(text=text):
                print(text, end="", flush=True)
            case ToolUse(summary=summary):
                print(f"\n● {summary}")
    print()


def options(**kw: str) -> SessionOptions:
    return SessionOptions(working_dir=DEMO, permission_mode="default", **kw)


async def start() -> None:
    DEMO.mkdir(exist_ok=True)
    (DEMO / "notes.md").write_text(NOTES)
    session = Session(options())
    try:
        await ask(session, "Read notes.md and make a todo list for the trip. Don't start on it yet.")
    finally:
        session.close()  # saved already; closing just tidies up
    print(f"\nSaved as {session.id}. Now run: uv run python examples/07_resume.py resume")


def show() -> None:
    saved = list_sessions(options())
    if not saved:
        print("No saved conversations yet. Run: uv run python examples/07_resume.py start")
    for info in saved:
        print(f"{info.id}  {info.updated:%Y-%m-%d %H:%M}  {info.messages:>3} messages  {info.title}")


async def resume(which: str) -> None:
    session = Session(options(resume=which))  # "latest", or an id from `list`
    try:
        report = session.resume_report
        assert report is not None
        print(f"Resumed {session.id}: {report.summary()}")
        print(f"Todo list: {[todo['content'] for todo in session.ctx.todos]}")
        if report.anything_changed:
            print(f"The model will be told:\n{report.note()}")
        await ask(session, "Where were we? Pick up the first item on the list.")
    finally:
        session.close()


async def main() -> None:
    command, *rest = sys.argv[1:] or ["start"]
    if command == "start":
        await start()
    elif command == "list":
        show()
    elif command == "resume":
        try:
            await resume(rest[0] if rest else "latest")
        except ValueError as e:  # nothing saved yet, or no such id
            print(e)
    else:
        print(__doc__)


if __name__ == "__main__":
    asyncio.run(main())
