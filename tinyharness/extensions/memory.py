"""Memory across sessions: notes about you and about this project that every session starts
with, which the agent keeps with the Memory tool and you can edit by hand (/memory)."""

from tinyharness.ext import CommandContext, Extension
from tinyharness.paths import pretty_path
from tinyharness.tools.memory import (
    LIMITS,
    TITLES,
    MemoryTool,
    Scope,
    memory_history_path,
    memory_path,
    read_history,
    read_notes,
    render,
    size,
)

GUIDANCE = (
    "Notes you saved in earlier sessions with the Memory tool: what the user wants and what "
    "you know about this project. Follow them. A note about the code can go stale: if one "
    "disagrees with what you see now, trust what you see and fix the note."
)


def setup(th: Extension) -> None:
    session = th.session
    if session is None:
        return
    root = session.workspace.root
    th.add_tool(MemoryTool(root))
    # Read once, now: the system prompt stays the same for the whole session (the cached
    # prefix), and notes saved during it show from the next session on.
    user, project = (read_notes(memory_path(s, root)) for s in ("user", "project"))
    th.prompt_section(
        "Memory",
        f"{GUIDANCE}\n\n## About the user\n{render(user)}\n\n## About this project\n{render(project)}",
    )

    @th.command("memory", help="show what the agent remembers, and where to edit it")
    async def show(arg: str, ctx: CommandContext) -> None:
        scopes: tuple[Scope, ...] = ("user", "project")
        if arg.strip().lower() == "history":
            rows: list[str] = []
            for scope in scopes:
                records = read_history(memory_history_path(scope, root))[-10:]
                rows.extend(
                    f"- {r.get('recorded_at', '?')} · {scope} · {r.get('action', '?')} · "
                    f"{r.get('memory_id', '?')} · {r.get('note') or r.get('previous') or '(empty)'}"
                    for r in records
                )
            ctx.say("Memory history (latest 10 per scope):\n" + ("\n".join(rows) or "(empty)"))
            return
        parts = []
        for scope in scopes:
            path = memory_path(scope, root)
            notes = read_notes(path)
            parts.append(
                f"{TITLES[scope].capitalize()} ({size(notes)} of {LIMITS[scope]} characters), "
                f"{pretty_path(path)}:\n{render(notes)}"
            )
        ctx.say(
            "\n\n".join(parts)
            + "\n\nUser memory follows you across projects; project memory applies only here. "
            "Standing global instructions belong in ~/.tinyharness/AGENTS.md, and shared "
            "project rules belong in AGENTS.md. Edit the memory files to change their notes; "
            "new sessions read them. Use /memory history to audit changes."
        )
