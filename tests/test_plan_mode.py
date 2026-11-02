"""Plan mode: the model is told when it's in it and asks the user to switch modes to build;
a call to change things ends the turn there, and the switch stays the user's."""

import io

from rich.console import Console

from conftest import ScriptedLLM, make_ctx
from tinyharness.agent.history import Message
from tinyharness.agent.loop import PLAN_MODE_CALL
from tinyharness.cli.render import Renderer
from tinyharness.events import AssistantText, Done, ToolResult
from tinyharness.permissions import PermissionGate, PermissionMode
from tinyharness.session import PLAN_OFF, PLAN_ON, PLAN_TAG
from tinyharness.subagents import AgentSpec
from tinyharness.subagents.runner import run_subagent
from tinyharness.testing import scripted_session
from tinyharness.tools.registry import default_registry
from tinyharness.workspace import Workspace

NOTEBOOK = '{"path": "mlx_basics.ipynb", "content": "{}"}'


def user_messages(session) -> list[str]:
    return [m.content or "" for m in session.history.messages if m.role == "user"]


async def test_the_model_is_told_once_when_plan_mode_starts_and_once_when_it_ends(tmp_path):
    """Plan mode hides the tools that change things; without a word the model can only guess
    why. Said once per crossing, so the requests in between keep the same prefix."""
    session = scripted_session(tmp_path, ["a", "b", "c", "d"])
    try:
        [e async for e in session.run("first, no plan mode")]
        session.set_mode("plan")
        [e async for e in session.run("plan the parser")]
        [e async for e in session.run("and the tests?")]
        session.set_mode("default")
        [e async for e in session.run("go")]
        first, planning, again, building = user_messages(session)
        assert PLAN_TAG not in first
        assert planning.startswith(f"{PLAN_TAG}{PLAN_ON}</plan-mode>")
        assert PLAN_TAG not in again
        assert building.startswith(f"{PLAN_TAG}{PLAN_OFF}</plan-mode>")
    finally:
        session.close()


async def test_the_plan_mode_note_comes_back_when_compaction_drops_it(tmp_path):
    """The note is read from the conversation each turn, not remembered: a summary that
    replaced it must not leave the model planning without knowing it."""
    session = scripted_session(tmp_path, ["a", "b"])
    try:
        session.set_mode("plan")
        [e async for e in session.run("plan the parser")]
        system = session.history.messages[0]
        session.history.messages[:] = [
            system,
            Message(role="user", content="[prior conversation summary] ..."),
        ]
        [e async for e in session.run("and the tests?")]
        assert user_messages(session)[-1].startswith(f"{PLAN_TAG}{PLAN_ON}")
    finally:
        session.close()


async def test_a_call_to_change_things_in_plan_mode_ends_the_turn_for_the_user_to_switch(tmp_path):
    """Seen live: asked to "draft the notebook" in plan mode, Sonnet called Bash, was told
    only "unknown or unavailable tool", and printed the whole notebook into its reply. The
    call means it wants to build: the turn stops there, and switching is the user's."""
    (tmp_path / "README.md").write_text("# demo\n")
    session = scripted_session(
        tmp_path,
        [[("Read", '{"path": "README.md"}'), ("Write", NOTEBOOK)], "Here is the notebook as text: ..."],
    )
    try:
        session.set_mode("plan")
        events = [e async for e in session.run("draft the notebook")]
        results = {e.name: e for e in events if isinstance(e, ToolResult)}
        assert "# demo" in results["Read"].output  # its read-only neighbor still ran
        assert results["Write"].output == PLAN_MODE_CALL and not (tmp_path / "mlx_basics.ipynb").exists()
        assert not [e for e in events if isinstance(e, AssistantText)]  # no second reply to dump into
        assert events[-1] == Done(reason="plan_mode")
        assert session.mode == PermissionMode.PLAN
    finally:
        session.close()


async def test_the_cli_says_how_to_build_when_plan_mode_stopped_a_change(tmp_path):
    session = scripted_session(tmp_path, [[("Write", NOTEBOOK)]])
    try:
        session.set_mode("plan")
        console = Console(file=io.StringIO(), width=120)
        await Renderer(console).render(session.run("draft the notebook"))
        shown = console.file.getvalue()
        assert "Plan mode kept it from making changes" in shown and "shift+tab" in shown
    finally:
        session.close()


async def test_a_subagent_in_plan_mode_is_told_the_tool_is_unavailable_and_carries_on(tmp_path):
    """A subagent's job is research for the main agent; there's no user to hand over to."""
    answer = await run_subagent(
        AgentSpec("explorer", "explores"),
        "look around",
        ctx=make_ctx(tmp_path),
        llm=ScriptedLLM([[("Write", '{"path": "x.txt", "content": "x"}')], "found what you need"]),
        tools=default_registry(),
        gate=PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.PLAN),
    )
    assert "found what you need" in answer and not (tmp_path / "x.txt").exists()
