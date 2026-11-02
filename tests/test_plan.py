"""Plan mode's plan: Markdown the agent writes with the Plan tool, which the user reviews,
approves by leaving plan mode, and the build is held to (tools/plan.py, the plan-mode code
in session.py, docs/plans-design.md)."""

import contextlib
import json

import pytest

from conftest import call, make_ctx
from tinyharness.events import Done, PlanSaved, ToolResult
from tinyharness.permissions import PermissionGate, PermissionMode
from tinyharness.testing import scripted_session
from tinyharness.tools.base import ToolError
from tinyharness.tools.plan import PlanTool, plans_dir, revision, title_of
from tinyharness.workspace import Workspace

MARKDOWN = """# Token refresh

Refresh expired tokens without interrupting active sessions.

## Decisions
- Reuse the existing token store
- assumed: retry once after refreshing

## Steps
1. Add refresh to the auth client (`tinyharness/auth/client.py`); check: `uv run pytest -q tests/test_auth.py`
2. Handle refresh failures in the session: fall back to asking the user to log in.

## Out of scope
- Changing provider auth formats
"""
PLAN = {"plan": MARKDOWN}


async def test_the_plan_is_saved_as_markdown_in_a_private_file(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    tool = PlanTool(project)
    out = await call(tool, PLAN, make_ctx(project))
    path = plans_dir(project) / "token-refresh.md"
    assert tool.current is not None and tool.current.path == path and tool.current.title == "Token refresh"
    assert not path.is_relative_to(project)  # never in the project
    assert path.read_text() == MARKDOWN and "reviewing it now" in out


def test_the_title_is_the_first_heading_or_else_the_first_line():
    assert title_of("intro\n\n## Add flags\n") == "Add flags"
    assert title_of("Add flags to wc.py\n1. parse\n") == "Add flags to wc.py"
    assert title_of("") == "Plan"


async def test_the_same_plan_always_gives_the_same_revision(tmp_path):
    """Approval binds to a hash of the whole file; stray whitespace around it doesn't count."""
    first, second = PlanTool(tmp_path / "a"), PlanTool(tmp_path / "b")
    await call(first, PLAN, make_ctx(tmp_path))
    await call(second, {"plan": "\n\n" + MARKDOWN + "  \n"}, make_ctx(tmp_path))
    assert first.current and second.current
    assert first.current.written == second.current.written == revision(MARKDOWN)


async def test_an_empty_plan_is_refused_with_what_to_send(tmp_path):
    with pytest.raises(ToolError, match="'# Title' line"):
        await call(PlanTool(tmp_path), {"plan": "  \n"}, make_ctx(tmp_path))


async def test_a_ready_plan_ends_the_turn_for_the_users_review(tmp_path):
    """The plan is shown and the user decides; the model doesn't get another call to
    repeat or embellish it."""
    session = scripted_session(
        tmp_path / "w",
        [[("Plan", json.dumps(PLAN))], "(never asked: the user is reviewing the plan)"],
        permission_mode="plan",
    )
    try:
        events = [e async for e in session.run("plan token refresh")]
        assert [e.reason for e in events if isinstance(e, Done)] == ["plan_ready"]
        assert [m.role for m in session.history.messages][-1] == "tool"
        assert session.ctx.hand_back == ""  # spent: the next turn runs normally
    finally:
        session.close()


async def test_a_draft_saves_without_ending_the_turn(tmp_path):
    draft = {"plan": MARKDOWN, "ready": False}
    session = scripted_session(
        tmp_path / "w",
        [[("Plan", json.dumps(draft))], [("Plan", json.dumps(PLAN))], "(never asked)"],
        permission_mode="plan",
    )
    try:
        events = [e async for e in session.run("plan token refresh")]
        results = [e.output for e in events if isinstance(e, ToolResult)]
        assert "It's a draft" in results[0] and "reviewing it now" in results[1]
        assert [e.reason for e in events if isinstance(e, Done)] == ["plan_ready"]
    finally:
        session.close()


async def test_a_revision_never_overwrites_what_the_user_edited_by_hand(tmp_path):
    tool, ctx = PlanTool(tmp_path), make_ctx(tmp_path)
    await call(tool, PLAN, ctx)
    assert tool.current is not None
    edited = MARKDOWN.replace("retry once", "retry twice")
    tool.current.path.write_text(edited)
    assert tool.current.edited()
    with pytest.raises(ToolError, match="The user edited the plan") as refused:
        await call(tool, {"plan": MARKDOWN.replace("# Token refresh", "# Token refresh v2")}, ctx)
    assert "retry twice" in str(refused.value) and tool.current.path.read_text() == edited
    # Having been shown the edit, the next call is the revision that includes it.
    await call(tool, {"plan": edited.replace("retry twice", "retry twice, then ask")}, ctx)
    assert "retry twice, then ask" in tool.current.path.read_text()


async def test_another_sessions_plan_with_the_same_name_is_left_alone(tmp_path):
    first, second = PlanTool(tmp_path), PlanTool(tmp_path)
    await call(first, PLAN, make_ctx(tmp_path))
    await call(second, {"plan": "# Token refresh\n\nSomething else.\n"}, make_ctx(tmp_path))
    assert first.current and second.current
    assert second.current.path.name == "token-refresh-2.md"
    assert "Refresh expired tokens" in first.current.path.read_text()


async def test_plan_mode_offers_the_plan_tool_without_asking(tmp_path):
    """Plan mode blocks everything that changes something; the plan is tinyharness's own
    file outside the project, so writing it is part of planning."""
    gate = PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.PLAN)
    decision = await gate.check(PlanTool(tmp_path), PLAN, target_path=None)
    assert decision.allow


async def test_leaving_plan_mode_approves_the_exact_valid_revision(tmp_path):
    session = scripted_session(tmp_path / "w", ["ok"], permission_mode="plan")
    try:
        await call(session.plan_tool, PLAN, make_ctx(session.workspace.root))
        assert session.plan_tool.current is not None
        approved = session.plan_tool.current.path.read_text().replace("retry once", "retry twice")
        session.plan_tool.current.path.write_text(approved)

        notice = session.set_mode("default")

        assert "Plan approved" in notice
        assert session.mode == PermissionMode.DEFAULT
        assert session.plan_tool.current.approved == revision(approved)
        assert not session.plan_tool.current.edited()
    finally:
        session.close()


async def test_a_freeform_hand_edit_can_be_approved_without_trapping_the_user(tmp_path):
    session = scripted_session(tmp_path / "w", ["ok"], permission_mode="plan")
    try:
        await call(session.plan_tool, PLAN, make_ctx(session.workspace.root))
        assert session.plan_tool.current is not None
        freeform = "# Do the small thing\n\nFirst inspect it, then change it carefully.\n"
        session.plan_tool.current.path.write_text(freeform)

        notice = session.set_mode("default")

        assert session.mode == PermissionMode.DEFAULT
        assert "Plan approved" in notice
        assert session.plan_tool.current.approved == revision(freeform)
    finally:
        session.close()


async def test_a_plan_changed_after_approval_blocks_the_first_mutation(tmp_path):
    target = tmp_path / "w" / "built.txt"
    session = scripted_session(
        tmp_path / "w",
        [[("Write", json.dumps({"path": str(target), "content": "surprise"}))], "stopped"],
        permission_mode="plan",
    )
    try:
        await call(session.plan_tool, PLAN, make_ctx(session.workspace.root))
        session.set_mode("default")
        assert session.plan_tool.current is not None
        session.plan_tool.current.path.write_text(
            session.plan_tool.current.path.read_text().replace("retry once", "retry twice")
        )

        events = [event async for event in session.run("build it")]

        refused = [e for e in events if getattr(e, "source", "") == "plan"]
        assert refused and "changed since approval" in refused[0].reason
        assert not target.exists()
    finally:
        session.close()


async def test_an_approved_plan_is_restored_and_enforced_after_resume(tmp_path):
    work = tmp_path / "w"
    first = scripted_session(
        work,
        [[("Plan", json.dumps(PLAN))], "saved"],
        permission_mode="plan",
    )
    try:
        [event async for event in first.run("plan token refresh")]
        first.set_mode("default")
        assert first.plan_tool.current is not None
        plan_path = first.plan_tool.current.path
        approved = first.plan_tool.current.approved
    finally:
        first.close()

    resumed = scripted_session(work, ["continued"], resume="latest")
    try:
        assert resumed.plan_tool.current is not None
        assert resumed.plan_tool.current.path == plan_path
        assert resumed.plan_tool.current.approved == approved
        assert "approved plan" in resumed.ctx.plan_summary

        plan_path.write_text(plan_path.read_text().replace("retry once", "retry twice"))
        write = resumed.tools.get("Write")
        assert write is not None
        error = resumed._plan_guard(write, write.parse({"path": "x", "content": "x"}))
        assert "changed since approval" in error
    finally:
        resumed.close()


async def test_the_plan_is_shown_in_full_when_plan_mode_saves_it(tmp_path):
    session = scripted_session(
        tmp_path / "w",
        [[("Plan", json.dumps(PLAN))], "(never asked: the user is reviewing the plan)"],
        permission_mode="plan",
    )
    try:
        events = [e async for e in session.run("plan token refresh")]
        saved = [e for e in events if isinstance(e, PlanSaved)]
        assert len(saved) == 1 and saved[0].title == "Token refresh"
        assert saved[0].text == (plans_dir(tmp_path / "w") / "token-refresh.md").read_text()
        assert "## Steps" in saved[0].text
    finally:
        session.close()


def test_plan_is_offered_only_in_plan_mode_and_never_to_subagents(tmp_path):
    """Every tool's definition is sent with every request; Plan only matters in plan mode.
    A subagent's plan would replace the one the user is reviewing."""
    from tinyharness.subagents.runner import MAIN_AGENT_ONLY

    session = scripted_session(tmp_path / "w", ["ok"])
    try:
        offered = lambda: {t.name for t in session.tools.available(session.mode)}  # noqa: E731
        assert "Plan" not in offered()
        session.set_mode("plan")
        assert "Plan" in offered()
        assert "Plan" in MAIN_AGENT_ONLY
        assert not session.plan_tool.is_read_only(session.plan_tool.parse(PLAN))
        assert not session.plan_tool.runs_alongside(session.plan_tool.parse(PLAN))
    finally:
        session.close()


async def test_a_plan_mode_turn_that_asks_questions_ends_with_them(tmp_path):
    """Seen live: Sonnet asked its questions and said it would write the plan once they were
    answered; a harness reminder to save the plan then made it write one anyway, built on
    defaults it had just asked about. Nothing may send a question turn back to the model."""
    session = scripted_session(
        tmp_path / "w",
        ["Which Mac do you have? Once you answer, I'll write the plan.", "(should never be asked for)"],
        permission_mode="plan",
    )
    try:
        [e async for e in session.run("let's brainstorm")]
        replies = [m.content for m in session.history.messages if m.role == "assistant"]
        assert replies == ["Which Mac do you have? Once you answer, I'll write the plan."]
        assert session.plan_tool.current is None
    finally:
        session.close()


async def test_clear_ends_the_plan_so_the_new_conversation_isnt_held_to_it(tmp_path):
    """The new conversation can't see the old plan, so a guard on it could only confuse."""
    target = tmp_path / "w" / "new.txt"
    session = scripted_session(
        tmp_path / "w",
        [[("Write", json.dumps({"path": str(target), "content": "ok"}))], "done"],
        permission_mode="plan",
    )
    try:
        await call(session.plan_tool, PLAN, make_ctx(session.workspace.root))
        session.set_mode("default")
        assert session.plan_tool.current is not None
        session.plan_tool.current.path.write_text("edited after approval")
        session.clear()
        assert session.plan_tool.current is None and session.ctx.plan_summary == ""
        events = [event async for event in session.run("write the file")]
        assert not [e for e in events if getattr(e, "source", "") == "plan"]
        assert target.read_text() == "ok"
    finally:
        session.close()


@pytest.mark.parametrize("emptied", ["", None])
async def test_leaving_plan_mode_never_fails_when_the_plan_file_was_emptied(tmp_path, emptied):
    """Leaving plan mode is the user's choice. An emptied or deleted plan file means they're
    building without a plan, so the plan ends instead of keeping them in plan mode."""
    session = scripted_session(tmp_path / "w", ["ok"], permission_mode="plan")
    try:
        await call(session.plan_tool, PLAN, make_ctx(session.workspace.root))
        assert session.plan_tool.current is not None
        path = session.plan_tool.current.path
        path.unlink() if emptied is None else path.write_text(emptied)
        notice = session.set_mode("default")
        assert session.mode == PermissionMode.DEFAULT and "builds without one" in notice
        assert session.plan_tool.current is None and session.plan_tool.build_error() == ""
    finally:
        session.close()


async def test_leaving_plan_mode_says_when_no_plan_was_saved(tmp_path):
    """Seen live: Qwen 4B planned in its replies and never called Plan. Plan mode still
    works (nothing is enforced), and the user is told rather than left to guess."""
    session = scripted_session(tmp_path / "w", ["Plan: 1. add flags 2. test"], permission_mode="plan")
    try:
        assert session.set_mode("default") == ""  # just cycling through: nothing to say
        session.set_mode("plan")
        [e async for e in session.run("plan it")]
        assert "No plan was saved" in session.set_mode("default")
        assert session.plan_tool.build_error() == ""
    finally:
        session.close()


async def test_approving_a_plan_leaves_the_build_to_the_user(tmp_path):
    """Seen live: approving started building in default mode at once, with no chance to
    pick auto first. Leaving plan mode approves; the user picks the mode and says when."""
    session = scripted_session(tmp_path / "w", ["(never asked)"], permission_mode="plan")
    try:
        await call(session.plan_tool, PLAN, make_ctx(session.workspace.root))
        notice = session.set_mode("default")  # Shift+Tab or /mode: approve now, build later
        assert "Plan approved" in notice and "say “build it”" in notice
        assert session.set_mode("auto") == ""  # approval holds across the switch to auto
        assert session.plan_tool.current is not None and session.plan_tool.build_error() == ""
        assert not any(m.role == "assistant" for m in session.history.messages)
    finally:
        session.close()


async def test_a_plan_between_tags_in_the_reply_is_saved_for_review(tmp_path):
    """Seen live: Qwen 4B planned well in its replies but never called Plan. A plan between
    <plan> tags is saved as if it had, and handed over for review the same way."""
    reply = f"Here's the plan.\n\n<plan>\n{MARKDOWN}</plan>\n\nTell me what to change."
    session = scripted_session(tmp_path / "w", [reply], permission_mode="plan")
    try:
        events = [e async for e in session.run("plan token refresh")]
        saved = [e for e in events if isinstance(e, PlanSaved)]
        assert len(saved) == 1 and saved[0].in_reply and saved[0].title == "Token refresh"
        assert [e.reason for e in events if isinstance(e, Done)] == ["plan_ready"]
        assert session.plan_tool.current is not None
        assert session.plan_tool.current.path.read_text() == MARKDOWN
        assert "Plan approved" in session.set_mode("default")
    finally:
        session.close()


GEMMA_FENCED = (
    "Okay, here's a plan to add persistence.\n\n```markdown\n# Stock Tracker Persistence\n\n"
    "**Steps:**\n\n1.  **Read on startup.**\n2.  **Write on exit.**\n```\n\nReady (true)\n"
)
GEMMA_BARE = (
    "Here's the plan:\n\n# Add Persistence\n\n**Steps:**\n\n1. save_data\n2. load_data\n\n"
    "```python\n# a comment, not a title\nimport json\n```\n\nLet me know when you're ready."
)


@pytest.mark.parametrize(
    "reply, plan",
    [
        (
            GEMMA_FENCED,
            "# Stock Tracker Persistence\n\n**Steps:**\n\n1.  **Read on startup.**\n2.  **Write on exit.**",
        ),
        (GEMMA_BARE, GEMMA_BARE[GEMMA_BARE.index("# Add") :].strip()),
        ("Sure.\n```python\n# load the file\nx = 1\n```", ""),  # a code comment isn't a title
        ("Before I plan:\n1. **Format**: notebook or modules?\n2. Which model?", ""),  # questions
        ("<plan>Edit a.py, then run the tests.</plan>", "Edit a.py, then run the tests."),  # any text
        ("# Persist\n\n1. Save.\n\n<plan>done</plan>", "# Persist\n\n1. Save."),  # seen live: qwen3-4b-2507
        (
            "Here's the plan.\n\n**# Plan: Persist it**\n\n**Goal:** keep it.",
            "# Plan: Persist it\n\n**Goal:** keep it.",
        ),
        ("Here's the plan:\n\n**Title: Persist it**\n\n1. Save.", "# Persist it\n\n1. Save."),
    ],
)
def test_a_plan_written_as_markdown_in_the_reply_is_found(reply, plan):
    """Seen live: Gemma 3 12B never called Plan or used the tags; it wrote its plan in the
    reply under a '# Title', fenced or not, and three of its six plans were lost."""
    from tinyharness.session import plan_in_reply

    assert plan_in_reply(reply) == plan


async def test_a_markdown_plan_in_the_reply_is_saved_for_review(tmp_path):
    session = scripted_session(tmp_path / "w", [GEMMA_FENCED], permission_mode="plan")
    try:
        events = [e async for e in session.run("plan persistence")]
        saved = [e for e in events if isinstance(e, PlanSaved)]
        assert len(saved) == 1 and saved[0].title == "Stock Tracker Persistence"
        assert [e.reason for e in events if isinstance(e, Done)] == ["plan_ready"]
    finally:
        session.close()


async def test_tags_count_only_in_plan_mode_and_never_over_a_hand_edit(tmp_path):
    reply = f"<plan>\n{MARKDOWN}</plan>"
    outside = scripted_session(tmp_path / "a", [reply])
    try:
        events = [e async for e in outside.run("go")]
        assert not [e for e in events if isinstance(e, PlanSaved)] and outside.plan_tool.current is None
    finally:
        outside.close()

    session = scripted_session(tmp_path / "b", [reply], permission_mode="plan")
    try:
        await call(session.plan_tool, PLAN, make_ctx(session.workspace.root))
        assert session.plan_tool.current is not None
        session.plan_tool.current.path.write_text("# Mine\n")
        events = [e async for e in session.run("revise it")]
        assert not [e for e in events if isinstance(e, PlanSaved)]
        assert session.plan_tool.current.path.read_text() == "# Mine\n"
    finally:
        session.close()


# ------------------------------------------------------------- the hand-off in the CLI


def _plan_repl(tmp_path, monkeypatch, choice, *, mode="auto", turns=None):
    """A REPL whose model saves a plan, then builds; the hand-off picker answers `choice`."""
    import io

    from rich.console import Console

    from tinyharness.cli import repl as repl_module
    from tinyharness.cli.render import Renderer
    from tinyharness.cli.theme import THEME

    session = scripted_session(
        tmp_path / "w", turns or [[("Plan", json.dumps(PLAN))], "built"], permission_mode=mode
    )
    session.set_mode("plan")
    asked = []

    async def pick(title, options, current=None, **_):
        asked.append((title, [value for value, _ in options], current))
        return choice

    monkeypatch.setattr(repl_module, "pick", pick)
    console = Console(file=io.StringIO(), width=100, theme=THEME)
    return repl_module.Repl(session, console, Renderer(console)), asked


async def test_a_ready_plan_asks_once_whether_to_build_it_in_the_mode_used_before(tmp_path, monkeypatch):
    """Seen live: approving meant pressing shift+tab through default to reach auto, then
    typing "build it". One question now, starting on the mode the user planned from."""
    repl, asked = _plan_repl(tmp_path, monkeypatch, "auto")
    try:
        await repl.handle("plan the token refresh")
        assert asked == [("Build this plan?", ["auto", "default", "plan"], "auto")]
        session = repl.session
        assert session.mode == PermissionMode.AUTO and repl.auto_confirmed
        assert session.plan_tool.current is not None and session.plan_tool.build_error() == ""
        build = [m for m in session.history.messages if m.role == "user"][-1].content
        assert "<approved-plan" in build and build.endswith("Build the approved plan.")
        assert session.history.messages[-1].content == "built"
    finally:
        repl.session.close()


async def test_keeping_on_planning_leaves_the_plan_unapproved_and_builds_nothing(tmp_path, monkeypatch):
    for choice in ("plan", None):  # the option, or Esc
        repl, asked = _plan_repl(tmp_path / str(choice), monkeypatch, choice, mode="default")
        try:
            await repl.handle("plan the token refresh")
            assert asked[0][2] == "default"
            assert repl.session.mode == PermissionMode.PLAN
            assert repl.session.history.messages[-1].role != "user"  # no build turn
            assert "draft plan" in repl.session.plan_tool.build_error()
            assert "Still planning" in repl.console.file.getvalue()
        finally:
            repl.session.close()


async def test_a_turn_without_a_ready_plan_asks_nothing(tmp_path, monkeypatch):
    repl, asked = _plan_repl(tmp_path, monkeypatch, "auto", turns=["Which token store do you use?"])
    try:
        await repl.handle("plan the token refresh")
        assert asked == [] and repl.session.mode == PermissionMode.PLAN
    finally:
        repl.session.close()


async def test_a_plan_left_waiting_for_review_resumes_in_plan_mode(tmp_path):
    """Seen in review: plan, "keep planning", quit, `tinyharness -c`, "build it". The session
    came back in default mode with the draft, every write was refused as unapproved, and
    only Shift+Tab into plan mode and out again approved it."""
    work = tmp_path / "w"
    first = scripted_session(work, [[("Plan", json.dumps(PLAN))], "saved"], permission_mode="plan")
    try:
        [event async for event in first.run("plan token refresh")]
    finally:
        first.close()

    resumed = scripted_session(work, ["continued"], resume="latest", permission_mode="auto")
    try:
        assert resumed.mode == PermissionMode.PLAN and resumed.mode_before_plan == PermissionMode.AUTO
        assert "Plan approved" in resumed.set_mode("auto")
        assert resumed.plan_tool.build_error() == ""
    finally:
        resumed.close()


async def test_the_build_question_comes_back_on_resume(tmp_path, monkeypatch):
    repl, asked = _plan_repl(tmp_path, monkeypatch, "plan", mode="default")
    try:
        await repl.handle("plan the token refresh")  # "keep planning"
    finally:
        repl.session.close()

    from tinyharness.cli import repl as repl_module

    resumed = scripted_session(tmp_path / "w", ["built"], resume="latest")
    monkeypatch.setattr(repl_module, "patch_stdout", lambda **_: contextlib.nullcontext())
    again = repl_module.Repl(resumed, repl.console, repl.renderer)
    again.running = False  # stop after the hand-off: no input box in a test
    try:
        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        with create_pipe_input() as keys, create_app_session(input=keys, output=DummyOutput()):
            await again.loop()  # no console needed (Windows CI)
        assert [title for title, _, _ in asked] == ["Build this plan?", "Build this plan?"]
    finally:
        resumed.close()


def test_the_resume_preview_and_picker_show_what_the_user_typed():
    """Seen in review: the preview's first line was `<plan-mode>You are in plan mode…`."""
    from tinyharness.agent.history import typed_text

    sent = (
        "<plan-mode>You are in plan mode: …</plan-mode>\n\n"
        '<approved-plan path="p.md" revision="ab">\nfollow it\n</approved-plan>\n\n'
        "<session-resumed>\n2 files changed\n</session-resumed>\n\nbuild it"
    )
    assert typed_text(sent) == "build it"
    assert typed_text("what does <plan-mode> mean?") == "what does <plan-mode> mean?"
