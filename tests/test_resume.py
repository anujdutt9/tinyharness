"""Saving conversations and resuming them, including noticing what changed in between."""

import json
import subprocess

import pytest

from conftest import ScriptedLLM
from tinyharness import Session, SessionOptions
from tinyharness.agent.history import Message, ToolCall
from tinyharness.agent.sessionlog import replay
from tinyharness.agent.world import STALE_MARK, UNFINISHED
from tinyharness.paths import move_legacy_state, project_dir, state_dir
from tinyharness.session import list_sessions


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    work = tmp_path / "work"
    work.mkdir()
    return work


def options(work, **kw) -> SessionOptions:
    return SessionOptions(config_path=work.parent / "none.toml", working_dir=work, **kw)


def script(session: Session, *turns) -> ScriptedLLM:
    """Replace the model's replies (the real client stays, for everything else)."""
    llm = ScriptedLLM(list(turns))
    session.llm.stream = llm.stream
    return llm


async def turn(session: Session, prompt: str, *turns) -> list:
    script(session, *turns)
    return [e async for e in session.run(prompt)]


def contents(session: Session) -> list[tuple[str, str | None]]:
    return [(m.role, m.content) for m in session.history.messages[1:]]  # without the system prompt


# ---- round trip ----


async def test_a_resumed_conversation_is_the_same_conversation(work):
    (work / "notes.txt").write_text(("x" * 50 + "\n") * 300)  # longer than the audit log keeps
    first = Session(options(work, todo_list=True))
    await turn(first, "read notes", [("Read", '{"path": "notes.txt"}')], "It's all x.")
    await turn(first, "and now?", [("TodoWrite", json.dumps({"todos": [TODO]}))], "Planned.")
    before, todos = contents(first), first.ctx.todos
    first.close()

    resumed = Session(options(work, resume="latest", todo_list=True))
    try:
        assert resumed.id == first.id
        assert contents(resumed) == before
        assert any(len(c or "") > 10_000 for _, c in before)  # lossless, not the 4000-char audit copy
        assert resumed.ctx.todos == todos
        assert str((work / "notes.txt").resolve()) in resumed.ctx.read_paths  # unchanged: may be edited
        assert resumed.resume_report is not None and not resumed.resume_report.anything_changed

        llm = script(resumed, "ok")
        [_ async for _ in resumed.run("continue")]
        assert llm.requests[0][-1].content == "continue"  # nothing changed: no note
    finally:
        resumed.close()


TODO = {"content": "Check", "status": "in_progress", "activeForm": "Checking"}


async def test_compaction_survives_a_resume(work):
    first = Session(options(work))
    for i in range(3):
        await turn(first, f"question {i}", f"answer {i}")
    first.compactor.keep_tail = 2
    assert await first.compactor.summarize(first.history)
    first._sync()
    before = contents(first)
    first.close()

    resumed = Session(options(work, resume=first.id))
    try:
        assert contents(resumed)[0][0] == "system" and "summary" in contents(resumed)[0][1]
        assert contents(resumed) == before
    finally:
        resumed.close()


async def test_clearing_is_saved_too(work):
    first = Session(options(work))
    await turn(first, "old topic", "old answer")
    first.clear()
    await turn(first, "new topic", "new answer")
    first.close()
    resumed = Session(options(work, resume="latest"))
    try:
        assert [c for _, c in contents(resumed)] == ["new topic", "new answer"]
    finally:
        resumed.close()


def test_a_log_cut_short_by_a_crash_still_replays(work):
    records = [
        {"event": "msg", "message": Message(role="user", content="hi").to_record()},
        {"event": "msg", "message": Message(role="assistant", content="hello").to_record()},
    ]
    path = state_dir(work) / "sessions" / "1-crash.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + '\n{"event": "msg", "mess')
    assert [
        m.content for m in replay(list(map(json.loads, path.read_text().splitlines()[:2]))).history.messages
    ] == [
        "hi",
        "hello",
    ]
    assert list_sessions(options(work))[0].id == "1-crash"


# ---- what changed while away ----


async def test_changed_files_are_flagged_and_their_old_contents_stubbed(work):
    (work / "a.py").write_text("A = 1\n")
    (work / "b.py").write_text("B = 1\n")
    first = Session(options(work))
    await turn(first, "read both", [("Read", '{"path": "a.py"}'), ("Read", '{"path": "b.py"}')], "Read them.")
    first.close()

    (work / "a.py").write_text("A = 2\n")  # changed by someone else
    resumed = Session(options(work, resume="latest"))
    try:
        report = resumed.resume_report
        assert report.changed == ["a.py"] and report.stale_results == 1
        results = [m.content for m in resumed.history.messages if m.role == "tool"]
        assert results[0].startswith(STALE_MARK) and "A = 1" not in results[0]
        assert "B = 1" in results[1]  # unchanged: kept as is
        assert str((work / "a.py").resolve()) not in resumed.ctx.read_paths  # must Read before editing

        llm = script(resumed, "ok")
        [_ async for _ in resumed.run("go on")]
        prompt = llm.requests[0][-1].content
        assert prompt.startswith("<session-resumed>") and "a.py" in prompt and prompt.endswith("go on")
    finally:
        resumed.close()

    again = Session(options(work, resume="latest"))  # reported once is enough; the stub was saved
    try:
        assert not again.resume_report.anything_changed
        assert next(m for m in again.history.messages if m.role == "tool").content.startswith(STALE_MARK)
        assert str((work / "a.py").resolve()) not in again.ctx.read_paths  # still has to read it again
    finally:
        again.close()

    (work / "a.py").write_text("A = 3\n")  # a new change is news again
    third = Session(options(work, resume="latest"))
    try:
        assert third.resume_report.changed == ["a.py"]
    finally:
        third.close()


async def test_the_agents_own_edits_are_not_changes(work):
    (work / "a.py").write_text("A = 1\n")
    first = Session(options(work, permission_mode="default"))
    edit = json.dumps({"path": "a.py", "old_string": "A = 1", "new_string": "A = 2"})
    await turn(first, "bump A", [("Read", '{"path": "a.py"}')], [("Edit", edit)], "Done.")
    first.close()
    assert (work / "a.py").read_text() == "A = 2\n"
    resumed = Session(options(work, resume="latest"))
    try:
        assert resumed.resume_report.changed == []
    finally:
        resumed.close()


async def test_deleted_files_are_reported(work):
    (work / "gone.txt").write_text("bye")
    first = Session(options(work))
    await turn(first, "read it", [("Read", '{"path": "gone.txt"}')], "ok")
    first.close()
    (work / "gone.txt").unlink()
    resumed = Session(options(work, resume="latest"))
    try:
        assert resumed.resume_report.missing == ["gone.txt"]
        assert "No longer exist: gone.txt" in resumed.resume_report.note()
    finally:
        resumed.close()


async def test_a_tool_call_without_a_result_is_flagged(work):
    first = Session(options(work))
    await turn(first, "start", "ok")
    first.history.append(
        Message(role="assistant", tool_calls=[ToolCall("c1", "Bash", '{"command": "make"}')])
    )
    first._sync()  # the process died here: no result, no repair
    first.transcript.close()

    resumed = Session(options(work, resume="latest"))
    try:
        assert resumed.resume_report.unfinished == 1
        assert resumed.history.messages[-1].content == UNFINISHED
        assert "may or may not have run" in resumed.resume_report.note()
    finally:
        resumed.close()


def git(work, *args):
    subprocess.run(["git", "-C", str(work), *args], check=True, capture_output=True)


@pytest.fixture
def repo(work):
    git(work, "init", "-q", "-b", "main")
    git(work, "config", "user.email", "t@example.com")
    git(work, "config", "user.name", "t")
    git(work, "commit", "-q", "--allow-empty", "-m", "one")
    return work


async def test_new_commits_are_reported(repo):
    first = Session(options(repo))
    await turn(first, "hi", "hello")
    first.close()
    git(repo, "commit", "-q", "--allow-empty", "-m", "two")
    git(repo, "commit", "-q", "--allow-empty", "-m", "three")
    resumed = Session(options(repo, resume="latest"))
    try:
        assert resumed.resume_report.commits_since == 2
        assert "2 new commits on main" in resumed.resume_report.note()
    finally:
        resumed.close()


async def test_a_branch_switch_is_reported(repo):
    first = Session(options(repo))
    await turn(first, "hi", "hello")
    first.close()
    git(repo, "checkout", "-q", "-b", "feature")
    resumed = Session(options(repo, resume="latest"))
    try:
        assert "branch main → feature" in resumed.resume_report.note()
    finally:
        resumed.close()


async def test_changed_instructions_are_reported_and_used(work):
    (work / "AGENTS.md").write_text("Use tabs.")
    first = Session(options(work))
    await turn(first, "hi", "hello")
    first.close()
    (work / "AGENTS.md").write_text("Use spaces.")
    resumed = Session(options(work, resume="latest"))
    try:
        assert resumed.resume_report.instructions_changed
        assert "Use spaces." in resumed.history.messages[0].content  # a fresh system prompt
    finally:
        resumed.close()


# ---- which model, which conversation ----


async def test_resuming_uses_the_conversations_model_unless_told_otherwise(work, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    first = Session(options(work))
    await turn(first, "hi", "hello")
    first.set_model("anthropic:claude-haiku-4-5")
    first.close()

    resumed = Session(options(work, resume="latest"))
    assert (resumed.config.provider, resumed.config.model) == ("anthropic", "claude-haiku-4-5")
    resumed.close()
    chosen = Session(options(work, resume="latest", model="openai:gpt-6-luna"))
    assert chosen.config.model == "gpt-6-luna"
    chosen.close()
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    no_key = Session(options(work, resume="latest"))
    assert no_key.config.provider == "openai"  # no key for it any more: stay on what's available
    no_key.close()


async def test_listing_and_errors(work):
    with pytest.raises(ValueError, match="no saved conversation to resume"):
        Session(options(work, resume="latest"))
    with pytest.raises(ValueError, match="no saved conversation 'nope'"):
        Session(options(work, resume="nope"))
    Session(options(work)).close()  # nothing said: not listed
    for prompt in ("first thing", "second thing"):
        s = Session(options(work))
        await turn(s, prompt, "ok")
        s.close()
    sessions = list_sessions(options(work))
    assert [s.title for s in sessions] == ["second thing", "first thing"]
    assert sessions[0].messages == 3  # system, user, assistant


# ---- where it's all kept ----


async def test_nothing_is_written_into_the_project(work):
    subprocess.run(["git", "-C", str(work), "init", "-q"], check=True)
    (work / "app.py").write_text("x = 1\n")
    session = Session(options(work))
    await turn(session, "read it", [("Read", '{"path": "app.py"}')], "ok")
    session.close()
    assert sorted(p.name for p in work.iterdir()) == [".git", "app.py"]
    assert list(state_dir(work).joinpath("sessions").glob("*.jsonl"))


def test_logs_from_older_versions_move_out_of_the_project(work):
    old = project_dir(work)
    (old / "sessions").mkdir(parents=True)
    (old / "sessions" / "1-old.jsonl").write_text(
        json.dumps({"event": "msg", "message": Message(role="user", content="from before").to_record()})
        + "\n"
    )
    (old / "tmp" / "stale").mkdir(parents=True)
    Session(options(work)).close()
    assert not old.exists()  # nothing else was in it
    assert [s.title for s in list_sessions(options(work))] == ["from before"]


def test_your_project_config_stays_put(work):
    old = project_dir(work)
    (old / "sessions").mkdir(parents=True)
    (old / "agents").mkdir()
    Session(options(work)).close()
    assert sorted(p.name for p in old.iterdir()) == ["agents"]


def test_long_project_paths_get_short_unique_names(tmp_path):
    a, b = tmp_path / ("x" * 150) / "one", tmp_path / ("x" * 150) / "two"
    assert state_dir(a) != state_dir(b) and len(state_dir(a).name) <= 111


def test_project_state_directory_has_a_readable_collision_resistant_name(tmp_path):
    first, second = tmp_path / "one" / "app", tmp_path / "two" / "app"
    assert state_dir(first).name.startswith("app-")
    assert len(state_dir(first).name.removeprefix("app-")) == 16
    assert state_dir(first) != state_dir(second)
    assert str(tmp_path).replace("/", "-") not in state_dir(first).name


def test_path_named_project_state_moves_to_the_readable_name(tmp_path):
    from tinyharness.paths import _path_named_state_dir

    work = tmp_path / "my app"
    old = _path_named_state_dir(work)
    (old / "sessions").mkdir(parents=True)
    (old / "sessions" / "saved.jsonl").write_text("saved\n")

    move_legacy_state(work)

    assert not old.exists()
    assert (state_dir(work) / "sessions" / "saved.jsonl").read_text() == "saved\n"


# ---- /clear starts a new conversation ----


async def test_clear_keeps_the_old_conversation_resumable(work):
    """The /clear command starts a new conversation without losing the old one."""
    first = Session(options(work))
    await turn(first, "build the deck", "Built it.")
    old_id, before = first.id, contents(first)
    first.clear()
    assert first.id != old_id and contents(first) == []
    await turn(first, "something else", "Sure.")
    first.close()

    latest = Session(options(work, resume="latest"))
    try:
        assert [c for _, c in contents(latest)] == ["something else", "Sure."]
    finally:
        latest.close()
    old = Session(options(work, resume=old_id))
    try:
        assert contents(old) == before
    finally:
        old.close()


async def test_continue_skips_a_conversation_emptied_by_clear(work):
    """Logs from before: /clear emptied the conversation in place, and -c then found
    "nothing to resume" instead of the conversation before it."""
    kept = Session(options(work))
    await turn(kept, "keep me", "Kept.")
    kept_id = kept.id
    kept.close()
    emptied = Session(options(work))
    await turn(emptied, "clear me", "Ok.")
    emptied.history.reset()  # what /clear used to do
    emptied._sync()
    emptied.close()

    resumed = Session(options(work, resume="latest"))
    try:
        assert resumed.id == kept_id
    finally:
        resumed.close()


def test_clearing_an_empty_conversation_makes_no_new_one(work):
    session = Session(options(work))
    try:
        before = session.id
        session.clear()
        assert session.id == before
    finally:
        session.close()


async def test_after_resume_a_file_changed_by_the_user_must_be_read_again(work):
    """Resume restored which files were read but not their versions, so a Write could
    overwrite what the user changed after resuming without any error."""
    (work / "plan.md").write_text("v1\n")
    first = Session(options(work))
    await turn(first, "read it", [("Read", '{"path": "plan.md"}')], "ok")
    first.close()

    resumed = Session(options(work, resume="latest", permission_mode="default"))
    try:
        (work / "plan.md").write_text("v2 from the user\n")  # after resuming
        events = await turn(
            resumed, "rewrite it", [("Write", '{"path": "plan.md", "content": "agent"}')], "done"
        )
        from tinyharness.events import ToolResult

        result = next(e for e in events if isinstance(e, ToolResult))
        assert result.is_error and "changed since you last read it" in result.output
        assert (work / "plan.md").read_text() == "v2 from the user\n"
    finally:
        resumed.close()
