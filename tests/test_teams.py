"""Subagents in parallel, their progress, and agent teams."""

import asyncio
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

from conftest import ScriptedLLM, make_ctx
from tinyharness import MissingAPIKeyError, Session, SessionOptions, Tool, ToolInput
from tinyharness.agent.compaction import Compactor
from tinyharness.agent.history import History, Message
from tinyharness.agent.loop import LoopDeps, run_agent_loop
from tinyharness.agent.transcript import Transcript
from tinyharness.cli.render import Renderer
from tinyharness.events import SubagentUpdate, ToolResult
from tinyharness.permissions import Decision, PermissionGate, PermissionMode, PermissionRequest
from tinyharness.subagents import AgentSpec, parse_agent_md
from tinyharness.subagents.runner import run_subagent
from tinyharness.tools.base import ToolError
from tinyharness.tools.registry import ToolRegistry, default_registry
from tinyharness.tools.team import Board, TaskState, TeamTool, report, run_team
from tinyharness.workspace import Workspace

READER = AgentSpec("reader", "reads", tools=["Read"])
WRITER = AgentSpec("writer", "writes", tools=["Read", "Write"])
AGENTS = {"reader": READER, "writer": WRITER}


def is_read_only(spec: AgentSpec) -> bool:
    return "Write" not in spec.tools


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = tmp_path / "config.toml"
    config.write_text('[extensions]\nenabled = ["agents"]\n')
    s = Session(SessionOptions(config_path=config, working_dir=tmp_path / "work"))
    yield s
    s.close()


# ---- which subagents may run side by side ----


def test_every_agent_runs_alongside_others(session):
    """Look-only agents and writing ones alike: a writer's writing calls take turns on the
    session's lock, not its whole run."""
    agent = session.tools.get("Agent")
    assert agent.is_read_only(agent.parse({"subagent_type": "explore", "prompt": "x"}))
    assert not agent.is_read_only(agent.parse({"subagent_type": "general-purpose", "prompt": "x"}))
    for name in ("explore", "general-purpose"):
        assert agent.runs_alongside(agent.parse({"subagent_type": name, "prompt": "x"}))


async def test_subagents_keep_plan_mode(session, monkeypatch):
    seen = []

    async def fake_run(spec, task, *, gate, **kwargs):
        seen.append(gate.mode)
        return "ok"

    monkeypatch.setattr("tinyharness.session.run_subagent", fake_run)
    spec = session.agents["general-purpose"]
    await session._spawn_subagent(spec, "x", session.ctx)
    session.set_mode("plan")
    await session._spawn_subagent(spec, "x", session.ctx)
    assert seen == [PermissionMode.DEFAULT, PermissionMode.PLAN]


async def test_permission_questions_are_asked_one_at_a_time(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    open_questions, most = 0, 0

    async def ask(request):
        nonlocal open_questions, most
        open_questions += 1
        most = max(most, open_questions)
        await asyncio.sleep(0.02)
        open_questions -= 1
        return Decision(True)

    s = Session(
        SessionOptions(config_path=tmp_path / "none.toml", working_dir=tmp_path, permission_callback=ask)
    )
    request = PermissionRequest("WebFetch", {}, "WebFetch(x)", None, "")
    await asyncio.gather(*(s.gate.callback(request) for _ in range(3)))
    assert most == 1
    s.close()


# ---- subagent progress ----


async def test_a_subagent_reports_its_progress(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    updates = []
    ctx = make_ctx(tmp_path)
    ctx.notify = updates.append
    llm = ScriptedLLM([[("Read", '{"path": "notes.txt"}')], "it says hello"])
    gate = PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.BYPASS)
    answer = await run_subagent(
        READER, "read notes", ctx=ctx, llm=llm, tools=default_registry(), gate=gate, label="notes"
    )
    assert answer == "it says hello"
    # writing the Read call, running it, waiting for the model's answer, done
    assert [(u.status, u.phase) for u in updates] == [
        ("started", "tool"),
        ("working", "writing"),
        ("working", "tool"),
        ("working", "waiting"),
        ("done", "tool"),
    ]
    assert updates[1].tool == "Read" and updates[1].detail == "notes.txt"
    assert updates[2].detail.startswith("Read(") and updates[-1].detail == "1 tool call"
    assert {u.label for u in updates} == {"notes"}
    assert {u.model for u in updates} == {"fake"}  # the client's model, since none was given


def test_the_renderer_shows_running_subagents_then_a_line_when_done():
    console = Console(width=100, record=True)
    renderer = Renderer(console)
    renderer.handle(
        SubagentUpdate(
            "r1",
            "researcher",
            "research-a: Laptop A",
            "working",
            "WebSearch(aero)",
            2,
            5.0,
            model="anthropic/claude-haiku-4.5",
        )
    )
    live = Console(width=100, record=True)
    live.print(renderer)
    running = live.export_text()
    assert "researcher  research-a: Laptop A" in running
    assert "claude-haiku-4.5 · WebSearch(aero) · 2 tools · 5s" in running
    renderer.handle(
        SubagentUpdate(
            "r1",
            "researcher",
            "research-a: Laptop A",
            "done",
            "4 tool calls",
            4,
            21.0,
            model="claude-haiku-4-5",
        )
    )
    assert not renderer._agents
    renderer.handle(ToolResult("Team", "report"))  # finished teammates fold into their call's row
    done = console.export_text()
    assert "✓ Team" in done and "1 agent · 4 tools" in done
    assert "research-a: Laptop A" not in done  # no line each once they're done


# ---- teammates' messages reach a running agent ----


async def test_messages_arrive_between_steps(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    inbox = [['<teammate-message from="b">\nuse port 8080\n</teammate-message>']]
    llm = ScriptedLLM([[("Read", '{"path": "a.txt"}')], "done"])
    history = History([Message(role="system", content="test")])
    with Transcript(tmp_path, "t") as transcript:
        deps = LoopDeps(
            llm=llm,
            tools=ToolRegistry(default_registry()._tools.values()),
            gate=PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.BYPASS),
            compactor=Compactor(llm),
            transcript=transcript,
            ctx=make_ctx(tmp_path),
            inbox=lambda: inbox.pop() if inbox else [],
        )
        [e async for e in run_agent_loop("go", history, deps)]
    assert '<teammate-message from="b">\nuse port 8080\n</teammate-message>' in [
        m.content for m in llm.requests[0]
    ]


# ---- teams ----


def team_tool(spawner=None):
    return TeamTool(AGENTS, spawner or (lambda *a, **k: None), is_read_only)


def plan(*tasks):
    return {
        "goal": "compare",
        "tasks": [
            dict(zip(("id", "agent", "depends_on"), t, strict=False))
            | {"title": t[0], "prompt": f"do {t[0]}"}
            for t in tasks
        ],
    }


@pytest.mark.parametrize(
    ("tasks", "error"),
    [
        ((("a", "reader", []), ("a", "reader", [])), "duplicate"),
        ((("a", "reader", ["zzz"]), ("b", "reader", [])), "unknown task"),
        ((("a", "reader", ["b"]), ("b", "reader", ["a"])), "cycle"),
        ((("a", "nobody", []), ("b", "reader", [])), "unknown agent"),
    ],
)
def test_a_bad_team_plan_is_explained(tasks, error):
    tool = team_tool()
    with pytest.raises(ToolError, match=error):
        tool._board(tool.parse(plan(*tasks)))


def test_a_team_is_read_only_only_if_every_member_is():
    tool = team_tool()
    assert tool.is_read_only(tool.parse(plan(("a", "reader", []), ("b", "reader", []))))
    assert not tool.is_read_only(tool.parse(plan(("a", "reader", []), ("b", "writer", []))))


async def test_a_team_runs_in_dependency_order_with_independent_tasks_in_parallel(tmp_path):
    running, most_readers, writers_at_once, most_writers, started = set(), 0, 0, 0, []
    prompts = {}

    async def spawner(spec, prompt, ctx, /, *, label, extra_tools, inbox):
        nonlocal most_readers, writers_at_once, most_writers
        task_id = label.split(":")[0]
        started.append(task_id)
        prompts[task_id] = prompt
        assert {t.name for t in extra_tools} == {"TeamBoard", "TeamMessage"}
        if spec is WRITER:
            writers_at_once += 1
            most_writers = max(most_writers, writers_at_once)
        running.add(task_id)
        most_readers = max(most_readers, len([t for t in running if t.startswith("r")]))
        await asyncio.sleep(0.03)
        running.discard(task_id)
        if spec is WRITER:
            writers_at_once -= 1
        return f"result of {task_id}"

    tool = team_tool(spawner)
    args = tool.parse(
        plan(
            ("r1", "reader", []),
            ("r2", "reader", []),
            ("r3", "reader", []),
            ("w1", "writer", ["r1"]),
            ("w2", "writer", ["r2"]),
            ("sum", "reader", ["w1", "w2", "r3"]),
        )
    )
    out = await tool.run(args, make_ctx(tmp_path))
    assert most_readers == 3  # independent readers ran together
    assert most_writers == 2  # writers too: only their writing calls take turns
    assert started[-1] == "sum"
    assert "result of w1" in prompts["sum"] and "result of r3" in prompts["sum"]
    assert out.startswith("Team finished (6 done)")


async def test_a_failed_teammate_is_retried_once_then_its_dependents_get_what_finished(tmp_path):
    attempts, prompts = [], {}

    async def spawner(spec, prompt, ctx, /, *, label, extra_tools, inbox):
        task = label.split(":")[0]
        attempts.append(task)
        prompts.setdefault(task, []).append(prompt)
        if task == "a":
            raise RuntimeError("site was down")
        return f"result of {task}"

    tool = team_tool(spawner)
    out = await tool.run(
        tool.parse(plan(("a", "reader", []), ("c", "reader", []), ("sum", "reader", ["a", "c"]))),
        make_ctx(tmp_path),
    )
    assert attempts.count("a") == 2  # started once more
    assert "previous attempt at this task failed (RuntimeError: site was down)" in prompts["a"][1]
    assert "## a: a · reader · failed" in out and "site was down" in out
    assert "## sum: sum · reader · done" in out  # one failed analyst doesn't sink the summary
    assert "result of c" in prompts["sum"][0] and "a did not finish" in prompts["sum"][0]


async def test_a_task_is_skipped_only_if_none_of_its_dependencies_finished(tmp_path):
    async def spawner(spec, prompt, ctx, /, *, label, extra_tools, inbox):
        if label.startswith("a:"):
            raise RuntimeError("site was down")
        return "fine"

    tool = team_tool(spawner)
    out = await tool.run(
        tool.parse(plan(("a", "reader", []), ("b", "reader", ["a"]), ("c", "reader", []))), make_ctx(tmp_path)
    )
    assert "## a: a · reader · failed" in out
    assert "## b: b · reader · skipped" in out
    assert "## c: c · reader · done" in out


def test_messages_reach_running_teammates_only():
    board = Board(
        "g",
        {
            "a": TaskState("a", "A", "p", "reader", [], status="running"),
            "b": TaskState("b", "B", "p", "reader", [], status="waiting"),
            "c": TaskState("c", "C", "p", "reader", [], status="done"),
        },
    )
    assert board.post("a", "all", "API is at /v2") == "sent to b"
    assert board.take_inbox("b") == ['<teammate-message from="a">\nAPI is at /v2\n</teammate-message>']
    assert board.take_inbox("b") == []
    with pytest.raises(ToolError, match="no task"):
        board.post("a", "zzz", "hi")
    assert "a → all: API is at /v2" in board.render()
    assert "a → all: API is at /v2" in report(board)


async def test_cancelling_a_team_stops_every_teammate(tmp_path):
    cancelled = []

    async def spawner(spec, prompt, ctx, /, *, label, extra_tools, inbox):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(label)
            raise
        return ""

    tool = team_tool(spawner)
    board = tool._board(tool.parse(plan(("a", "reader", []), ("b", "reader", []))))
    job = asyncio.create_task(run_team(board, AGENTS, spawner, make_ctx(tmp_path)))
    await asyncio.sleep(0.02)
    job.cancel()
    with pytest.raises(asyncio.CancelledError):
        await job
    assert sorted(cancelled) == ["a: a", "b: b"]


async def test_auto_mode_is_confirmed_once_per_session(session, monkeypatch):
    import io

    from tinyharness.cli import repl as repl_module
    from tinyharness.cli.repl import Repl

    answers = ["no", "yes"]
    asked = []

    async def pick(title, options, current=None, **kwargs):
        asked.append(title)
        return answers.pop(0)

    monkeypatch.setattr(repl_module, "pick", pick)
    console = Console(file=io.StringIO())
    repl = Repl(session, console, Renderer(console))
    assert not await repl.enter_auto_mode()
    assert session.mode == PermissionMode.DEFAULT  # "no" leaves it alone
    assert await repl.enter_auto_mode()
    assert session.mode == PermissionMode.AUTO
    session.set_mode("ask")
    assert await repl.enter_auto_mode()
    assert len(asked) == 2  # not asked again once agreed


async def test_saying_no_to_auto_mode_does_not_trap_shift_tab(session, monkeypatch):
    import io

    from tinyharness.cli import repl as repl_module
    from tinyharness.cli.repl import Repl

    asked = []

    async def pick(title, options, current=None, **kwargs):
        asked.append(title)
        return None  # Esc, same as "no"

    monkeypatch.setattr(repl_module, "pick", pick)
    console = Console(file=io.StringIO())
    repl = Repl(session, console, Renderer(console))
    session.set_mode("default")
    assert repl.next_mode() == PermissionMode.AUTO  # the next press would ask
    await repl.shift_tab_to_auto()
    assert session.mode == PermissionMode.DEFAULT  # "no" keeps the mode
    cycle = []
    for _ in range(4):  # the presses after that go round without auto, and never ask again
        session.set_mode(repl.next_mode())
        cycle.append(session.mode)
    assert cycle == [
        PermissionMode.PLAN,
        PermissionMode.ASK,
        PermissionMode.DEFAULT,
        PermissionMode.PLAN,
    ]
    assert len(asked) == 1


# ---- agents defined in code (SDK) ----


def test_agents_can_be_passed_in_code(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    reviewer = AgentSpec(
        name="reviewer",
        description="Reviews a diff for bugs.",
        tools=["Read", "Grep"],
        permission_mode="default",
    )
    explore = AgentSpec(name="explore", description="My own explorer.", tools=["Read"])
    config = tmp_path / "config.toml"
    config.write_text('[extensions]\nenabled = ["agents"]\n')
    options = SessionOptions(config_path=config, working_dir=tmp_path / "work", agents=[reviewer, explore])
    session = Session(options)
    try:
        assert session.agents["reviewer"] is reviewer
        assert reviewer.permission_mode == PermissionMode.DEFAULT
        assert session.agents["explore"].description == "My own explorer."  # replaces the built-in
        assert "reviewer" in session.tools.get("Agent").description
    finally:
        session.close()


def test_code_agents_work_without_file_agents(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    only = AgentSpec(name="only", description="The only one.")
    options = SessionOptions(
        config_path=tmp_path / "none.toml",
        working_dir=tmp_path / "work",
        enable_subagents=False,
        agents=[only],
    )
    session = Session(options)
    try:
        assert list(session.agents) == ["only"]
    finally:
        session.close()


def test_sdk_exports():
    import tinyharness

    assert tinyharness.AgentSpec is AgentSpec
    assert tinyharness.SubagentUpdate is SubagentUpdate


# ---- which model a subagent runs on ----


def test_subagent_model_settings(session, monkeypatch):
    assert session.config.provider == "openai"
    assert session.llm_for(None) == (session.llm, None)
    assert session.llm_for("inherit") == (session.llm, None)
    assert session.llm_for("small") == (session.llm, session.config.small_model)
    assert session.llm_for("gpt-cheap") == (session.llm, "gpt-cheap")  # same provider
    assert session.llm_for("openai:gpt-cheap") == (session.llm, "gpt-cheap")

    with pytest.raises(MissingAPIKeyError):
        session.llm_for("anthropic:claude-haiku-4-5")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    client, model = session.llm_for("anthropic:claude-haiku-4-5")
    assert client.provider.name == "anthropic" and client.model == "claude-haiku-4-5" and model is None
    client, _ = session.llm_for("anthropic:")
    assert client.model == session.config.providers["anthropic"].default_model


async def test_subagents_run_on_their_own_model(session, monkeypatch):
    seen = []

    async def fake_run(spec, task, *, llm, model, **kwargs):
        seen.append((llm.provider.name, model))
        return "ok"

    monkeypatch.setattr("tinyharness.session.run_subagent", fake_run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    for model in (None, "small", "anthropic:claude-haiku-4-5"):
        await session._spawn_subagent(AgentSpec("a", "b", model=model), "x", session.ctx)
    assert seen == [("openai", None), ("openai", session.config.small_model), ("anthropic", None)]


async def test_a_subagent_without_a_key_for_its_model_fails_as_a_tool_error(session):
    with pytest.raises(ToolError, match="can't start the a agent: No API key for Anthropic"):
        await session._spawn_subagent(AgentSpec("a", "b", model="anthropic:claude-x"), "x", session.ctx)


# ---- tools ----


def test_agents_without_a_tool_list_get_every_tool(tmp_path, session):
    (tmp_path / "helper.md").write_text("---\nname: helper\ndescription: Helps.\n---\nHelp.")
    spec = parse_agent_md(tmp_path / "helper.md")
    assert spec.tools is None and AgentSpec("x", "y").tools is None
    is_read_only = session.tools.get("Agent").agent_is_read_only
    assert not is_read_only(spec)  # it can write, like the session
    assert is_read_only(AgentSpec("none", "no tools", tools=[]))


async def test_agents_without_a_tool_list_get_every_tool_but_spawning_and_todos(tmp_path):
    offered = []

    class Recorder(ScriptedLLM):
        async def stream(self, messages, tools, **kwargs):
            offered.append({t.name for t in tools})
            async for chunk in super().stream(messages, tools, **kwargs):
                yield chunk

    registry = default_registry()
    registry.register(TeamTool({}, None, lambda spec: False))
    gate = PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.BYPASS)
    await run_subagent(
        AgentSpec("all", "everything"),
        "hi",
        ctx=make_ctx(tmp_path),
        llm=Recorder(["done"]),
        tools=registry,
        gate=gate,
    )
    assert offered[0] == set(registry.names()) - {"Team", "TodoWrite"}


def test_custom_tools_can_be_passed_in_options(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    clock = ClockTool()
    options = SessionOptions(config_path=tmp_path / "none.toml", working_dir=tmp_path / "work", tools=[clock])
    session = Session(options)
    try:
        assert session.tools.get("Clock") is clock
    finally:
        session.close()


class ClockInput(ToolInput):
    pass


class ClockTool(Tool[ClockInput]):
    name = "Clock"
    description = "The time."
    Input = ClockInput
    read_only = True

    async def run(self, args, ctx):
        return "noon"


def _live(renderer) -> str:
    live = Console(width=120, record=True)
    live.print(renderer)
    return live.export_text()


def test_a_running_subagents_clock_keeps_ticking_between_updates(monkeypatch):
    """Updates only come when the subagent does something; a long model call mustn't
    look frozen."""
    import tinyharness.cli.render as render

    now = [1000.0]
    monkeypatch.setattr(render.time, "monotonic", lambda: now[0])
    renderer = Renderer(Console(width=120, record=True))
    renderer.handle(
        SubagentUpdate("r1", "general-purpose", "verify the deck", "working", "Grep(svg)", 4, 251.0)
    )
    assert "4 tools · 4m 11s" in _live(renderer)
    now[0] += 180
    assert "4 tools · 7m 11s" in _live(renderer)


@pytest.mark.parametrize(
    ("phase", "tool", "detail", "shown"),
    [
        ("thinking", "", "", "Thinking"),
        ("waiting", "", "", "Working"),
        ("writing", "Write", "deck.html", "Writing deck.html"),
        ("tool", "", "Grep(svg)", "Grep(svg)"),
    ],
)
def test_a_running_subagent_says_what_it_is_doing(phase, tool, detail, shown):
    renderer = Renderer(Console(width=120, record=True))
    update = SubagentUpdate("r1", "general-purpose", "verify", "working", detail, 4, 3.0, "m", phase, tool)
    renderer.handle(update)
    assert f"m · {shown} · 4 tools" in _live(renderer)


async def test_a_subagent_cannot_touch_the_main_todo_list(tmp_path):
    """The list is the main agent's plan on the user's screen; a subagent's TodoWrite
    would replace it."""
    ctx = make_ctx(tmp_path)
    ctx.todos = [{"content": "Ship it", "activeForm": "Shipping it", "status": "in_progress"}]
    theirs = '{"todos": [{"content": "Check", "activeForm": "Checking", "status": "in_progress"}]}'
    llm = ScriptedLLM([[("TodoWrite", theirs)], "checked"])
    gate = PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.BYPASS)
    anything = AgentSpec("helper", "does anything")  # no tools list: every tool it may have
    await run_subagent(anything, "check", ctx=ctx, llm=llm, tools=default_registry(), gate=gate)
    assert [t["content"] for t in ctx.todos] == ["Ship it"]


# ---- what a running subagent has been doing ----


async def test_a_subagent_reports_its_recent_calls_and_its_log(tmp_path):
    (tmp_path / "notes.txt").write_text("hello\n")
    updates = []
    ctx = make_ctx(tmp_path)
    ctx.notify = updates.append
    llm = ScriptedLLM(
        [[("Read", '{"path": "notes.txt"}')], [("Read", '{"path": "missing.txt"}')], "it says hello"]
    )
    gate = PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.BYPASS)
    await run_subagent(READER, "read notes", ctx=ctx, llm=llm, tools=default_registry(), gate=gate)
    first, second = updates[-1].recent
    assert first.summary.startswith("Read(") and first.done and first.ok and first.result == "1\thello"
    assert second.done and not second.ok and "file not found" in second.result
    assert all(u.log == updates[0].log for u in updates) and Path(updates[0].log).exists()
    running = next(u for u in updates if u.recent and not u.recent[-1].done)  # while a call runs
    assert running.recent[-1].summary.startswith("Read(")


def running_agent(tools_used: int, *calls, name="general-purpose", run_id="a1"):
    from tinyharness.events import SubagentCall

    return SubagentUpdate(
        run_id,
        name,
        "Verify deck rendering",
        "working",
        "Bash(node m.mjs)",
        tools_used,
        388.0,
        "opus",
        recent=tuple(SubagentCall(*call) for call in calls),
    )


def test_a_running_subagent_shows_its_last_few_calls():
    console = Console(width=110, record=True)
    renderer = Renderer(console)
    renderer.handle(
        running_agent(
            21,
            ("Edit(.deck-verify/measure.mjs, -1/+3 lines)", True, True, "edited", 0.2),
            ("Bash(node .deck-verify/measure.mjs)", True, False, "(exit code 1)", 4.0),
            ("Bash(node .deck-verify/measure.mjs 2>&1)", False, True, "", 12.0),
        )
    )
    live = [line.rstrip() for line in renderer.live_ansi(110).splitlines()]
    assert live[1].strip() == "… 18 earlier"
    assert live[2].strip().startswith("✓ Edit") and live[2].rstrip().endswith("edited")
    assert live[3].strip().startswith("✗ Bash") and live[3].endswith("(exit code 1) · 4s")
    assert live[4].strip()[0] not in "✓✗" and live[4].endswith("12s")  # still running


def test_with_several_subagents_each_shows_only_its_latest_call():
    console = Console(width=110, record=True)
    renderer = Renderer(console)
    for n in range(2):
        renderer.handle(
            running_agent(
                5,
                ("Read(a.py)", True, True, "1\tx", 0.1),
                ("Grep(x)", True, True, "3 matches", 0.1),
                run_id=f"a{n}",
            )
        )
    live = renderer.live_ansi(110)
    assert live.count("Grep") == 2 and "Read(" not in live and live.count("4 earlier") == 2


async def test_teammates_know_the_request_and_to_name_their_files_uniquely(tmp_path):
    prompts = {}

    async def spawner(spec, prompt, ctx, /, *, label, extra_tools, inbox):
        prompts[label.split(":")[0]] = prompt
        return "ok"

    ctx = make_ctx(tmp_path)
    ctx.request = "Why did revenue drop in July?"
    tool = team_tool(spawner)
    await tool.run(tool.parse(plan(("a", "reader", []), ("b", "reader", []))), ctx)
    assert "Why did revenue drop in July?" in prompts["a"]
    assert "starting with 'a'" in prompts["a"] and "never make up data" not in prompts["a"]


async def test_a_chain_plan_and_a_folder_still_to_be_made_are_accepted(tmp_path):
    """Team used to refuse a plan where no two tasks run at once ("do it yourself") and
    one naming a path whose folder didn't exist yet (an out/ folder the first task makes)."""
    tool = team_tool()
    tool.spawner = _ok_spawner
    chain = tool.parse({"goal": "g", "tasks": [
        {"id": "a", "title": "a", "prompt": f"write {tmp_path}/out/data.csv", "agent": "reader"},
        {"id": "b", "title": "b", "prompt": f"chart {tmp_path}/out/data.csv", "agent": "reader",
         "depends_on": ["a"]},
    ]})  # fmt: skip
    assert (await tool.run(chain, make_ctx(tmp_path))).startswith("Team finished (2 done)")


async def _ok_spawner(spec, prompt, ctx, /, *, label, extra_tools, inbox):
    return "ok"


async def test_a_subagent_knows_where_it_works_and_the_date(tmp_path):
    import datetime

    llm = ScriptedLLM(["done"])
    gate = PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.BYPASS)
    await run_subagent(READER, "x", ctx=make_ctx(tmp_path), llm=llm, tools=default_registry(), gate=gate)
    system = llm.requests[0][0].content
    assert f"Working directory: {tmp_path}" in system
    assert datetime.date.today().isoformat() in system


def test_an_agent_file_can_set_its_step_limit(tmp_path):
    path = tmp_path / "analyst.md"
    path.write_text("---\nname: analyst\ndescription: d\nmax_steps: 60\n---\nYou analyse.")
    spec = parse_agent_md(path)
    assert spec is not None and spec.max_steps == 60


def test_a_subagents_multi_line_command_stays_on_one_row():
    # Seen live: a teammate's SQL spilled over several rows and broke the layout.
    console = Console(width=110, record=True)
    renderer = Renderer(console)
    sql = 'Bash(sqlite3 nova_ai.db "\n-- 2. Invoice summary\nSELECT\n  COUNT(*) FROM invoices")'
    renderer.handle(running_agent(3, (sql, True, True, "Total invoices|7675", 0.2)))
    live = [Text.from_ansi(line).plain.rstrip() for line in renderer.live_ansi(110).splitlines()]
    row = next(line for line in live if "sqlite3" in line)
    assert 'sqlite3 nova_ai.db " (+3 lines)' in row and row.endswith("Total invoices|7675")
    assert not any(line.strip().startswith(("-- 2.", "SELECT")) for line in live)


async def test_the_shared_context_comes_first_and_the_same_for_every_teammate(tmp_path):
    prompts = {}

    async def spawner(spec, prompt, ctx, /, *, label, extra_tools, inbox):
        prompts[label.split(":")[0]] = prompt
        return "ok"

    tool = team_tool(spawner)
    args = plan(("a", "reader", []), ("b", "reader", []))
    args["context"] = "Tables: customers(customer_id, plan, status). MRR = sum of active contracts."
    await tool.run(tool.parse(args), make_ctx(tmp_path))
    a, b = prompts["a"], prompts["b"]
    assert "<team-context>\nTables: customers" in a
    shared = a[: a.index("Your task")]
    assert b.startswith(shared)  # word for word, so a prefix cache can reuse it
    assert a.index("<team-context>") < a.index("Your task (a")  # its own task comes last


def test_an_oversized_shared_context_is_refused():
    from pydantic import ValidationError

    from tinyharness.tools.team import CONTEXT_CHARS

    tool = team_tool()
    args = plan(("a", "reader", []), ("b", "reader", []))
    args["context"] = "x" * (CONTEXT_CHARS + 1)
    with pytest.raises(ValidationError):
        tool.parse(args)


def test_a_plan_with_any_parallel_step_is_accepted():
    tool = team_tool()
    # a then (b, c) together then d: a diamond has a parallel step
    diamond = plan(
        ("a", "reader", []), ("b", "reader", ["a"]), ("c", "reader", ["a"]), ("d", "reader", ["b", "c"])
    )
    assert len(tool._board(tool.parse(diamond)).tasks) == 4


def test_a_running_subagents_status_stays_on_one_row_during_a_multi_line_command():
    # Seen live: "Bash(sqlite3 nova_ai.db <<EOF\n-- Check if…" broke the agent's line in two.
    console = Console(width=140, record=True)
    renderer = Renderer(console)
    update = SubagentUpdate(
        "a2", "general-purpose", "data-analysis: Data Analysis\n& Insights", "working",
        "Bash(sqlite3 nova_ai.db <<EOF\n-- Check if tables exist\nSELECT 1;\nEOF)", 7, 209.0, "gemma-4-26b-a4b-it",
    )  # fmt: skip
    renderer.handle(update)
    live = [Text.from_ansi(line).plain.rstrip() for line in renderer.live_ansi(140).splitlines()]
    agent_line = next(line for line in live if "general-purpose" in line)
    assert "Bash(sqlite3 nova_ai.db <<EOF …" in agent_line and agent_line.endswith("3m 29s")
    assert "Data Analysis & Insights" in agent_line
    assert not any(line.lstrip().startswith(("-- Check", "SELECT", "EOF")) for line in live)


async def test_a_subagent_keeps_its_own_record_of_the_files_it_has_read(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    ctx = make_ctx(tmp_path)
    ctx.read_paths.add(str(tmp_path / "notes.txt"))  # the lead read it, the subagent didn't
    llm = ScriptedLLM(
        [[("Edit", '{"path": "notes.txt", "old_string": "hello", "new_string": "hi"}')], "done"]
    )
    gate = PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.BYPASS)
    await run_subagent(WRITER_EDIT, "edit", ctx=ctx, llm=llm, tools=default_registry(), gate=gate)
    assert (tmp_path / "notes.txt").read_text() == "hello"  # refused: read it first
    assert "before editing it" in llm.requests[1][-1].content
    assert ctx.read_paths == {str(tmp_path / "notes.txt")}  # the lead's record is untouched


WRITER_EDIT = AgentSpec("editor", "edits", tools=["Read", "Edit"])


# ---- result integrity (from tinyharness's own logs) ----


async def test_the_team_board_waits_for_news_instead_of_being_polled(monkeypatch):
    from tinyharness.tools import team
    from tinyharness.tools.team import BoardInput, TeamBoardTool

    monkeypatch.setattr(team, "BOARD_WAIT_SECONDS", 5)
    board = Board(
        "goal",
        {
            "a": TaskState("a", "A", "p", "general-purpose", []),
            "b": TaskState("b", "B", "p", "general-purpose", []),
        },
    )
    tool = TeamBoardTool(board)
    await tool.run(BoardInput(), None)  # the first look is immediate
    loop = asyncio.get_running_loop()
    loop.call_later(0.2, board.post, "b", "a", "schema is in schema.sql")
    start = loop.time()
    seen = await tool.run(BoardInput(), None)  # nothing new: waits until b posts
    assert 0.15 < loop.time() - start < 2 and "schema is in schema.sql" in seen


async def test_a_failed_subagent_hands_back_its_work_so_far(tmp_path):
    from tinyharness.llm.types import LLMError
    from tinyharness.subagents.runner import SubagentFailed

    (tmp_path / "data.txt").write_text("42\n")

    class Breaks(ScriptedLLM):
        async def stream(self, messages, tools=None, **kw):
            if len(self.requests) >= 1:
                self.requests.append(messages)
                raise LLMError("LLM error 502: upstream died")
            async for chunk in super().stream(messages, tools, **kw):
                yield chunk

    llm = Breaks([[("Read", '{"path": "data.txt"}')]])
    gate = PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.BYPASS)
    with pytest.raises(SubagentFailed) as failed:
        await run_subagent(
            AgentSpec("general-purpose", "x"),
            "read it",
            ctx=make_ctx(tmp_path),
            llm=llm,
            tools=default_registry(),
            gate=gate,
        )
    message = str(failed.value)
    assert "upstream died" in message and "Read(" in message and "transcript" in message


def test_subagents_see_the_skills_too(tmp_path):
    """Teammates load skills the same way the lead does: from the listing, with Read. A
    report builder couldn't use the user's design skill when only the lead could see it."""
    from tinyharness.skills import Skill, skills_section
    from tinyharness.subagents.runner import subagent_system_prompt
    from tinyharness.subagents.specs import builtin_agents

    ctx = make_ctx(tmp_path)
    ctx.skills_prompt = skills_section(
        [Skill("design", "Design pages well.", "Use a grid.", source=tmp_path / "SKILL.md")]
    )
    prompt = subagent_system_prompt(builtin_agents()["general-purpose"], ctx)
    assert f"- design: Design pages well. ({tmp_path / 'SKILL.md'})" in prompt


async def test_a_long_result_is_handed_on_whole_through_a_file(tmp_path):
    """Dependency results were cut at 4,000 characters with no way to get the rest, so a
    report builder saw only the start of each analyst's findings."""
    from tinyharness.workspace.scratch import Scratch

    findings = "FINDING " + "x" * 9_000 + " LAST LINE"
    seen = {}

    async def spawner(spec, prompt, ctx, /, *, label, extra_tools, inbox):
        seen[label] = prompt
        return findings if label.startswith("a:") else "report written"

    board = Board(
        "goal",
        {
            "a": TaskState("a", "A", "analyze", "general-purpose", []),
            "b": TaskState("b", "B", "write", "general-purpose", ["a"]),
        },
    )
    ctx = make_ctx(tmp_path)
    ctx.scratch = Scratch(tmp_path, "s1")
    await run_team(board, {"general-purpose": AgentSpec("general-purpose", "x")}, spawner, ctx)
    path = board.tasks["a"].result_file
    assert path and Path(path).read_text().endswith("LAST LINE")
    assert f"the whole result is in {path}" in seen["b: B"]
    assert path in report(board)
