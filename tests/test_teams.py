"""Subagents in parallel, their progress, and agent teams."""

import asyncio

import pytest
from rich.console import Console

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
    s = Session(SessionOptions(config_path=tmp_path / "none.toml", working_dir=tmp_path / "work"))
    yield s
    s.close()


# ---- which subagents may run side by side ----


def test_only_read_only_agents_run_in_parallel(session):
    agent = session.tools.get("Agent")
    assert agent.is_read_only(agent.parse({"subagent_type": "explore", "prompt": "x"}))
    assert agent.is_read_only(agent.parse({"subagent_type": "researcher", "prompt": "x"}))
    assert not agent.is_read_only(agent.parse({"subagent_type": "general-purpose", "prompt": "x"}))


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
    assert [u.status for u in updates] == ["started", "working", "done"]
    assert updates[1].detail.startswith("Read(") and updates[-1].detail == "1 tool call"
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
    inbox = [["[message from teammate b] use port 8080"]]
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
    assert "[message from teammate b] use port 8080" in [m.content for m in llm.requests[0]]


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


async def test_a_team_runs_in_dependency_order_with_readers_in_parallel(tmp_path):
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
    assert most_writers == 1  # writers never overlapped
    assert started[-1] == "sum"
    assert "result of w1" in prompts["sum"] and "result of r3" in prompts["sum"]
    assert out.startswith("Team finished (6 done)")


async def test_when_a_task_fails_its_dependents_are_skipped(tmp_path):
    async def spawner(spec, prompt, ctx, /, *, label, extra_tools, inbox):
        if label.startswith("a:"):
            raise RuntimeError("site was down")
        return "fine"

    tool = team_tool(spawner)
    out = await tool.run(
        tool.parse(plan(("a", "reader", []), ("b", "reader", ["a"]), ("c", "reader", []))), make_ctx(tmp_path)
    )
    assert "## a: a · reader · failed" in out and "site was down" in out
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
    assert board.take_inbox("b") == ["[message from teammate a] API is at /v2"]
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
    job = asyncio.create_task(run_team(board, AGENTS, spawner, is_read_only, make_ctx(tmp_path)))
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
    session.set_mode("default")
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
    session.set_mode("acceptEdits")
    assert repl.next_mode() == PermissionMode.AUTO  # the next press would ask
    await repl.shift_tab_to_auto()
    assert session.mode == PermissionMode.ACCEPT_EDITS  # "no" keeps the mode
    cycle = []
    for _ in range(4):  # the presses after that go round without auto, and never ask again
        session.set_mode(repl.next_mode())
        cycle.append(session.mode)
    assert cycle == [
        PermissionMode.PLAN,
        PermissionMode.DEFAULT,
        PermissionMode.ACCEPT_EDITS,
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
        permission_mode="acceptEdits",
    )
    explore = AgentSpec(name="explore", description="My own explorer.", tools=["Read"])
    options = SessionOptions(
        config_path=tmp_path / "none.toml", working_dir=tmp_path / "work", agents=[reviewer, explore]
    )
    session = Session(options)
    try:
        assert session.agents["reviewer"] is reviewer
        assert reviewer.permission_mode == PermissionMode.ACCEPT_EDITS
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
    assert not session._agent_is_read_only(spec)  # it can write, like the session
    assert session._agent_is_read_only(AgentSpec("none", "no tools", tools=[]))


async def test_agents_without_a_tool_list_get_every_tool_but_spawning(tmp_path):
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
    assert offered[0] == set(registry.names()) - {"Team"}


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
