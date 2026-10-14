"""Extensions: one file with setup(th) adds tools, prompt sections, commands and event
handlers, all registered at session start (docs/extension-api.md)."""

import textwrap

import pytest

from conftest import ScriptedLLM, make_ctx
from tinyharness import Session, SessionOptions
from tinyharness.agent.history import Message
from tinyharness.events import ToolResult
from tinyharness.ext import Extension, function_tool

HELLO = '''
"""Greets people."""
from tinyharness.ext import Extension

NEEDS = ["network:api.example.com"]


def setup(th: Extension) -> None:
    @th.tool(read_only=True)
    async def hello(name: str, excited: bool = False) -> str:
        """Greet someone by name.

        Parameters
        ----------
        name : str
            Person to greet.
        """
        return f"Hello, {name}{'!' if excited else '.'}"

    th.prompt_section("Greetings", "Use the hello tool to greet people by name.")

    @th.on("tool_call")
    async def no_force_push(call, ctx):
        if call.tool == "Bash" and "push --force" in call.input.get("command", ""):
            return ctx.block("force-push is off in this project")

    @th.on("tool_result")
    async def remind(result, ctx):
        if result.get("tool") == "hello":
            return ctx.note("Greeted.")

    @th.on("user_prompt")
    async def ticket(prompt, ctx):
        return ctx.note("Current ticket: ABC-1")

    @th.command("hi", help="say hi")
    async def hi(arg, ctx):
        ctx.say(f"hi {arg}")
'''


def write_ext(home, name, source):
    folder = home / "extensions"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.py").write_text(textwrap.dedent(source))


def make_session(tmp_path, monkeypatch, enabled, **options):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = tmp_path / "config.toml"
    config.write_text(f"[extensions]\nenabled = {enabled!r}\n")
    return Session(SessionOptions(config_path=config, working_dir=tmp_path / "work", **options))


async def test_a_function_becomes_a_tool_with_a_schema_from_its_signature(tmp_path):
    async def lookup(city: str, days: int = 3, ctx=None) -> str:
        """Look up the weather.

        Parameters
        ----------
        city : str
            The city's name.
        days : int, default=3
            Number of forecast days.
        """
        return f"{city} for {days} days, {ctx is not None}"

    tool = function_tool(lookup, read_only=True)
    schema = tool.input_schema()
    assert tool.name == "lookup" and tool.description == "Look up the weather." and tool.read_only
    assert schema["required"] == ["city"]
    assert schema["properties"]["city"]["description"] == "The city's name."
    assert schema["properties"]["days"]["description"] == "Number of forecast days."
    assert schema["properties"]["days"]["default"] == 3 and "ctx" not in schema["properties"]
    assert await tool.run(tool.parse({"city": "Oslo"}), make_ctx(tmp_path)) == "Oslo for 3 days, True"


def test_function_tool_keeps_compact_parameter_docs_compatible():
    async def lookup(city: str) -> str:
        """Look up a city.

        city: City name.
        """
        return city

    schema = function_tool(lookup).input_schema()
    assert schema["properties"]["city"]["description"] == "City name."


def test_a_tool_must_be_async():
    def sync_tool(x: str) -> str:
        return x

    with pytest.raises(TypeError, match="async"):
        Extension("x").tool(sync_tool)


async def test_an_extension_adds_its_tool_prompt_section_command_and_handlers(
    tmp_path, monkeypatch, isolated_home
):
    write_ext(isolated_home, "hello", HELLO)
    session = make_session(tmp_path, monkeypatch, ["hello"])
    try:
        [ext] = session.extensions
        assert ext.description == "Greets people." and ext.needs == ["network:api.example.com"]
        assert "hello" in session.tools.names() and session.extension_command("hi") is not None
        assert "# Greetings\n\nUse the hello tool" in session._system_prompt()
        assert session.extension_tokens(ext) > 20

        llm = ScriptedLLM(
            [
                [("hello", '{"name": "Ada", "excited": true}')],
                [("Bash", '{"command": "git push --force"}')],
                "done",
            ]
        )
        session.llm.stream = llm.stream
        events = [e async for e in session.run("greet Ada, then push")]
        results = [e for e in events if isinstance(e, ToolResult)]
        assert results[0].output.startswith("Hello, Ada!") and "Greeted." in results[0].output
        assert results[1].is_error and "force-push is off" in results[1].output
        first_user = next(m for m in session.history.messages if m.role == "user")
        assert first_user.content.endswith("Current ticket: ABC-1")
    finally:
        session.close()


async def test_a_turn_end_handler_can_keep_the_turn_going(tmp_path, monkeypatch, isolated_home):
    write_ext(
        isolated_home,
        "check",
        """
        def setup(th):
            seen = []

            @th.on("turn_end")
            async def once_more(reply, ctx):
                if not seen:
                    seen.append(1)
                    return ctx.block("run the tests first")
        """,
    )
    session = make_session(tmp_path, monkeypatch, ["check"])
    try:
        session.llm.stream = ScriptedLLM(["done", "tests pass, done"]).stream
        [e async for e in session.run("go")]
        texts = [m.content for m in session.history.messages if m.role == "user"]
        assert any("run the tests first" in (t or "") for t in texts)
        assert session.history.messages[-1].content == "tests pass, done"
    finally:
        session.close()


def test_a_broken_or_missing_extension_is_reported_and_the_session_starts(
    tmp_path, monkeypatch, isolated_home
):
    write_ext(isolated_home, "broken", "def setup(th):\n    raise RuntimeError('boom')\n")
    write_ext(isolated_home, "nosetup", "X = 1\n")
    session = make_session(tmp_path, monkeypatch, ["broken", "nosetup", "nowhere"])
    try:
        errors = {e.name: e.error for e in session.extension_errors}
        assert errors == {
            "broken": "RuntimeError: boom",
            "nosetup": "has no setup(th) function",
            "nowhere": "not found",
        }
        assert session.extensions == [] and "Read" in session.tools.names()
    finally:
        session.close()


def test_a_projects_extensions_load_only_in_a_trusted_folder(tmp_path, monkeypatch):
    project_ext = tmp_path / "work" / ".tinyharness" / "extensions"
    project_ext.mkdir(parents=True)
    (project_ext / "local.py").write_text(
        "def setup(th):\n    th.prompt_section('Local', 'from the project')\n"
    )
    untrusted = make_session(tmp_path, monkeypatch, ["local"], trust_project=False)
    trusted = make_session(tmp_path, monkeypatch, ["local"], trust_project=True)
    try:
        assert [e.name for e in untrusted.extension_errors] == ["local"]
        assert [e.name for e in trusted.extensions] == ["local"]
    finally:
        untrusted.close()
        trusted.close()


def test_switching_extensions_mid_session_updates_tools_and_prompt(tmp_path, monkeypatch, isolated_home):
    write_ext(isolated_home, "hello", HELLO)
    session = make_session(tmp_path, monkeypatch, [])
    try:
        session.history.append(Message(role="system", content=session._system_prompt()))
        assert "hello" not in session.tools.names()
        session.set_extensions(["hello"])
        assert "hello" in session.tools.names() and "# Greetings" in session.history.messages[0].content
        hooks_with = len(session.hooks.hooks)
        session.set_extensions([])
        assert (
            "hello" not in session.tools.names() and "# Greetings" not in session.history.messages[0].content
        )
        assert len(session.hooks.hooks) == hooks_with - 3  # its handlers went with it
    finally:
        session.close()


def test_an_extension_can_take_away_a_core_tool(tmp_path, monkeypatch, isolated_home):
    write_ext(isolated_home, "noglob", "def setup(th):\n    th.remove_tool('Glob')\n")
    session = make_session(tmp_path, monkeypatch, ["noglob"])
    try:
        assert "Glob" not in session.tools.names() and "Grep" in session.tools.names()
    finally:
        session.close()


async def test_run_agent_names_the_agents_it_knows(tmp_path, monkeypatch):
    from tinyharness.tools.base import ToolError

    session = make_session(tmp_path, monkeypatch, [])
    try:
        with pytest.raises(ToolError, match=r"unknown agent 'nope'; available: .*general-purpose"):
            await session.run_agent("nope", "do it")
    finally:
        session.close()


@pytest.mark.parametrize(
    ("name", "tools", "section"),
    [
        ("web", ["WebSearch", "WebFetch"], "# Web"),
        ("pdf", ["Pdf"], None),
        ("todo", ["TodoWrite"], "# Planning"),
    ],
)
def test_built_in_extensions_add_their_tools_and_guidance(tmp_path, monkeypatch, name, tools, section):
    off = make_session(tmp_path, monkeypatch, [])
    on = make_session(tmp_path, monkeypatch, [name])
    try:
        assert not set(tools) & set(off.tools.names())
        assert set(tools) <= set(on.tools.names())
        if section:
            assert section in on._system_prompt() and section not in off._system_prompt()
    finally:
        off.close()
        on.close()


def test_the_older_todo_list_setting_turns_on_the_todo_extension(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = tmp_path / "config.toml"
    config.write_text("todo_list = true\n")
    session = Session(SessionOptions(config_path=config, working_dir=tmp_path / "work"))
    try:
        assert [e.name for e in session.extensions] == ["todo"] and "TodoWrite" in session.tools.names()
    finally:
        session.close()


def test_agents_and_teams_are_an_extension_built_on_run_agent(tmp_path, monkeypatch):
    off = make_session(tmp_path, monkeypatch, [])
    on = make_session(tmp_path, monkeypatch, ["agents"])
    try:
        assert not {"Agent", "Team"} & set(off.tools.names()) and "# Subagents" not in off._system_prompt()
        assert {"Agent", "Team"} <= set(on.tools.names()) and "# Subagents" in on._system_prompt()
    finally:
        off.close()
        on.close()


async def test_the_agent_tool_runs_its_subagent_through_run_agent(tmp_path, monkeypatch):
    session = make_session(tmp_path, monkeypatch, ["agents"])
    calls = []

    async def fake_run_agent(agent, task, **kwargs):
        calls.append((agent, task))
        return "found it"

    monkeypatch.setattr(session, "run_agent", fake_run_agent)
    try:
        agent = session.tools.get("Agent")
        out = await agent.run(agent.parse({"subagent_type": "explore", "prompt": "find x"}), session.ctx)
        assert "found it" in str(out) and calls == [("explore", "find x")]
    finally:
        session.close()


async def test_an_extension_given_in_code_runs_in_a_scripted_session(tmp_path):
    """SDK users pass extensions as setup functions; tinyharness.testing runs them against
    a scripted model with no key or network, as the extension-api docs show."""
    from tinyharness.testing import scripted_session

    def greeter(th):
        """Greets."""

        @th.tool(read_only=True)
        async def hello(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}."

    session = scripted_session(tmp_path / "w", [[("hello", '{"name": "Ada"}')], "done"], extensions=[greeter])
    try:
        assert [e.name for e in session.extensions] == ["greeter"]
        results = [e.output async for e in session.run("greet Ada") if isinstance(e, ToolResult)]
        assert results == ["Hello, Ada."]
    finally:
        session.close()


def test_an_extension_in_config_and_in_code_loads_once(tmp_path, monkeypatch):
    from tinyharness.extensions import agents

    session = make_session(tmp_path, monkeypatch, ["agents"], extensions=[agents])
    try:
        assert [e.name for e in session.extensions] == ["agents"]
        assert session._system_prompt().count("# Subagents") == 1
    finally:
        session.close()


NEEDY = '''
"""Syncs notes."""
NEEDS = ["network:api.example.com", "write:{notes}"]


def setup(th):
    @th.tool(network=True)
    async def fetch_notes(url: str) -> str:
        """Fetch notes."""
        return "notes"

    th.prompt_section("Notes", "Use fetch_notes.")
'''


def test_needs_must_be_hosts_or_folders(tmp_path, monkeypatch, isolated_home):
    write_ext(isolated_home, "bad", 'NEEDS = ["everything"]\ndef setup(th):\n    pass\n')
    session = make_session(tmp_path, monkeypatch, ["bad"])
    try:
        assert "NEEDS entries are 'network:<host>' or 'write:<path>'" in session.extension_errors[0].error
    finally:
        session.close()


async def test_an_extensions_needs_apply_only_once_approved_and_until_it_changes(
    tmp_path, monkeypatch, isolated_home
):
    """Declared needs used to be shown and never enforced. Unapproved, the extension loads
    but its tools ask; approved, its own tools reach its hosts and folders without asking;
    a changed file asks again."""
    from tinyharness.permissions import Decision

    notes = tmp_path / "notes"
    write_ext(isolated_home, "sync", NEEDY.replace("{notes}", str(notes)))
    asked = []

    async def user(request):
        asked.append(request.tool)
        return Decision(False, "no", "user")

    session = make_session(tmp_path, monkeypatch, ["sync"], permission_callback=user)
    try:
        [ext] = session.extensions
        tool = session.tools.get("fetch_notes")
        args = {"url": "https://api.example.com/notes?all=1"}
        assert not ext.granted
        assert not (await session.gate.check(tool, args, None, target_host="api.example.com")).allow
        session.grant_needs("sync")
        assert (await session.gate.check(tool, args, None, target_host="api.example.com")).allow
        assert not (await session.gate.check(tool, args, None, target_host="other.example")).allow
        assert asked == ["fetch_notes", "fetch_notes"]
    finally:
        session.close()

    again = make_session(tmp_path, monkeypatch, ["sync"])
    try:
        assert again.extensions[0].granted  # remembered
    finally:
        again.close()
    write_ext(isolated_home, "sync", NEEDY.replace("{notes}", str(notes)) + "\n# changed\n")
    changed = make_session(tmp_path, monkeypatch, ["sync"])
    try:
        assert not changed.extensions[0].granted  # its code changed: ask again
    finally:
        changed.close()


def test_an_extension_given_in_code_has_its_needs(tmp_path, monkeypatch):
    def mine(th):
        pass

    mine.NEEDS = ["network:api.example.com"]  # type: ignore[attr-defined]
    session = make_session(tmp_path, monkeypatch, [], extensions=[mine])
    try:
        assert session.extensions[0].granted and session.extensions[0].hosts == ["api.example.com"]
    finally:
        session.close()


async def test_th_run_gets_the_extensions_folders_and_hosts_once_approved(
    tmp_path, monkeypatch, isolated_home
):
    """th.run runs a command through the same sandbox as Bash; approved needs widen that
    sandbox for this extension only: its folders writable, its hosts on its own proxy."""
    from tinyharness.tools import bash

    notes = tmp_path / "notes"
    write_ext(isolated_home, "sync", NEEDY.replace("{notes}", str(notes)))
    session = make_session(tmp_path, monkeypatch, ["sync"])
    seen = []

    async def record(self, args, ctx):
        seen.append((self.policy.write_roots, self.proxy.allowed if self.proxy else ()))
        return bash.Execution("ok\n", 0, False, True, [], [])

    monkeypatch.setattr(bash.BashTool, "execute", record)
    session.sandbox = session.sandbox or object()  # the sandbox's presence is what matters here
    try:
        [ext] = session.extensions
        await ext.run("sync-notes")
        session.grant_needs("sync")
        result = await ext.run("sync-notes")
        (before_roots, before_hosts), (after_roots, after_hosts) = seen
        assert notes.resolve() not in before_roots and "api.example.com" not in before_hosts
        assert notes.resolve() in after_roots and "api.example.com" in after_hosts
        assert "pypi.org" in after_hosts and result.code == 0 and result.output == "ok\n"
    finally:
        session.close()
