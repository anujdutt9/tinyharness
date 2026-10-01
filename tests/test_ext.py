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

        name: who to greet
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

        city: the city's name
        """
        return f"{city} for {days} days, {ctx is not None}"

    tool = function_tool(lookup, read_only=True)
    schema = tool.input_schema()
    assert tool.name == "lookup" and tool.description == "Look up the weather." and tool.read_only
    assert schema["required"] == ["city"] and schema["properties"]["city"]["description"] == "the city's name"
    assert schema["properties"]["days"]["default"] == 3 and "ctx" not in schema["properties"]
    assert await tool.run(tool.parse({"city": "Oslo"}), make_ctx(tmp_path)) == "Oslo for 3 days, True"


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
        ("open", ["Open"], "# Showing things"),
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
