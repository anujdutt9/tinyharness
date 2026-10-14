"""Seeing when a skill is loaded, and loading one asked for as /name mid-message."""

import json

import pytest

from tinyharness.events import SkillLoaded, ToolResult
from tinyharness.skills import Skill
from tinyharness.testing import ScriptedLLM, scripted_session

BODY = "Step 1: ask what the design is for. Step 2: build it."


@pytest.fixture
def session(tmp_path):
    session = scripted_session(tmp_path / "w", [])
    folder = tmp_path / "skills" / "claude-design-skill"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(f"---\nname: claude-design-skill\n---\n{BODY}\n")
    session.skills.add(Skill("claude-design-skill", "Design things.", BODY, source=folder / "SKILL.md"))
    session.workspace.add_read_root(folder)
    yield session
    session.close()


def play(session, turns):
    llm = ScriptedLLM(turns)
    session.llm.stream = llm.stream
    return llm


async def events(session, prompt):
    return [e async for e in session.run(prompt)]


async def test_a_slash_name_mid_message_loads_the_skill(session):
    play(session, ["ok"])
    got = await events(session, "Build the report and use /claude-design-skill for the layout.")
    loaded = [e for e in got if isinstance(e, SkillLoaded)]
    assert [(e.name, e.how) for e in loaded] == [("claude-design-skill", "named")] and loaded[0].tokens > 0
    sent = next(m.content for m in session.history.messages if m.role == "user")
    assert sent.startswith("# Skill: claude-design-skill") and BODY in sent
    assert sent.endswith("User message: Build the report and use /claude-design-skill for the layout.")


@pytest.mark.parametrize(
    "prompt",
    [
        "Don't use the claude-design-skill for this one.",  # a bare name isn't a request
        "What's in ~/.tinyharness/skills/claude-design-skill/SKILL.md?",  # nor a path
        "Try /claude-design-skills",  # nor another word
    ],
)
async def test_only_a_slash_name_loads_it(session, prompt):
    play(session, ["ok"])
    got = await events(session, prompt)
    assert not [e for e in got if isinstance(e, SkillLoaded)]
    assert [m.content for m in session.history.messages if m.role == "user"] == [prompt]


async def test_a_skill_already_in_the_conversation_isnt_loaded_again(session):
    play(session, ["ok", "ok again"])
    await events(session, "use /claude-design-skill")
    got = await events(session, "and /claude-design-skill again for the second page")
    assert not [e for e in got if isinstance(e, SkillLoaded)]


async def test_running_a_skill_says_so(session):
    play(session, ["ok"])
    got = [e async for e in session.run_skill("claude-design-skill", "a dashboard")]
    assert isinstance(got[0], SkillLoaded) and got[0].how == "command"
    # its own text names the skill: that isn't a second request for it
    assert sum(isinstance(e, SkillLoaded) for e in got) == 1


async def test_invoked_skill_enables_the_extensions_it_requires(session):
    skill = session.skills.get("claude-design-skill")
    assert skill is not None
    skill.required_extensions = ["agents"]
    play(session, ["ok"])
    await events(session, "please use /claude-design-skill")
    assert "agents" in session.config.extensions.enabled
    assert session.tools.get("Agent") is not None


async def test_resumed_skill_reenables_the_extensions_it_requires(tmp_path):
    """Seen live: a resumed multi-agent skill lost Agent because it was already in context."""
    root = tmp_path / "work"
    folder = root / ".tinyharness" / "skills" / "needs-agents"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        "---\nname: needs-agents\ndescription: delegate work\n"
        "requires-extensions: [agents]\n---\nUse Agent.\n"
    )
    first = scripted_session(root, ["done"])
    await events(first, "use /needs-agents")
    assert first.tools.get("Agent") is not None
    first.close()

    resumed = scripted_session(root, [], resume="latest")
    try:
        assert "agents" in resumed.config.extensions.enabled
        assert resumed.tools.get("Agent") is not None
    finally:
        resumed.close()


async def test_invoked_skill_registers_its_exact_network_set(session):
    skill = session.skills.get("claude-design-skill")
    assert skill is not None
    skill.network_hosts = ["assets.example.com", "fonts.example.com"]
    play(session, ["ok"])
    await events(session, "please use /claude-design-skill")
    assert session.gate.network_bundles == {
        "claude-design-skill": ("assets.example.com", "fonts.example.com")
    }


async def test_the_model_reading_a_skill_file_is_shown(session):
    path = str(session.skills.get("claude-design-skill").source)
    play(session, [[("Read", json.dumps({"path": path}))], "done"])
    got = await events(session, "design me a landing page")
    i = next(i for i, e in enumerate(got) if isinstance(e, SkillLoaded))
    assert got[i].how == "read" and isinstance(got[i - 1], ToolResult)
    # and it counts as loaded: naming it now doesn't send it again
    assert session.skills_named("now use /claude-design-skill") == []


@pytest.mark.parametrize(
    ("how", "note"), [("command", ""), ("named", "named in your message"), ("read", "the model read it")]
)
def test_the_cli_shows_a_loaded_skill(how, note):
    import io

    from rich.console import Console

    from tinyharness.cli.render import Renderer
    from tinyharness.cli.theme import THEME

    console = Console(file=io.StringIO(), width=100, theme=THEME)
    Renderer(console).handle(SkillLoaded("claude-design-skill", 9_200, how))
    out = console.file.getvalue()
    assert "skill claude-design-skill loaded · 9.2k tokens" in out and note in out
