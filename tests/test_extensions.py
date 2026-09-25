"""Skills, agents, hooks, plugins and AGENT.md are loaded from tinyharness's own
directories only; another tool's home (~/.claude) is never read."""

from pathlib import Path

import pytest

from tinyharness.agent.prompts import load_agent_md
from tinyharness.hooks import load_hooks
from tinyharness.paths import home_dir
from tinyharness.plugins import load_plugins
from tinyharness.skills import load_skills
from tinyharness.subagents import builtin_agents, load_agents


@pytest.fixture
def fake_home(tmp_path, monkeypatch) -> Path:
    home = tmp_path / "homedir"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    claude = home / ".claude"
    (claude / "skills" / "trap").mkdir(parents=True)
    (claude / "skills" / "trap" / "SKILL.md").write_text("---\nname: trap\n---\nbody")
    (claude / "agents").mkdir()
    (claude / "agents" / "trap.md").write_text("---\nname: trap\n---\nbody")
    (claude / "CLAUDE.md").write_text("trap content")
    return home


@pytest.fixture
def project(tmp_path) -> Path:
    p = tmp_path / "project"
    p.mkdir()
    return p


def write_skill(root: Path, name: str, description: str = "d") -> None:
    (root / name).mkdir(parents=True, exist_ok=True)
    (root / name / "SKILL.md").write_text(f"---\nname: {name}\ndescription: {description}\n---\nbody")


def test_home_dir_override(isolated_home):
    assert home_dir() == isolated_home


def test_home_dir_default(monkeypatch):
    monkeypatch.delenv("TINYHARNESS_HOME")
    assert home_dir() == Path.home() / ".tinyharness"


def test_dot_claude_is_ignored(fake_home, project):
    assert all(s.name != "trap" for s in load_skills(project))
    assert "trap" not in load_agents(project)
    assert "trap content" not in load_agent_md(project)


def test_user_skill_wins_over_project_skill(isolated_home, project):
    write_skill(isolated_home / "skills", "deploy", "trusted")
    write_skill(project / ".tinyharness" / "skills", "deploy", "shadow")
    write_skill(project / ".tinyharness" / "skills", "lint")
    skills = {s.name: s for s in load_skills(project)}
    assert skills["deploy"].description == "trusted"
    assert "lint" in skills


def test_custom_agents_extend_builtins(isolated_home, project):
    agents_dir = project / ".tinyharness" / "agents"
    agents_dir.mkdir(parents=True)
    (agents_dir / "reviewer.md").write_text(
        "---\nname: reviewer\ndescription: reviews\ntools: [Read]\npermission_mode: plan\n---\nBe picky."
    )
    agents = load_agents(project)
    assert set(builtin_agents()) <= set(agents)
    assert agents["reviewer"].tools == ["Read"]
    assert agents["reviewer"].system_prompt == "Be picky."


def test_hooks_loaded_from_home_and_project(isolated_home, project):
    isolated_home.mkdir(parents=True)
    (isolated_home / "hooks.toml").write_text('[[hook]]\nevent = "PreToolUse"\ncommand = "a.sh"\n')
    (project / ".tinyharness").mkdir()
    (project / ".tinyharness" / "hooks.toml").write_text(
        '[[hook]]\nevent = "Stop"\ncommand = "b.sh"\n[[hook]]\nevent = "Bogus"\ncommand = "c.sh"\n'
    )
    assert [h.command for h in load_hooks(project)] == ["a.sh", "b.sh"]


def test_plugin_bundles_skills_agents_and_hooks(isolated_home, project):
    plugin = isolated_home / "plugins" / "kit"
    write_skill(plugin / "skills", "fmt")
    (plugin / "agents").mkdir()
    (plugin / "agents" / "helper.md").write_text("---\nname: helper\n---\nhelp")
    (plugin / "hooks.toml").write_text('[[hook]]\nevent = "Stop"\ncommand = "x.sh"\n')
    (plugin / "plugin.toml").write_text('name = "kit"\n')
    [loaded] = load_plugins(project)
    assert loaded.name == "kit"
    assert [s.name for s in loaded.skills] == ["fmt"]
    assert [a.name for a in loaded.agents] == ["helper"]
    assert [h.command for h in loaded.hooks] == ["x.sh"]


def test_agent_md_layers(isolated_home, project):
    isolated_home.mkdir(parents=True)
    (isolated_home / "AGENT.md").write_text("user context")
    (project / "CLAUDE.md").write_text("project context")
    out = load_agent_md(project)
    assert out.index("user context") < out.index("project context")
