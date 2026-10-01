from tinyharness.skills import parse_skill_md


def test_skill_with_frontmatter(tmp_path):
    sk_dir = tmp_path / "greet"
    sk_dir.mkdir()
    (sk_dir / "SKILL.md").write_text(
        "---\n"
        "name: greet\n"
        "description: say hi formally\n"
        "allowed-tools: [Read]\n"
        "---\n"
        "Always greet the user as 'Esteemed colleague'.\n"
    )
    s = parse_skill_md(sk_dir / "SKILL.md")
    assert s.name == "greet"
    assert s.description == "say hi formally"
    assert s.allowed_tools == ["Read"]
    assert "Esteemed" in s.body


def test_skill_without_frontmatter(tmp_path):
    sk_dir = tmp_path / "plain"
    sk_dir.mkdir()
    (sk_dir / "SKILL.md").write_text("Just a plain body.\n")
    s = parse_skill_md(sk_dir / "SKILL.md")
    assert s.name == "plain"
    assert "plain body" in s.body


def test_malformed_frontmatter(tmp_path):
    sk_dir = tmp_path / "bad"
    sk_dir.mkdir()
    (sk_dir / "SKILL.md").write_text("---\n: not valid: yaml: [\n---\nbody\n")
    s = parse_skill_md(sk_dir / "SKILL.md")
    assert s is not None  # graceful degrade, no crash
    assert s.name == "bad"


async def test_skill_tool_loads_a_guide_bundled_inside_a_skill(tmp_path):
    # Seen live: a skill said "invoke `discovery-questions`" (its references/discovery-questions.md),
    # the model asked the Skill tool for it, and got "unknown skill".
    import pytest

    from conftest import call, make_ctx
    from tinyharness.skills import SkillRegistry
    from tinyharness.tools.base import ToolError
    from tinyharness.tools.skill import SkillTool

    sk = tmp_path / "design"
    (sk / "references").mkdir(parents=True)
    (sk / "SKILL.md").write_text(
        "---\nname: design\ndescription: Design pages.\n---\nSee `discovery-questions`."
    )
    (sk / "references" / "discovery-questions.md").write_text("Ask one round of questions.")
    tool = SkillTool(SkillRegistry([parse_skill_md(sk / "SKILL.md")]))
    out = await call(tool, {"name": "discovery-questions"}, make_ctx(tmp_path))
    text = out if isinstance(out, str) else out.text
    assert "bundled with the skill design" in text and "Ask one round of questions." in text
    with pytest.raises(ToolError, match="Installed skills: design"):
        await call(tool, {"name": "../SKILL"}, make_ctx(tmp_path))
