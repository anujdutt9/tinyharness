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
