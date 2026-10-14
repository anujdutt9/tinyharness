from tinyharness.skills import Skill, build_skill_prompt, parse_skill_md


def test_skill_with_frontmatter(tmp_path):
    sk_dir = tmp_path / "greet"
    sk_dir.mkdir()
    (sk_dir / "SKILL.md").write_text(
        "---\n"
        "name: greet\n"
        "description: say hi formally\n"
        "allowed-tools: [Read]\n"
        "requires-extensions: [agents]\n"
        "requires-network: [query1.finance.yahoo.com, query2.finance.yahoo.com]\n"
        "---\n"
        "Always greet the user as 'Esteemed colleague'.\n"
    )
    s = parse_skill_md(sk_dir / "SKILL.md")
    assert s.name == "greet"
    assert s.description == "say hi formally"
    assert s.allowed_tools == ["Read"]
    assert s.required_extensions == ["agents"]
    assert s.network_hosts == ["query1.finance.yahoo.com", "query2.finance.yahoo.com"]
    assert "Esteemed" in s.body


def test_skill_network_requirements_only_accept_exact_hosts(tmp_path):
    sk_dir = tmp_path / "market"
    sk_dir.mkdir()
    (sk_dir / "SKILL.md").write_text(
        "---\n"
        "requires-network: [QUERY1.FINANCE.YAHOO.COM., '*.yahoo.com', 'https://bad.example']\n"
        "---\nbody\n"
    )
    skill = parse_skill_md(sk_dir / "SKILL.md")
    assert skill is not None
    assert skill.network_hosts == ["query1.finance.yahoo.com"]


def test_skill_prompt_tells_the_model_about_declared_network_hosts():
    prompt = build_skill_prompt(
        Skill("market", "market data", "Fetch it.", network_hosts=["api.example.com"])
    )
    assert "exact network destinations: api.example.com" in prompt
    assert "never request a wildcard" in prompt


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
