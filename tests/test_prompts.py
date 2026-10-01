"""Prompt files load, and every placeholder is filled by the code that uses them."""

import pytest

from tinyharness.agent.prompts import build_system_prompt
from tinyharness.prompts import PROMPTS_DIR, load_prompt, placeholders
from tinyharness.subagents import builtin_agents

# Every prompt file and the variables its caller provides.
PROMPT_VARIABLES = {
    "system": {"working_dir", "scratch_dir", "model_name", "model_id", "provider", "date", "assets_dir"},
    "compaction": set(),
    "classifier": set(),
}


def test_every_prompt_file_is_covered():
    assert {p.stem for p in PROMPTS_DIR.glob("*.md")} == set(PROMPT_VARIABLES)


@pytest.mark.parametrize(("name", "variables"), PROMPT_VARIABLES.items())
def test_placeholders_match_callers(name, variables):
    assert placeholders(name) == variables
    rendered = load_prompt(name, **{v: f"<{v}>" for v in variables})
    assert rendered and "{" + "working_dir}" not in rendered


def test_missing_variable_is_an_error():
    with pytest.raises(KeyError, match="scratch_dir"):
        load_prompt("system", working_dir="/repo")


def test_json_braces_are_left_alone():
    assert '{"decision": "allow"' in load_prompt("classifier")


def test_system_prompt_includes_agent_md(tmp_path):
    (tmp_path / "AGENT.md").write_text("Always use tabs.")
    prompt = build_system_prompt(
        tmp_path,
        tmp_path / "scratch",
        model_name="Model X",
        model_id="x-1",
        provider="Acme",
        extra="Be brief.",
    )
    assert f"Working directory: {tmp_path}" in prompt
    assert "Always use tabs." in prompt
    assert prompt.endswith("Be brief.")


def test_builtin_agents_load_from_markdown():
    agents = builtin_agents()
    assert set(agents) == {"explore", "plan", "general-purpose", "researcher"}
    assert "Write" not in agents["explore"].tools
    assert "Write" not in agents["researcher"].tools and "WebSearch" in agents["researcher"].tools
    assert all(a.permission_mode is None for a in agents.values())  # they follow the session's mode
    assert "Bash" in agents["general-purpose"].tools  # delegated coding work can run tests
    assert "Bash" not in agents["explore"].tools and "Bash" not in agents["plan"].tools
    assert all(a.description and a.system_prompt for a in agents.values())


def test_system_prompt_names_the_model(tmp_path):
    prompt = build_system_prompt(
        tmp_path,
        tmp_path / "s",
        model_name="Claude Opus 5.5",
        model_id="claude-opus-5-5",
        provider="Anthropic",
    )
    assert "Claude Opus 5.5 (model id `claude-opus-5-5`, provided by Anthropic)" in prompt


def _prompt(work):
    return build_system_prompt(work, work / "scratch", model_name="M", model_id="m", provider="P")


def test_agents_md_is_read_like_agent_md(tmp_path):
    """AGENTS.md is what most other agents read; a repo set up for them works as is."""
    (tmp_path / "AGENTS.md").write_text("Run make check before finishing.")
    assert "Run make check before finishing." in _prompt(tmp_path)


def test_the_model_is_told_when_there_is_no_instructions_file(tmp_path):
    """Otherwise it goes looking (a failed Read of AGENTS.md)."""
    assert "no instructions file" in _prompt(tmp_path)
    (tmp_path / "AGENT.md").write_text("Use tabs.")
    assert "no instructions file" not in _prompt(tmp_path)


def test_the_prompt_says_which_os_shell_and_git_state_it_runs_in(tmp_path):
    import platform
    import subprocess

    from tinyharness.agent.prompts import environment

    assert "not a git repository" in environment(tmp_path)
    subprocess.run(["git", "init", "-q", "-b", "work"], cwd=tmp_path, check=True)
    env = environment(tmp_path)
    assert "on branch `work`" in env and "Shell for Bash:" in env
    if platform.system() == "Darwin":
        assert "sed -i ''" in env


def test_subagents_get_the_projects_instructions(tmp_path):
    from conftest import make_ctx
    from tinyharness.subagents.runner import subagent_system_prompt
    from tinyharness.subagents.specs import AgentSpec

    (tmp_path / "AGENTS.md").write_text("Run tests with `make test`. Use pnpm, never npm.")
    prompt = subagent_system_prompt(AgentSpec("worker", "does work"), make_ctx(tmp_path))
    assert "use pnpm, never npm" in prompt.lower() and str(tmp_path) in prompt
