"""Prompt files load, and every placeholder is filled by the code that uses them."""

import pytest

from tinyharness.agent.prompts import build_system_prompt
from tinyharness.prompts import PROMPTS_DIR, load_prompt, placeholders
from tinyharness.subagents import builtin_agents

# Every prompt file and the variables its caller provides.
PROMPT_VARIABLES = {
    "system": {"working_dir", "scratch_dir", "model_name", "model_id", "provider"},
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
    assert set(agents) == {"explore", "plan", "general-purpose"}
    assert "Write" not in agents["explore"].tools
    assert agents["general-purpose"].permission_mode == "default"
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
