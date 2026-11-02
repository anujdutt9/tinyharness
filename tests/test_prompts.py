"""Prompt files load, and every placeholder is filled by the code that uses them."""

import pytest

from tinyharness.agent.prompts import build_system_prompt
from tinyharness.prompts import PROMPTS_DIR, load_prompt, placeholders
from tinyharness.subagents import builtin_agents

# Every prompt file and the variables its caller provides.
PROMPT_VARIABLES = {
    "system": {"model_name", "model_id", "provider"},
    "compaction": set(),
    "compaction_system": set(),
    "compaction_update": set(),
    "classifier": set(),
    "aside": set(),
    "next_prompt": set(),
}


def test_every_prompt_file_is_covered():
    assert {p.stem for p in PROMPTS_DIR.glob("*.md")} == set(PROMPT_VARIABLES)


@pytest.mark.parametrize(("name", "variables"), PROMPT_VARIABLES.items())
def test_placeholders_match_callers(name, variables):
    assert placeholders(name) == variables
    rendered = load_prompt(name, **{v: f"<{v}>" for v in variables})
    assert rendered and "{" + "working_dir}" not in rendered


def test_missing_variable_is_an_error():
    with pytest.raises(KeyError, match="provider"):
        load_prompt("system", model_name="M", model_id="m")


def test_json_braces_are_left_alone():
    assert '{"risk": "low"' in load_prompt("classifier")


def test_system_prompt_includes_agents_md(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Always use tabs.")
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
    assert prompt.index("Be brief.") > prompt.index("Always use tabs.")


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
    assert "Claude Opus 5.5\n(model id `claude-opus-5-5`, via Anthropic)" in prompt


def _prompt(work):
    return build_system_prompt(work, work / "scratch", model_name="M", model_id="m", provider="P")


def test_agents_md_is_read(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Run make check before finishing.")
    assert "Run make check before finishing." in _prompt(tmp_path)


def test_the_model_is_told_when_there_is_no_instructions_file(tmp_path):
    """Otherwise it goes looking (a failed Read of AGENTS.md)."""
    assert "no instructions file" in _prompt(tmp_path)
    (tmp_path / "AGENTS.md").write_text("Use tabs.")
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


def test_pages_are_written_the_usual_way(tmp_path):
    """Seen live: told pages "must work offline" with a "bundled Chart.js", Sonnet ran
    `find /` for the file for two minutes, then tried to download it. Artifact validation
    needs a general contained workflow, not a rule for one browser or one demo."""
    prompt = load_prompt("system", **{v: f"<{v}>" for v in PROMPT_VARIABLES["system"]})
    assert "offline" not in prompt and "bundled" not in prompt and "Screenshot" not in prompt
    assert "non-interactive tool" in prompt and "private cache and $TMPDIR" in prompt
    assert "network_host" in prompt and "playwright" not in prompt.lower()


def test_declared_installs_are_kept_separate_and_use_the_venv_directly():
    """Seen live: bundling activation, Git/file setup and an editable install made its
    downloads look unrelated, so Auto asked about every later install and network read."""
    prompt = load_prompt("system", **{v: f"<{v}>" for v in PROMPT_VARIABLES["system"]})
    assert "declared dependency install in its own Bash call" in prompt
    assert "--python .venv/bin/python" in prompt


def test_reading_a_file_is_not_described_as_showing_it_to_the_user():
    """Seen live: the model read a 372-line report, then falsely said it was shown above."""
    prompt = load_prompt("system", **{v: f"<{v}>" for v in PROMPT_VARIABLES["system"]})
    assert "never claim" in prompt and "merely because you used Read" in prompt
    assert "Open is always available" in prompt


def test_the_system_prompt_carries_no_one_demos_instructions():
    """A business-database Team example, "delegate without being asked", "re-run a
    subagent's query" and "never do mental math" were in every session's prompt, from
    one data-analysis demo. They pushed models into costly teams and double work."""
    prompt = load_prompt("system", **{v: f"<{v}>" for v in PROMPT_VARIABLES["system"]})
    for phrase in ("without being asked", "customers and", "run the query again", "Mental math"):
        assert phrase not in prompt


def test_what_changes_per_session_comes_last(tmp_path):
    """The scratch folder is new in every session; placed mid-prompt it made everything
    after it (AGENTS.md, skills, extension sections) miss the provider's cache each time."""
    a = build_system_prompt(tmp_path, tmp_path / "s1", model_name="M", model_id="m", provider="P", extra="X")
    b = build_system_prompt(tmp_path, tmp_path / "s2", model_name="M", model_id="m", provider="P", extra="X")
    shared = a[: a.index("# This session")]
    assert b.startswith(shared) and "X" in shared and a.rstrip().endswith(a.splitlines()[-1])
    assert a.splitlines()[-1].startswith("Today's date:")


def test_scratch_is_for_throwaway_files_and_what_builds_a_deliverable_stays_with_it(tmp_path):
    """Seen live: a notebook landed in the project but its 748-line generator went to scratch,
    deleted at session end, so the notebook couldn't be regenerated or changed from it."""
    prompt = build_system_prompt(tmp_path, tmp_path / "s", model_name="M", model_id="m", provider="p")
    assert "scripts and sources that produce them" in prompt
    assert "only for throwaway files" in prompt and "don't point them here" in prompt


def test_an_unfamiliar_name_is_looked_up_before_asking_or_guessing():
    """Seen live: asked to fine-tune "a laya decision model", the agent asked what laya meant
    and built the whole brainstorm on a wrong guess, with WebSearch available all along."""
    prompt = load_prompt("system", **{v: f"<{v}>" for v in PROMPT_VARIABLES["system"]})
    assert "don't recognize" in prompt and "look it up with WebSearch or WebFetch before asking" in prompt
    # Seen live next: it looked Laya up, then opened by explaining Laya to the user who named it.
    assert "for your own understanding" in prompt and "only when\n  the user asked for it" in prompt


def test_answers_lead_with_the_answer_and_dont_restate_what_the_user_knows():
    """Seen live: after looking up "laya", the reply opened with five bullets explaining Laya
    to the user who had named it, before getting to the use cases they asked for."""
    prompt = load_prompt("system", **{v: f"<{v}>" for v in PROMPT_VARIABLES["system"]})
    assert "lead with the answer" in prompt and "Don't restate what the user already knows" in prompt
