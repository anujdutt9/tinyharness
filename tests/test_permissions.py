import pytest

from tinyharness.hooks import Hook, HookRunner
from tinyharness.permissions import Decision, PermissionGate, PermissionMode, PermissionRules, Rule
from tinyharness.tools.agent import AgentTool
from tinyharness.tools.read import ReadTool
from tinyharness.tools.write import WriteTool
from tinyharness.workspace import Scratch, Workspace

READ, WRITE = ReadTool(), WriteTool()


class User:
    """A scripted permission callback that records what it was asked."""

    def __init__(self, *answers: Decision) -> None:
        self.answers = list(answers)
        self.requests = []

    async def __call__(self, request):
        self.requests.append(request)
        return self.answers.pop(0) if self.answers else Decision(True, "ok", "user")


YES = Decision(True, "ok", "user")
REMEMBER = Decision(True, "ok", "user", remember=True)
NO = Decision(False, "no", "user")


def gate(tmp_path, mode=PermissionMode.DEFAULT, **kwargs) -> PermissionGate:
    return PermissionGate(workspace=Workspace(tmp_path), mode=mode, **kwargs)


# ---- what needs asking ----


async def test_reads_inside_the_project_never_prompt(tmp_path):
    user = User()
    d = await gate(tmp_path, callback=user).check(READ, {}, "src/app.py")
    assert d.allow and d.source == "mode"
    assert user.requests == []


async def test_side_effect_free_tools_without_a_path_never_prompt(tmp_path):
    agent_tool = AgentTool({}, spawner=None)
    assert (await gate(tmp_path).check(agent_tool, {}, None)).allow


async def test_reads_outside_the_project_prompt(tmp_path):
    user = User(YES)
    d = await gate(tmp_path, callback=user).check(READ, {}, "/etc/hosts")
    assert d.allow and len(user.requests) == 1


async def test_edits_prompt_in_default_mode(tmp_path):
    user = User(YES)
    await gate(tmp_path, callback=user).check(WRITE, {}, "a.py", summary="Write(a.py, 1 lines)")
    [request] = user.requests
    assert request.summary == "Write(a.py, 1 lines)"
    assert request.remember_label == "allow all edits in this project for this session"


async def test_no_callback_means_deny(tmp_path):
    d = await gate(tmp_path).check(WRITE, {}, "a.py")
    assert not d.allow and d.source == "prompt"


# ---- modes ----


async def test_accept_edits_only_inside_the_project(tmp_path):
    g = gate(tmp_path, PermissionMode.ACCEPT_EDITS)
    assert (await g.check(WRITE, {}, "in.txt")).source == "mode"
    assert (await g.check(WRITE, {}, "/tmp/elsewhere")).source == "prompt"


async def test_bypass_allows_everything(tmp_path):
    assert (await gate(tmp_path, PermissionMode.BYPASS).check(WRITE, {}, "/etc/x")).allow


async def test_plan_blocks_writes(tmp_path):
    d = await gate(tmp_path, PermissionMode.PLAN, callback=User(YES)).check(WRITE, {}, "x.txt")
    assert not d.allow and d.source == "mode"


def test_shift_tab_cycle():
    mode = PermissionMode.DEFAULT
    seen = [mode := mode.next() for _ in range(3)]
    assert seen == [PermissionMode.ACCEPT_EDITS, PermissionMode.PLAN, PermissionMode.DEFAULT]


# ---- "don't ask again" ----


async def test_remembering_an_edit_switches_to_accept_edits(tmp_path):
    user = User(REMEMBER)
    g = gate(tmp_path, callback=user)
    await g.check(WRITE, {}, "src/a.py")
    assert g.mode == PermissionMode.ACCEPT_EDITS
    assert (await g.check(WRITE, {}, "tests/b.py")).allow  # different dir, same tool family
    assert len(user.requests) == 1


async def test_remembering_an_outside_path_covers_that_directory(tmp_path_factory):
    project, outside = tmp_path_factory.mktemp("project"), tmp_path_factory.mktemp("downloads")
    user = User(REMEMBER)
    g = gate(project, callback=user)
    assert user.answers  # sanity
    await g.check(READ, {}, str(outside / "a.pdf"))
    assert user.requests[0].remember_label == f"allow Read in {outside}/ for this session"
    d = await g.check(READ, {}, str(outside / "b.pdf"))
    assert d.allow and d.source == "rule"
    assert g.mode == PermissionMode.DEFAULT  # outside paths never flip the mode


async def test_plain_yes_is_not_remembered(tmp_path):
    user = User(YES, YES)
    g = gate(tmp_path, callback=user)
    await g.check(WRITE, {}, "a.py")
    await g.check(WRITE, {}, "a.py")
    assert len(user.requests) == 2


async def test_denials_are_not_remembered(tmp_path):
    user = User(NO, YES)
    g = gate(tmp_path, callback=user)
    assert not (await g.check(WRITE, {}, "a.py")).allow
    assert (await g.check(WRITE, {}, "a.py")).allow


# ---- configured rules ----


@pytest.mark.parametrize(
    ("rule", "tool", "path", "expected"),
    [
        ("Edit", "Edit", "/p/any.py", True),
        ("Edit", "Write", "/p/any.py", False),
        ("Edit(src/*)", "Edit", "/p/src/deep/x.py", True),
        ("Edit(src/*)", "Edit", "/p/tests/x.py", False),
        ("*(/etc/*)", "Read", "/etc/hosts", True),
        ("Read(*.env)", "Read", "/p/.env", True),
    ],
)
def test_rule_matching(rule, tool, path, expected):
    from pathlib import Path

    assert Rule.parse(rule, Path("/p")).matches(tool, path) is expected


def test_invalid_rule_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        Rule.parse("Edit(src/*", tmp_path)


async def test_config_rules_allow_and_deny_beats_allow(tmp_path):
    rules = PermissionRules.parse(allow=["Write"], deny=["Write(secrets/*)"], root=tmp_path)
    g = gate(tmp_path, rules=rules, callback=User())
    assert (await g.check(WRITE, {}, "notes.md")).source == "rule"
    d = await g.check(WRITE, {}, "secrets/key.txt")
    assert not d.allow and d.source == "rule"


async def test_deny_rules_apply_even_to_reads(tmp_path):
    rules = PermissionRules.parse(allow=[], deny=["Read(private/*)"], root=tmp_path)
    assert not (await gate(tmp_path, rules=rules).check(READ, {}, "private/x.md")).allow


# ---- other layers ----


async def test_scratch_dir_is_writable_without_asking(tmp_path):
    scratch = Scratch(tmp_path, "s1")
    d = await gate(tmp_path, scratch=scratch).check(WRITE, {}, str(scratch.dir / "notes.md"))
    assert d.allow


async def test_skill_assets_are_readable_without_asking(tmp_path):
    project, skill_dir = tmp_path / "proj", tmp_path / "skill"
    project.mkdir()
    skill_dir.mkdir()
    g = PermissionGate(workspace=Workspace(project, read_roots=[skill_dir]))
    assert (await g.check(READ, {}, str(skill_dir / "ref.md"))).allow


async def test_hook_deny_beats_bypass(tmp_path):
    hooks = HookRunner([Hook(event="PreToolUse", type="python", callable=lambda e, p: {"decision": "deny"})])
    d = await gate(tmp_path, PermissionMode.BYPASS, hooks=hooks).check(WRITE, {}, "x.txt")
    assert not d.allow and d.source == "hook"
