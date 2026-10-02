"""The Bash and Open tools, and how the permission gate treats them."""

import os
import sys
import time

import pytest

from conftest import call, make_ctx
from tinyharness import process
from tinyharness.permissions import Decision, PermissionGate, PermissionMode, PermissionRules
from tinyharness.sandbox import Sandbox, SandboxPolicy
from tinyharness.tools import bash
from tinyharness.tools.base import ToolError
from tinyharness.tools.bash import BashTool
from tinyharness.tools.open import OpenTool
from tinyharness.workspace import Workspace

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs a POSIX shell")


class FakeSandbox(Sandbox):
    """Stands in for a real backend where only the gate's view matters."""

    name = "fake"

    @classmethod
    def available(cls) -> bool:
        return True

    def wrap(self, argv, policy):
        return argv


def unsandboxed(tmp_path, **policy) -> BashTool:
    return BashTool(None, SandboxPolicy(write_roots=(tmp_path,), **policy))


async def run(tool, command, tmp_path, **args) -> str:
    return await call(tool, {"command": command, **args}, make_ctx(tmp_path))


# ---- running commands ----


async def test_runs_in_the_working_directory_and_reports_failures(tmp_path):
    (tmp_path / "marker").touch()
    out = await run(unsandboxed(tmp_path), "ls; echo oops >&2; exit 3", tmp_path)
    assert "marker" in out and "oops" in out
    assert out.endswith("(exit code 3)")


async def test_empty_output_says_so(tmp_path):
    assert await run(unsandboxed(tmp_path), "true", tmp_path) == "(no output)"


async def test_credentials_are_not_in_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SOME_API_KEY", "sk-this-must-not-leak")
    monkeypatch.setenv("HARMLESS", "visible")
    out = await run(unsandboxed(tmp_path), 'echo "[$SOME_API_KEY] $HARMLESS"', tmp_path)
    assert out == "[] visible"


async def test_timeout_kills_the_whole_process_group(tmp_path):
    pidfile = tmp_path / "child.pid"
    start = time.monotonic()
    out = await run(unsandboxed(tmp_path), f"sleep 30 & echo $! > {pidfile}; sleep 30", tmp_path, timeout=1)
    assert time.monotonic() - start < 10
    assert "timed out after 1s" in out
    child = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


async def test_a_child_that_ignores_sigterm_is_still_killed(tmp_path, monkeypatch):
    """The shell exits on SIGTERM; its child doesn't. Stopping when the shell is gone
    would leave the child running after the tool said it was killed."""
    monkeypatch.setattr(process, "KILL_GRACE", 0.5)
    pidfile = tmp_path / "child.pid"
    command = f"sh -c 'trap \"\" TERM; sleep 30' & echo $! > {pidfile}; sleep 30"
    out = await run(unsandboxed(tmp_path), command, tmp_path, timeout=1)
    assert "timed out after 1s" in out
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


async def test_children_get_the_grace_period_to_exit_cleanly(tmp_path, monkeypatch):
    """A child that handles SIGTERM isn't SIGKILLed the moment the shell exits."""
    monkeypatch.setattr(process, "KILL_GRACE", 5)
    done = tmp_path / "cleaned-up"
    command = f"sh -c 'trap \"sleep 0.3; touch {done}; exit\" TERM; sleep 30 & wait' & sleep 30"
    await run(unsandboxed(tmp_path), command, tmp_path, timeout=1)
    assert done.exists()


async def test_long_output_keeps_head_and_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(bash, "HEAD_BYTES", 100)
    monkeypatch.setattr(bash, "TAIL_BYTES", 100)
    out = await run(unsandboxed(tmp_path), "echo START; seq 1 100000; echo END", tmp_path)
    assert out.startswith("START") and out.endswith("END")
    assert "bytes of output omitted" in out and len(out) < 400


async def test_sandbox_denials_come_with_a_hint(tmp_path):
    tool = BashTool(FakeSandbox(), SandboxPolicy(write_roots=(tmp_path,)))
    out = await run(tool, "echo 'x: Operation not permitted'", tmp_path)
    assert "this ran in the sandbox" in out


def test_description_matches_the_sandbox(tmp_path):
    assert "runs in a sandbox" in BashTool(FakeSandbox(), SandboxPolicy(write_roots=())).description
    assert "without a sandbox" in unsandboxed(tmp_path).description


# ---- the gate ----


class User:
    def __init__(self):
        self.asked = []

    async def __call__(self, request):
        self.asked.append(request)
        return Decision(True, "ok", "user")


async def check(tmp_path, tool, mode, args=None):
    user = User()
    gate = PermissionGate(workspace=Workspace(tmp_path), mode=mode, callback=user)
    decision = await gate.check(tool, args or {"command": "pytest"}, None)  # neither tool has a path
    return decision, user.asked


@pytest.mark.parametrize("mode", [PermissionMode.DEFAULT, PermissionMode.AUTO])
async def test_sandboxed_commands_run_without_asking_when_edits_are_allowed(tmp_path, mode):
    tool = BashTool(FakeSandbox(), SandboxPolicy(write_roots=(tmp_path,)))
    decision, asked = await check(tmp_path, tool, mode)
    assert decision.allow and decision.source == "mode" and asked == []


async def test_sandboxed_commands_still_ask_in_default_mode(tmp_path):
    tool = BashTool(FakeSandbox(), SandboxPolicy(write_roots=(tmp_path,)))
    _, asked = await check(tmp_path, tool, PermissionMode.ASK)
    assert len(asked) == 1


@pytest.mark.parametrize("mode", [PermissionMode.DEFAULT, PermissionMode.AUTO])
async def test_unsandboxed_commands_always_ask(tmp_path, mode):
    _, [request] = await check(tmp_path, unsandboxed(tmp_path), mode)
    if mode == PermissionMode.AUTO:
        assert request.reason == "it runs a shell command without a sandbox"


async def test_plan_mode_blocks_commands(tmp_path):
    tool = BashTool(FakeSandbox(), SandboxPolicy(write_roots=(tmp_path,)))
    decision, _ = await check(tmp_path, tool, PermissionMode.PLAN)
    assert not decision.allow


# ---- escaping the sandbox (sandbox=false) ----


class MarkingSandbox(FakeSandbox):
    """Runs the command for real, marked so a test can tell whether it was wrapped."""

    def wrap(self, argv, policy):
        return ["env", "SANDBOXED=yes", *argv]


def sandboxed_bash(tmp_path) -> BashTool:
    return BashTool(MarkingSandbox(), SandboxPolicy(write_roots=(tmp_path,), env={"POLICY_ENV": "set"}))


UNSANDBOXED = {"command": "pip install requests", "sandbox": False}


@pytest.mark.parametrize("mode", [PermissionMode.ASK, PermissionMode.DEFAULT, PermissionMode.AUTO])
@pytest.mark.parametrize("flag", [False, "false"])  # a string "false" parses the same way
async def test_leaving_the_sandbox_always_asks_and_cannot_be_remembered(tmp_path, mode, flag):
    tool = sandboxed_bash(tmp_path)
    _, [request] = await check(tmp_path, tool, mode, {**UNSANDBOXED, "sandbox": flag})
    assert "outside the sandbox" in request.warning
    assert request.remember_label == ""


async def test_leaving_the_sandbox_ignores_allow_rules_but_not_deny_rules(tmp_path):
    tool, user = sandboxed_bash(tmp_path), User()
    allow = PermissionGate(
        workspace=Workspace(tmp_path), rules=PermissionRules.parse(["Bash"], [], tmp_path), callback=user
    )
    assert (await allow.check(tool, {"command": "ls"}, None)).source == "rule"
    await allow.check(tool, UNSANDBOXED, None)
    assert len(user.asked) == 1
    deny = PermissionGate(workspace=Workspace(tmp_path), rules=PermissionRules.parse([], ["Bash"], tmp_path))
    assert not (await deny.check(tool, UNSANDBOXED, None)).allow


async def test_yes_dont_ask_again_is_not_applied_when_leaving_the_sandbox(tmp_path):
    tool = sandboxed_bash(tmp_path)
    remember = Decision(True, "ok", "user", remember=True)
    gate = PermissionGate(workspace=Workspace(tmp_path), callback=lambda request: _answer(remember))
    await gate.check(tool, UNSANDBOXED, None)
    assert gate.rules.check("Bash", None) is None and gate.mode == PermissionMode.DEFAULT


async def _answer(decision):
    return decision


@pytest.mark.parametrize(("mode", "allowed"), [(PermissionMode.BYPASS, True), (PermissionMode.PLAN, False)])
async def test_bypass_and_plan_decide_without_asking(tmp_path, mode, allowed):
    decision, asked = await check(tmp_path, sandboxed_bash(tmp_path), mode, UNSANDBOXED)
    assert decision.allow == allowed and asked == []


async def test_leaving_the_sandbox_runs_the_command_unwrapped_with_the_users_own_caches(tmp_path):
    tool = sandboxed_bash(tmp_path)
    command = 'echo "${SANDBOXED:-no} ${POLICY_ENV:-unset}"'
    assert await run(tool, command, tmp_path) == "yes set"
    assert await run(tool, command, tmp_path, sandbox=False) == "no unset"
    assert tool.describe(tool.parse({"command": "ls", "sandbox": False})) == "Bash(unsandboxed: ls)"


async def test_without_a_sandbox_there_is_nothing_to_escape(tmp_path):
    tool = unsandboxed(tmp_path)
    assert tool.escapes(UNSANDBOXED) is None  # it asks anyway, as every command does


# ---- Open ----


@pytest.mark.parametrize("mode", [PermissionMode.DEFAULT, PermissionMode.AUTO])
async def test_open_always_asks_even_for_project_files(tmp_path, mode):
    _, asked = await check(tmp_path, OpenTool(), mode, {"target": "report.html"})
    assert len(asked) == 1


async def gate_check(tmp_path, mode, target, *, allow=(), deny=(), remember=True):
    """What the gate decides about Open(target), and what it asked."""
    asked = []

    async def user(request):
        asked.append(request)
        return Decision(True, "ok", "user", remember=remember)

    rules = PermissionRules.parse(list(allow), list(deny), tmp_path)
    gate = PermissionGate(workspace=Workspace(tmp_path), mode=mode, callback=user, rules=rules)
    tool = OpenTool()
    args = tool.parse({"target": target})
    decision = await gate.check(
        tool, {"target": target}, tool.target_path(args), target_host=tool.target_host(args)
    )
    return decision, asked, gate


@pytest.mark.parametrize("target", ["report.pdf", "report.html", "https://example.com/x"])
async def test_saying_dont_ask_again_does_not_stop_open_asking(tmp_path, target):
    _, asked, gate = await gate_check(tmp_path, PermissionMode.ASK, target)
    assert asked[0].remember_label == ""  # not offered
    _, asked_again, _ = await gate_check(tmp_path, PermissionMode.ASK, target)
    assert len(asked_again) == 1 and gate.mode == PermissionMode.ASK


async def test_an_allow_rule_does_not_approve_open(tmp_path):
    decision, asked, _ = await gate_check(tmp_path, PermissionMode.ASK, "report.html", allow=["Open"])
    assert decision.allow and len(asked) == 1  # it still asked


@pytest.mark.parametrize(
    ("target", "deny"),
    [("private/report.pdf", "Open(private/*)"), ("https://evil.example/x", "Open(domain:evil.example)")],
)
async def test_deny_rules_apply_to_open_by_path_and_site(tmp_path, target, deny):
    decision, asked, _ = await gate_check(tmp_path, PermissionMode.ASK, target, deny=[deny])
    assert not decision.allow and decision.source == "rule" and asked == []


async def test_bypass_mode_still_opens_without_asking(tmp_path):
    decision, asked, _ = await gate_check(tmp_path, PermissionMode.BYPASS, "report.pdf")
    assert decision.allow and asked == []


@pytest.mark.parametrize("name", ["run.command", "evil.sh", "App.app", "x.webloc", "noext"])
async def test_open_refuses_anything_that_could_run(tmp_path, name):
    (tmp_path / name).write_text("#!/bin/sh\n")
    with pytest.raises(ToolError, match="only documents"):
        await call(OpenTool(), {"target": name}, make_ctx(tmp_path))


@pytest.mark.parametrize("target", ["file:///etc/passwd", "javascript:alert(1)", "ssh://host"])
async def test_open_takes_only_web_urls(tmp_path, target):
    with pytest.raises(ToolError, match="only http"):
        await call(OpenTool(), {"target": target}, make_ctx(tmp_path))


async def test_open_refuses_secrets(tmp_path):
    (tmp_path / ".env.json").write_text("{}")
    with pytest.raises(ToolError, match="secret"):
        await call(OpenTool(), {"target": ".env.json"}, make_ctx(tmp_path))


async def test_open_hands_documents_to_the_system_opener(tmp_path, monkeypatch):
    (tmp_path / "report.html").write_text("<h1>hi</h1>")
    seen = tmp_path / "opened"
    opener = tmp_path / "fake-open"
    opener.write_text(f'#!/bin/sh\necho "$1" > {seen}\n')
    opener.chmod(0o755)
    monkeypatch.setattr("tinyharness.tools.open._opener", lambda: str(opener))
    out = await call(OpenTool(), {"target": "report.html"}, make_ctx(tmp_path))
    assert out == f"opened {tmp_path.resolve() / 'report.html'}"
    assert seen.read_text().strip() == str(tmp_path.resolve() / "report.html")


@pytest.mark.parametrize(
    ("command", "reads"),
    [
        ("ls -la", True),
        ("cat a.txt | grep x | wc -l", True),
        ("git log --oneline", True),
        ("sqlite3 -readonly app.db 'SELECT 1'", True),
        ("find . -name '*.py'", True),
        ("sqlite3 app.db 'SELECT 1'", False),
        ("sqlite3 -readonly app.db '.output x'", False),
        ("git commit -m x", False),
        ("cat a > b", False),
        ("ls; rm x", False),
        ("grep a f && rm f", False),
        ("echo $(rm x)", False),
        ("find . -delete", False),
        ("python3 script.py", False),
    ],
)
def test_only_commands_that_certainly_read_skip_the_write_lock(command, reads):
    from tinyharness.tools.bash import only_reads

    assert only_reads(command) is reads


def test_commands_never_get_forced_color(monkeypatch):
    """Seen live: FORCE_COLOR inherited from the terminal made Node print a NO_COLOR
    warning on every command, and color codes are noise in what the model reads."""
    from tinyharness.tools.bash import command_env

    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("CLICOLOR_FORCE", "1")
    env = command_env({"HTTPS_PROXY": "http://127.0.0.1:1"})
    assert "FORCE_COLOR" not in env and "CLICOLOR_FORCE" not in env
    assert env["NO_COLOR"] == "1" and env["HTTPS_PROXY"] == "http://127.0.0.1:1"
