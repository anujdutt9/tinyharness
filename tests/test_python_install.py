"""Cross-platform preparation of Python package installations."""

import os
import shlex
import sys
from pathlib import Path

import pytest

from conftest import make_ctx
from tinyharness.permissions import PermissionGate, PermissionMode
from tinyharness.sandbox import Sandbox, SandboxPolicy
from tinyharness.tools.bash import BashInput, BashTool, _python_install_environment
from tinyharness.workspace import Workspace


def environment_python(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


class FakeSandbox(Sandbox):
    name = "fake"

    @classmethod
    def available(cls) -> bool:
        return True

    def wrap(self, argv, policy):
        return argv


def test_fresh_environment_keeps_the_bootstrap_interpreter_separate(tmp_path, monkeypatch):
    """Seen live: rewriting sys.executable made the not-yet-created venv bootstrap itself."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    command, _, environment = _python_install_environment(
        f"{shlex.quote(sys.executable)} -m pip install requests", SandboxPolicy.for_project(tmp_path)
    )
    assert environment is not None
    bootstrap, install = command.split(" && ", 1)
    selected = environment_python(environment)
    assert " -m venv " in bootstrap and str(selected) not in bootstrap
    assert install == f"{shlex.quote(str(selected))} -m pip install requests"


def test_only_the_parsed_install_launcher_is_rewritten(tmp_path, monkeypatch):
    """Seen live: an earlier echo was rewritten while the package installer was left unchanged."""
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    environment = tmp_path / ".venv"
    selected = environment_python(environment)
    selected.parent.mkdir(parents=True)
    selected.touch()
    launcher = environment_python(tmp_path.parent / "launcher")
    original = f"echo {shlex.quote(str(launcher))} && {shlex.quote(str(launcher))} -m pip install requests"
    command, _, chosen = _python_install_environment(original, SandboxPolicy.for_project(tmp_path))
    assert chosen == environment
    assert command == (
        f"echo {shlex.quote(str(launcher))} && {shlex.quote(str(selected))} -m pip install requests"
    )


def test_nested_env_split_launcher_is_rewritten_at_its_parsed_location(tmp_path, monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    environment = tmp_path / ".venv"
    selected = environment_python(environment)
    selected.parent.mkdir(parents=True)
    selected.touch()
    launcher = environment_python(tmp_path.parent / "launcher")
    original = f'env -S "{launcher.as_posix()} -m pip install requests"'
    command, _, _ = _python_install_environment(original, SandboxPolicy.for_project(tmp_path))
    assert str(launcher) not in command and str(selected) in command


@pytest.mark.parametrize("option", ["-S", "--split-string"])
async def test_env_split_string_install_requires_approval(tmp_path, option):
    """Seen live: env -S could execute pip install without an acquisition boundary."""
    tool = BashTool(object(), SandboxPolicy.for_project(tmp_path))
    args = BashInput(command=f'env {option} "pip install requests"')
    boundary = tool.approval_boundary(args)
    assert boundary is not None
    gate = PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.DEFAULT)
    decision = await gate.check(tool, args.model_dump(), None, approval_boundary=boundary)
    assert not decision.allow and decision.source == "prompt"


def test_selected_environment_paths_match_the_platform(tmp_path, monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    _, env, environment = _python_install_environment(
        "pip install requests", SandboxPolicy.for_project(tmp_path)
    )
    assert environment is not None
    assert Path(env["PATH"].split(os.pathsep)[0]) == environment_python(environment).parent


async def test_install_actually_creates_the_platform_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.delenv("CONDA_PREFIX", raising=False)
    tool = BashTool(FakeSandbox(), SandboxPolicy.for_project(tmp_path))
    output = await tool.run(
        BashInput(command="pip install --no-index package-that-does-not-exist"), make_ctx(tmp_path)
    )
    assert environment_python(tmp_path / ".venv").is_file()
    assert "No matching distribution" in output and "exit code" in output
