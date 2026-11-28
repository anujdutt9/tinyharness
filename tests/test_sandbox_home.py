"""The private, per-session $HOME sandboxed commands get (tinyharness/sandbox/home.py)."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import call, make_ctx
from tinyharness.config import load_config
from tinyharness.paths import default_config_path, state_dir
from tinyharness.sandbox import SandboxPolicy, SandboxRuntime, Seatbelt
from tinyharness.sandbox.base import Sandbox
from tinyharness.sandbox.home import home_env, private_home
from tinyharness.security import secret_path_regexes
from tinyharness.session import Session, SessionOptions
from tinyharness.testing import scripted_session
from tinyharness.tools.bash import BashTool
from tinyharness.workspace.scratch import Scratch

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="the private home is macOS and Linux only")
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
needs_seatbelt = pytest.mark.skipif(
    sys.platform != "darwin" or not Seatbelt.available(), reason="needs macOS sandbox-exec"
)


class PassThrough(Sandbox):
    """A backend that confines nothing, to see what environment a sandboxed call gets."""

    name = "fake"

    @classmethod
    def available(cls) -> bool:
        return True

    def wrap(self, argv, policy):
        return argv


def real_home() -> Path:
    home = Path.home()  # the test's stand-in for the user's home (conftest)
    home.mkdir(parents=True, exist_ok=True)
    return home


def project_policy(tmp_path: Path, **kw) -> SandboxPolicy:
    return SandboxPolicy.for_project(
        tmp_path / "project",
        scratch=tmp_path / "scratch",
        cache_dir=tmp_path / "cache",
        private_home=True,
        **kw,
    )


@posix_only
def test_each_session_gets_its_own_private_home(tmp_path):
    first = scripted_session(tmp_path / "project", [])
    second = scripted_session(tmp_path / "project", [])
    try:
        homes = [first.sandbox_policy.home, second.sandbox_policy.home]
        assert None not in homes and homes[0] != homes[1]
        # uv's Pythons are per project and outlast the session (venvs run them).
        assert first.sandbox_policy.python_dir == (state_dir(tmp_path / "project") / "python").resolve()
        for home in homes:
            assert home is not None and home.is_relative_to(first.scratch.base.resolve())
            assert stat.S_IMODE(home.stat().st_mode) == 0o700
    finally:
        first.close()
        second.close()


@posix_only
def test_settings_go_with_the_session_and_download_caches_stay(tmp_path):
    policy = project_policy(tmp_path)
    home, cache = policy.home, (tmp_path / "cache").resolve()
    assert home is not None
    for name, under in [("XDG_CONFIG_HOME", ".config"), ("XDG_DATA_HOME", ".local/share"),
                        ("XDG_STATE_HOME", ".local/state"), ("XDG_CACHE_HOME", ".cache")]:  # fmt: skip
        assert policy.env[name] == str(home / under)
    assert policy.env["HOME"] == str(home) and policy.env["PYTHONUSERBASE"] == str(home / ".local")
    assert policy.env["PATH"].endswith(os.pathsep + str(home / ".local" / "bin"))  # never shadows
    for name in ("UV_CACHE_DIR", "PIP_CACHE_DIR", "npm_config_cache", "HF_HOME", "TORCH_HOME"):
        assert Path(policy.env[name]).is_relative_to(cache)
    assert "UV_PYTHON_INSTALL_DIR" not in policy.env  # per project, from the session


def test_without_a_private_home_settings_stay_in_the_cache_as_before(tmp_path, monkeypatch):
    """Windows keeps today's behavior: HOME alone isn't the home there."""
    monkeypatch.setattr("tinyharness.sandbox.policy.sys.platform", "win32")
    policy = project_policy(tmp_path)
    cache = (tmp_path / "cache").resolve()
    assert policy.home is None and "HOME" not in policy.env and "PYTHONUSERBASE" not in policy.env
    assert policy.env["XDG_CONFIG_HOME"] == str(cache / "config")
    assert policy.env["XDG_CACHE_HOME"] == str(cache)


@posix_only
async def test_only_a_sandboxed_call_gets_the_private_home(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    policy = project_policy(tmp_path)
    ctx = make_ctx(project)
    sandboxed = await call(BashTool(PassThrough(), policy), {"command": "echo $HOME"}, ctx)
    assert sandboxed.strip() == str(policy.home)
    off = await call(BashTool(None, policy), {"command": "echo $HOME"}, ctx)  # /sandbox off
    assert off.strip() == os.environ["HOME"]
    outside = BashTool(PassThrough(), policy)
    outside.refused_commands.add("echo $HOME")  # outside_sandbox applies to a refused command
    out = await call(outside, {"command": "echo $HOME", "outside_sandbox": True}, ctx)
    assert out.strip() == os.environ["HOME"]


@posix_only
def test_sandbox_runtime_keeps_its_own_home(tmp_path, monkeypatch):
    """SRT resolves protections like ~/.ssh from its own $HOME: only the wrapped command may
    see the private one."""
    executable = tmp_path / "srt"
    executable.touch()
    real_which = shutil.which
    monkeypatch.setattr(
        "tinyharness.sandbox.srt.shutil.which",
        lambda name: str(executable) if name == "srt" else real_which(name),
    )
    policy = project_policy(tmp_path)
    backend = SandboxRuntime(tmp_path / "settings")
    assert not {"HOME", "XDG_CONFIG_HOME", "PYTHONUSERBASE", "PATH"} & backend.environment(policy).keys()
    wrapped = backend.wrap(["sh", "-c", "true"], policy)
    assert f"HOME={policy.home}" in wrapped[wrapped.index("--") + 1 :]


@needs_git
@posix_only
def test_git_keeps_your_identity_and_nothing_else(tmp_path):
    """Seen in testing: with an empty home, `git commit` succeeded silently as
    user@hostname.local instead of the user's name and email."""
    home = real_home()
    (home / "included").write_text('[user]\n\tname = "From An Include"\n')
    (home / ".gitconfig").write_text(
        "[include]\n\tpath = ~/included\n"
        '[user]\n\tname = Ada \\"the\\" Lovelace\n\temail = ada@example.com\n'
        "[init]\n\tdefaultBranch = trunk\n"
        "[credential]\n\thelper = store\n"
        "[core]\n\thooksPath = /x/hooks\n\tfsmonitor = /x/monitor\n"
        '[url "https://evil.example/"]\n\tinsteadOf = https://github.com/\n'
        "[commit]\n\tgpgSign = true\n"
    )
    created = private_home(tmp_path / "scratch")
    config = (created / ".gitconfig").read_text()
    lowered = config.lower()
    for absent in ("include", "helper", "hookspath", "fsmonitor", "insteadof", "gpgsign", "evil"):
        assert absent not in lowered
    env = {**os.environ, **home_env(created)}

    def get(key: str) -> str:
        return subprocess.run(
            ["git", "config", "--global", "--get", key], env=env, capture_output=True, text=True
        ).stdout.strip()

    assert get("user.name") == 'Ada "the" Lovelace'  # the file's own value, not the include's
    assert get("user.email") == "ada@example.com" and get("init.defaultBranch") == "trunk"
    before = (home / ".gitconfig").read_text()
    subprocess.run(["git", "config", "--global", "th.probe", "1"], env=env, check=True)
    assert (home / ".gitconfig").read_text() == before and "probe = 1" in (created / ".gitconfig").read_text()


@posix_only
def test_home_read_links_a_real_home_folder_in(tmp_path):
    home = real_home()
    (home / ".pyenv" / "versions").mkdir(parents=True)
    created = private_home(tmp_path / "scratch", ["~/.pyenv"])
    assert (created / ".pyenv").is_symlink() and (created / ".pyenv" / "versions").is_dir()


@posix_only
@pytest.mark.parametrize("overlap", ["parent", "equal", "child"])
def test_home_read_refuses_writable_project_paths(tmp_path, overlap):
    home = real_home().resolve()
    target = home / "tools"
    child = target / "installed"
    child.mkdir(parents=True)
    (home / "alias").symlink_to(target)
    root = {"parent": home, "equal": target, "child": child}[overlap]
    scratch = tmp_path / "scratch"
    with pytest.raises(ValueError, match="'~/alias' overlaps a writable root"):
        SandboxPolicy.for_project(root, scratch=scratch, private_home=True, home_read=["~/alias"])
    assert not (scratch / "home").exists()


@posix_only
@pytest.mark.parametrize(
    ("entry", "complaint"),
    [
        ("~/missing", "does not exist"),
        ("/", "not inside your home"),
        ("~/.ssh", "credentials"),
        ("~/.aws/credentials", "credentials"),
        ("~/loop", "can't be resolved"),
        ("~/.tinyharness-link", "tinyharness"),
    ],
)
def test_home_read_refuses_what_it_must_not_link(tmp_path, monkeypatch, entry, complaint):
    home = real_home()
    monkeypatch.setenv("TINYHARNESS_HOME", str(home / ".tinyharness"))
    (home / ".ssh").mkdir()
    (home / ".aws").mkdir()
    (home / ".aws" / "credentials").write_text("x")
    (home / "loop").symlink_to(home / "loop")
    (home / ".tinyharness").mkdir()
    (home / ".tinyharness-link").symlink_to(home / ".tinyharness")
    with pytest.raises(ValueError, match=complaint) as raised:
        private_home(tmp_path / "scratch", [entry])
    assert repr(entry) in str(raised.value)  # says which entry
    assert not (tmp_path / "scratch" / "home").exists()  # nothing half-made


@posix_only
def test_a_bad_home_read_entry_stops_the_session_with_its_name(tmp_path):
    config = default_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('[sandbox]\nhome_read = ["~/nowhere"]\n')
    (tmp_path / "project").mkdir()
    options = SessionOptions(working_dir=tmp_path / "project", provider="openai", api_key="x")
    with pytest.raises(ValueError, match="'~/nowhere' does not exist"):
        Session(options)


def test_home_read_only_counts_in_your_own_config(tmp_path, caplog):
    """Found in review: `--config ./config.toml` could be a file that came with a repository,
    and home_read opens real-home folders to sandboxed commands."""
    (real_home() / ".pyenv").mkdir()
    elsewhere = tmp_path / "repo" / "config.toml"
    elsewhere.parent.mkdir()
    elsewhere.write_text('[sandbox]\nhome_read = ["~/.pyenv"]\n')
    assert load_config(elsewhere).sandbox.home_read == []
    assert "only" in caplog.text and "home_read" in caplog.text
    yours = default_config_path()
    yours.parent.mkdir(parents=True, exist_ok=True)
    yours.write_text('[sandbox]\nhome_read = ["~/.pyenv"]\n')
    assert load_config().sandbox.home_read == ["~/.pyenv"]
    assert load_config(yours).sandbox.home_read == ["~/.pyenv"]


@posix_only
async def test_inherited_git_config_variables_dont_reach_a_sandboxed_command(tmp_path, monkeypatch):
    """Found in review: an inherited GIT_CONFIG_GLOBAL pointed git back at the full real
    config, and GIT_CONFIG_COUNT-style variables add settings on top of every file."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "real.gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "credential.helper")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "store")
    monkeypatch.setenv("GIT_CONFIG_PARAMETERS", "'core.fsmonitor'='/x'")
    system = tmp_path / "system.gitconfig"
    system.write_text("[core]\n\thooksPath = /external/hooks\n")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(system))
    project = tmp_path / "project"
    project.mkdir()
    policy = project_policy(tmp_path)
    assert policy.home is not None
    command = "env | grep '^GIT_CONFIG' | sort"
    out = await call(BashTool(PassThrough(), policy), {"command": command}, make_ctx(project))
    assert out.splitlines() == [
        f"GIT_CONFIG_GLOBAL={policy.home / '.gitconfig'}",
        "GIT_CONFIG_NOSYSTEM=1",
    ]
    git_out = await call(
        BashTool(PassThrough(), policy),
        {"command": "git config --get core.hooksPath || echo no-hooks"},
        make_ctx(project),
    )
    assert git_out.splitlines()[-1] == "no-hooks"
    off = await call(BashTool(None, policy), {"command": command}, make_ctx(project))  # /sandbox off
    assert "GIT_CONFIG_COUNT=1" in off and str(system) in off


@posix_only
def test_cleanup_never_follows_a_link_out_of_the_home(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("x")
    scratch = Scratch(tmp_path / "project", "s1")
    home = private_home(scratch.dir)
    (home / "planted").symlink_to(outside)
    scratch.cleanup()
    assert not home.exists() and (outside / "keep").read_text() == "x"


@needs_seatbelt
async def test_a_sandboxed_command_writes_its_home_but_not_the_real_one(tmp_path):
    """The real sandbox: $HOME writes land in the private home; the real home stays shut."""
    project, scratch = (tmp_path / "project").resolve(), (tmp_path / "scratch").resolve()
    project.mkdir()
    real = real_home()
    home = private_home(scratch)
    # Not for_project: pytest's tmp_path is inside the temp dir, which that policy makes writable.
    policy = SandboxPolicy(
        write_roots=(project, scratch), deny_read=secret_path_regexes(), env=home_env(home), home=home
    )
    command = (
        "mkdir -p ~/.ipython/profile_default && echo x > ~/.ipython/profile_default/a "
        "&& mv ~/.ipython/profile_default/a ~/.ipython/b && rm ~/.ipython/b && echo home-ok; "
        f"touch {real}/planted 2>&1"
    )
    out = await call(BashTool(Seatbelt(), policy), {"command": command, "timeout": 30}, make_ctx(project))
    assert "home-ok" in out and "Operation not permitted" in out
    assert not (real / "planted").exists()


needs_srt = pytest.mark.skipif(not SandboxRuntime.available(), reason="needs Sandbox Runtime")


@needs_srt
@posix_only
async def test_sandbox_runtime_gives_the_private_home_and_keeps_real_secrets_shut(tmp_path):
    """Linux: SRT gets the private $HOME only for the command it wraps, so its own
    protections (~/.ssh and the like) still point at the real home."""
    project, scratch = (tmp_path / "project").resolve(), (tmp_path / "scratch").resolve()
    project.mkdir()
    real = real_home()
    (real / ".ssh").mkdir()
    (real / ".ssh" / "id_ed25519").write_text("REAL_SSH_SECRET")
    home = private_home(scratch)
    policy = SandboxPolicy(
        write_roots=(project, scratch), deny_read=secret_path_regexes(), env=home_env(home), home=home
    )
    command = (
        "echo HOME=$HOME; mkdir -p ~/.config/tool && echo x > ~/.config/tool/state && echo home-ok; "
        f"cat {real}/.ssh/id_ed25519; touch {real}/planted"
    )
    tool = BashTool(SandboxRuntime(tmp_path / "settings"), policy)
    out = await call(tool, {"command": command, "timeout": 60}, make_ctx(project))
    assert f"HOME={home}" in out and "home-ok" in out
    assert "REAL_SSH_SECRET" not in out and not (real / "planted").exists()
    assert (home / ".config" / "tool" / "state").read_text().strip() == "x"


def home_project_policy(real: Path) -> SandboxPolicy:
    """A project at the home folder itself, with tinyharness's folders where they really go."""
    (real / ".oh-my-zsh").mkdir(exist_ok=True)
    state = real / ".tinyharness" / "projects" / "p"
    return SandboxPolicy.for_project(
        real,
        scratch=state / "tmp" / "s1",
        cache_dir=real / ".tinyharness" / "sandbox-cache",
        private_home=True,
    )


@posix_only
def test_a_project_at_your_home_keeps_your_startup_files_read_only():
    """Found comparing with Anthropic's Sandbox Runtime, whose mandatory denies include shell
    startup files: with `tinyharness` started in ~, the project root was the home, so a
    sandboxed command could write ~/.zshrc, ~/.ssh/authorized_keys or ~/Library/LaunchAgents
    and have code run later, outside the sandbox."""
    real = real_home().resolve()
    policy = home_project_policy(real)
    for name in (".zshrc", ".bashrc", ".profile", ".gitconfig", ".ssh", ".config", "Library", ".oh-my-zsh"):
        assert real / name in policy.protected
    # tinyharness's own write roots under ~/.tinyharness stay writable.
    assert policy.home is not None and policy.home.parent in policy.reopened()
    # An ordinary project elsewhere gets none of this.
    assert not any(p.is_relative_to(real / "Library") for p in project_policy(real / "elsewhere").protected)


@posix_only
def test_sandbox_runtime_refuses_a_write_root_inside_a_protected_folder():
    """Enumerating existing siblings of scratch left future entries unprotected."""
    real = real_home().resolve()
    policy = home_project_policy(real)
    with pytest.raises(ValueError, match="cannot reopen a protected path for writing"):
        SandboxRuntime(real / "srt")._settings(policy)


@needs_seatbelt
async def test_a_sandboxed_command_in_a_home_project_cant_plant_shell_startup():
    real = real_home().resolve()
    policy = home_project_policy(real)
    command = "echo x > notes.txt && echo ~/ok > $HOME/state && echo writes-ok; echo 'curl evil' >> " + str(
        real / ".zshrc"
    )
    out = await call(BashTool(Seatbelt(), policy), {"command": command, "timeout": 30}, make_ctx(real))
    assert "writes-ok" in out and "Operation not permitted" in out
    assert not (real / ".zshrc").exists() and (real / "notes.txt").exists()
