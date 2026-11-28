"""The OS sandbox: shared policy, platform backends, and real macOS confinement."""

import json
import os
import re
import shlex
import sys
from pathlib import Path

import pytest

from conftest import call, make_ctx
from tinyharness.paths import home_dir, state_dir
from tinyharness.sandbox import BACKENDS, SandboxPolicy, SandboxRuntime, Seatbelt, setup_hint
from tinyharness.sandbox.policy import PROTECTED_IN_PROJECT
from tinyharness.sandbox.seatbelt import profile
from tinyharness.security import public_path_regexes, secret_path_regexes
from tinyharness.tools.bash import BashTool

posix = pytest.mark.skipif(sys.platform == "win32", reason="the fake srt is a shell script")

not_windows = pytest.mark.skipif(sys.platform == "win32", reason="tests the macOS Seatbelt profile")


def native_text(path: object) -> str:
    return str(path).replace("\\", "/")


# ---- the policy ----


def test_native_seatbelt_is_preferred_when_macos_has_both_backends():
    assert (Seatbelt, SandboxRuntime) == BACKENDS


def test_missing_sandbox_has_platform_specific_setup_help(monkeypatch):
    # SRT isn't installed. (Faking sys.platform also sends shutil.which down Windows paths.)
    monkeypatch.setattr("tinyharness.sandbox.unusable_hint", lambda: None)
    monkeypatch.setattr("tinyharness.sandbox.sys.platform", "linux")
    assert "bubblewrap" in setup_hint() and "sandbox-runtime" in setup_hint()
    monkeypatch.setattr("tinyharness.sandbox.sys.platform", "win32")
    assert "Git Bash" in setup_hint() and "windows-install" in setup_hint()


def test_linux_setup_help_names_the_install_command_for_its_package_manager(monkeypatch):
    """A banner that only says "not available" leaves you to find the packages yourself."""
    monkeypatch.setattr("tinyharness.sandbox.sys.platform", "linux")
    monkeypatch.setattr(
        "tinyharness.sandbox.shutil.which", lambda name: "/usr/bin/dnf" if name == "dnf" else None
    )
    assert setup_hint().startswith("sudo dnf install bubblewrap socat ripgrep, then npm install")


def test_project_policy_uses_private_scratch_temp_and_tool_caches(tmp_path):
    root = tmp_path / "proj"
    scratch = root / ".tinyharness" / "tmp" / "s1"
    cache = tmp_path / "cache"
    policy = SandboxPolicy.for_project(root, scratch=scratch, cache_dir=cache)
    real = root.resolve()
    assert real in policy.write_roots and scratch.resolve() in policy.write_roots
    assert cache.resolve() in policy.write_roots and cache.is_dir()
    assert Path("/tmp").resolve() not in policy.write_roots
    assert {real / ".tinyharness", real / ".git/hooks", real / ".git/config", real / ".vscode"} <= set(
        policy.protected
    )
    assert policy.reopened() == (scratch.resolve(),)  # scratch sits under the protected .tinyharness
    assert policy.env["UV_CACHE_DIR"] == str(cache.resolve() / "uv")
    assert policy.env["PLAYWRIGHT_BROWSERS_PATH"] == str(cache.resolve() / "playwright")
    assert policy.env["HF_HOME"] == str(cache.resolve() / "huggingface")
    assert policy.env["TMPDIR"] == str(scratch.resolve() / "tmp")
    assert (scratch / "tmp").is_dir()
    fixture = scratch.resolve() / "tmp" / "pytest-of-user" / "pytest-0" / "test_auth" / "auth.json"
    assert any(re.search(pattern, fixture.as_posix()) for pattern in policy.allow_read)
    srt_scratch = tmp_path / "srt-scratch"
    srt_policy = SandboxPolicy.for_project(root, scratch=srt_scratch)
    settings = SandboxRuntime(tmp_path / "settings")._settings(srt_policy)
    assert f"{srt_scratch.resolve().as_posix()}/tmp/pytest-of-*/**" in [
        native_text(path) for path in settings["filesystem"]["allowRead"]
    ]
    assert not policy.allow_localhost


def test_sandbox_runtime_translates_the_shared_policy(tmp_path, monkeypatch):
    executable = tmp_path / "srt"
    executable.touch()
    monkeypatch.setattr("tinyharness.sandbox.srt.shutil.which", lambda name: str(executable))
    root = tmp_path / "project"
    protected = root / ".git/hooks"
    policy = SandboxPolicy(
        write_roots=(root,),
        project_roots=(root,),
        protected=(protected,),
        deny_read=secret_path_regexes(),
        allow_read=public_path_regexes(),
        allowed_hosts=("assets.example.com",),
        allow_localhost=True,
    )
    backend = SandboxRuntime(tmp_path / "settings")
    wrapped = backend.wrap(["python", "build.py"], policy)
    settings = json.loads(Path(wrapped[2]).read_text())

    assert wrapped[:2] == [str(executable), "--settings"] and wrapped[3:5] == ["--", "python"]
    assert settings["network"]["allowedDomains"] == ["assets.example.com"]
    assert settings["network"]["allowLocalBinding"] is True
    assert settings["filesystem"]["allowWrite"] == [str(root)]
    assert settings["filesystem"]["denyWrite"] == [str(protected)]
    denied = [native_text(path) for path in settings["filesystem"]["denyRead"]]
    allowed = [native_text(path) for path in settings["filesystem"]["allowRead"]]
    assert any(path.endswith("/.ssh") for path in denied)
    assert f"{root.as_posix()}/**/.env" in denied
    assert f"{root.parent.as_posix()}/**/id_ed25519" in denied
    assert f"{root.parent.as_posix()}/**/.aws/**" in denied
    assert f"{root.as_posix()}/**/.env.example" in allowed
    if os.name != "nt":
        assert (Path(wrapped[2]).stat().st_mode & 0o777) == 0o600


def test_sandbox_runtime_refuses_an_untranslatable_read_rule(tmp_path):
    policy = SandboxPolicy(write_roots=(tmp_path,), deny_read=(r"/custom-(secret|token)$",))
    with pytest.raises(ValueError, match="cannot translate read policy regex"):
        SandboxRuntime(tmp_path / "settings")._settings(policy)


def test_sandbox_runtime_requires_a_posix_shell_on_windows(monkeypatch):
    monkeypatch.setattr("tinyharness.sandbox.srt.sys.platform", "win32")
    monkeypatch.setattr(
        "tinyharness.sandbox.srt.shutil.which",
        lambda name: "C:/npm/srt.cmd" if name == "srt" else None,
    )
    assert not SandboxRuntime.available()


def test_sandbox_runtime_requires_its_linux_helpers(monkeypatch):
    monkeypatch.setattr("tinyharness.sandbox.srt.sys.platform", "linux")
    present = {"srt", "bwrap", "socat", "rg"}
    monkeypatch.setattr(
        "tinyharness.sandbox.srt.shutil.which",
        lambda name: f"/usr/bin/{name}" if name in present else None,
    )
    monkeypatch.setattr("tinyharness.sandbox.srt.cannot_run", lambda: None)
    assert SandboxRuntime.installed() and SandboxRuntime.available()
    present.remove("socat")
    assert not SandboxRuntime.installed() and not SandboxRuntime.available()


SECCOMP_FAILURE = (
    "apply-seccomp: write /proc/self/setgroups (nested userns is capability-restricted; "
    "caller must provide CAP_SYS_ADMIN): Permission denied"
)


def fake_srt(tmp_path, monkeypatch, script: str) -> None:
    """An `srt` on PATH that runs `script`, with the probe's cache cleared around the test."""
    from tinyharness.sandbox import srt

    executable = tmp_path / "bin" / "srt"
    executable.parent.mkdir()
    executable.write_text("#!/bin/sh\n" + script)
    executable.chmod(0o755)
    real_which = __import__("shutil").which
    monkeypatch.setattr(
        srt.shutil, "which", lambda name: str(executable) if name == "srt" else real_which(name)
    )
    srt.cannot_run.cache_clear()


@posix
def test_an_installed_sandbox_runtime_that_cant_run_a_command_is_not_available(tmp_path, monkeypatch):
    """Seen in CI: on Ubuntu 26.04, AppArmor's bwrap profile denies CAP_SYS_ADMIN to
    bubblewrap's children, so SRT's seccomp helper aborted every command before it ran.
    SRT counted as available because it was installed, and every Bash call failed."""
    from tinyharness.sandbox import srt

    fake_srt(tmp_path, monkeypatch, f"echo 'startup message' >&2; echo '{SECCOMP_FAILURE}' >&2; exit 1\n")
    try:
        assert srt.cannot_run() == SECCOMP_FAILURE
        monkeypatch.setattr(SandboxRuntime, "installed", classmethod(lambda cls: True))
        assert not SandboxRuntime.available()
        hint = setup_hint()  # the startup warning, /sandbox, and the `required = true` error
        assert "installed but can't run commands here" in hint and SECCOMP_FAILURE in hint
        assert "sandbox-runtime/issues/429" in hint and "apt-get install" not in hint
    finally:
        srt.cannot_run.cache_clear()


@posix
def test_a_working_sandbox_runtime_passes_its_probe(tmp_path, monkeypatch):
    from tinyharness.sandbox import srt

    # Like SRT: run what follows `--`.
    fake_srt(tmp_path, monkeypatch, 'while [ "$1" != "--" ]; do shift; done; shift; exec "$@"\n')
    try:
        assert srt.cannot_run() is None
    finally:
        srt.cannot_run.cache_clear()


def test_sandbox_runtime_keeps_control_socket_temp_short(tmp_path, monkeypatch):
    """Seen in CI: a long private TMPDIR exceeded Linux's Unix-socket path limit."""
    executable = tmp_path / "srt"
    executable.touch()
    real_which = __import__("shutil").which
    monkeypatch.setattr(
        "tinyharness.sandbox.srt.shutil.which",
        lambda name: str(executable) if name == "srt" else real_which(name),
    )
    scratch = tmp_path / ("long-session-name-" * 6)
    policy = SandboxPolicy.for_project(tmp_path, scratch=scratch, cache_dir=tmp_path / "cache")
    backend = SandboxRuntime(tmp_path / "settings")
    wrapped = backend.wrap(["sh", "-c", "echo ok"], policy)
    outer = backend.environment(policy)
    assert not {"TMPDIR", "TMP", "TEMP"} & outer.keys()
    assert outer["UV_CACHE_DIR"] == policy.env["UV_CACHE_DIR"]
    after_separator = wrapped[wrapped.index("--") + 1 :]
    assert any(value == f"TMPDIR={policy.env['TMPDIR']}" for value in after_separator)


needs_srt = pytest.mark.skipif(not SandboxRuntime.available(), reason="needs Sandbox Runtime")


@needs_srt
async def test_sandbox_runtime_really_confines_files_and_credentials(tmp_path):
    root = tmp_path / "project"
    sibling = tmp_path / "other-project"
    (root / ".aws").mkdir(parents=True)
    sibling.mkdir()
    secrets = {
        root / ".env": "PROJECT_ENV_SECRET",
        root / "id_ed25519": "PROJECT_SSH_SECRET",
        root / ".aws/credentials": "PROJECT_AWS_SECRET",
        sibling / ".env": "SIBLING_ENV_SECRET",
    }
    for path, value in secrets.items():
        path.write_text(value)
    tool = BashTool(
        SandboxRuntime(tmp_path / "settings"),
        SandboxPolicy.for_project(root),
    )

    assert await call(tool, {"command": "touch inside"}, make_ctx(root)) == "(no output)"
    outside = tmp_path / "outside"
    out = await call(tool, {"command": f"touch {shlex.quote(str(outside))}"}, make_ctx(root))
    assert (root / "inside").exists() and not outside.exists() and "exit code" in out
    for path, value in secrets.items():
        out = await call(tool, {"command": f"cat {shlex.quote(str(path))}"}, make_ctx(root))
        assert value not in out and "exit code" in out


@needs_srt
async def test_sandbox_runtime_really_blocks_unlisted_network(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    script = "import socket; socket.create_connection(('1.1.1.1', 443), timeout=3); print('LEAKED')"
    tool = BashTool(SandboxRuntime(tmp_path / "settings"), SandboxPolicy.for_project(root))
    out = await call(tool, {"command": f'{sys.executable} -c "{script}"'}, make_ctx(root))
    assert "LEAKED" not in out and "exit code" in out


def test_extra_write_dirs_are_protected_like_the_project(tmp_path):
    other = tmp_path / "other"
    policy = SandboxPolicy.for_project(tmp_path / "proj", extra_write=[other])
    assert other.resolve() in policy.write_roots
    assert other.resolve() / ".git/hooks" in policy.protected


def test_secret_reads_can_be_opted_out_of(monkeypatch):
    assert any(".env" in r for r in secret_path_regexes())
    monkeypatch.setenv("TINYHARNESS_ALLOW_SECRET_READS", "1")
    assert secret_path_regexes() == ()


# ---- the Seatbelt profile ----


def _policy(root: Path, **kw) -> SandboxPolicy:
    root = root.resolve()
    return SandboxPolicy(write_roots=(root,), protected=(root / ".git/hooks",), **kw)


@not_windows
def test_profile_denies_network_and_writes_then_allows_the_roots(tmp_path):
    text = profile(_policy(tmp_path))
    assert text.index("(deny network*)") < text.index("(deny file-write*)")
    allow = text.index(f'(allow file-write*\n  (subpath "{tmp_path.resolve()}")')
    assert text.index("(deny file-write*)") < allow
    assert "localhost" not in text
    assert "(deny appleevent-send)" in text and "launchservicesd" in text


@not_windows
def test_profile_protected_paths_come_after_the_roots_they_are_in(tmp_path):
    text = profile(_policy(tmp_path))
    allow = text.index(f'(subpath "{tmp_path.resolve()}")')
    assert text.index(f'(subpath "{tmp_path.resolve() / ".git/hooks"}")') > allow


@not_windows
def test_profile_localhost_is_opt_in(tmp_path):
    assert '(remote ip "localhost:*")' in profile(_policy(tmp_path, allow_localhost=True))


@not_windows
def test_profile_quotes_paths(tmp_path):
    odd = tmp_path / 'a "quoted" dir'
    assert r"a \"quoted\" dir" in profile(_policy(odd))


# ---- the real thing (macOS only) ----

needs_seatbelt = pytest.mark.skipif(
    sys.platform != "darwin" or not Seatbelt.available(), reason="needs macOS sandbox-exec"
)


@pytest.fixture
def ws(tmp_path):
    ws = tmp_path / "ws"
    (ws / ".git" / "hooks").mkdir(parents=True)
    return ws


async def sandboxed(ws: Path, command: str, **policy) -> str:
    # Not for_project: pytest's tmp_path is inside the temp dir, which that policy makes writable.
    root = ws.resolve()
    policy.setdefault("deny_read", secret_path_regexes())
    policy.setdefault("protected", tuple(root / rel for rel in PROTECTED_IN_PROJECT))
    tool = BashTool(Seatbelt(), SandboxPolicy(write_roots=(root,), **policy))
    return await call(tool, {"command": command, "timeout": 30}, make_ctx(ws))


@needs_seatbelt
async def test_the_default_project_policy_is_a_valid_profile(tmp_path):
    policy = SandboxPolicy.for_project(
        tmp_path, scratch=tmp_path / ".tinyharness/tmp/s", cache_dir=tmp_path / "c"
    )
    out = await call(BashTool(Seatbelt(), policy), {"command": "echo hi"}, make_ctx(tmp_path))
    assert out == "hi"


@pytest.mark.parametrize("backend_name", ["seatbelt", "srt"])
async def test_sandboxed_pytest_reads_fixtures_from_its_private_tmp(tmp_path, backend_name):
    """Seen live: the child TMPDIR was absent from pytest's synthetic-secret exception."""
    project = tmp_path / "project"
    project.mkdir()
    if backend_name == "seatbelt":
        if not Seatbelt.available():
            pytest.skip("needs macOS sandbox-exec")
        backend = Seatbelt()
    else:
        if not SandboxRuntime.available():
            pytest.skip("needs Sandbox Runtime")
        backend = SandboxRuntime(tmp_path / "settings")
    test_file = project / "test_fixture.py"
    test_file.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "def test_fake_credentials(tmp_path):\n"
        "    assert tmp_path.resolve().is_relative_to(Path(os.environ['TMPDIR']).resolve())\n"
        "    for name in ('.env', 'auth.json'):\n"
        "        path = tmp_path / name\n"
        "        path.write_text('synthetic-secret')\n"
        "        assert path.read_text() == 'synthetic-secret'\n"
    )
    # Sessions keep scratch in their state directory, outside the project's protected .tinyharness.
    scratch = state_dir(project) / "tmp" / "sandbox-test"
    policy = SandboxPolicy.for_project(project, scratch=scratch, cache_dir=home_dir() / "sandbox-cache")
    tool = BashTool(backend, policy)
    command = f"{shlex.quote(sys.executable)} -m pytest -q -p no:cacheprovider {test_file.name}"
    out = await call(tool, {"command": command, "timeout": 60}, make_ctx(project))
    assert "1 passed" in out, out


@needs_seatbelt
async def test_writes_only_inside_the_project(ws, tmp_path):
    out = await sandboxed(ws, f"touch inside.txt && touch {tmp_path / 'outside.txt'}")
    assert (ws / "inside.txt").exists()
    assert not (tmp_path / "outside.txt").exists()
    assert "Operation not permitted" in out and "(exit code 1)" in out


@needs_seatbelt
async def test_reads_anything_but_secrets(ws, tmp_path):
    (tmp_path / "notes.txt").write_text("public")
    (ws / ".env").write_text("TOKEN=hunter2")
    ssh = Path.home() / ".ssh"  # the fake HOME from conftest
    ssh.mkdir(parents=True)
    (ssh / "id_ed25519").write_text("PRIVATE KEY")
    out = await sandboxed(ws, f"cat {tmp_path / 'notes.txt'}; cat .env; cat {ssh / 'id_ed25519'}")
    assert "public" in out
    assert "hunter2" not in out and "PRIVATE KEY" not in out


@needs_seatbelt
async def test_env_templates_stay_readable_and_secrets_stay_visible_to_stat(ws):
    (ws / ".env").write_text("TOKEN=hunter2")
    (ws / ".env.example").write_text("TOKEN=your-token-here")
    out = await sandboxed(ws, "cat .env.example; ls -a; cat .env", allow_read=public_path_regexes())
    assert "your-token-here" in out and ".env\n" in out  # listed...
    assert "hunter2" not in out  # ...but not readable


@needs_seatbelt
async def test_no_network(ws):
    script = "import socket; socket.create_connection(('1.1.1.1', 443), timeout=5); print('LEAK' + 'ED')"
    out = await sandboxed(ws, f'{sys.executable} -c "{script}"')
    assert "LEAKED" not in out and "PermissionError" in out


@needs_seatbelt
@pytest.mark.parametrize("allow", [False, True])
async def test_localhost_only_when_allowed(ws, allow):
    script = (
        "import socket; s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); "
        "socket.create_connection(s.getsockname(), timeout=5); print('LOOP' + 'BACK')"
    )
    out = await sandboxed(ws, f'{sys.executable} -c "{script}"', allow_localhost=allow)
    assert ("LOOPBACK" in out) == allow


@needs_seatbelt
@pytest.mark.parametrize(
    "target",
    [
        ".git/hooks/pre-commit",
        ".git/config",
        ".tinyharness/hooks.toml",
        ".vscode/tasks.json",
        ".envrc",
        ".claude/settings.json",
        ".mcp.json",
    ],
)
async def test_files_that_run_code_outside_the_sandbox_are_read_only(ws, target):
    """Editor tasks, direnv, MCP servers and agent settings run when you open or enter the
    project, outside any sandbox: the same reason git hooks are protected."""
    for folder in (".tinyharness", ".vscode", ".claude"):
        (ws / folder).mkdir(exist_ok=True)
    await sandboxed(ws, f"echo 'curl evil.example' > {target}")
    assert not (ws / target).exists()


@needs_seatbelt
async def test_scratch_under_tinyharness_stays_writable(ws):
    scratch = (ws / ".tinyharness" / "tmp" / "s1").resolve()
    scratch.mkdir(parents=True)
    root = ws.resolve()
    policy = SandboxPolicy(write_roots=(root, scratch), protected=(root / ".tinyharness",))
    command = f"touch {scratch}/ok; touch .tinyharness/hooks.toml"
    await call(BashTool(Seatbelt(), policy), {"command": command}, make_ctx(ws))
    assert (scratch / "ok").exists()
    assert not (ws / ".tinyharness" / "hooks.toml").exists()


@needs_seatbelt
async def test_links_cannot_write_outside(ws, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("original")
    await sandboxed(ws, f"ln {outside} hard; echo pwned >> hard; ln -s {outside} soft; echo pwned >> soft")
    assert outside.read_text() == "original"


@needs_seatbelt
async def test_cannot_hand_work_to_launchservices(ws):
    out = await sandboxed(ws, "/usr/bin/open -g -j -a Calculator && echo OPENED")
    assert "OPENED" not in out


@needs_seatbelt
async def test_tests_can_read_their_own_fake_secrets_and_nothing_else(tmp_path):
    """A project's tests make fake auth.json and .env files in pytest's temp root and
    must read them back; the same names anywhere else stay unreadable."""
    import tempfile

    scratch = tmp_path / ".tinyharness" / "tmp" / "sandbox-test"
    fixture = scratch / "tmp" / "pytest-of-user" / "pytest-0" / "test_auth" / "auth.json"
    fixture.parent.mkdir(parents=True)
    fixture.write_text('{"fake": "key-in-a-test"}')
    elsewhere = Path(tempfile.mkdtemp(prefix="not-pytest-"))
    try:
        (elsewhere / "auth.json").write_text('{"real": "hunter2"}')
        policy = SandboxPolicy.for_project(tmp_path, scratch=scratch, cache_dir=tmp_path / "c")
        out = await call(
            BashTool(Seatbelt(), policy),
            {"command": f"cat {fixture}; cat {elsewhere / 'auth.json'}"},
            make_ctx(tmp_path),
        )
        assert "key-in-a-test" in out
        assert "hunter2" not in out
    finally:
        (elsewhere / "auth.json").unlink()
        elsewhere.rmdir()


def test_shared_caches_write_the_users_own_cache(tmp_path, monkeypatch):
    """Opt-in: the agent and the user's terminal share uv's cache instead of each keeping one."""
    from tinyharness.sandbox import SandboxPolicy

    monkeypatch.setenv("HF_HOME", str(tmp_path / "my-hf"))
    for unset in ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR"):  # CI's setup-uv sets them
        monkeypatch.delenv(unset, raising=False)
    policy = SandboxPolicy.for_project(
        tmp_path / "p", cache_dir=tmp_path / "c", shared_caches=["uv", "huggingface"]
    )
    assert policy.env["HF_HOME"] == str((tmp_path / "my-hf").resolve())  # the user's own setting
    assert policy.env["UV_CACHE_DIR"] == str(Path("~/.cache/uv").expanduser().resolve())
    assert Path(policy.env["UV_CACHE_DIR"]) in policy.write_roots
    assert policy.env["PIP_CACHE_DIR"].startswith(str((tmp_path / "c").resolve()))  # not shared
    assert SandboxPolicy.for_project(tmp_path / "p", cache_dir=tmp_path / "c").shared_caches == ()


_SOCKETS = r"""
import socket, sys, threading
srv = socket.socket(socket.AF_UNIX); srv.bind("own.sock"); srv.listen(1)
threading.Thread(target=lambda: (lambda c: c.sendall(c.recv(4)))(srv.accept()[0]), daemon=True).start()
own = socket.socket(socket.AF_UNIX); own.connect("own.sock"); own.sendall(b"ping"); print("own", own.recv(4))
try:
    socket.socket(socket.AF_UNIX).connect(sys.argv[1]); print("outside connected")
except OSError as e:
    print("outside", e.errno)
"""


@needs_seatbelt
async def test_a_commands_own_unix_sockets_work_and_others_stay_closed(ws):
    """Seen live: SGLang's engine couldn't reach its own scheduler over a ZMQ `ipc://`
    socket in the session's private temp folder: `(deny network*)` covers Unix sockets."""
    import socket
    import tempfile

    outside = Path(tempfile.mkdtemp(dir="/tmp")) / "svc.sock"  # short: socket paths max out at 104
    server = socket.socket(socket.AF_UNIX)
    server.bind(str(outside))
    server.listen(1)
    try:
        (ws / "sockets.py").write_text(_SOCKETS)
        out = await sandboxed(ws, f"python3 sockets.py {outside}")
    finally:
        server.close()
        outside.unlink()
    assert "own b'ping'" in out and "outside 1" in out


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS frameworks")
def test_metal_may_write_its_own_scratch_folder_but_not_the_rest_of_the_users_temp(tmp_path):
    """Seen live: PyTorch on Apple GPUs printed "You don't have permission to save the file
    mpsgraph-..." on every graph: Metal writes there, under the user's own temp directory,
    whatever TMPDIR says. Only that one folder opens up."""
    policy = SandboxPolicy.for_project(tmp_path, scratch=tmp_path / "scratch")
    metal = [r for r in policy.write_roots if r.name == "com.apple.MetalPerformanceShadersGraph"]
    assert len(metal) == 1 and metal[0].parent.name == "T"
    assert metal[0].parent not in policy.write_roots
