import csv
import os
import shlex
import subprocess
import sys

import pytest

from tinyharness.hooks import Hook, HookRunner


@pytest.mark.asyncio
async def test_python_hook_deny():
    def deny(event, payload):
        return {"decision": "deny", "reason": "no"}

    runner = HookRunner([Hook(event="PreToolUse", type="python", callable=deny)])
    out = await runner.fire("PreToolUse", {"tool": "Write"})
    assert out["decision"] == "deny"


@pytest.mark.asyncio
async def test_matcher_filters_tool():
    def deny(event, payload):
        return {"decision": "deny"}

    runner = HookRunner([Hook(event="PreToolUse", type="python", matcher={"tool": "Write"}, callable=deny)])
    out = await runner.fire("PreToolUse", {"tool": "Read"})
    assert out is None


@pytest.mark.asyncio
async def test_deny_wins_over_allow():
    def allow(e, p):
        return {"decision": "allow"}

    def deny(e, p):
        return {"decision": "deny"}

    runner = HookRunner(
        [
            Hook(event="PreToolUse", type="python", callable=allow),
            Hook(event="PreToolUse", type="python", callable=deny),
        ]
    )
    out = await runner.fire("PreToolUse", {"tool": "Write"})
    assert out["decision"] == "deny"


# ---- what a shell hook gets, and how it's stopped ----

KEY = "sk-or-v1-" + "a1b2c3d4" * 4  # key-shaped, and not configured anywhere


async def test_a_shell_hook_never_receives_credentials(tmp_path, monkeypatch):
    """Hooks may forward their input (an audit log, a chat); keys are masked out of it."""
    monkeypatch.setenv("MISTRAL_API_KEY", "plainvalue0123456789")  # no recognizable shape
    seen = tmp_path / "payload.json"
    capture = f"cat > {shlex.quote(str(seen))}"
    runner = HookRunner([Hook(event="PreToolUse", command=f"sh -c {shlex.quote(capture)}")], cwd=tmp_path)
    args = {"command": f"curl -H 'Bearer {KEY}' https://x.example/?k=plainvalue0123456789"}
    await runner.fire("PreToolUse", {"tool": "Bash", "args": args})
    payload = seen.read_text()
    assert KEY not in payload and "plainvalue0123456789" not in payload
    assert "[REDACTED API KEY]" in payload and "[REDACTED MISTRAL_API_KEY]" in payload


async def test_a_hook_that_times_out_is_stopped_with_its_children(tmp_path, monkeypatch):
    from tinyharness import process

    monkeypatch.setattr(process, "KILL_GRACE", 0.5)
    pidfile = tmp_path / "child.pid"
    script = tmp_path / "hook.sh"
    # Let the child report its native PID. MSYS `$!` is an emulated PID that Windows
    # process APIs cannot inspect, which hid whether taskkill stopped the real child.
    child = (
        "import os, pathlib, signal, time; "
        f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"
    )
    python = shlex.quote(os.path.abspath(sys.executable).replace("\\", "/"))
    script.write_text(f"{python} -c {shlex.quote(child)} & sleep 30\n")
    timeout = 1.5 if os.name == "nt" else 0.5
    runner = HookRunner(
        [Hook(event="Stop", command=f"sh {shlex.quote(str(script))}")], cwd=tmp_path, timeout=timeout
    )
    verdict = await runner.fire("Stop", {})
    assert verdict == {"decision": "ask", "reason": f"hook timed out after {timeout:g}s"}
    pid = int(pidfile.read_text())
    if os.name == "nt":
        listed = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert not any(len(row) > 1 and row[1] == str(pid) for row in csv.reader(listed.stdout.splitlines()))
    else:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
