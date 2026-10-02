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
    runner = HookRunner([Hook(event="PreToolUse", command=f"sh -c 'cat > {seen}'")], cwd=tmp_path)
    args = {"command": f"curl -H 'Bearer {KEY}' https://x.example/?k=plainvalue0123456789"}
    await runner.fire("PreToolUse", {"tool": "Bash", "args": args})
    payload = seen.read_text()
    assert KEY not in payload and "plainvalue0123456789" not in payload
    assert "[REDACTED API KEY]" in payload and "[REDACTED MISTRAL_API_KEY]" in payload


async def test_a_hook_that_times_out_is_stopped_with_its_children(tmp_path, monkeypatch):
    import os

    from tinyharness import process

    monkeypatch.setattr(process, "KILL_GRACE", 0.5)
    pidfile = tmp_path / "child.pid"
    script = tmp_path / "hook.sh"
    script.write_text(f"sh -c 'trap \"\" TERM; sleep 30' & echo $! > {pidfile}; sleep 30\n")
    runner = HookRunner([Hook(event="Stop", command=f"sh {script}")], cwd=tmp_path, timeout=0.5)
    verdict = await runner.fire("Stop", {})
    assert verdict == {"decision": "ask", "reason": "hook timed out after 0.5s"}
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)
