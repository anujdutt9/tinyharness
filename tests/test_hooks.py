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
