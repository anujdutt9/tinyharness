from tinyharness.llm.client import _apply_thinking


def test_anthropic_gets_thinking_param():
    payload = {"model": "claude-sonnet-4-6"}
    _apply_thinking(payload, payload["model"], 4096)
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert "reasoning_effort" not in payload


def test_openai_reasoner_gets_effort_string():
    payload = {"model": "o3-mini"}
    _apply_thinking(payload, payload["model"], 1024)
    assert payload["reasoning_effort"] == "low"
    assert "thinking" not in payload


def test_effort_thresholds():
    p1 = {"model": "o1"}
    _apply_thinking(p1, "o1", 1024)
    assert p1["reasoning_effort"] == "low"
    p2 = {"model": "o1"}
    _apply_thinking(p2, "o1", 8192)
    assert p2["reasoning_effort"] == "medium"
    p3 = {"model": "o1"}
    _apply_thinking(p3, "o1", 16384)
    assert p3["reasoning_effort"] == "high"


def test_zero_budget_is_noop():
    payload = {"model": "claude-sonnet-4"}
    _apply_thinking(payload, payload["model"], 0)
    assert "thinking" not in payload
    assert "reasoning_effort" not in payload


def test_unknown_model_silently_skipped():
    payload = {"model": "llama-3.3-70b"}
    _apply_thinking(payload, payload["model"], 4096)
    assert "thinking" not in payload
    assert "reasoning_effort" not in payload


def test_deepseek_reasoner_gets_effort():
    payload = {"model": "deepseek-reasoner"}
    _apply_thinking(payload, payload["model"], 4096)
    assert payload["reasoning_effort"] == "medium"
