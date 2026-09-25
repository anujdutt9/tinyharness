"""Prompt caching: cache_control marker on the system message."""

from tinyharness.llm.client import _apply_prompt_caching


def test_marks_string_system_message():
    payload = {
        "messages": [
            {"role": "system", "content": "you are an agent"},
            {"role": "user", "content": "hi"},
        ]
    }
    _apply_prompt_caching(payload)
    sys_msg = payload["messages"][0]
    assert isinstance(sys_msg["content"], list)
    block = sys_msg["content"][0]
    assert block["type"] == "text"
    assert block["text"] == "you are an agent"
    assert block["cache_control"] == {"type": "ephemeral"}
    # User message unchanged.
    assert payload["messages"][1] == {"role": "user", "content": "hi"}


def test_idempotent_when_already_marked():
    """Running twice doesn't double-wrap."""
    payload = {"messages": [{"role": "system", "content": "x"}]}
    _apply_prompt_caching(payload)
    snapshot = payload["messages"][0]["content"]
    _apply_prompt_caching(payload)
    assert payload["messages"][0]["content"] == snapshot


def test_skips_when_no_system_message():
    payload = {"messages": [{"role": "user", "content": "hi"}]}
    _apply_prompt_caching(payload)
    assert payload["messages"][0] == {"role": "user", "content": "hi"}


def test_skips_empty_system():
    payload = {"messages": [{"role": "system", "content": ""}, {"role": "user", "content": "hi"}]}
    _apply_prompt_caching(payload)
    # No mutation when system is empty.
    assert payload["messages"][0]["content"] == ""


def test_existing_structured_content_marked_at_tail():
    """If the user already passed structured content, mark the last block."""
    payload = {
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "first part"},
                    {"type": "text", "text": "second part"},
                ],
            },
        ]
    }
    _apply_prompt_caching(payload)
    blocks = payload["messages"][0]["content"]
    assert len(blocks) == 2
    assert "cache_control" not in blocks[0]
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}
