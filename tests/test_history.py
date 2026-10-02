"""The history is always sent in a shape providers accept (agent/history.py `well_formed`)."""

import re

from tinyharness.agent.history import MISSING_RESULT, History, Message, ToolCall, well_formed

ANTHROPIC_IDS = re.compile(r"[A-Za-z0-9_-]{1,64}")
MISTRAL_IDS = re.compile(r"[A-Za-z0-9]{9}")


def assistant(*ids, content=None):
    return Message(role="assistant", content=content, tool_calls=[ToolCall(i, "Read", "{}") for i in ids])


def result(call_id, content="ok"):
    return Message(role="tool", tool_call_id=call_id, content=content)


def roles(messages):
    return [(m.role, m.tool_call_id or m.content) for m in messages]


def test_a_call_without_a_result_is_answered_before_the_next_message():
    messages = [
        Message(role="user", content="go"),
        assistant("a", "b"),
        result("a"),
        Message(role="user", content="next"),
    ]
    fixed, added = well_formed(messages)
    assert added == 1
    assert roles(fixed)[1:5] == [("assistant", None), ("tool", "a"), ("tool", "b"), ("user", "next")]
    assert fixed[3].content == MISSING_RESULT
    assert len(messages) == 4  # the original list is left alone


def test_a_result_that_answers_no_call_is_dropped():
    messages = [Message(role="user", content="go"), result("ghost"), assistant("a"), result("a"), result("a")]
    fixed, _ = well_formed(messages)
    assert roles(fixed) == [("user", "go"), ("assistant", None), ("tool", "a")]


def test_an_id_reused_by_a_later_turn_is_answered_in_each_turn():
    """Some servers number calls per reply (call_0, call_1...)."""
    messages = [assistant("call_0"), result("call_0"), assistant("call_0")]
    fixed, added = well_formed(messages)
    assert added == 1 and [m.role for m in fixed] == ["assistant", "tool", "assistant", "tool"]


def test_an_empty_assistant_message_is_dropped():
    messages = [
        Message(role="user", content="go"),
        Message(role="assistant", content="  "),
        Message(role="user", content="?"),
    ]
    assert [m.content for m in well_formed(messages)[0]] == ["go", "?"]


def test_ids_another_provider_wouldnt_accept_are_replaced_consistently():
    messages = [assistant("functions.Read:0"), result("functions.Read:0")]
    for pattern in (ANTHROPIC_IDS, MISTRAL_IDS):
        fixed, _ = well_formed(messages, valid_id=pattern)
        call_id = fixed[0].tool_calls[0].id
        assert pattern.fullmatch(call_id) and fixed[1].tool_call_id == call_id
        assert (
            well_formed(messages, valid_id=pattern)[0][0].tool_calls[0].id == call_id
        )  # stable for the cache
    assert messages[0].tool_calls[0].id == "functions.Read:0"


def test_valid_ids_are_left_alone():
    messages = [assistant("toolu_01ABC"), result("toolu_01ABC")]
    assert well_formed(messages, valid_id=ANTHROPIC_IDS)[0][0] is messages[0]


def test_repair_fixes_the_stored_history_in_place():
    history = History([Message(role="user", content="go"), assistant("a")])
    assert history.repair() == 1
    assert history.messages[-1].tool_call_id == "a"
