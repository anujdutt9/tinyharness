"""Tool calls from models that send them in the wrong shape (llm/textcalls.py, Tool.parse).

The cases come from a plan-mode run of 11 models (2026-11-01): Gemma 3 12B never made a real
call, Llama 3.1 8B sent a list argument as JSON text."""

import json

import pytest

from tinyharness.llm.textcalls import recover_calls, repair_arguments
from tinyharness.tools.ask import AskTool
from tinyharness.tools.write import WriteTool

SCHEMAS = {
    "Plan": {"properties": {"plan": {"type": "string"}, "ready": {"type": "boolean"}}},
    "Read": {"properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}},
}


def calls(text):
    rest, found = recover_calls(text, SCHEMAS.get, lambda name: name in SCHEMAS)
    return rest, [(c.name, json.loads(c.arguments)) for c in found]


@pytest.mark.parametrize(
    "text",
    [
        # The API's own shape, as Gemma 3 wrote it in a ```python block.
        '```python\n[\n  {"type": "function", "function": {"name": "Plan", "parameters": {"plan": "# T"}}}\n]\n```',
        # Flat, in a ```tool_code block: the arguments beside the name.
        '```tool_code\n[\n  {"name": "Plan", "plan": "# T"}\n]\n```',
        # Flat, with the tool's name in "type".
        '```python\n[{"type": "Plan", "plan": "# T"}]\n```',
        # Arguments as JSON text, as the API sends them; no fence.
        '{"name": "Plan", "arguments": "{\\"plan\\": \\"# T\\"}"}',
    ],
)
def test_calls_written_as_json_are_recovered(text):
    """Seen live: Gemma 3 12B wrote every plan this way, so all of them were lost."""
    assert calls("Here's the plan.\n\n" + text) == ("Here's the plan.", [("Plan", {"plan": "# T"})])


@pytest.mark.parametrize(
    "text",
    [
        '```json\n{"name": "myapp", "version": "1.0.0"}\n```',  # a package.json: no such tool
        '```json\n{"name": "Read", "version": "1.0.0"}\n```',  # "version" isn't a Read parameter
        '```json\n[{"name": "Plan", "plan": "# T"}, {"name": "Nope"}]\n```',  # all or nothing
        'Call it like this:\n```json\n{"name": "Plan", "plan": "# T"}\n```\nThat saves it.',  # not last
    ],
)
def test_json_that_isnt_a_call_is_left_alone(text):
    assert calls(text) == (text, [])


def test_a_list_argument_sent_as_json_text_is_decoded():
    """Seen live: Llama 3.1 8B sent Ask's `questions` as the string "[{...}]", and the call
    was refused as invalid. A string argument is never touched, though: it's text the tool
    takes as given."""
    questions = [{"header": "Format", "question": "Which?", "options": [{"label": "a"}, {"label": "b"}]}]
    args = AskTool(None).parse({"questions": json.dumps(questions)})
    assert args.questions[0].header == "Format"
    write = WriteTool().parse({"path": "x.json", "content": '["a", "b"]'})
    assert write.content == '["a", "b"]'


@pytest.mark.parametrize(
    "broken, fixed",
    [
        ('{"path": "a.py", "limit": 5,}', {"path": "a.py", "limit": 5}),  # trailing comma
        ("{'path': 'a.py'}", {"path": "a.py"}),  # single quotes
        ('{"path": "a.py" "limit": 5}', {"path": "a.py", "limit": 5}),  # a missing comma
    ],
)
def test_broken_json_syntax_is_repaired(broken, fixed):
    assert json.loads(repair_arguments(broken)) == fixed


def test_a_repair_never_completes_cut_off_arguments():
    """Completing them would run a call the model never finished (guarantee 2)."""
    assert repair_arguments('{"path": "notes.md", "content": "# Notes\\n\\nThe first') is None


async def test_a_json_example_the_user_asked_for_is_not_run(tmp_path):
    """Found in review: "Here is the requested example:" and a fenced Bash call, as the
    reply's last thing, was run. JSON with no call marker is taken only for tools that
    change nothing and reach nothing outside (Read, Glob, Grep, Plan, Ask)."""
    from tinyharness.events import ToolUse
    from tinyharness.testing import scripted_session

    example = (
        'Here is the requested example:\n\n```json\n{"name": "Bash", "arguments": '
        '{"command": "rm -rf build"}}\n```'
    )
    session = scripted_session(tmp_path / "w", [example])
    try:
        events = [e async for e in session.run("show me what a Bash tool call looks like")]
        assert not [e for e in events if isinstance(e, ToolUse)]
        assert session.history.messages[-1].content == example
    finally:
        session.close()

    read = '```json\n{"name": "Read", "arguments": {"path": "a.py"}}\n```'
    assert recover_calls(read, SCHEMAS.get)[1] == []  # no tool allowed: nothing is taken
    assert [c.name for c in recover_calls(read, SCHEMAS.get, lambda name: True)[1]] == ["Read"]
