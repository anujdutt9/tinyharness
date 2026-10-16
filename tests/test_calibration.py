"""Tool-call calibration: how a model calls tools is measured once, not guessed from its
name; text-mode models get tools in the prompt and write their calls (llm/calibration.py)."""

import pytest

from tinyharness.agent.history import Message, ToolCall
from tinyharness.llm.calibration import CalibrationStore, as_text, probe, tools_section
from tinyharness.llm.client import LLMClient
from tinyharness.llm.providers import load_catalog
from tinyharness.llm.types import ModelInfo, StreamChunk, ToolSpec


def replying(*chunks):
    async def stream(messages, tools):
        for chunk in chunks:
            yield chunk

    return stream


async def test_the_probe_tells_native_calls_from_calls_written_as_text():
    native = StreamChunk(
        tool_call_delta={"index": 0, "id": "c", "function": {"name": "get_time", "arguments": "{}"}}
    )
    written = StreamChunk(
        text='Sure.\n<tool_call>{"name": "get_time", "arguments": {"timezone": "UTC"}}</tool_call>'
    )
    assert await probe(replying(native)) == "native"
    assert await probe(replying(written)) == "text"
    assert await probe(replying(StreamChunk(text="It's noon."))) == "native"  # can't tell: keep native


def test_the_history_is_sent_as_text_for_a_text_mode_model():
    history = [
        Message(role="user", content="read a.py"),
        Message(role="assistant", content="Reading.", tool_calls=[ToolCall("1", "Read", '{"path": "a.py"}')]),
        Message(role="tool", tool_call_id="1", content="x = 1"),
        Message(role="user", content="and now?"),
    ]
    out = as_text(history)
    assert [m.role for m in out] == ["user", "assistant", "user"]
    assert out[1].content == 'Reading.\n<call>{"name": "Read", "arguments": {"path": "a.py"}}</call>'
    assert (
        out[1].tool_calls == []
        and out[2].content == '<tool_result name="Read">\nx = 1\n</tool_result>\n\nand now?'
    )


def test_the_tools_section_shows_each_tool_and_how_to_call_it():
    text = tools_section(
        [ToolSpec("Read", "Read a file.", {"type": "object", "properties": {"path": {"type": "string"}}})]
    )
    assert "<call>" in text and "## Read" in text and '"path":{"type":"string"}' in text


async def test_a_text_mode_request_has_no_tool_definitions(tmp_path):
    seen = []
    client = LLMClient(load_catalog()["openrouter"], "k", "some/model", tool_calls="text")

    def capture(request):
        seen.append(request)
        raise RuntimeError("stop here")

    client.adapter.payload = capture
    tools = [ToolSpec("Read", "Read a file.", {"type": "object", "properties": {}})]
    with pytest.raises(RuntimeError, match="stop here"):
        [
            c
            async for c in client.stream(
                [Message(role="system", content="S"), Message(role="user", content="hi")], tools
            )
        ]
    [request] = seen
    assert request.tools == [] and request.system.startswith("S\n\n# Tools") and "## Read" in request.system


async def test_a_probed_format_is_cached_and_the_config_overrides_it(tmp_path, monkeypatch):
    store = CalibrationStore(tmp_path / "tool-calls.json")
    client = LLMClient(load_catalog()["openrouter"], "k", "small/model", calibration=store)
    probes = []

    async def fake_probe(stream):
        probes.append(1)
        return "text"

    monkeypatch.setattr("tinyharness.llm.client.probe", fake_probe)
    assert await client.tool_format() == "text" and await client.tool_format() == "text"
    fresh = LLMClient(load_catalog()["openrouter"], "k", "small/model", calibration=store)
    assert await fresh.tool_format() == "text" and len(probes) == 1  # from the cache
    forced = LLMClient(
        load_catalog()["openrouter"], "k", "small/model", calibration=store, tool_calls="native"
    )
    assert await forced.tool_format() == "native"
    assert (
        await LLMClient(load_catalog()["openrouter"], "k", "m").tool_format() == "native"
    )  # no store: no probe


async def test_a_failed_probe_uses_the_interface_and_isnt_remembered(tmp_path, monkeypatch):
    store = CalibrationStore(tmp_path / "tool-calls.json")

    async def failing(stream):
        raise ConnectionError("offline")

    monkeypatch.setattr("tinyharness.llm.client.probe", failing)
    client = LLMClient(load_catalog()["openrouter"], "k", "m", calibration=store)
    assert await client.tool_format() == "native" and store.get("openrouter:m") is None


async def test_a_model_known_not_to_accept_native_tools_uses_text_tools(tmp_path):
    """Seen live: Ollama accepted gemma3:270m during setup, then rejected the first
    agent turn because tinyharness sent native tools the model does not support."""
    from tinyharness.llm.models import ModelDirectory

    directory = ModelDirectory(tmp_path)
    directory.store("ollama", [ModelInfo("gemma3:270m", native_tools=False)])
    client = LLMClient(load_catalog()["ollama"], "", "gemma3:270m", models=directory, tool_calls="auto")

    assert await client.tool_format() == "text"

    forced = LLMClient(load_catalog()["ollama"], "", "gemma3:270m", models=directory, tool_calls="native")
    assert await forced.tool_format() == "native"


def test_text_mode_calls_use_a_tag_server_parsers_leave_alone():
    """Seen live: hosted servers stripped <tool_call> text from replies even with no tools in
    the request, so a text-mode model's calls arrived as empty replies. Ours is <call>."""
    from tinyharness.llm.textcalls import recover_calls

    schemas = {"Read": {"type": "object", "properties": {"path": {"type": "string"}}}}
    text = 'Reading it.\n<call>{"name": "Read", "arguments": {"path": "a.py"}}</call>'
    rest, calls = recover_calls(text, schemas.get)
    assert rest == "Reading it." and [(c.name, c.arguments) for c in calls] == [("Read", '{"path": "a.py"}')]
