"""Each provider adapter, end to end through LLMClient, against recorded-style streams.

The streams follow each provider's documented event format. A fake HTTP transport
captures the request body so we can check what would be sent.
"""

import json

import httpx
import pytest

from tinyharness.agent.history import Message, ToolCall
from tinyharness.llm.client import LLMClient, ToolCallAccumulator
from tinyharness.llm.models import ModelDirectory
from tinyharness.llm.providers import load_catalog
from tinyharness.llm.types import ContextOverflowError, LLMError, ModelInfo, ToolSpec

READ = ToolSpec("Read", "read a file", {"type": "object", "properties": {"path": {"type": "string"}}})


def sse(*events, named: bool = False) -> str:
    """Server-sent events; `named` adds the `event:` lines Anthropic and Gemini send."""
    lines = []
    for e in events:
        if named:
            lines.append(f"event: {e.get('type') or e.get('event_type')}")
        lines.append(f"data: {json.dumps(e)}\n")
    return "\n".join(lines) + "\n"


class Server:
    """A fake provider: records request bodies, answers with the given stream."""

    def __init__(self, *bodies: str, status: int = 200) -> None:
        self.bodies = list(bodies)
        self.status = status
        self.requests: list[dict] = []
        self.headers: list[httpx.Headers] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content or b"{}"))
        self.headers.append(request.headers)
        return httpx.Response(self.status, text=self.bodies.pop(0) if self.bodies else "")


def client(provider: str, server: Server, model: str = "m1", tmp_path=None, **kwargs) -> LLMClient:
    info = load_catalog()[provider]
    directory = ModelDirectory(tmp_path) if tmp_path else None
    return LLMClient(
        info, "key-123", model, transport=httpx.MockTransport(server), models=directory, **kwargs
    )


async def run(llm: LLMClient, messages: list[Message], tools=(READ,)):
    text, calls, state = [], ToolCallAccumulator(), None
    async for chunk in llm.stream(messages, list(tools)):
        text.append(chunk.text)
        if chunk.tool_call_delta:
            calls.add(chunk.tool_call_delta)
        state = chunk.state or state
    return "".join(text), calls.calls(), state


CONVERSATION = [
    Message(role="system", content="be brief"),
    Message(role="user", content="read a.py"),
]


def after_tool_call(reply_state, call_id="c1") -> list[Message]:
    """The conversation after the model called Read and got a result."""
    return [
        *CONVERSATION,
        Message(
            role="assistant",
            content="Reading.",
            tool_calls=[ToolCall(call_id, "Read", '{"path": "a.py"}')],
            state=reply_state,
        ),
        Message(role="tool", tool_call_id=call_id, content="print(1)"),
    ]


# ---- Anthropic Messages ----

ANTHROPIC_STREAM = sse(
    {"type": "message_start", "message": {"id": "msg_1"}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "SIG=="}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Reading."}},
    {"type": "content_block_stop", "index": 1},
    {
        "type": "content_block_start",
        "index": 2,
        "content_block": {"type": "tool_use", "id": "c1", "name": "Read", "input": {}},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": '{"path": '},
    },
    {
        "type": "content_block_delta",
        "index": 2,
        "delta": {"type": "input_json_delta", "partial_json": '"a.py"}'},
    },
    {"type": "content_block_stop", "index": 2},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
    {"type": "message_stop"},
    named=True,
)


async def test_anthropic_request_and_stream(tmp_path):
    server = Server(ANTHROPIC_STREAM)
    llm = client("anthropic", server, "claude-x", tmp_path=tmp_path)
    llm.directory.store(
        "anthropic",
        [
            ModelInfo(
                "claude-x",
                context=1_000_000,
                max_output=64_000,
                adaptive_thinking=True,
                efforts=frozenset({"low", "high"}),
            )
        ],
    )
    llm.effort = "high"
    text, [call], state = await run(llm, CONVERSATION)

    body = server.requests[0]
    assert body["system"] == "be brief"
    assert body["max_tokens"] == 64_000  # from the model list, not a guess
    assert body["tools"] == [{"name": "Read", "description": "read a file", "input_schema": READ.parameters}]
    assert body["thinking"]["type"] == "adaptive"
    assert body["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
    assert body["output_config"] == {"effort": "high"}
    assert body["cache_control"] == {"type": "ephemeral"}
    assert server.headers[0]["x-api-key"] == "key-123"
    assert "thinking-binding-controls" in server.headers[0]["anthropic-beta"]

    assert text == "Reading."
    assert (call.id, call.name, json.loads(call.arguments)) == ("c1", "Read", {"path": "a.py"})
    assert state.owner == "anthropic:claude-x"
    assert state.data[0] == {"type": "thinking", "thinking": "", "signature": "SIG=="}


async def test_anthropic_sends_signed_thinking_back_unchanged(tmp_path):
    first = Server(ANTHROPIC_STREAM)
    _, _, state = await run(client("anthropic", first, "claude-x"), CONVERSATION)

    second = Server(sse({"type": "message_stop"}, named=True))
    await run(client("anthropic", second, "claude-x"), after_tool_call(state))
    messages = second.requests[0]["messages"]
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == state.data  # thinking block and signature, as received
    assert messages[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "print(1)"}],
    }


async def test_saved_reasoning_is_not_sent_to_a_different_model(tmp_path):
    first = Server(ANTHROPIC_STREAM)
    _, _, state = await run(client("anthropic", first, "claude-x"), CONVERSATION)

    second = Server(sse({"type": "message_stop"}, named=True))
    await run(client("anthropic", second, "claude-y"), after_tool_call(state))
    assistant = second.requests[0]["messages"][1]["content"]
    assert [b["type"] for b in assistant] == ["text", "tool_use"]  # rebuilt; no thinking


async def test_anthropic_skips_settings_the_model_list_says_are_unsupported(tmp_path):
    server = Server(sse({"type": "message_stop"}, named=True))
    llm = client("anthropic", server, "small", tmp_path=tmp_path)
    llm.directory.store("anthropic", [ModelInfo("small", adaptive_thinking=False, efforts=frozenset())])
    llm.effort = "high"
    await run(llm, CONVERSATION)
    assert "thinking" not in server.requests[0] and "output_config" not in server.requests[0]


# ---- OpenAI Responses ----

RESPONSES_STREAM = sse(
    {"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1"}},
    {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC"},
    },
    {"type": "response.output_text.delta", "output_index": 1, "delta": "Reading."},
    {
        "type": "response.output_item.added",
        "output_index": 2,
        "item": {"type": "function_call", "call_id": "c1", "name": "Read"},
    },
    {"type": "response.function_call_arguments.delta", "output_index": 2, "delta": '{"path": "a.py"}'},
    {
        "type": "response.completed",
        "response": {
            "output": [
                {"type": "reasoning", "id": "rs_1", "encrypted_content": "ENC"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Reading."}],
                },
                {"type": "function_call", "call_id": "c1", "name": "Read", "arguments": '{"path": "a.py"}'},
            ]
        },
    },
)


async def test_responses_request_stream_and_echo():
    server = Server(RESPONSES_STREAM, sse({"type": "response.completed", "response": {"output": []}}))
    llm = client("openai", server, "gpt-x", effort="low")
    text, [call], state = await run(llm, CONVERSATION)

    body = server.requests[0]
    assert body["instructions"] == "be brief"
    assert body["store"] is False and body["include"] == ["reasoning.encrypted_content"]
    assert body["reasoning"] == {"effort": "low"}
    assert body["tools"][0]["name"] == "Read" and body["tools"][0]["strict"] is False
    assert text == "Reading." and call.arguments == '{"path": "a.py"}'

    await run(llm, after_tool_call(state))
    items = server.requests[1]["input"]
    assert items[1:4] == state.data  # the whole output, encrypted reasoning included
    assert items[4] == {"type": "function_call_output", "call_id": "c1", "output": "print(1)"}


async def test_responses_accepts_arguments_that_arrive_whole():
    """xAI sends a function call's arguments in one piece, with no deltas."""
    stream = sse(
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "function_call", "call_id": "c1", "name": "Read"},
        },
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "c1",
                "name": "Read",
                "arguments": '{"path":"b.py"}',
            },
        },
        {"type": "response.completed", "response": {}},
    )
    _, [call], _ = await run(client("xai", Server(stream)), CONVERSATION)
    assert call.arguments == '{"path":"b.py"}'


# ---- Gemini Interactions ----

GEMINI_STREAM = sse(
    {"event_type": "interaction.created", "interaction": {"id": "v1_1"}},
    {"event_type": "step.start", "index": 0, "step": {"type": "thought"}},
    {"event_type": "step.delta", "index": 0, "delta": {"type": "thought_signature", "signature": "GSIG"}},
    {"event_type": "step.stop", "index": 0},
    {"event_type": "step.start", "index": 1, "step": {"type": "model_output"}},
    {"event_type": "step.delta", "index": 1, "delta": {"type": "text", "text": "Reading."}},
    {"event_type": "step.stop", "index": 1},
    {
        "event_type": "step.start",
        "index": 2,
        "step": {"type": "function_call", "id": "c1", "name": "Read", "arguments": {}},
    },
    {
        "event_type": "step.delta",
        "index": 2,
        "delta": {"type": "arguments_delta", "arguments": '{"path": "a.py"}'},
    },
    {"event_type": "step.stop", "index": 2},
    {"event_type": "interaction.completed", "interaction": {"status": "requires_action"}},
    named=True,
)


async def test_gemini_request_stream_and_echo():
    server = Server(GEMINI_STREAM, sse({"event_type": "interaction.completed", "interaction": {}}))
    llm = client("gemini", server, "gemini-x", effort="max")
    text, [call], state = await run(llm, CONVERSATION)

    body = server.requests[0]
    assert body["system_instruction"] == "be brief"
    assert body["store"] is False
    assert body["input"] == [{"type": "user_input", "content": [{"type": "text", "text": "read a.py"}]}]
    assert body["tools"][0] == {
        "type": "function",
        "name": "Read",
        "description": "read a file",
        "parameters": READ.parameters,
    }
    assert body["generation_config"] == {"thinking_level": "high"}
    assert server.headers[0]["x-goog-api-key"] == "key-123"
    assert text == "Reading." and call.id == "c1" and json.loads(call.arguments) == {"path": "a.py"}

    await run(llm, after_tool_call(state))
    steps = server.requests[1]["input"]
    assert steps[1] == {"type": "thought", "signature": "GSIG"}
    assert steps[3]["arguments"] == {"path": "a.py"}
    assert steps[4] == {
        "type": "function_result",
        "name": "Read",
        "call_id": "c1",
        "result": [{"type": "text", "text": "print(1)"}],
    }


async def test_gemini_accepts_arguments_that_arrive_whole():
    stream = sse(
        {
            "event_type": "step.start",
            "index": 0,
            "step": {"type": "function_call", "id": "c1", "name": "Read", "arguments": {"path": "z.py"}},
        },
        {"event_type": "step.stop", "index": 0},
        named=True,
    )
    _, [call], state = await run(client("gemini", Server(stream)), CONVERSATION)
    assert json.loads(call.arguments) == {"path": "z.py"}
    assert state is None  # nothing signed to send back


# ---- Chat Completions ----


def chat_chunk(delta=None, finish=None, **extra):
    return {"choices": [{"delta": delta or {}, "finish_reason": finish}], **extra}


async def test_chat_request_and_openrouter_reasoning_details():
    stream = (
        sse(
            chat_chunk({"reasoning_details": [{"type": "reasoning.text", "text": "Let me ", "index": 0}]}),
            chat_chunk(
                {
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "look.", "signature": "S", "index": 0}
                    ]
                }
            ),
            chat_chunk({"content": "Reading."}),
            chat_chunk(
                {
                    "tool_calls": [
                        {"index": 0, "id": "c1", "function": {"name": "Read", "arguments": '{"path":"a.py"}'}}
                    ]
                }
            ),
            chat_chunk(finish="tool_calls"),
        )
        + "data: [DONE]\n\n"
    )
    server = Server(stream, sse(chat_chunk(finish="stop")))
    llm = client("openrouter", server, "vendor/model", effort="medium")
    text, [call], state = await run(llm, CONVERSATION)

    body = server.requests[0]
    assert body["messages"][0] == {"role": "system", "content": "be brief"}
    assert body["reasoning"] == {"effort": "medium"}  # OpenRouter's effort_param
    assert body["cache_control"] == {"type": "ephemeral"}  # OpenRouter's extra_body
    assert text == "Reading." and call.name == "Read"
    assert state.data == {
        "reasoning_details": [
            {"type": "reasoning.text", "text": "Let me look.", "signature": "S", "index": 0}
        ]
    }

    await run(llm, after_tool_call(state))
    assistant = server.requests[1]["messages"][2]
    assert assistant["reasoning_details"] == state.data["reasoning_details"]
    assert assistant["tool_calls"][0]["function"]["name"] == "Read"


async def test_chat_keeps_deepseek_reasoning_content():
    stream = sse(
        chat_chunk({"reasoning_content": "hmm"}), chat_chunk({"content": "hi"}), chat_chunk(finish="stop")
    )
    server = Server(stream)
    _, _, state = await run(client("deepseek", server, effort="high"), CONVERSATION)
    assert server.requests[0]["reasoning_effort"] == "high"
    assert state.data == {"reasoning_content": "hmm"}


async def test_chat_mid_stream_error_is_raised():
    stream = sse({"error": {"message": "upstream overloaded"}, "choices": []})
    with pytest.raises(LLMError, match="overloaded"):
        await run(client("openrouter", Server(stream)), CONVERSATION)


# ---- errors and model lists ----


@pytest.mark.parametrize(
    ("provider", "body"),
    [
        (
            "anthropic",
            '{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 250000 tokens > 200000 maximum"}}',
        ),
        (
            "openai",
            '{"error":{"message":"Your input exceeds the context window of this model.","code":"context_length_exceeded"}}',
        ),
        (
            "openrouter",
            '{"error":{"message":"too long","metadata":{"error_type":"context_length_exceeded"}}}',
        ),
    ],
)
async def test_context_overflow_is_recognized(provider, body):
    with pytest.raises(ContextOverflowError):
        await run(client(provider, Server(body, status=400)), CONVERSATION)


async def test_other_400s_are_plain_errors():
    with pytest.raises(LLMError) as err:
        await run(
            client("openai", Server('{"error":{"message":"bad tool schema"}}', status=400)), CONVERSATION
        )
    assert not isinstance(err.value, ContextOverflowError)


async def test_anthropic_model_list_is_paginated_and_read_for_limits(tmp_path):
    pages = [
        {
            "data": [
                {
                    "id": "a",
                    "display_name": "A",
                    "max_input_tokens": 1000,
                    "max_tokens": 64,
                    "capabilities": {
                        "thinking": {"types": {"adaptive": {"supported": True}}},
                        "effort": {"supported": True, "low": {"supported": True}},
                    },
                }
            ],
            "has_more": True,
            "last_id": "a",
        },
        {"data": [{"id": "b", "max_input_tokens": 2000, "capabilities": None}], "has_more": False},
    ]
    server = Server(*(json.dumps(p) for p in pages))
    models = await client("anthropic", server, tmp_path=tmp_path).list_models(refresh=True)
    assert [(m.id, m.context) for m in models] == [("a", 1000), ("b", 2000)]
    assert models[0].adaptive_thinking and models[0].efforts == {"low"}
    assert models[1].adaptive_thinking is None  # unknown, not "no"


async def test_gemini_model_list(tmp_path):
    body = {
        "models": [
            {
                "name": "models/gemini-x",
                "displayName": "Gemini X",
                "inputTokenLimit": 1048576,
                "supportedGenerationMethods": ["generateContent"],
            },
            {"name": "models/embed", "supportedGenerationMethods": ["embedContent"]},
        ]
    }
    models = await client("gemini", Server(json.dumps(body)), tmp_path=tmp_path).list_models(refresh=True)
    assert [(m.id, m.label, m.context) for m in models] == [("gemini-x", "Gemini X", 1048576)]


async def test_openai_style_model_lists_read_any_context_field(tmp_path):
    body = {
        "data": [
            {"id": "router/model", "name": "Model", "context_length": 200000},  # OpenRouter, xAI
            {"id": "groq-model", "context_window": 131072},  # Groq, DeepSeek
            {"id": "mistral-model", "max_context_length": 262144},  # Mistral
            {"id": "gpt-x"},  # OpenAI doesn't say
        ]
    }
    models = await client("openrouter", Server(json.dumps(body)), tmp_path=tmp_path).list_models(refresh=True)
    assert [m.context for m in models] == [200000, 131072, 262144, None]


# ---- checking a model before switching ----

from tinyharness.llm.client import Ping  # noqa: E402

IMAGE_MODEL_404 = (
    '{"error":{"message":"The model `gpt-image-x` does not exist or you do not have access to it.",'
    '"type":"invalid_request_error","code":"model_not_found"}}'
)


async def test_ping_reports_status_and_the_providers_message():
    check = await client("openai", Server(IMAGE_MODEL_404, status=404)).ping("gpt-image-x")
    assert (check.ok, check.status) == (False, 404)
    assert check.message.startswith("The model `gpt-image-x` does not exist")
    assert check.model_unusable


@pytest.mark.parametrize(
    ("check", "unusable"),
    [
        (Ping(True, 200, "auth ok"), False),
        (Ping(False, 404, "not found"), True),
        (Ping(False, 400, "This model does not support tools"), True),
        (Ping(False, 400, "max_tokens is too small"), False),  # our test request, not the model
        (Ping(False, 503, "overloaded"), False),
        (Ping(False, None, "ReadTimeout"), False),
    ],
)
def test_only_a_clear_answer_blocks_a_switch(check, unusable):
    assert check.model_unusable is unusable


# ---- "the model is reasoning" signal ----


async def reasoning_flags(provider: str, stream: str) -> int:
    llm = client(provider, Server(stream))
    return sum([chunk.reasoning async for chunk in llm.stream(CONVERSATION, [READ])])


async def test_each_api_signals_when_the_model_reasons():
    assert await reasoning_flags("anthropic", ANTHROPIC_STREAM) == 1
    assert await reasoning_flags("openai", RESPONSES_STREAM) == 1
    assert await reasoning_flags("gemini", GEMINI_STREAM) == 1
    chat = sse(
        chat_chunk({"reasoning": "hmm", "reasoning_details": [{"type": "reasoning.text", "text": "hmm"}]}),
        chat_chunk({"reasoning": " more"}),
        chat_chunk({"content": "hi"}),
    )
    assert await reasoning_flags("openrouter", chat) == 1  # once, not per delta


async def test_no_reasoning_signal_for_a_plain_reply():
    assert (
        await reasoning_flags("openrouter", sse(chat_chunk({"content": "hello"}), chat_chunk(finish="stop")))
        == 0
    )
