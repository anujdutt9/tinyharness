"""Each provider adapter, end to end through LLMClient, against recorded-style streams.

The streams follow each provider's documented event format. A fake HTTP transport
captures the request body so we can check what would be sent.
"""

import json

import httpx
import pytest

from tinyharness.agent.history import Image, Message, ToolCall
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
        {"event_type": "interaction.completed", "interaction": {"status": "requires_action"}},
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


async def test_chat_keeps_plain_reasoning_and_sends_it_back():
    """Groq (gpt-oss, Qwen), vLLM and Ollama stream reasoning only as `reasoning`."""
    stream = sse(
        chat_chunk({"reasoning": "The user wants "}),
        chat_chunk({"reasoning": "a.py read."}),
        chat_chunk(
            {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "Read", "arguments": "{}"}}]}
        ),
        chat_chunk(finish="tool_calls"),
    )
    server = Server(stream, sse(chat_chunk(finish="stop")))
    llm = client("groq", server, "openai/gpt-oss-120b")
    _, _, state = await run(llm, CONVERSATION)
    assert state.data == {"reasoning": "The user wants a.py read."}

    await run(llm, after_tool_call(state))
    assert server.requests[1]["messages"][2]["reasoning"] == "The user wants a.py read."


async def test_openrouter_reasoning_is_sent_back_once():
    """OpenRouter streams `reasoning` as a readable copy of `reasoning_details`."""
    stream = sse(
        chat_chunk(
            {"reasoning": "hmm", "reasoning_details": [{"type": "reasoning.text", "text": "hmm", "index": 0}]}
        ),
        chat_chunk({"content": "hi"}),
        chat_chunk(finish="stop"),
    )
    _, _, state = await run(client("openrouter", Server(stream)), CONVERSATION)
    assert "reasoning" not in state.data and state.data["reasoning_details"][0]["text"] == "hmm"


async def test_chat_mid_stream_error_is_raised():
    stream = sse({"error": {"message": "invalid tool schema"}, "choices": []})
    server = Server(stream, stream)
    with pytest.raises(LLMError, match="invalid tool schema"):
        await run(client("openrouter", server), CONVERSATION)
    assert len(server.requests) == 1  # a problem with the request isn't retried


async def test_a_busy_error_inside_the_stream_is_retried_before_any_output():
    busy = sse({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}, named=True)
    server = Server(busy, ANTHROPIC_STREAM)
    text, _, _ = await run(client("anthropic", server), CONVERSATION)
    assert len(server.requests) == 2 and text


async def test_a_busy_error_that_persists_is_raised():
    stream = sse({"error": {"message": "upstream overloaded"}, "choices": []})
    with pytest.raises(LLMError, match="overloaded"):
        await run(client("openrouter", Server(*[stream] * 4)), CONVERSATION)


async def test_a_body_that_is_not_an_event_stream_is_reported():
    """A proxy's error page, or a server that ignored `stream: true`, with status 200."""
    body = '{"object": "error", "message": "model not loaded"}'
    with pytest.raises(LLMError, match=r"didn't answer with an event stream.*model not loaded"):
        await run(client("openrouter", Server(body)), CONVERSATION)


async def test_an_event_split_over_several_data_lines_is_joined():
    event = json.dumps(chat_chunk({"content": "hello"}, finish="stop"), indent=2)
    stream = "".join(f"data: {line}\n" for line in event.splitlines()) + "\ndata: [DONE]\n\n"
    text, _, _ = await run(client("openrouter", Server(stream)), CONVERSATION)
    assert text == "hello"


async def test_stop_reasons_are_normalized_and_the_route_is_reported():
    stream = (
        sse({**chat_chunk({"content": "hi"}, finish="length"), "provider": "Novita"}) + "data: [DONE]\n\n"
    )
    chunks = [c async for c in client("openrouter", Server(stream)).stream(CONVERSATION, [READ])]
    finish = next(c for c in chunks if c.finish_reason)
    assert (finish.finish_reason, finish.finish_detail) == ("length", "length")
    assert chunks[-1].route == "Novita"


@pytest.mark.parametrize(
    ("provider", "stream"),
    [
        ("openrouter", sse(chat_chunk({"content": "The answer is"}))),  # no finish_reason, no [DONE]
        ("openai", sse({"type": "response.output_text.delta", "output_index": 0, "delta": "The answer is"})),
        ("anthropic", "\n".join(ANTHROPIC_STREAM.split("\n")[:6]) + "\n\n"),
        ("gemini", "\n".join(GEMINI_STREAM.split("\n")[:4]) + "\n\n"),
        ("openrouter", "data: {not json\n\n"),  # malformed data is skipped; that alone isn't a reply
    ],
)
async def test_a_stream_that_is_cut_off_is_an_error(provider, stream):
    """A proxy closing the connection mid-reply must not look like a finished answer."""
    with pytest.raises(LLMError, match=r"incomplete response from .*: the stream ended without a finish"):
        await run(client(provider, Server(stream)), CONVERSATION)


@pytest.mark.parametrize(
    ("provider", "stream"),
    [
        (
            "openrouter",
            sse(chat_chunk({"content": "hi"})) + "data: [DONE]\n\n",
        ),  # servers without finish_reason
        ("anthropic", sse({"type": "message_stop"}, named=True)),
    ],
)
async def test_a_stream_that_says_it_is_done_is_complete(provider, stream):
    await run(client(provider, Server(stream)), CONVERSATION)


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
            {"id": "planner", "object": "model", "max_model_len": 32768},  # vLLM
            {"id": "gpt-x"},  # OpenAI doesn't say
        ]
    }
    models = await client("openrouter", Server(json.dumps(body)), tmp_path=tmp_path).list_models(refresh=True)
    assert [m.context for m in models] == [200000, 131072, 262144, 32768, None]


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
        chat_chunk(finish="stop"),
    )
    assert await reasoning_flags("openrouter", chat) == 1  # once, not per delta


async def test_no_reasoning_signal_for_a_plain_reply():
    assert (
        await reasoning_flags("openrouter", sse(chat_chunk({"content": "hello"}), chat_chunk(finish="stop")))
        == 0
    )


# ---- readable reasoning, where providers share it ----


async def reasoning_text(provider: str, stream: str, **kwargs) -> tuple[str, list[dict]]:
    server = Server(stream)
    llm = client(provider, server, **kwargs)
    text = "".join([chunk.reasoning_text async for chunk in llm.stream(CONVERSATION, [READ])])
    return text, server.requests


async def test_anthropic_thinking_text():
    stream = sse(
        {"type": "message_start", "message": {"usage": {"input_tokens": 5, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Check "},
        },
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "a.py."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        named=True,
    )
    assert (await reasoning_text("anthropic", stream))[0] == "Check a.py."


async def test_openai_summaries_only_when_asked_for():
    stream = sse(
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning"}},
        {"type": "response.reasoning_summary_text.delta", "delta": "Look at a.py"},
        {"type": "response.completed", "response": {"output": []}},
    )
    text, [request] = await reasoning_text("openai", stream)
    assert "summary" not in request.get("reasoning", {})  # not asked for by default
    llm_server = Server(stream)
    llm = client("openai", llm_server)
    llm.show_reasoning = True
    text = "".join([chunk.reasoning_text async for chunk in llm.stream(CONVERSATION, [READ])])
    assert text == "Look at a.py" and llm_server.requests[0]["reasoning"]["summary"] == "auto"


@pytest.mark.parametrize(
    ("provider", "delta"),
    [
        ("openrouter", {"reasoning": "Look at a.py"}),
        ("deepseek", {"reasoning_content": "Look at a.py"}),
        (
            "mistral",
            {"content": [{"type": "thinking", "thinking": [{"type": "text", "text": "Look at a.py"}]}]},
        ),
    ],
)
async def test_chat_reasoning_text(provider, delta):
    stream = sse(chat_chunk(delta), chat_chunk({"content": "hi"}), chat_chunk(finish="stop"))
    assert (await reasoning_text(provider, stream))[0] == "Look at a.py"


async def test_gemini_thought_summaries():
    stream = sse(
        {"event_type": "step.start", "index": 0, "step": {"type": "thought"}},
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {"type": "thought_summary", "content": {"type": "text", "text": "Look at a.py"}},
        },
        {"event_type": "step.stop", "index": 0},
        {"event_type": "interaction.completed", "interaction": {"status": "completed"}},
        named=True,
    )
    assert (await reasoning_text("gemini", stream))[0] == "Look at a.py"


# ---- background work on the small model ----


def recording(llm: LLMClient, fail_on: str | None = None, error: Exception | None = None) -> list:
    """Replace llm.one_shot with a fake that records the model each call used."""
    used = []

    async def one_shot(prompt, *, system=None, model=None, web_search=False):
        used.append(model or llm.model)
        if model == fail_on:
            raise error
        return "ok"

    llm.one_shot = one_shot
    return used


async def test_background_work_uses_the_small_model():
    llm = client("anthropic", Server(), model="claude-sonnet-5")
    used = recording(llm)
    assert await llm.one_shot_small("summarize") == "ok"
    assert used == ["claude-haiku-4-5"]


async def test_background_work_falls_back_to_the_main_model():
    llm = client("anthropic", Server(), model="claude-sonnet-5")
    used = recording(llm, fail_on="claude-haiku-4-5", error=ContextOverflowError("too long"))
    assert await llm.one_shot_small("summarize") == "ok"
    assert used == ["claude-haiku-4-5", "claude-sonnet-5"]


async def test_background_work_does_not_retry_a_network_failure():
    from tinyharness.llm.types import ProviderConnectionError

    llm = client("anthropic", Server(), model="claude-sonnet-5")
    used = recording(
        llm,
        fail_on="claude-haiku-4-5",
        error=ProviderConnectionError("offline", host="api.anthropic.com", code="ECONNRESET"),
    )
    with pytest.raises(ProviderConnectionError):
        await llm.one_shot_small("summarize")
    assert used == ["claude-haiku-4-5"]


async def test_unreadable_stream_events_are_skipped_but_reported(caplog):
    stream = sse(chat_chunk({"content": "Hello"})) + "data: {not json\n\n" + sse(chat_chunk(finish="stop"))
    text, _, _ = await run(client("openrouter", Server(stream)), CONVERSATION)
    assert text == "Hello"
    assert "1 stream event(s) from https://openrouter.ai" in caplog.text
    assert "may be missing some text" in caplog.text


# ---- images ----

SHOT = Image("image/png", "iVBORw0KGgo=", "slide-1.png")
CHART = Image("image/png", "iVBORw0KGgp=", "chart.png")


def with_images() -> list[Message]:
    """The user attaches a screenshot; the model reads two images in one batch."""
    return [
        Message(role="system", content="be brief"),
        Message(role="user", content="what's wrong with slide 1?", images=[SHOT]),
        Message(
            role="assistant",
            tool_calls=[
                ToolCall("c1", "Read", '{"path": "chart.png"}'),
                ToolCall("c2", "Read", '{"path": "a.py"}'),
            ],
        ),
        Message(role="tool", tool_call_id="c1", content="chart.png: image/png image", images=[CHART]),
        Message(role="tool", tool_call_id="c2", content="print(1)"),
    ]


async def sent(provider: str) -> dict:
    server = Server(ANY_REPLY[provider])
    await run(client(provider, server), with_images())
    return server.requests[0]


ANY_REPLY = {
    "anthropic": sse({"type": "message_stop"}, named=True),
    "openrouter": sse(chat_chunk(finish="stop")),
    "openai": sse({"type": "response.completed", "response": {"output": []}}),
    "gemini": sse({"event_type": "interaction.completed", "interaction": {}}, named=True),
}


async def test_anthropic_gets_images_in_the_user_message_and_the_tool_result():
    body = await sent("anthropic")
    first, _, results = body["messages"]
    assert first["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": SHOT.data},
    }
    chart = results["content"][0]
    assert chart["type"] == "tool_result" and chart["content"][1]["source"]["data"] == CHART.data


async def test_chat_sends_tool_images_after_every_tool_message():
    """Tool messages take only text, and nothing may come between the ones that answer a
    single assistant turn: the images follow in a user message."""
    messages = (await sent("openrouter"))["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "tool", "user"]
    assert messages[1]["content"][1] == {"type": "image_url", "image_url": {"url": SHOT.data_url()}}
    follow_up = messages[-1]["content"]
    assert follow_up[0]["text"] == "The images from the tool results above: chart.png"
    assert follow_up[1]["image_url"]["url"] == "data:image/png;base64," + CHART.data


async def test_responses_sends_input_images():
    items = (await sent("openai"))["input"]
    assert items[0]["content"][1] == {"type": "input_image", "image_url": SHOT.data_url()}
    kinds = [i.get("type") or i.get("role") for i in items]
    assert kinds[-3:] == ["function_call_output", "function_call_output", "user"]
    assert items[-1]["content"][1]["image_url"] == CHART.data_url()


async def test_gemini_sends_image_parts():
    steps = (await sent("gemini"))["input"]
    assert steps[0]["content"][1] == {"type": "image", "data": SHOT.data, "mime_type": "image/png"}
    assert [s["type"] for s in steps[-3:]] == ["function_result", "function_result", "user_input"]
    assert steps[-1]["content"][1]["data"] == CHART.data


# ---- retries ----

OK_REPLY = sse({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}) + "data: [DONE]\n\n"


class Flaky(Server):
    """Answers with `failures` (status, headers) first, then OK_REPLY."""

    def __init__(self, *failures: tuple[int, dict[str, str]]) -> None:
        super().__init__()
        self.failures = list(failures)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content or b"{}"))
        if self.failures:
            status, headers = self.failures.pop(0)
            return httpx.Response(status, headers=headers, text='{"error": "busy"}')
        return httpx.Response(200, text=OK_REPLY)


async def test_rate_limits_and_overload_are_retried(monkeypatch):
    waits = []

    async def sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr("tinyharness.llm.client.asyncio.sleep", sleep)
    monkeypatch.setattr("tinyharness.llm.client.RETRY_MAX_SECONDS", 60.0)
    server = Flaky((429, {"Retry-After": "2"}), (503, {}))
    text, _, _ = await run(client("deepseek", server), CONVERSATION)
    assert text == "hi" and len(server.requests) == 3
    assert waits[0] == 2.0  # the server's Retry-After wins over backoff


async def test_retries_give_up_with_the_last_error():
    server = Flaky(*[(503, {})] * 5)
    with pytest.raises(LLMError, match="503"):
        await run(client("deepseek", server, retries=2), CONVERSATION)
    assert len(server.requests) == 3


async def test_a_client_error_is_not_retried():
    server = Flaky((400, {}))
    with pytest.raises(LLMError, match="400"):
        await run(client("deepseek", server), CONVERSATION)
    assert len(server.requests) == 1


def test_retry_delay_honors_retry_after_and_caps_backoff(monkeypatch):
    from tinyharness.llm.client import retry_delay

    monkeypatch.setattr("tinyharness.llm.client.RETRY_MAX_SECONDS", 60.0)
    assert retry_delay("7", 0) == 7.0
    assert retry_delay("600", 0) == 60.0
    assert retry_delay("Wed, 21 Oct 2026 07:28:00 GMT", 0) <= 60.0
    assert 0 <= retry_delay(None, 10) <= 60.0


# ---- a self-hosted server that needs no key ----


async def test_a_keyless_provider_sends_no_authorization_header():
    from tinyharness.llm.providers import Provider

    local = Provider(name="local", base_url="http://127.0.0.1:8000/v1", models=["planner"])
    assert not local.needs_key and local.has_key and local.default_model == "planner"
    server = Server(OK_REPLY)
    llm = LLMClient(local, "", "planner", transport=httpx.MockTransport(server))
    text, _, _ = await run(llm, CONVERSATION)
    assert text == "hi" and "authorization" not in server.headers[0]


def test_a_silent_connection_times_out_long_before_the_request_timeout():
    from tinyharness.llm.client import STALL_SECONDS

    llm = client("deepseek", Server(), timeout=1800.0)
    timeouts = llm._timeouts()
    assert timeouts.read == STALL_SECONDS < 1800.0 and timeouts.connect == 1800.0


def test_parallel_calls_without_an_index_stay_separate():
    """Some OpenAI-compatible servers leave `index` out; two calls became one 'ReadGlob'."""
    acc = ToolCallAccumulator()
    acc.add({"id": "a", "function": {"name": "Read", "arguments": '{"path": "x"}'}})
    acc.add({"id": "b", "function": {"name": "Glob", "arguments": '{"pattern": "*"}'}})
    assert [(c.id, c.name, c.arguments) for c in acc.calls()] == [
        ("a", "Read", '{"path": "x"}'),
        ("b", "Glob", '{"pattern": "*"}'),
    ]


def test_a_name_repeated_in_every_delta_is_not_doubled():
    acc = ToolCallAccumulator()
    acc.add({"index": 0, "id": "a", "function": {"name": "Read", "arguments": '{"pa'}})
    acc.add({"index": 0, "function": {"name": "Read", "arguments": 'th": "x"}'}})
    assert [(c.name, c.arguments) for c in acc.calls()] == [("Read", '{"path": "x"}')]


# ---- found by review of the audit fixes ----


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (
            400,
            '{"error": {"message": "maximum context length is 32768 tokens. However, you requested 33268 tokens (32768 in the messages, 500 in the completion)"}}',
        ),
        (401, '{"error": {"message": "invalid api key; upstream unavailable"}}'),
        (404, '{"error": {"message": "model not found upstream"}}'),
    ],
)
async def test_a_request_error_is_never_retried(status, body):
    server = Server(body, body, body, body, status=status)
    with pytest.raises(LLMError):
        await run(client("openrouter", server), CONVERSATION)
    assert len(server.requests) == 1


async def test_events_stream_as_they_arrive_even_without_blank_lines():
    from tinyharness.llm.client import _sse_data

    seen = []

    async def lines():
        for i in range(3):
            seen.append(f"sent {i}")
            yield f'data: {{"n": {i}}}'

    async for data in _sse_data(lines(), []):
        seen.append(f"got {data}")
    assert seen == ["sent 0", 'got {"n": 0}', "sent 1", 'got {"n": 1}', "sent 2", 'got {"n": 2}']


def test_ids_inside_echoed_reasoning_are_left_alone():
    import re

    from tinyharness.agent.history import Message, ReasoningState, ToolCall, well_formed

    call = Message(
        role="assistant",
        tool_calls=[ToolCall("fc:abc.1", "Read", "{}")],
        state=ReasoningState("gemini:g", [{"id": "fc:abc.1"}]),
    )
    result = Message(role="tool", tool_call_id="fc:abc.1", content="ok")
    pattern = re.compile(r"[A-Za-z0-9_-]{1,64}")
    same, _ = well_formed([call, result], valid_id=pattern, owner="gemini:g")
    assert same[0].tool_calls[0].id == same[1].tool_call_id == "fc:abc.1"  # matches the echoed state
    other, _ = well_formed([call, result], valid_id=pattern, owner="anthropic:c")
    assert pattern.fullmatch(other[0].tool_calls[0].id) and other[1].tool_call_id == other[0].tool_calls[0].id


async def test_opt_in_reasoning_models_think_at_medium_by_default(tmp_path):
    """Seen live: Gemma 4 on OpenRouter ran agent tasks with zero reasoning tokens, because
    its thinking is a per-request switch and nothing switched it on."""
    from tinyharness.llm.adapters.base import Adapter
    from tinyharness.llm.models import ModelDirectory

    models, _ = Adapter.parse_models(
        None, {"data": [{"id": "g4", "supported_parameters": ["tools", "reasoning"]}, {"id": "plain"}]}
    )
    assert [m.reasoning for m in models] == [True, None]
    directory = ModelDirectory(tmp_path)
    directory.store("openrouter", models)

    def sent(model, effort=None, tools=(READ,)):
        server = Server(OK_REPLY)
        llm = client("openrouter", server, model=model, tmp_path=tmp_path)
        llm.effort = effort
        return server, llm

    for model, effort, tools, expected in [
        ("g4", None, (READ,), "medium"),  # the default, on an agent's turn
        ("g4", None, (), None),  # not for the harness's own quick calls
        ("g4", "high", (READ,), "high"),  # the user's choice wins
        ("g4", "none", (READ,), None),  # and "none" turns it off
        ("plain", None, (READ,), None),  # models without the switch are left alone
    ]:
        server, llm = sent(model, effort)
        await run(llm, CONVERSATION, tools=tools)
        assert server.requests[0].get("reasoning", {}).get("effort") == expected, (model, effort, tools)


def test_an_older_model_cache_is_refetched(tmp_path):
    import json

    from tinyharness.llm.models import ModelDirectory

    (tmp_path / "openrouter.json").write_text(json.dumps({"fetched_at": 9e12, "models": [{"id": "g4"}]}))
    directory = ModelDirectory(tmp_path)
    assert directory.info("openrouter", "g4") is not None and not directory.is_fresh("openrouter")
