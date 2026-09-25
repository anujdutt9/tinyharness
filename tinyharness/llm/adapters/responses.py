"""OpenAI Responses (`POST /responses`): OpenAI and xAI.

Requests are stateless (`store: false`): the whole history is sent every time.
Reasoning comes back encrypted (`include: reasoning.encrypted_content`); the
reply's full `output` list is kept and sent back as-is on the next request, which
is how reasoning carries across tool calls.
"""

from __future__ import annotations

import json
from typing import Any

from tinyharness.llm.adapters.base import Adapter, StreamParser
from tinyharness.llm.types import LLMError, Request, StreamChunk


class ResponsesAdapter(Adapter):
    name = "responses"
    request_path = "/responses"

    def payload(self, request: Request) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        for m in request.messages:
            if m.role == "tool":
                items.append(
                    {"type": "function_call_output", "call_id": m.tool_call_id, "output": m.content or ""}
                )
            elif m.role == "assistant" and (output := request.echo(m)) is not None:
                items.extend(output)  # exactly as received, reasoning included
            elif m.role == "assistant":
                if m.content:
                    items.append({"role": "assistant", "content": m.content})
                items.extend(
                    {
                        "type": "function_call",
                        "call_id": c.id,
                        "name": c.name,
                        "arguments": c.arguments or "{}",
                    }
                    for c in m.tool_calls
                )
            else:
                items.append({"role": "user", "content": m.content or ""})

        payload: dict[str, Any] = {
            "model": request.model,
            "input": items,
            "stream": True,
            "store": False,
            "include": ["reasoning.encrypted_content"],
        }
        if request.system:
            payload["instructions"] = request.system
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                    "strict": False,  # strict mode needs every field required; ours have defaults
                }
                for t in request.tools
            ]
        if request.effort:
            payload["reasoning"] = {"effort": request.effort}
        return payload

    def parser(self) -> StreamParser:
        return _ResponsesParser()

    def ping_payload(self, model: str) -> dict[str, Any]:
        return {"model": model, "input": "ping", "max_output_tokens": 16, "store": False}


class _ResponsesParser(StreamParser):
    def __init__(self) -> None:
        self.output: list[dict[str, Any]] = []
        self.streamed_arguments: set[int] = set()

    def feed(self, event: dict[str, Any]) -> list[StreamChunk]:
        match event.get("type"):
            case "response.output_text.delta":
                return [StreamChunk(text=event.get("delta", ""))]
            case "response.output_item.added" if event["item"].get("type") == "reasoning":
                return [StreamChunk(reasoning=True)]
            case "response.output_item.added" if event["item"].get("type") == "function_call":
                item = event["item"]
                delta = {
                    "index": event["output_index"],
                    "id": item["call_id"],
                    "function": {"name": item["name"]},
                }
                return [StreamChunk(tool_call_delta=delta)]
            case "response.function_call_arguments.delta":
                self.streamed_arguments.add(event["output_index"])
                delta = {"index": event["output_index"], "function": {"arguments": event.get("delta", "")}}
                return [StreamChunk(tool_call_delta=delta)]
            case "response.output_item.done":
                return self._item_done(event["output_index"], event["item"])
            case "response.completed":
                self.output = event.get("response", {}).get("output") or self.output
                return [StreamChunk(finish_reason="stop")]
            case "response.incomplete":
                details = event.get("response", {}).get("incomplete_details") or {}
                return [StreamChunk(finish_reason=details.get("reason", "incomplete"))]
            case "response.failed" | "error":
                error = event.get("response", {}).get("error") or event.get("error") or event
                raise LLMError(f"LLM error: {error.get('message', error)}")
        return []

    def _item_done(self, index: int, item: dict[str, Any]) -> list[StreamChunk]:
        self.output.append(item)
        # Some providers (xAI) send a call's arguments whole here instead of as deltas.
        if (
            item.get("type") == "function_call"
            and index not in self.streamed_arguments
            and item.get("arguments")
        ):
            args = item["arguments"] if isinstance(item["arguments"], str) else json.dumps(item["arguments"])
            return [StreamChunk(tool_call_delta={"index": index, "function": {"arguments": args}})]
        return []

    def state(self) -> Any:
        return self.output or None
