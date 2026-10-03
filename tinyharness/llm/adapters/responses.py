"""OpenAI Responses (`POST /responses`): OpenAI and xAI.

Requests are stateless (`store: false`): the whole history is sent every time.
Reasoning comes back encrypted (`include: reasoning.encrypted_content`); the
reply's full `output` list is kept and sent back as-is on the next request, which
is how reasoning carries across tool calls.
"""

from __future__ import annotations

import json
from typing import Any

from tinyharness.agent.history import Image
from tinyharness.llm.adapters.base import Adapter, StreamParser
from tinyharness.llm.types import LLMError, Request, StreamChunk


class ResponsesAdapter(Adapter):
    name = "responses"
    request_path = "/responses"

    def payload(self, request: Request) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        pending: list[Image] = []  # from tool results: sent in a user message after them
        for m in request.messages:
            if m.role == "tool":
                items.append(
                    {"type": "function_call_output", "call_id": m.tool_call_id, "output": m.content or ""}
                )
                pending += m.images
                continue
            _flush_images(items, pending)
            if m.role == "assistant" and (output := request.echo(m)) is not None:
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
            elif m.images:
                parts = [{"type": "input_text", "text": m.content or ""}, *map(_image_part, m.images)]
                items.append({"role": "user", "content": parts})
            else:
                items.append({"role": "user", "content": m.content or ""})
        _flush_images(items, pending)

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
        if request.max_tokens:
            payload["max_output_tokens"] = request.max_tokens
        if request.effort:
            payload["reasoning"] = {"effort": request.effort}
        if request.show_reasoning:  # opt-in: some organizations must be verified to get these
            payload.setdefault("reasoning", {})["summary"] = "auto"
        if request.web_search:
            self.add_web_search(payload)
        return payload

    def add_web_search(self, payload: dict[str, Any]) -> None:
        payload.setdefault("tools", []).append({"type": "web_search"})

    def parser(self) -> StreamParser:
        return _ResponsesParser()

    def ping_payload(self, model: str) -> dict[str, Any]:
        return {"model": model, "input": "ping", "max_output_tokens": 16, "store": False}


class _ResponsesParser(StreamParser):
    def __init__(self) -> None:
        self.output: list[dict[str, Any]] = []
        self.streamed_arguments: set[int] = set()
        self.refused = False

    def feed(self, event: dict[str, Any]) -> list[StreamChunk]:
        match event.get("type"):
            case "response.output_text.delta":
                return [StreamChunk(text=event.get("delta", ""))]
            case "response.refusal.delta":  # shown like text, so a refusal isn't an empty reply
                return [StreamChunk(text=event.get("delta", ""))]
            case "response.refusal.done":
                self.refused = True
            case "response.reasoning_summary_text.delta":
                return [StreamChunk(reasoning_text=event.get("delta", ""))]
            case "response.reasoning_summary_part.done":
                return [StreamChunk(reasoning_text="\n\n")]  # between summary paragraphs
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
                return [StreamChunk(finish_reason="refusal" if self.refused else "stop")]
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


def _image_part(image: Image) -> dict[str, Any]:
    return {"type": "input_image", "image_url": image.data_url()}


def _flush_images(items: list[dict[str, Any]], pending: list[Image]) -> None:
    """Images from tool results, in a user message after them."""
    if pending:
        names = ", ".join(image.name or "image" for image in pending)
        text = {"type": "input_text", "text": f"The images from the tool results above: {names}"}
        items.append({"role": "user", "content": [text, *map(_image_part, pending)]})
        pending.clear()
