"""Anthropic Messages (`POST /v1/messages`).

- Thinking: `thinking: {type: "adaptive"}` when the model list says the model
  supports it, with effort in `output_config`. Thinking blocks come back signed
  and are sent back unchanged (the whole reply's content blocks are kept).
- Signed thinking is bound to everything before it, so any edit to earlier
  messages (our compaction, or switching models) would invalidate it. We opt in to
  `prefix_mismatch_behavior: "drop_block"` so the API drops such blocks instead of
  rejecting the request.
- Caching: the top-level `cache_control` lets the API place the breakpoint itself.
"""

from __future__ import annotations

import json
from typing import Any

from tinyharness.llm.adapters.base import Adapter, StreamParser, parse_arguments
from tinyharness.llm.types import LLMError, ModelInfo, Request, StreamChunk

API_VERSION = "2023-06-01"
BETAS = "thinking-binding-controls-2026-08-01"
DEFAULT_MAX_TOKENS = 16_000  # used only when the model list hasn't told us the model's limit


class AnthropicAdapter(Adapter):
    name = "anthropic"
    request_path = "/messages"

    def headers(self, api_key: str) -> dict[str, str]:
        return {"x-api-key": api_key, "anthropic-version": API_VERSION, "anthropic-beta": BETAS}

    def payload(self, request: Request) -> dict[str, Any]:
        info = request.info or ModelInfo(id=request.model)
        payload: dict[str, Any] = {
            "model": request.model,
            "max_tokens": info.max_output or DEFAULT_MAX_TOKENS,
            "messages": _messages(request),
            "stream": True,
            "cache_control": {"type": "ephemeral"},
        }
        if request.system:
            payload["system"] = request.system
        if request.tools:
            payload["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in request.tools
            ]
        if info.adaptive_thinking:
            payload["thinking"] = {
                "type": "adaptive",
                "block_binding": {"prefix_mismatch_behavior": "drop_block"},
            }
        if request.effort and (info.efforts is None or request.effort in info.efforts):
            payload["output_config"] = {"effort": request.effort}
        return payload

    def parser(self) -> StreamParser:
        return _AnthropicParser()

    def ping_payload(self, model: str) -> dict[str, Any]:
        return {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]}

    def parse_models(self, body: dict[str, Any]) -> tuple[list[ModelInfo], dict[str, str] | None]:
        models = []
        for item in body.get("data") or []:
            caps = item.get("capabilities") or {}
            thinking = (caps.get("thinking") or {}).get("types") or {}
            effort = caps.get("effort") or {}
            efforts = frozenset(k for k, v in effort.items() if isinstance(v, dict) and v.get("supported"))
            models.append(
                ModelInfo(
                    id=item["id"],
                    label=item.get("display_name", ""),
                    context=item.get("max_input_tokens") or None,
                    max_output=item.get("max_tokens") or None,
                    adaptive_thinking=bool((thinking.get("adaptive") or {}).get("supported"))
                    if caps
                    else None,
                    efforts=(efforts if effort.get("supported") else frozenset()) if caps else None,
                )
            )
        more = {"after_id": body["last_id"]} if body.get("has_more") and body.get("last_id") else None
        return models, more


def _messages(request: Request) -> list[dict[str, Any]]:
    """Tool results go in a user message; consecutive user-side content is merged into one."""
    out: list[dict[str, Any]] = []

    def user(block: dict[str, Any]) -> None:
        if out and out[-1]["role"] == "user":
            out[-1]["content"].append(block)
        else:
            out.append({"role": "user", "content": [block]})

    for m in request.messages:
        if m.role == "tool":
            user({"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content or ""})
        elif m.role == "assistant":
            blocks = request.echo(m)
            if blocks is None:
                blocks = [{"type": "text", "text": m.content}] if m.content else []
                blocks += [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": parse_arguments(c.arguments)}
                    for c in m.tool_calls
                ]
            out.append({"role": "assistant", "content": blocks})
        else:
            user({"type": "text", "text": m.content or ""})
    return out


class _AnthropicParser(StreamParser):
    def __init__(self) -> None:
        self.blocks: dict[int, dict[str, Any]] = {}
        self.json: dict[int, list[str]] = {}

    def feed(self, event: dict[str, Any]) -> list[StreamChunk]:
        match event.get("type"):
            case "content_block_start":
                index, block = event["index"], dict(event["content_block"])
                self.blocks[index] = block
                if block["type"] in ("thinking", "redacted_thinking"):
                    return [StreamChunk(reasoning=True)]
                if block["type"] == "tool_use":
                    block.pop("input", None)
                    self.json[index] = []
                    return [
                        StreamChunk(
                            tool_call_delta={
                                "index": index,
                                "id": block["id"],
                                "function": {"name": block["name"]},
                            }
                        )
                    ]
            case "content_block_delta":
                return self._delta(event["index"], event["delta"])
            case "content_block_stop":
                block = self.blocks.get(event["index"], {})
                if block.get("type") == "tool_use":
                    block["input"] = parse_arguments("".join(self.json.get(event["index"], [])))
            case "message_delta":
                if reason := (event.get("delta") or {}).get("stop_reason"):
                    return [StreamChunk(finish_reason=reason)]
            case "error":
                error = event.get("error") or {}
                raise LLMError(f"LLM error: {error.get('message', json.dumps(error))}")
        return []

    def _delta(self, index: int, delta: dict[str, Any]) -> list[StreamChunk]:
        block = self.blocks.setdefault(index, {})
        match delta.get("type"):
            case "text_delta":
                block["text"] = block.get("text", "") + delta["text"]
                return [StreamChunk(text=delta["text"])]
            case "input_json_delta":
                self.json.setdefault(index, []).append(delta["partial_json"])
                return [
                    StreamChunk(
                        tool_call_delta={"index": index, "function": {"arguments": delta["partial_json"]}}
                    )
                ]
            case "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta["thinking"]
            case "signature_delta":
                block["signature"] = delta["signature"]
        return []

    def state(self) -> Any:
        """The reply's content blocks, kept only when it has thinking to send back."""
        blocks = [self.blocks[i] for i in sorted(self.blocks)]
        if any(b.get("type") in ("thinking", "redacted_thinking") for b in blocks):
            return blocks
        return None
