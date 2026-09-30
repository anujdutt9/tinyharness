"""Google Gemini Interactions (`POST /v1beta/interactions`).

A reply is a list of steps: `thought` (signed), `model_output` and
`function_call`. Requests are stateless (`store: false`), and the steps are sent
back exactly as received, which carries the thought signatures across tool calls.
"""

from __future__ import annotations

import json
from typing import Any

from tinyharness.llm.adapters.base import Adapter, StreamParser, parse_arguments, tool_names
from tinyharness.llm.types import LLMError, ModelInfo, Request, StreamChunk

THINKING_LEVELS = {"low": "low", "medium": "medium", "high": "high", "max": "high"}


class GeminiAdapter(Adapter):
    name = "gemini"
    request_path = "/interactions"

    def headers(self, api_key: str) -> dict[str, str]:
        return {"x-goog-api-key": api_key}

    def payload(self, request: Request) -> dict[str, Any]:
        names = tool_names(request)
        steps: list[dict[str, Any]] = []
        for m in request.messages:
            if m.role == "tool":
                steps.append(
                    {
                        "type": "function_result",
                        "name": names.get(m.tool_call_id or "", ""),
                        "call_id": m.tool_call_id,
                        "result": [{"type": "text", "text": m.content or ""}],
                    }
                )
            elif m.role == "assistant" and (saved := request.echo(m)) is not None:
                steps.extend(saved)  # exactly as received, thought signatures included
            elif m.role == "assistant":
                if m.content:
                    steps.append({"type": "model_output", "content": [{"type": "text", "text": m.content}]})
                steps.extend(
                    {
                        "type": "function_call",
                        "id": c.id,
                        "name": c.name,
                        "arguments": parse_arguments(c.arguments),
                    }
                    for c in m.tool_calls
                )
            else:
                steps.append({"type": "user_input", "content": [{"type": "text", "text": m.content or ""}]})

        payload: dict[str, Any] = {"model": request.model, "input": steps, "stream": True, "store": False}
        if request.system:
            payload["system_instruction"] = request.system
        if request.tools:
            payload["tools"] = [
                {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
                for t in request.tools
            ]
        if request.effort:
            payload["generation_config"] = {"thinking_level": THINKING_LEVELS[request.effort]}
        if request.web_search:
            self.add_web_search(payload)
        return payload

    def add_web_search(self, payload: dict[str, Any]) -> None:
        payload.setdefault("tools", []).append({"type": "google_search"})

    def parser(self) -> StreamParser:
        return _GeminiParser()

    def ping_payload(self, model: str) -> dict[str, Any]:
        return {"model": model, "input": "ping", "store": False}

    def parse_models(self, body: dict[str, Any]) -> tuple[list[ModelInfo], dict[str, str] | None]:
        models = [
            ModelInfo(
                id=str(item["name"]).removeprefix("models/"),
                label=item.get("displayName", ""),
                context=item.get("inputTokenLimit") or None,
                max_output=item.get("outputTokenLimit") or None,
            )
            for item in body.get("models") or []
            if "generateContent" in (item.get("supportedGenerationMethods") or ["generateContent"])
        ]
        token = body.get("nextPageToken")
        return models, ({"pageToken": token} if token else None)


class _GeminiParser(StreamParser):
    def __init__(self) -> None:
        self.steps: dict[int, dict[str, Any]] = {}
        self.arguments: dict[int, list[str]] = {}

    def feed(self, event: dict[str, Any]) -> list[StreamChunk]:
        match event.get("event_type"):
            case "step.start":
                return self._start(event["index"], dict(event["step"]))
            case "step.delta":
                return self._delta(event["index"], event["delta"])
            case "step.stop":
                return self._stop(event["index"])
            case "interaction.completed":
                return [StreamChunk(finish_reason=(event.get("interaction") or {}).get("status", "stop"))]
            case "error":
                error = event.get("error") or {}
                raise LLMError(f"LLM error: {error.get('message', json.dumps(error))}")
        return []

    def _start(self, index: int, step: dict[str, Any]) -> list[StreamChunk]:
        self.steps[index] = step
        if step.get("type") == "thought":
            return [StreamChunk(reasoning=True)]
        if step.get("type") == "function_call":
            self.arguments[index] = []
            delta = {"index": index, "id": step["id"], "function": {"name": step["name"]}}
            return [StreamChunk(tool_call_delta=delta)]
        return []

    def _delta(self, index: int, delta: dict[str, Any]) -> list[StreamChunk]:
        step = self.steps.setdefault(index, {"type": "model_output"})
        match delta.get("type"):
            case "text":
                content = step.setdefault("content", [])
                if not content or content[-1].get("type") != "text":
                    content.append({"type": "text", "text": ""})
                content[-1]["text"] += delta["text"]
                return [StreamChunk(text=delta["text"])]
            case "arguments_delta":
                self.arguments.setdefault(index, []).append(delta["arguments"])
                return [
                    StreamChunk(
                        tool_call_delta={"index": index, "function": {"arguments": delta["arguments"]}}
                    )
                ]
            case "thought_signature":
                step["signature"] = delta["signature"]
            case "thought_summary":
                content = delta.get("content") or {}
                step.setdefault("summary", []).append(content)
                if isinstance(content, dict) and content.get("text"):
                    return [StreamChunk(reasoning_text=str(content["text"]))]
        return []

    def _stop(self, index: int) -> list[StreamChunk]:
        step = self.steps.get(index, {})
        if step.get("type") != "function_call":
            return []
        streamed = "".join(self.arguments.get(index, []))
        if streamed:
            step["arguments"] = parse_arguments(streamed)
            return []
        # Arguments that arrived whole with step.start instead of as deltas.
        whole = json.dumps(step.get("arguments") or {})
        return [StreamChunk(tool_call_delta={"index": index, "function": {"arguments": whole}})]

    def state(self) -> Any:
        """The reply's steps, kept only when there are signed thoughts to send back."""
        steps = [self.steps[i] for i in sorted(self.steps)]
        return steps if any(s.get("type") == "thought" for s in steps) else None
