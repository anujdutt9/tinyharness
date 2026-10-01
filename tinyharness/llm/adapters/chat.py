"""OpenAI Chat Completions (`POST /chat/completions`).

The format most providers accept: OpenRouter, DeepSeek, Mistral, Groq and nearly
every gateway. Providers extend it with their own reasoning fields, which must be
sent back unchanged on the next request. All three variants are kept:

    reasoning_details   OpenRouter (list of typed items, some signed)
    reasoning_content   DeepSeek (text)
    reasoning           Groq, vLLM, Ollama (text)
    content: [chunks]   Mistral ("thinking" and "text" chunks in one list)

OpenRouter streams `reasoning` too, as a readable copy of `reasoning_details`; it's
sent back only when a reply had nothing else, so the same reasoning never goes twice.
"""

from __future__ import annotations

from typing import Any

from tinyharness.agent.history import Image
from tinyharness.llm.adapters.base import Adapter, StreamParser
from tinyharness.llm.types import LLMError, Request, StreamChunk


class ChatAdapter(Adapter):
    name = "chat"
    request_path = "/chat/completions"

    def payload(self, request: Request) -> dict[str, Any]:
        messages: list[dict[str, Any]] = (
            [{"role": "system", "content": request.system}] if request.system else []
        )
        pending: list[Image] = []  # from tool results: tool messages are text-only here
        for m in request.messages:
            if m.role == "tool":
                messages.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content or ""})
                pending += m.images
                continue
            _flush_images(messages, pending)
            if m.role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
                if m.tool_calls:
                    msg["tool_calls"] = [c.to_dict() for c in m.tool_calls]
                msg.update(request.echo(m) or {})
                messages.append(msg)
            elif m.images:
                parts = [{"type": "text", "text": m.content or ""}, *map(_image_part, m.images)]
                messages.append({"role": "user", "content": parts})
            else:
                messages.append({"role": "user", "content": m.content or ""})
        _flush_images(messages, pending)

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},  # token counts in the last chunk
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in request.tools
            ]
        if request.effort:
            _set_path(payload, self.provider.effort_param, request.effort)
        if request.web_search:
            self.add_web_search(payload)
        return payload

    def add_web_search(self, payload: dict[str, Any]) -> None:
        if not isinstance(self.provider.web_search, dict):
            raise NotImplementedError
        payload.update(self.provider.web_search)

    def parser(self) -> StreamParser:
        return _ChatParser()

    def ping_payload(self, model: str) -> dict[str, Any]:
        return {"model": model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 1}


def _image_part(image: Image) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": image.data_url()}}


def _flush_images(messages: list[dict[str, Any]], pending: list[Image]) -> None:
    """Images from a batch of tool results, in a user message after all of them (nothing
    may come between the tool messages that answer one assistant turn)."""
    if pending:
        names = ", ".join(image.name or "image" for image in pending)
        text = {"type": "text", "text": f"The images from the tool results above: {names}"}
        messages.append({"role": "user", "content": [text, *map(_image_part, pending)]})
        pending.clear()


def _set_path(payload: dict[str, Any], dotted: str, value: Any) -> None:
    """Set `a.b.c` in a nested dict: providers disagree on where the effort goes."""
    *parents, leaf = dotted.split(".")
    for key in parents:
        payload = payload.setdefault(key, {})
    payload[leaf] = value


class _ChatParser(StreamParser):
    def __init__(self) -> None:
        self.text: list[str] = []
        self.reasoning_content: list[str] = []
        self.reasoning: list[str] = []  # the plain `reasoning` field
        self.reasoning_details: dict[int, dict[str, Any]] = {}
        self.thinking_chunks: list[str] = []  # Mistral
        self.reasoning_seen = False

    def feed(self, event: dict[str, Any]) -> list[StreamChunk]:
        if error := event.get("error"):  # OpenRouter reports mid-stream errors this way
            raise LLMError(f"LLM error: {error.get('message', error) if isinstance(error, dict) else error}")
        choices = event.get("choices") or []
        if not choices:
            return []
        delta = choices[0].get("delta") or {}
        finish = choices[0].get("finish_reason")
        chunks: list[StreamChunk] = []

        content = delta.get("content")
        if isinstance(content, list):  # Mistral: typed chunks
            for part in content:
                if part.get("type") == "thinking":
                    self.thinking_chunks.extend(t.get("text", "") for t in part.get("thinking") or [])
                elif part.get("type") == "text" and part.get("text"):
                    chunks.append(self._text(part["text"]))
        elif content:
            chunks.append(self._text(content))

        reasoning = bool(
            delta.get("reasoning")  # OpenRouter's readable reasoning text
            or delta.get("reasoning_content")
            or delta.get("reasoning_details")
            or (isinstance(content, list) and any(p.get("type") == "thinking" for p in content))
        )
        if reasoning and not self.reasoning_seen:
            self.reasoning_seen = True
            chunks.insert(0, StreamChunk(reasoning=True))
        if readable := _readable_reasoning(delta, content):
            chunks.insert(1 if reasoning else 0, StreamChunk(reasoning_text=readable))
        if delta.get("reasoning_content"):
            self.reasoning_content.append(delta["reasoning_content"])
        if isinstance(delta.get("reasoning"), str):
            self.reasoning.append(delta["reasoning"])
        for i, detail in enumerate(delta.get("reasoning_details") or []):
            self._merge_detail(detail.get("index", i), detail)
        chunks.extend(StreamChunk(tool_call_delta=tc) for tc in delta.get("tool_calls") or [])
        if finish:
            chunks.append(StreamChunk(finish_reason=finish))
        return chunks

    def _text(self, text: str) -> StreamChunk:
        self.text.append(text)
        return StreamChunk(text=text)

    def _merge_detail(self, index: int, delta: dict[str, Any]) -> None:
        """reasoning_details stream in pieces; text-like fields concatenate, the rest overwrite."""
        item = self.reasoning_details.setdefault(index, {})
        for key, value in delta.items():
            if key in ("text", "summary", "data") and isinstance(value, str):
                item[key] = item.get(key, "") + value
            elif value is not None:
                item[key] = value

    def state(self) -> Any:
        state: dict[str, Any] = {}
        if self.reasoning_details:
            state["reasoning_details"] = [self.reasoning_details[i] for i in sorted(self.reasoning_details)]
        if self.reasoning_content:
            state["reasoning_content"] = "".join(self.reasoning_content)
        if self.thinking_chunks:
            state["content"] = [
                {"type": "thinking", "thinking": [{"type": "text", "text": "".join(self.thinking_chunks)}]},
                {"type": "text", "text": "".join(self.text)},
            ]
        if self.reasoning and not state:  # only form of it (Groq, vLLM, Ollama); see the docstring
            state["reasoning"] = "".join(self.reasoning)
        return state or None


def _readable_reasoning(delta: dict[str, Any], content: Any) -> str:
    """The reasoning text in one chunk, from whichever field this provider uses."""
    if isinstance(delta.get("reasoning"), str):  # OpenRouter
        return str(delta["reasoning"])
    if isinstance(delta.get("reasoning_content"), str):  # DeepSeek
        return str(delta["reasoning_content"])
    if isinstance(content, list):  # Mistral
        return "".join(
            t.get("text", "") for p in content if p.get("type") == "thinking" for t in p.get("thinking") or []
        )
    return ""
