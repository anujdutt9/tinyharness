"""Tool calls that arrive in the wrong place or the wrong shape, made usable.

Open models are served behind a tool-call parser (vLLM's, OpenRouter's upstreams'). When
that parser trips, the call arrives as text in the model's own format, and a harness that
only reads `tool_calls` takes it for the final answer. Seen in tinyharness's logs from
Qwen3-Coder and Qwen3.5 (a teammate's "result" was the XML of the call it meant to make).

Two formats cover the open models in use:

    <tool_call>{"name": "Read", "arguments": {"path": "a.py"}}</tool_call>      Hermes, Qwen2.5
    <tool_call><function=Read><parameter=path>a.py</parameter></function></tool_call>  Qwen3-Coder

`recover_calls` only takes calls to tools that exist, so prose about tool calls (or an
example in a code block) isn't run.

`repair_arguments` fixes the small, unambiguous ways arguments come back malformed:
wrapped in a code fence, JSON encoded twice, two objects run together, or empty.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from typing import Any

from tinyharness.agent.history import ToolCall

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
_FUNCTION = re.compile(r"<function=([\w.-]+)>(.*?)(?:</function>|(?=<function=)|$)", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
_FENCE = re.compile(r"^```[\w-]*\s*\n?(.*?)\n?```$", re.DOTALL)


def recover_calls(text: str, schema_of: Callable[[str], dict[str, Any] | None]) -> tuple[str, list[ToolCall]]:
    """Tool calls written into `text`, and the text without them. `schema_of(name)` is the
    tool's JSON schema, or None if there's no such tool (then nothing is taken)."""
    start = _first_tag(text)
    if start < 0:
        return text, []
    calls: list[ToolCall] = []
    for body in _TOOL_CALL.findall(text[start:]) or [text[start:]]:
        if body.lstrip().startswith("{"):
            call = _from_json(body, schema_of)
            if call is not None:
                calls.append(call)
            continue
        for name, params in _FUNCTION.findall(body):
            if (schema := schema_of(name)) is not None:
                calls.append(_call(name, _from_parameters(params, schema)))
    if not calls:
        return text, []
    return text[:start].rstrip(), calls


def _first_tag(text: str) -> int:
    """Where the first call starts, ignoring any inside a code block (an example)."""
    for match in re.finditer(r"<tool_call>|<function=", text):
        if text.count("```", 0, match.start()) % 2 == 0:
            return match.start()
    return -1


def _from_json(body: str, schema_of: Callable[[str], dict[str, Any] | None]) -> ToolCall | None:
    try:
        value, _ = json.JSONDecoder().raw_decode(body.strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict) or schema_of(str(value.get("name"))) is None:
        return None
    arguments = value.get("arguments", value.get("parameters", {}))
    return _call(str(value["name"]), arguments if isinstance(arguments, dict) else {})


def _from_parameters(params: str, schema: dict[str, Any]) -> dict[str, Any]:
    """`<parameter=k>v</parameter>` pairs as arguments. Values are text; a parameter the
    schema says isn't a string (a number, a list) is read as JSON when it parses."""
    properties = schema.get("properties") or {}
    arguments: dict[str, Any] = {}
    for key, value in _PARAMETER.findall(params):
        if (properties.get(key) or {}).get("type", "string") != "string":
            try:
                arguments[key] = json.loads(value)
                continue
            except json.JSONDecodeError:
                pass
        arguments[key] = value
    return arguments


def _call(name: str, arguments: dict[str, Any]) -> ToolCall:
    return ToolCall(f"call_{uuid.uuid4().hex[:12]}", name, json.dumps(arguments, ensure_ascii=False))


def repair_arguments(arguments: str) -> str | None:
    """`arguments` made into a JSON object if one of the known slips explains why it isn't
    one; None if not (the model is then told what's wrong)."""
    text = arguments.strip()
    if not text:
        return "{}"
    if match := _FENCE.match(text):
        text = match.group(1).strip()
    try:
        value, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return None
    if isinstance(value, str):  # encoded twice: '"{\"path\": ...}"'
        return repair_arguments(value)
    if not isinstance(value, dict):
        return None
    # Something after the object: a second object run together with it, or stray text.
    # The first object is the call; anything else is dropped.
    return json.dumps(value, ensure_ascii=False) if end < len(text) or text != arguments else arguments
