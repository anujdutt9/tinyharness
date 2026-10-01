"""Tool calls that arrive in the wrong place or the wrong shape, made usable.

Open models are served behind a tool-call parser (vLLM's, OpenRouter's upstreams'). When
that parser trips, the call arrives as text in the model's own format, and a harness that
only reads `tool_calls` takes it for the final answer. Seen in tinyharness's logs from
Qwen3-Coder and Qwen3.5 (a teammate's "result" was the XML of the call it meant to make).

Three formats cover the open models in use:

    <tool_call>{"name": "Read", "arguments": {"path": "a.py"}}</tool_call>      Hermes, Qwen2.5
    <tool_call><function=Read><parameter=path>a.py</parameter></function></tool_call>  Qwen3-Coder
    <|tool_call>call:read{path: "a.py"}<tool_call|>                             Gemma 4

`recover_calls` is strict, since running a tool the model only described would be worse
than missing a call: the calls must name tools that exist, start a line outside any code
block or inline code, and be the last thing in the reply (a model that meant to call a
tool stops there; one explaining the format keeps writing).

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

# Each body stops at the next opening tag, so unclosed tags can't make a search quadratic.
# <call> is tinyharness's own text protocol (llm/calibration.py): a tag no server-side
# tool-call parser strips out, as some do with <tool_call> even when no tools are sent.
_TOOL_CALL = re.compile(
    r"<(?:tool_call|call)>((?:(?!</?(?:tool_call|call)>).)*)(?:</(?:tool_call|call)>|$)", re.DOTALL
)
_CLOSE = re.compile(r"</(?:tool_call|call)>")
_FUNCTION = re.compile(r"<function=([\w.-]+)>((?:(?!<function=|</function>).)*)(?:</function>)?", re.DOTALL)
_PARAMETER = re.compile(
    r"<parameter=([\w.-]+)>\n?((?:(?!<parameter=|</parameter>).)*?)\n?</parameter>", re.DOTALL
)
_GEMMA = re.compile(
    r"<\|tool_call>\s*call:([\w.-]+)\s*(\{(?:(?!<tool_call\|>).)*\})\s*<tool_call\|>", re.DOTALL
)
_START = re.compile(r"^[ \t]*(<tool_call>|<call>|<function=|<\|tool_call>)", re.MULTILINE)
MAX_CHARS = 200_000  # a reply longer than this isn't scanned
_FENCE = re.compile(r"^```[\w-]*\s*\n?(.*?)\n?```$", re.DOTALL)


def recover_calls(text: str, schema_of: Callable[[str], dict[str, Any] | None]) -> tuple[str, list[ToolCall]]:
    """Tool calls written into `text`, and the text without them. `schema_of(name)` is the
    tool's JSON schema, or None if there's no such tool (then nothing is taken)."""
    start = _first_tag(text)
    if start < 0:
        return text, []
    tail = text[start:]
    if tail.startswith("<|tool_call>"):
        return _from_gemma(text, start, schema_of)
    # Nothing but calls from here on, and none of it code: otherwise it's prose about calls.
    if "```" in tail or "~~~" in tail or _CLOSE.sub("", _FUNCTION.sub("", _TOOL_CALL.sub("", tail))).strip():
        return text, []
    calls: list[ToolCall] = []
    for body in _TOOL_CALL.findall(tail) or [tail]:
        body = body.strip()
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


def _from_gemma(
    text: str, start: int, schema_of: Callable[[str], dict[str, Any] | None]
) -> tuple[str, list[ToolCall]]:
    """Gemma 4's own format: `call:name{key: "value"}`, keys bare, strings in plain quotes or
    its `<|"|>` delimiter. Taken only if every call parses and names a real tool."""
    tail = text[start:]
    if _GEMMA.sub("", tail).strip():
        return text, []
    calls: list[ToolCall] = []
    for name, body in _GEMMA.findall(tail):
        if schema_of(name) is None:
            return text, []
        try:
            arguments = json.loads(_gemma_json(body))
        except json.JSONDecodeError:
            return text, []
        if not isinstance(arguments, dict):
            return text, []
        calls.append(_call(name, arguments))
    return text[:start].rstrip(), calls


# A string in Gemma's arguments: raw between its <|"|> delimiters (quotes and newlines
# included, as in a shell command), or an ordinary JSON string.
_GEMMA_STRING = re.compile(r'<\|"\|>(.*?)<\|"\|>|"(?:[^"\\]|\\.)*"', re.DOTALL)


def _gemma_json(body: str) -> str:
    """Gemma's `{key: <|"|>raw text<|"|>, n: 3}` as JSON: raw strings encoded, bare keys
    quoted (only outside strings, so a "word:" inside a command stays as it is)."""
    out, last = [], 0
    for match in _GEMMA_STRING.finditer(body):
        out.append(_quote_keys(body[last : match.start()]))
        out.append(json.dumps(match.group(1)) if match.group(1) is not None else match.group(0))
        last = match.end()
    out.append(_quote_keys(body[last:]))
    return "".join(out)


def _quote_keys(fragment: str) -> str:
    return re.sub(r"([{,]\s*)([A-Za-z_][\w-]*)\s*:", r'\1"\2":', fragment)


def looks_like_a_call(text: str) -> bool:
    """Whether a reply ends in something shaped like a tool call (one `recover_calls`
    couldn't use: a tool that doesn't exist, arguments that don't parse), as opposed to
    prose that mentions the format."""
    start = _first_tag(text)
    return start >= 0 and not re.search(
        r"\n\s*\n\s*[A-Za-z]", _CLOSE.split(text[start:].split("<tool_call|>")[-1])[-1]
    )


def _first_tag(text: str) -> int:
    """Where the first call starts: at the start of a line, outside any code block."""
    if len(text) > MAX_CHARS:
        return -1
    for match in _START.finditer(text):
        before = text[: match.start()]
        if before.count("```") % 2 == 0 and before.count("~~~") % 2 == 0:
            return match.start(1)
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
