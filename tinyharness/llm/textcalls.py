"""Tool calls that arrive in the wrong place or the wrong shape, made usable.

Open models are served behind a tool-call parser (vLLM's, OpenRouter's upstreams'). When
that parser trips, the call arrives as text in the model's own format, and a harness that
only reads `tool_calls` takes it for the final answer. Seen in tinyharness's logs from
Qwen3-Coder and Qwen3.5 (a teammate's "result" was the XML of the call it meant to make).

These formats cover the open models in use:

    <tool_call>{"name": "Read", "arguments": {"path": "a.py"}}</tool_call>      JSON-style calls
    <tool_call><function=Read><parameter=path>a.py</parameter></function></tool_call>  Qwen3-Coder
    <|tool_call>call:read{path: "a.py"}<tool_call|>                             Gemma 4
    ```json [{"type": "function", "function": {"name": "Read", ...}}] ```       the API's own JSON (Gemma 3)

`recover_calls` is strict, since running a tool the model only described would be worse
than missing a call: the calls must name tools that exist, start a line outside any code
block or inline code, and be the last thing in the reply (a model that meant to call a
tool stops there; one explaining the format keeps writing). JSON with no call marker
could just as well be an example the user asked for, so it's taken only for tools whose
calls change nothing and reach nothing outside.

`repair_arguments` fixes the small, unambiguous ways arguments come back malformed:
wrapped in a code fence, JSON encoded twice, two objects run together, or empty; then, as
a last resort, broken syntax that json-repair can mend without changing any string.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Iterator
from typing import Any

import json_repair

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
# A whole reply that is a call's JSON, or the tail of one: `{"name": ..., "arguments": {...}}`
# or `"arguments": {...}}`. The arguments object is what tells it from JSON the user asked for.
MAX_BARE_CALL = 20_000  # chars: a longer reply is an answer, not a stray call
_BARE_CALL = re.compile(
    r'^(?:\{\s*"(?:name|tool|function)"\s*:.*|\{?\s*)"(?:arguments|parameters)"\s*:\s*\{.*\}$', re.DOTALL
)


def recover_calls(
    text: str,
    schema_of: Callable[[str], dict[str, Any] | None],
    from_json: Callable[[str], bool] = lambda name: False,
) -> tuple[str, list[ToolCall]]:
    """Tool calls written into `text`, and the text without them. `schema_of(name)` is the
    tool's JSON schema, or None if there's no such tool (then nothing is taken). Calls with
    no marker, only JSON, are taken only for tools `from_json(name)` allows: such JSON can
    be an example the user asked for (`Tool.taken_from_json`)."""
    start = _first_tag(text)
    if start < 0:
        return _from_openai_json(text, schema_of, from_json)
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


# The reply's last thing: a fenced block (```json, ```python) or bare JSON.
_LAST_FENCE = re.compile(r"(?:^|\n)```[\w-]*[ \t]*\n(.*?)\n?```\s*$", re.DOTALL)


def _from_openai_json(
    text: str, schema_of: Callable[[str], dict[str, Any] | None], from_json: Callable[[str], bool]
) -> tuple[str, list[ToolCall]]:
    """Calls written as JSON (`_json_call` lists the shapes), alone or in a list, as the
    reply's last thing. Seen live: Gemma 3 12B never made a real call and wrote its plans
    this way in a ```python or ```tool_code block, so every plan was lost. Taken only if
    every item is a call to a real tool."""
    if len(text) > MAX_BARE_CALL:
        return text, []
    if match := _LAST_FENCE.search(text):
        found = _json_value(match.group(1))
        before = text[: match.start()]
    else:  # bare: JSON from the start of a line to the end of the reply
        found, before = None, text
        for start in (0, *(m.end() for m in re.finditer(r"\n(?=[ \t]*[\[{])", text))):
            if (found := _json_value(text[start:])) is not None:
                before = text[:start]
                break
    if found is None:
        return text, []
    value = found
    calls: list[ToolCall] = []
    for item in value if isinstance(value, list) else [value]:
        call = _json_call(item, schema_of)
        if call is None or not from_json(call.name):
            return text, []
        calls.append(call)
    return (before.rstrip(), calls) if calls else (text, [])


def _json_value(body: str) -> Any:
    """`body` as one JSON object or list, or None."""
    if body.strip()[:1] not in ("{", "["):
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _json_call(item: Any, schema_of: Callable[[str], dict[str, Any] | None]) -> ToolCall | None:
    """One call in any of the shapes seen: the API's, `{"name", "arguments"}`, or flat, with
    the arguments beside the name (`{"name": "Plan", "plan": ...}`, or the name in "type").
    A flat one is taken only if every other key is one of the tool's own parameters."""
    if not isinstance(item, dict):
        return None
    if item.get("type") == "function" and isinstance(item.get("function"), dict):
        item = item["function"]
    name = item.get("name") if isinstance(item.get("name"), str) else item.get("type")
    if not isinstance(name, str) or (schema := schema_of(name)) is None:
        return None
    if "arguments" in item or "parameters" in item:
        arguments = item.get("arguments", item.get("parameters"))
        if isinstance(arguments, str):  # the API sends arguments as JSON text
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
    else:
        arguments = {k: v for k, v in item.items() if k not in ("name", "type")}
        if not set(arguments) <= set(schema.get("properties") or {}):
            return None
    return _call(name, arguments) if isinstance(arguments, dict) else None


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
    prose that mentions the format. That includes a reply that is nothing but a call's
    JSON with no tags: seen live (Qwen2.5 7B), ` "arguments": {"prompt": "..."}}` with
    no tool name was shown as the turn's answer, twice."""
    if len(text) <= MAX_BARE_CALL and _BARE_CALL.match(_FENCE.sub(r"\1", text.strip()).strip()):
        return True
    start = _first_tag(text)
    return start >= 0 and not re.search(
        r"\n\s*\n\s*[A-Za-z]", _CLOSE.split(text[start:].split("<tool_call|>")[-1])[-1]
    )


TAGS = ("<tool_call>", "<call>", "<function=", "<|tool_call>")


def call_start(text: str) -> int:
    """Where a call written as text starts in `text` (so far), or -1. While a reply streams,
    text from here on is held back: shown only if it turns out not to be a call."""
    return _first_tag(text)


def may_start_a_call(line: str) -> bool:
    """Whether a line still being written could turn out to be a call's opening tag."""
    line = line.lstrip(" \t")
    return bool(line) and any(tag.startswith(line) for tag in TAGS)


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
        return _repair_hybrid_parameters(text) or _repair_syntax(text)
    if isinstance(value, str):  # encoded twice: '"{\"path\": ...}"'
        return repair_arguments(value)
    if not isinstance(value, dict):
        return None
    # Something after the object: a second object run together with it, or stray text.
    # The first object is the call; anything else is dropped.
    return json.dumps(value, ensure_ascii=False) if end < len(text) or text != arguments else arguments


def _repair_syntax(text: str) -> str | None:
    """Last resort: the JSON's syntax repaired (a missing comma or quote, a trailing comma,
    single quotes) by json-repair, but only where that can't change what a tool gets. The
    text must end its object (cut-off arguments aren't completed: that would run a call the
    model never finished), and every string in the result must be in the original word for
    word, so a file's content or a command is never altered."""
    if not text.endswith("}"):
        return None
    try:
        value = json_repair.loads(text)
    except Exception:  # json-repair should return something for anything; don't trust that
        return None
    if not isinstance(value, dict) or not value or not all(_verbatim(s, text) for s in _strings(value)):
        return None
    return json.dumps(value, ensure_ascii=False)


def _strings(value: Any) -> Iterator[str]:
    """Every string in a JSON value: keys and values, at any depth."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _verbatim(string: str, text: str) -> bool:
    """Whether `string` is in `text` as written: raw, or JSON-escaped."""
    return string in text or json.dumps(string, ensure_ascii=False)[1:-1] in text


_HYBRID_PARAMETER = re.compile(
    r'^(.*)\s*<parameter\s+name="([\w.-]+)">\s*(.*?)(?:</parameter>)?\s*$', re.DOTALL
)


def _repair_hybrid_parameters(text: str) -> str | None:
    """Repair a native call that switches from JSON to an XML parameter for its last field.

    Seen live: a hosted parser returned a Plan call whose object ended after `decisions`,
    followed by `<parameter name="out_of_scope">[...]}`. Only accept it when the JSON prefix
    becomes a complete object by adding one brace and the suffix is exactly one JSON value
    followed by that object's closing brace.
    """
    match = _HYBRID_PARAMETER.fullmatch(text)
    if match is None:
        return None
    prefix, name, suffix = match.groups()
    try:
        base = json.loads(prefix.rstrip().rstrip(",") + "}")
        value, end = json.JSONDecoder().raw_decode(suffix)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(base, dict) or name in base or suffix[end:].strip() not in ("", "}"):
        return None
    base[name] = value
    return json.dumps(base, ensure_ascii=False)
