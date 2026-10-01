"""Types shared by every provider adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from tinyharness.agent.history import Message, ReasoningState
from tinyharness.llm.usage import Price, Usage

Effort = Literal["none", "low", "medium", "high", "max"]  # "none": no reasoning, even where it's the default
EFFORTS: tuple[Effort, ...] = ("low", "medium", "high", "max")
# For models whose reasoning is off unless asked for (ModelInfo.reasoning): the effort used
# when the user hasn't chosen one. A model with reasoning on scales it to the step, so a
# simple step still costs little; with it off, it never thinks, however hard the task.
DEFAULT_OPT_IN_EFFORT: Effort = "medium"


class LLMError(RuntimeError):
    pass


class ContextOverflowError(LLMError):
    """The request didn't fit in the model's context window."""


class RunawayOutputError(LLMError):
    """The model's reply stopped going anywhere while it still streamed (see agent/runaway.py)."""


class RepeatedFailureError(LLMError):
    """The agent kept making a tool call that failed the same way (see agent/loop.py)."""


class ImageTooLargeError(LLMError):
    """The provider rejected an image in the request as too large (pixels or bytes)."""


class ProviderConnectionError(LLMError):
    """The provider couldn't be reached, or the connection failed mid-reply."""

    def __init__(self, message: str, *, host: str, code: str, slow_reply: bool = False) -> None:
        super().__init__(message)
        self.host = host
        self.code = code  # "ECONNRESET", "EAI_NONAME", "CERTIFICATE_VERIFY_FAILED", "ReadTimeout", …
        self.slow_reply = slow_reply  # connected, but the model took longer than the timeout


FinishReason = Literal["stop", "tool_calls", "length", "filtered", "refused", "paused", "other"]

# Why a reply ended, in each API's own words -> one shared set the agent loop acts on.
# The vocabularies don't overlap, so a single table covers every adapter.
_FINISH: dict[str, FinishReason] = {
    # finished normally
    "stop": "stop",  # Chat Completions
    "end_turn": "stop",  # Anthropic
    "stop_sequence": "stop",
    "completed": "stop",  # Gemini interaction status
    # stopped to run tools
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",  # legacy Chat Completions
    "tool_use": "tool_calls",  # Anthropic
    "requires_action": "tool_calls",  # Gemini
    # ran out of output tokens: the reply (and any tool call in it) is cut off
    "length": "length",
    "max_tokens": "length",  # Anthropic
    "max_output_tokens": "length",  # OpenAI Responses (incomplete_details.reason)
    "model_context_window_exceeded": "length",  # Anthropic
    "incomplete": "length",  # Gemini, and Responses without a reason
    # the provider withheld (part of) the reply
    "content_filter": "filtered",
    "safety": "filtered",
    "refusal": "refused",  # Anthropic
    "pause_turn": "paused",  # Anthropic: a server-side tool is still working; send it back to resume
}


def normalize_finish(raw: str) -> FinishReason:
    return _FINISH.get(raw.lower(), "other")


@dataclass
class StreamChunk:
    """One piece of a streamed reply, in the same shape for every provider."""

    text: str = ""
    tool_call_delta: dict[str, Any] | None = None  # {index, id?, function: {name?, arguments?}}
    finish_reason: str | None = None  # one of FinishReason, once LLMClient has normalized it
    finish_detail: str = ""  # the provider's own word for it, for the session log
    route: str = ""  # on the last chunk: which upstream served it, when a router says (OpenRouter)
    state: ReasoningState | None = None  # on the last chunk: data the provider needs back next turn
    reasoning: bool = False  # the model has started reasoning (shown as "Thinking")
    reasoning_text: str = ""  # readable reasoning (or a summary of it), when the provider shares it
    usage: Usage | None = None  # on the last chunk: tokens used, when the provider reports them


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ModelInfo:
    """What a provider's model list says about a model. Unknown fields stay None."""

    id: str
    label: str = ""
    context: int | None = None  # input window, in tokens
    max_output: int | None = None
    adaptive_thinking: bool | None = None
    efforts: frozenset[str] | None = None  # supported reasoning efforts
    reasoning: bool | None = None  # reasoning can be switched on per request, and is off unless it is
    price: Price | None = None  # when the model list says (OpenRouter's does)

    @property
    def display(self) -> str:
        return self.label or self.id


@dataclass
class Request:
    """A provider-neutral model request; each adapter turns it into its own API's format."""

    model: str
    system: str
    messages: list[Message]  # user / assistant / tool, in order
    tools: list[ToolSpec] = field(default_factory=list)
    effort: Effort | None = None  # None: the model's default
    info: ModelInfo | None = None  # what the provider told us about the model, if anything
    owner: str = ""  # "provider:model"; saved reasoning is only sent back to its owner
    web_search: bool = False  # let the model search the web with the provider's own search tool
    show_reasoning: bool = False  # ask for readable reasoning where that's opt-in (OpenAI summaries)

    def echo(self, message: Message) -> Any:
        """The saved provider data for `message`, if it came from this same model."""
        state = message.state
        return state.data if state is not None and state.owner == self.owner else None
