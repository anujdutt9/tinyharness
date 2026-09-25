"""Types shared by every provider adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from tinyharness.agent.history import Message, ReasoningState

Effort = Literal["low", "medium", "high", "max"]
EFFORTS: tuple[Effort, ...] = ("low", "medium", "high", "max")


class LLMError(RuntimeError):
    pass


class ContextOverflowError(LLMError):
    """The request didn't fit in the model's context window."""


@dataclass
class StreamChunk:
    """One piece of a streamed reply, in the same shape for every provider."""

    text: str = ""
    tool_call_delta: dict[str, Any] | None = None  # {index, id?, function: {name?, arguments?}}
    finish_reason: str | None = None
    state: ReasoningState | None = None  # on the last chunk: data the provider needs back next turn
    reasoning: bool = False  # the model has started reasoning (shown as "Thinking")


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

    def echo(self, message: Message) -> Any:
        """The saved provider data for `message`, if it came from this same model."""
        state = message.state
        return state.data if state is not None and state.owner == self.owner else None
