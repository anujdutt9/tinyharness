from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from tinyharness.agent.history import Message
from tinyharness.config import Config
from tinyharness.llm.client import StreamChunk
from tinyharness.llm.providers import load_catalog
from tinyharness.tools.base import Tool, ToolContext
from tinyharness.workspace import Workspace

# ---- opt-in tests ----
# `terminal` tests drive the real CLI in an emulated terminal (pyte). They're slower and
# only matter when changing the screen layout, so they run only with `--terminal`.


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--terminal", action="store_true", help="also run the emulated-terminal tests")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "terminal: drives the CLI in an emulated terminal (run with --terminal)"
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--terminal"):
        return
    skip = pytest.mark.skip(reason="emulated-terminal test: run with --terminal")
    for item in items:
        if "terminal" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Never touch the real ~/.tinyharness or ~/.config, or pick up a developer's API keys."""
    home = tmp_path / "_home"
    monkeypatch.setenv("HOME", str(tmp_path / "_user"))
    monkeypatch.setenv("TINYHARNESS_HOME", str(home))
    for var in ("TINYHARNESS_MODEL", "TINYHARNESS_ALLOW_SECRET_READS", "XDG_CONFIG_HOME"):
        monkeypatch.delenv(var, raising=False)
    for provider in load_catalog().values():
        monkeypatch.delenv(provider.api_key_env, raising=False)
    return home


def make_ctx(root: Path, workspace: Workspace | None = None) -> ToolContext:
    return ToolContext(workspace=workspace or Workspace(root), config=Config(working_dir=root))


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return make_ctx(tmp_path)


async def call(tool: Tool[Any], args: dict[str, Any], ctx: ToolContext) -> str:
    """Validate raw arguments the way the agent loop does, then run the tool."""
    return await tool.run(tool.parse(args), ctx)


class ScriptedLLM:
    """A fake LLM that replays a fixed list of turns.

    Each turn is either a string (plain text reply) or a list of
    (tool_name, arguments_json) tool calls.
    """

    def __init__(self, turns: list[str | list[tuple[str, str]]]) -> None:
        self.turns = list(turns)
        self.requests: list[list[Message]] = []
        self.model = "fake"

    async def stream(
        self, messages: list[Message], tools: Any = None, **_: Any
    ) -> AsyncIterator[StreamChunk]:
        self.requests.append(list(messages))
        turn = self.turns.pop(0) if self.turns else "done"
        if isinstance(turn, str):
            yield StreamChunk(text=turn, finish_reason="stop")
            return
        for i, (name, arguments) in enumerate(turn):
            yield StreamChunk(
                tool_call_delta={
                    "index": i,
                    "id": f"call_{i}",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        yield StreamChunk(finish_reason="tool_calls")

    async def one_shot(self, prompt: str, **_: Any) -> str:
        return "summary of prior conversation"
