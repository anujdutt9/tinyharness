from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tinyharness.config import Config
from tinyharness.llm.providers import load_catalog
from tinyharness.testing import ScriptedLLM as ScriptedLLM  # tests import it from here
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


@pytest.fixture(autouse=True)
def instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retried requests (429/5xx) don't wait in tests."""
    monkeypatch.setattr("tinyharness.llm.client.RETRY_MAX_SECONDS", 0.0)


def make_ctx(root: Path, workspace: Workspace | None = None) -> ToolContext:
    return ToolContext(workspace=workspace or Workspace(root), config=Config(working_dir=root))


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return make_ctx(tmp_path)


async def call(tool: Tool[Any], args: dict[str, Any], ctx: ToolContext) -> str:
    """Validate raw arguments the way the agent loop does, then run the tool."""
    return await tool.run(tool.parse(args), ctx)


def png(width: int, height: int, *, noise: bool = False) -> bytes:
    """A real PNG of that size (noise makes it photo-like: it doesn't compress)."""
    import io
    import os

    from PIL import Image as PILImage

    if noise:
        picture = PILImage.frombytes("RGB", (width, height), os.urandom(width * height * 3))
    else:
        picture = PILImage.new("RGB", (width, height), (200, 30, 30))
    buffer = io.BytesIO()
    picture.save(buffer, "PNG")
    return buffer.getvalue()
