"""Screenshot: rendering a local page in a sandboxed headless browser."""

import os
import pwd
from pathlib import Path

import pytest

from conftest import call, make_ctx
from tinyharness.sandbox import SandboxPolicy, Seatbelt
from tinyharness.tools.base import ToolError
from tinyharness.tools.screenshot import ScreenshotTool, find_headless_shell

SHELL = "chromium_headless_shell-{}/chrome-headless-shell-mac-arm64/chrome-headless-shell"


def _install(root: Path, build: int) -> Path:
    exe = root / SHELL.format(build)
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    return exe


def test_finds_the_newest_headless_shell(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path))
    assert find_headless_shell() is None
    _install(tmp_path, 998)
    newest = _install(tmp_path, 1243)
    (tmp_path / "chromium-1243").mkdir()  # the full browser: not what we want
    assert find_headless_shell() == newest


def tool(**policy) -> ScreenshotTool:
    return ScreenshotTool(Seatbelt(), SandboxPolicy(write_roots=(), **policy))


async def test_says_how_to_install_the_browser_when_it_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp_path / "none"))
    (tmp_path / "page.html").write_text("<h1>hi</h1>")
    with pytest.raises(ToolError, match="npx playwright install chromium-headless-shell"):
        await call(tool(), {"target": "page.html"}, make_ctx(tmp_path))


@pytest.mark.parametrize(
    ("target", "error"),
    [
        ("https://example.com", "local pages only"),
        ("http://localhost:3000", "allow_localhost"),
        ("missing.html", "file not found"),
        (".env", "secret"),
    ],
)
async def test_refuses_what_it_cannot_or_should_not_render(tmp_path, target, error):
    (tmp_path / ".env").write_text("TOKEN=x")
    with pytest.raises(ToolError, match=error):
        await call(tool(), {"target": target}, make_ctx(tmp_path))


def test_permissions_follow_the_sandbox(tmp_path):
    sandboxed, bare = tool(), ScreenshotTool(None, SandboxPolicy(write_roots=()))
    args = {"target": "deck.html#3"}
    assert sandboxed.escapes(args) is None and sandboxed.contained(args)
    assert bare.escapes(args) is not None  # no sandbox: the page could reach the network, so it asks
    assert sandboxed.target_path(sandboxed.parse(args)) == "deck.html"  # the gate checks the file
    assert sandboxed.target_path(sandboxed.parse({"target": "http://localhost:8000"})) is None


def _real_browsers() -> str:
    """Where Playwright keeps browsers for the real user (tests fake HOME)."""
    home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    return os.environ.get("PLAYWRIGHT_BROWSERS_PATH") or str(home / "Library/Caches/ms-playwright")


@pytest.mark.skipif(not Seatbelt.available(), reason="needs Seatbelt")
async def test_renders_a_page_inside_the_sandbox_and_shows_it(tmp_path, monkeypatch):
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", _real_browsers())
    if find_headless_shell() is None:
        pytest.skip("Playwright's chrome-headless-shell isn't installed")
    (tmp_path / "page.html").write_text('<h1 style="color:red">hi</h1>')
    policy = SandboxPolicy.for_project(tmp_path)
    out = await call(
        ScreenshotTool(Seatbelt(), policy),
        {"target": "page.html#top", "width": 400, "height": 300},
        make_ctx(tmp_path),
    )
    assert "screenshot of page.html#top, 400x300" in out.text
    assert out.images[0].media_type == "image/png"


@pytest.mark.skipif(not Seatbelt.available(), reason="needs Seatbelt")
async def test_reports_the_pages_script_errors(tmp_path, monkeypatch):
    # Seen live: a CDN 404 left `feather` undefined, its call threw, and every chart after
    # it was missing from a screenshot that otherwise looked fine.
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", _real_browsers())
    if find_headless_shell() is None:
        pytest.skip("Playwright's chrome-headless-shell isn't installed")
    (tmp_path / "page.html").write_text("<h1>hi</h1>\n<script>\nfeather.replace();\n</script>")
    out = await call(
        ScreenshotTool(Seatbelt(), SandboxPolicy.for_project(tmp_path)),
        {"target": "page.html", "width": 400, "height": 300},
        make_ctx(tmp_path),
    )
    assert "line 3: Uncaught ReferenceError: feather is not defined" in out.text


def test_script_errors_note_keeps_uncaught_errors_only():
    from tinyharness.tools.screenshot import _script_errors_note

    log = (
        '[1001/1:INFO:CONSOLE:7] "hello", source: file:///p.html (7)\n'
        '[1001/1:INFO:CONSOLE:9] "Uncaught TypeError: x is null", source: file:///p.html (9)\n'
        '[1001/1:INFO:CONSOLE(9)] "Uncaught TypeError: x is null", source: file:///p.html (9)\n'
    )
    note = _script_errors_note(log)
    assert note.count("line 9: Uncaught TypeError: x is null") == 1 and "hello" not in note
    assert _script_errors_note('[1:INFO:CONSOLE:7] "hello", source: file:///p.html (7)') == ""
