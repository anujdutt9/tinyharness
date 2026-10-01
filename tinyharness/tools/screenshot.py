"""Screenshot: render a local page (an HTML file, a deck, a dev server) and show the model
the picture, so it can check visual work by looking at it.

It runs Playwright's chrome-headless-shell, a plain executable, inside the same OS
sandbox as Bash: no network (so a page can't send out what it has read), writes only
to the project, scratch and temp dirs. Google Chrome.app can't start there: a macOS app
needs LaunchServices, which the sandbox blocks so nothing can open programs outside it.
Chrome's own sandbox can't run inside ours, hence `--no-sandbox`; ours confines it.
Without an OS sandbox, every call asks.

Local pages only: files, and localhost when the sandbox allows it. Remote pages would
need the network; WebFetch reads those.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import url2pathname

from pydantic import Field

from tinyharness.assets import INLINE_SCRIPT
from tinyharness.images import MAX_SIDE
from tinyharness.process import kill_group
from tinyharness.sandbox import Sandbox, SandboxPolicy
from tinyharness.security import sanitized_env
from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput, ToolOutput
from tinyharness.tools.read import load_image, resolve_readable

INSTALL = "npx playwright install chromium-headless-shell"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
_BUILD = re.compile(r"chromium_headless_shell-(\d+)$")


class ScreenshotInput(ToolInput):
    target: str = Field(
        description="An HTML file (absolute, or relative to the working directory; may end in "
        "#fragment, e.g. deck.html#3) or a http://localhost URL"
    )
    # At most MAX_SIDE: a bigger picture would only be shrunk before you see it.
    width: int = Field(default=1280, ge=100, le=MAX_SIDE, description="Viewport width in pixels")
    height: int = Field(
        default=800,
        ge=100,
        le=MAX_SIDE,
        description="Viewport height in pixels (only this much is captured)",
    )
    timeout: int = Field(default=30, ge=1, le=120, description="Seconds before giving up")


class ScreenshotTool(Tool[ScreenshotInput]):
    name = "Screenshot"
    Input = ScreenshotInput
    read_only = True  # it writes only the picture, to the scratch directory
    description = (
        "Render a local web page (an HTML file, a slide deck, a dev server on localhost) in a "
        "headless browser and look at it: the screenshot is shown to you. Use it to check "
        "visual work instead of guessing from the source. Captures the viewport (width x "
        "height); for another slide or section, use its #fragment or a taller viewport. The "
        "page has no network, so remote fonts or scripts don't load."
    )

    def __init__(self, sandbox: Sandbox | None, policy: SandboxPolicy) -> None:
        self.sandbox = sandbox
        self.policy = policy

    def describe(self, args: ScreenshotInput) -> str:
        return f"Screenshot({args.target})"

    def target_path(self, args: ScreenshotInput) -> str | None:
        return None if _is_url(args.target) else _split(args.target)[0]

    def contained(self, args: dict[str, Any]) -> bool:
        return self.sandbox is not None

    def escapes(self, args: dict[str, Any]) -> str | None:
        if self.sandbox is not None:
            return None
        return "there's no sandbox here, so the page's scripts can reach the network and your files"

    async def run(self, args: ScreenshotInput, ctx: ToolContext) -> ToolOutput:
        url = self._url(args, ctx)
        shell = find_headless_shell()
        if shell is None:
            raise ToolError(
                "Playwright's chrome-headless-shell isn't installed. Install it once with "
                f"`{INSTALL}` (a download, so run it with Bash sandbox=false), then try again."
            )
        # Scratch, else temp: both writable in the sandbox, and neither litters the project.
        out_dir = ctx.scratch.dir if ctx.scratch is not None else Path(tempfile.gettempdir())
        out = out_dir / f"screenshot-{os.urandom(3).hex()}.png"
        argv = [
            str(shell),
            f"--window-size={args.width},{args.height}",
            "--hide-scrollbars",
            "--enable-logging=stderr",  # the page's console, so its script errors reach us
            f"--screenshot={out}",
            url,
        ]
        if self.sandbox is not None:
            argv = self.sandbox.wrap([argv[0], "--no-sandbox", *argv[1:]], self.policy)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=ctx.workspace.root,
            env={**sanitized_env(), **(self.policy.env if self.sandbox else {})},
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), args.timeout)
        except TimeoutError:
            await kill_group(proc)
            raise ToolError(f"the page didn't finish loading within {args.timeout}s") from None
        except BaseException:
            await kill_group(proc)
            raise
        if not out.exists():
            detail = _last_error(stderr.decode("utf-8", "replace"))
            raise ToolError(f"the browser made no screenshot (exit code {proc.returncode}){detail}")
        try:
            image = load_image(out)
        except ValueError as e:
            raise ToolError(f"{e}; use a smaller viewport") from None
        text = f"screenshot of {args.target}, {args.width}x{args.height}, saved as {out}"
        if note := _script_errors_note(stderr.decode("utf-8", "replace")):
            text += f"\n{note}"
        if note := _remote_assets_note(url):
            text += f"\n{note}"
        return ToolOutput(text, [image])

    def _url(self, args: ScreenshotInput, ctx: ToolContext) -> str:
        if _is_url(args.target):
            parts = urlsplit(args.target)
            if parts.hostname not in _LOCAL_HOSTS:
                raise ToolError(
                    "Screenshot shows local pages only (a file or http://localhost), since the "
                    "browser has no network. Use WebFetch to read a remote page."
                )
            if self.sandbox is not None and not self.policy.allow_localhost:
                raise ToolError(
                    "the sandbox blocks localhost; the user can allow it with `allow_localhost = true` "
                    "under [sandbox] in config.toml"
                )
            return args.target
        path_arg, fragment = _split(args.target)
        path = resolve_readable(args, path_arg, ctx)
        return path.as_uri() + fragment


_REMOTE_ASSET = re.compile(
    r"<(?:script|link)\b[^>]*\b(?:src|href)\s*=\s*[\"']((?:https?:)?//[^\"']+)", re.IGNORECASE
)


# chrome's log line for a console message: `...:INFO:CONSOLE:346] "Uncaught ...", source: <url> (346)`
# (older builds write `CONSOLE(346)]`).
_CONSOLE = re.compile(r':CONSOLE[(:]\d+\)?\] "(.*)", source: \S* \((\d+)\)$', re.MULTILINE)
MAX_SCRIPT_ERRORS = 5


def _script_errors_note(stderr: str) -> str:
    """The page's uncaught script errors. A picture can't show them: an error stops the rest
    of its script, so what that script would have drawn is just missing (seen live: a CDN
    404 left `feather` undefined, and the charts after that line never ran)."""
    errors = dict.fromkeys(
        f"line {line}: {message}"
        for message, line in _CONSOLE.findall(stderr)
        if message.startswith("Uncaught")
    )
    if not errors:
        return ""
    shown = list(errors)[:MAX_SCRIPT_ERRORS]
    return (
        "(script errors on this page: " + "; ".join(shown) + ". Each one stops the rest of its "
        "script, so whatever that script draws after it is missing. Fix these first.)"
    )


def _remote_assets_note(url: str) -> str:
    """What a local page tried to load from the network (which this browser doesn't have),
    so the agent knows a blank chart is missing its library, not broken code."""
    if not url.startswith("file:"):
        return ""
    try:
        html = Path(url2pathname(urlsplit(url).path)).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    remote = list(dict.fromkeys(_REMOTE_ASSET.findall(html)))
    if not remote:
        return ""
    shown = ", ".join(remote[:3]) + (f" and {len(remote) - 3} more" if len(remote) > 3 else "")
    return (
        f"(note: this page links {shown}, which can't load here: there's no network. Pages "
        f"should work offline: for charts use the bundled Chart.js and inline it with "
        f"`python3 {INLINE_SCRIPT} <page>`.)"
    )


def find_headless_shell() -> Path | None:
    """The newest chrome-headless-shell Playwright installed, or None."""
    exe = "chrome-headless-shell.exe" if sys.platform == "win32" else "chrome-headless-shell"
    for root in _browser_dirs():
        builds = [(int(m[1]), d) for d in _children(root) if (m := _BUILD.search(d.name))]
        for _, build in sorted(builds, reverse=True):
            for platform_dir in _children(build):
                if (candidate := platform_dir / exe).is_file():
                    return candidate
    return None


def _browser_dirs() -> list[Path]:
    """Where Playwright keeps browsers: $PLAYWRIGHT_BROWSERS_PATH, else a per-OS cache."""
    if custom := os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
        return [Path(custom).expanduser()]
    home = Path.home()
    if sys.platform == "darwin":
        return [home / "Library/Caches/ms-playwright"]
    if sys.platform == "win32":
        return [Path(os.environ.get("LOCALAPPDATA", home / "AppData/Local")) / "ms-playwright"]
    return [Path(os.environ.get("XDG_CACHE_HOME", home / ".cache")) / "ms-playwright"]


def _children(path: Path) -> list[Path]:
    try:
        return [p for p in path.iterdir() if p.is_dir()]
    except OSError:
        return []


def _is_url(target: str) -> bool:
    return urlsplit(target).scheme in ("http", "https")


def _split(target: str) -> tuple[str, str]:
    """`deck.html#3` -> (`deck.html`, `#3`); a path with no fragment keeps it all."""
    path, hash_, fragment = target.partition("#")
    return path, f"#{fragment}" if hash_ else ""


def _last_error(stderr: str) -> str:
    lines = [line for line in stderr.splitlines() if line.strip()]
    return f": {lines[-1][-300:]}" if lines else ""
