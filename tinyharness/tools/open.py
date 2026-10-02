"""Open: show the user a file or web page in their default app (browser, PDF viewer...).

This is the one sanctioned way out of the Bash sandbox, which blocks `open` and
`xdg-open` because `open evil.command` would run a script outside it. So Open only
takes documents (not programs), and it asks every time: no allow rule, mode or "don't
ask again" approves it (only bypassPermissions or a hook). A page opened in a browser
can reach the network, which the sandbox can't, so a model that wrote one could send
out what it has read. Deny rules still apply, to the file's path or the URL's site.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import Field

from tinyharness.security import is_secret_path, sanitized_env
from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput

# Documents only. Anything that could run code when opened (.command, .app, .sh,
# .terminal, .webloc, .scpt, installers...) is refused.
_PAGES = {".html", ".htm", ".svg"}
_DOCUMENTS = {".pdf", ".md", ".txt", ".csv", ".json", ".log"}
_IMAGES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
VIEWABLE = frozenset(_PAGES | _DOCUMENTS | _IMAGES)


class OpenInput(ToolInput):
    target: str = Field(
        description="A file path (absolute, or relative to the working directory) or an http(s) URL"
    )


class OpenTool(Tool[OpenInput]):
    name = "Open"
    Input = OpenInput
    description = (
        "Open a file or web page for the user in their default app: an HTML report or chart in "
        "the browser, a PDF, an image. Use it only when the user wants to see something. Files "
        f"must be documents ({', '.join(sorted(VIEWABLE))}); the user approves each one."
    )

    def describe(self, args: OpenInput) -> str:
        return f"Open({args.target})"

    def target_path(self, args: OpenInput) -> str | None:
        # Safe to report (so path deny rules and protected paths apply): `escapes` makes
        # every call ask, so no mode approves opening a project file by itself.
        return None if _is_url(args.target) else args.target

    def target_host(self, args: OpenInput) -> str | None:
        return urlsplit(args.target).hostname if _is_url(args.target) else None

    def escapes(self, args: dict[str, Any]) -> str | None:
        target = str(args.get("target", ""))
        if _is_url(target):
            return "it opens a web page in your browser, which can reach the network outside the sandbox"
        if Path(target).suffix.lower() in _PAGES:
            return (
                "it opens a page in your browser, where its scripts can reach the network outside the sandbox"
            )
        return "it opens a file in another app, outside the sandbox"

    def risk(self, args: dict[str, Any], path: Path | None) -> str | None:
        return "it opens something in another app, outside the sandbox"

    async def run(self, args: OpenInput, ctx: ToolContext) -> str:
        target = _check(args.target, ctx)
        opener = _opener()
        if opener is None:
            if sys.platform == "win32":
                os.startfile(target)  # type: ignore[attr-defined]  # Windows only
                return f"opened {target}"
            raise ToolError("no way to open files here (xdg-open isn't installed)")
        proc = await asyncio.create_subprocess_exec(
            opener,
            target,
            env=sanitized_env(),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, err = await asyncio.wait_for(proc.communicate(), 15)
        except TimeoutError:
            return f"opening {target} (the opener is still running)"
        if proc.returncode:
            raise ToolError(f"could not open {target}: {err.decode(errors='replace').strip()}")
        return f"opened {target}"


def _check(target: str, ctx: ToolContext) -> str:
    """The URL or absolute file path to open, if it's safe to hand to another app."""
    scheme = urlsplit(target).scheme.lower()
    if scheme in ("http", "https"):
        return target
    if scheme and len(scheme) > 1:  # (a one-letter "scheme" is a Windows drive)
        raise ToolError(f"only http(s) URLs and files can be opened, not {scheme}:")
    path = ctx.workspace.resolve(target)
    if path is None or not path.is_file():
        raise ToolError(f"file not found: {target}")
    if is_secret_path(path):
        raise ToolError(f"refusing to open secret-shaped path: {path}")
    if path.suffix.lower() not in VIEWABLE:
        raise ToolError(
            f"refusing to open {path.name}: only documents can be opened ({', '.join(sorted(VIEWABLE))})"
        )
    return str(path)


def _is_url(target: str) -> bool:
    return urlsplit(target).scheme.lower() in ("http", "https")


def _opener() -> str | None:
    if sys.platform == "darwin":
        return "/usr/bin/open"
    return shutil.which("xdg-open")
