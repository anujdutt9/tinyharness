"""Open: present an artifact safely, or explicitly launch it in another application.

Preview is the normal path: it emits a typed artifact event for the UI and has no side
effect. External mode is the sanctioned way out of the Bash sandbox, which blocks
`open` and `xdg-open`; it accepts documents only and asks every time.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field

from tinyharness.security import is_secret_path, sanitized_env
from tinyharness.tools.base import ArtifactRef, Tool, ToolContext, ToolError, ToolInput, ToolOutput

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
    mode: Literal["preview", "external"] = Field(
        description=(
            "Required. Use 'external' when the user explicitly asks to open or launch the "
            "artifact in a browser or viewer; it requires approval. Use 'preview' only to "
            "present a clickable artifact link without rendering or opening it."
        ),
    )


class OpenTool(Tool[OpenInput]):
    name = "Open"
    Input = OpenInput
    description = (
        "Present or open a file or web page for the user. Choose the mode explicitly. If the "
        "user says open, launch, browser, viewer, or otherwise asks to actually see it, use "
        "'external'; that always asks before launching the host application. Use 'preview' only "
        "when a clickable link is sufficient; preview does not render or open the artifact. "
        f"Files must be documents ({', '.join(sorted(VIEWABLE))})."
    )
    read_only = True

    def describe(self, args: OpenInput) -> str:
        return f"Open({args.mode}: {args.target})"

    def target_path(self, args: OpenInput) -> str | None:
        # Safe to report (so path deny rules and protected paths apply): `escapes` makes
        # every call ask, so no mode approves opening a project file by itself.
        return None if _is_url(args.target) else args.target

    def target_host(self, args: OpenInput) -> str | None:
        return urlsplit(args.target).hostname if _is_url(args.target) else None

    def escapes(self, args: dict[str, Any]) -> str | None:
        if args.get("mode") != "external":
            return None
        target = str(args.get("target", ""))
        if _is_url(target):
            return "it opens a web page in your browser, which can reach the network outside the sandbox"
        if Path(target).suffix.lower() in _PAGES:
            return (
                "it opens a page in your browser, where its scripts can reach the network outside the sandbox"
            )
        return "it opens a file in another app, outside the sandbox"

    def risk(self, args: dict[str, Any], path: Path | None) -> str | None:
        if args.get("mode") == "external":
            return "it opens something in another app, outside the sandbox"
        return None

    def is_read_only(self, args: OpenInput) -> bool:
        return args.mode == "preview"

    async def run(self, args: OpenInput, ctx: ToolContext) -> str | ToolOutput:
        target = _check(args.target, ctx)
        if args.mode == "preview":
            media_type = "text/html" if target.lower().endswith((".html", ".htm")) else ""
            if not media_type and not _is_url(target):
                media_type = mimetypes.guess_type(target)[0] or ""
            return ToolOutput(
                f"linked {target} for the user; it was not rendered or opened",
                artifacts=[
                    ArtifactRef(target, media_type, Path(target).name if not _is_url(target) else target)
                ],
            )
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
