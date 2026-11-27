"""Open: present an artifact, as a link or as an offer to open it in another application.

Neither mode launches anything. Preview shows a link. External offers the artifact: the UI
shows it with a key to press, and only that keypress opens it (`launch`), so a page the
model wrote never runs outside the sandbox without the user choosing it. The offer used to
be an approval prompt, which stopped the turn until it was answered: seen live, an agent
sat waiting for the user while it could have been checking its work. Documents only.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import shutil
import sys
from pathlib import Path
from typing import Literal
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
            "Required. 'external' offers the artifact to the user to open in their browser or "
            "viewer; they open it with a keypress when they choose, and your turn carries on. "
            "'preview' only presents a clickable link."
        ),
    )


class OpenTool(Tool[OpenInput]):
    name = "Open"
    Input = OpenInput
    description = (
        "Offer a file or web page to the user. 'external' offers to open it in their browser or "
        "viewer: use it when they ask to see something, or when a result they'll want to look at "
        "is ready. It doesn't open anything or wait: the user opens it with a keypress whenever "
        "they like, so don't offer the same thing twice. 'preview' only shows a clickable link. "
        f"Files must be documents ({', '.join(sorted(VIEWABLE))})."
    )
    read_only = True

    def describe(self, args: OpenInput) -> str:
        # Seen live: "Open external: report.html" put the mode's jargon on the user's screen.
        return f"Open({args.target})" if args.mode == "external" else f"Open(link: {args.target})"

    def target_path(self, args: OpenInput) -> str | None:
        # Reported so path deny rules and protected paths apply to offers as well as links.
        return None if _is_url(args.target) else args.target

    def target_host(self, args: OpenInput) -> str | None:
        return urlsplit(args.target).hostname if _is_url(args.target) else None

    async def run(self, args: OpenInput, ctx: ToolContext) -> str | ToolOutput:
        target = _check(args.target, ctx)
        media_type = "text/html" if target.lower().endswith((".html", ".htm")) else ""
        if not media_type and not _is_url(target):
            media_type = mimetypes.guess_type(target)[0] or ""
        title = Path(target).name if not _is_url(target) else target
        offer = args.mode == "external"
        said = "offered to the user; not open yet" if offer else "linked for the user; not rendered or opened"
        return ToolOutput(said, artifacts=[ArtifactRef(target, media_type, title, offer=offer)])


async def launch(target: str) -> str:
    """Open an offered artifact with the system's opener: only ever on the user's keypress.
    The target is checked again here, since the file may have changed since it was offered."""
    if not _is_url(target):
        path = Path(target)
        if not path.is_file():
            raise ToolError(f"file not found: {target}")
        if is_secret_path(path) or path.suffix.lower() not in VIEWABLE:
            raise ToolError(f"refusing to open {path.name}")
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
