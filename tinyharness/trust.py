"""Folder trust: a project's own code runs only once you've said you trust it.

Most of what a project can put in <project>/.tinyharness/ is text for the model
(skills, agents, AGENT.md) and is always loaded. Two things run code on your machine,
outside the sandbox, and are loaded only from a trusted project:

    hooks.toml   shell commands run at lifecycle events (SessionStart fires at launch)
    plugins/     bundles that can bring their own hooks

So a cloned repo can't run anything just because you started tinyharness in it. Only
projects that have hooks or plugins ask at all, once. The answer is remembered with
a fingerprint of those files, so it's asked again only if they change (say, a `git pull`
brings a new hook). The fingerprint covers every file in each plugin, and any file in the
project that a hook command names (`./scripts/check.sh`, `python tools/lint.py`), since
changing that script changes what the hook does. It can't follow what a script itself
imports or calls. Answers live in ~/.tinyharness/trusted.json, which nothing in a
project, and no sandboxed command, can write.

Fingerprinting happens at every launch, before the question, so a project can't make
it slow: past 50 MB or 5,000 entries in all, a folder can't be trusted (a partial
fingerprint would miss changes) and its hooks and plugins stay off. Entries and sizes are
counted as they're listed, before anything is read, and symlinked directories are
recorded, not followed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from tinyharness.hooks import load_hooks_file
from tinyharness.paths import home_dir, project_dir
from tinyharness.plugins import load_plugin

log = logging.getLogger(__name__)

MAX_FINGERPRINT_BYTES = 50_000_000
MAX_FINGERPRINT_ENTRIES = 5_000  # files and directories


@dataclass
class ProjectCode:
    """What in a project would run code if loaded."""

    root: Path
    hooks: list[str] = field(default_factory=list)  # "Event: command"
    plugins: list[str] = field(default_factory=list)  # plugin directory names
    fingerprint: str = ""
    too_large: str = ""  # why it can't be fingerprinted, and so can't be trusted

    @property
    def needs_trust(self) -> bool:
        return bool(self.hooks or self.plugins)

    def describe(self) -> list[str]:
        """One line per item, for the trust prompt."""
        lines = [f"hook  {h}" for h in self.hooks]
        lines += [f"plugin  {p}" for p in self.plugins]
        return lines


def scan(working_dir: Path) -> ProjectCode:
    root = Path(working_dir).expanduser().resolve()
    base = project_dir(root)
    code = ProjectCode(root)
    if base.resolve() == home_dir().expanduser().resolve():
        return code  # run from your home directory: these are your own, user-level settings
    digest = hashlib.sha256()
    commands: list[str] = []
    hooks_file = base / "hooks.toml"
    if hooks_file.is_file():
        data = hooks_file.read_bytes()
        digest.update(b"hooks\0" + data)
        try:
            entries = tomllib.loads(data.decode()).get("hook") or []
        except (tomllib.TOMLDecodeError, UnicodeDecodeError):
            entries = [{"event": "?", "command": "(unreadable hooks.toml)"}]
        code.hooks = [f"{e.get('event', '?')}: {e.get('command') or e.get('name') or '?'}" for e in entries]
        commands += [h.command for h in load_hooks_file(hooks_file) if h.command]
    plugins = base / "plugins"
    subs = sorted(p for p in plugins.iterdir() if p.is_dir()) if plugins.is_dir() else []
    code.plugins = [sub.name for sub in subs]
    for sub in subs:
        if plugin := load_plugin(sub):
            commands += [h.command for h in plugin.hooks if h.command]
    # Hooks run from the project root, so that's where the scripts they name are found.
    scripts = sorted({p for command in commands for p in _named_files(command, root)})

    # Everything that will be read is listed and totalled first; nothing is read if it's too much.
    budget = _Budget(root)
    try:
        trees = [(sub, _list_tree(sub, budget)) for sub in subs]
        for script in scripts:
            budget.add(script, script.stat().st_size)
        for sub, entries in trees:
            digest.update(b"plugin\0" + sub.name.encode() + _tree_digest(sub, entries))
        for script in scripts:
            digest.update(b"script\0" + str(script.relative_to(root)).encode() + b"\0" + _file_digest(script))
    except _TooLarge as e:
        code.too_large = str(e)
        return code
    except OSError as e:
        code.too_large = f"can't read {e.filename}: {e.strerror or e}"
        return code
    code.fingerprint = digest.hexdigest()[:24] if code.needs_trust else ""
    return code


class TrustStore:
    """Which projects you trust, and the fingerprint of their code when you did."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or home_dir() / "trusted.json"

    def trusts(self, code: ProjectCode) -> bool:
        if not code.needs_trust:
            return True
        return not code.too_large and self._load().get(str(code.root)) == code.fingerprint

    def trusted_before(self, code: ProjectCode) -> bool:
        """Trusted once, but its code has changed since."""
        return str(code.root) in self._load() and not self.trusts(code)

    def trust(self, code: ProjectCode) -> None:
        if code.too_large:
            log.warning("not trusting %s: %s", code.root, code.too_large)
            return
        data = self._load()
        data[str(code.root)] = code.fingerprint
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, indent=2, sort_keys=True))
        except OSError as e:
            log.warning("could not save folder trust to %s: %s", self.path, e)

    def _load(self) -> dict[str, str]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}


class _TooLarge(Exception):
    pass


class _Budget:
    """How much fingerprinting is left. Running out stops the listing where it is."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.bytes = 0
        self.entries = 0

    def add(self, path: Path, size: int = 0) -> None:
        self.bytes += size
        self.entries += 1
        if self.entries > MAX_FINGERPRINT_ENTRIES:
            raise _TooLarge(
                f"its plugins and hook scripts have more than {MAX_FINGERPRINT_ENTRIES:,} files "
                f"and folders to fingerprint (reached at {self._where(path)})"
            )
        if self.bytes > MAX_FINGERPRINT_BYTES:
            raise _TooLarge(
                f"its plugins and hook scripts come to more than {MAX_FINGERPRINT_BYTES // 1_000_000} MB "
                f"to fingerprint (reached at {self._where(path)})"
            )

    def _where(self, path: Path) -> Path:
        return path.relative_to(self.root) if path.is_relative_to(self.root) else path


def _list_tree(directory: Path, budget: _Budget) -> list[tuple[str, Path, bool]]:
    """(relative path, path, is a directory symlink) for each file and directory symlink,
    in a stable order. Every entry counts against the budget as it's read, so a huge
    directory stops the listing partway. Directory symlinks aren't followed: a link to /
    or a loop would never end."""
    entries = []
    skip = len(str(directory)) + 1  # the relative path is what follows "<directory>/"
    pending = [str(directory)]
    while pending:
        with os.scandir(pending.pop()) as listing:
            for entry in listing:
                if entry.is_dir(follow_symlinks=False):
                    budget.add(Path(entry.path))
                    pending.append(entry.path)
                elif entry.is_symlink() and entry.is_dir():
                    budget.add(Path(entry.path))
                    entries.append((entry.path[skip:], Path(entry.path), True))
                elif entry.is_file():  # not sockets, devices or broken links
                    budget.add(Path(entry.path), entry.stat().st_size)
                    entries.append((entry.path[skip:], Path(entry.path), False))
    return sorted(entries)


def _tree_digest(directory: Path, entries: list[tuple[str, Path, bool]]) -> bytes:
    digest = hashlib.sha256()
    for rel, path, is_link in entries:
        content = b"link\0" + os.readlink(path).encode() if is_link else _file_digest(path)
        digest.update(rel.encode() + b"\0" + content)
    return digest.digest()


def _file_digest(path: Path) -> bytes:
    """Every byte of the file, read in chunks: a size can't stand in for contents. Stops at
    the budget even if the file grew after its size was counted."""
    digest, read = hashlib.sha256(), 0
    try:
        with path.open("rb") as f:
            while chunk := f.read(1 << 20):
                read += len(chunk)
                if read > MAX_FINGERPRINT_BYTES:
                    raise _TooLarge(
                        f"{path.name} grew past {MAX_FINGERPRINT_BYTES // 1_000_000} MB while being read"
                    )
                digest.update(chunk)
    except OSError:
        return b"(unreadable)"
    return digest.digest()


def _named_files(command: str, root: Path) -> list[Path]:
    """Files inside `root` that a hook command names directly."""
    try:
        words = shlex.split(command)
    except ValueError:
        words = command.split()
    found = []
    for word in (part for w in words for part in w.split()):  # also inside `sh -c '...'`
        if word.startswith("-"):
            continue
        try:
            path = (root / word).resolve()
            if path.is_relative_to(root) and path.is_file():
                found.append(path)
        except (OSError, ValueError):
            continue
    return found
