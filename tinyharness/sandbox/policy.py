"""What a sandboxed command may do, independent of how it's enforced.

    writes   only under project, private scratch/cache and configured `write_roots`, and
             never under `protected` paths inside them
    reads    anywhere, except the contents of `deny_read` paths (credential-shaped files)
    network  none, except loopback when `allow_localhost`, and the allowlist proxy's port
             when `proxy_port` is set (sandbox/proxy.py decides which hosts)

Reads stay open on purpose: a coding agent needs your toolchain, other repos and
dotfiles. What keeps data from leaving is the network rule, not file isolation.

Each backend (sandbox/seatbelt.py, ...) translates this one policy for its platform.
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from tinyharness.security import public_path_regexes, secret_path_regexes

# Caches a user may choose to share with their own terminal ([sandbox] share_caches):
# each name sets these variables to where the tool keeps them outside tinyharness (the
# user's own setting when there is one). Sharing saves downloading and compiling twice,
# but a sandboxed command can then change files the tool loads later, outside the sandbox.
SHAREABLE_CACHES: dict[str, dict[str, str]] = {
    "uv": {"UV_CACHE_DIR": "~/.cache/uv", "UV_PYTHON_INSTALL_DIR": "~/.local/share/uv/python"},
    "pip": {"PIP_CACHE_DIR": "~/Library/Caches/pip" if sys.platform == "darwin" else "~/.cache/pip"},
    "npm": {"npm_config_cache": "~/.npm"},
    "huggingface": {"HF_HOME": "~/.cache/huggingface"},
    "torch": {"TORCH_HOME": "~/.cache/torch"},
}

# Inside a project, these would let a sandboxed command get code run *outside* the
# sandbox later: tinyharness hooks/plugins/skills, git hooks or config (core.hooksPath,
# core.fsmonitor), and the files other tools run from when you open or enter the project:
# editor tasks and settings, direnv, MCP servers, other agents' commands and hooks. The
# list follows Anthropic's Sandbox Runtime's mandatory denies, plus .envrc.
PROTECTED_IN_PROJECT = (
    ".tinyharness",
    ".git/hooks",
    ".git/config",
    ".gitmodules",
    ".vscode",
    ".idea",
    ".envrc",
    ".mcp.json",
    ".claude/settings.json",
    ".claude/settings.local.json",
    ".claude/commands",
    ".claude/agents",
)


@dataclass(frozen=True)
class SandboxPolicy:
    write_roots: tuple[Path, ...]
    project_roots: tuple[Path, ...] = ()  # roots whose credential-shaped files stay unreadable
    protected: tuple[Path, ...] = ()
    deny_read: tuple[str, ...] = ()  # regexes over absolute, symlink-resolved paths
    allow_read: tuple[str, ...] = ()  # exceptions to deny_read (.env.example)
    allow_localhost: bool = False
    allowed_hosts: tuple[str, ...] = ()  # exact names / *.suffixes the sandbox may reach
    proxy_port: int | None = None  # the allowlist proxy (sandbox/proxy.py), the one way out
    cache_dir: Path | None = None  # package managers cache here instead of the user's own cache
    env: dict[str, str] = field(default_factory=dict)  # extra environment for the command
    shared_caches: tuple[str, ...] = ()  # SHAREABLE_CACHES names writing the user's own cache

    @classmethod
    def for_project(
        cls,
        root: Path,
        *,
        scratch: Path | None = None,
        cache_dir: Path | None = None,
        extra_write: Iterable[Path] = (),
        allow_localhost: bool = False,
        allowed_hosts: Iterable[str] = (),
        shared_caches: Iterable[str] = (),
    ) -> SandboxPolicy:
        """Build a project policy with private writes and explicit network destinations."""
        projects = [_real(root), *(_real(p) for p in extra_write)]
        roots = list(projects)
        if scratch is not None:
            scratch = _real(scratch)
            roots.append(scratch)
        env: dict[str, str] = {}
        if cache_dir is not None:
            cache_dir = _real(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            roots.append(cache_dir)
            env = _cache_env(cache_dir)
        private_tmp: Path | None = None
        if scratch is not None:
            private_tmp = scratch / "tmp"
            private_tmp.mkdir(parents=True, exist_ok=True)
            env.update({"TMPDIR": str(private_tmp), "TMP": str(private_tmp), "TEMP": str(private_tmp)})
        roots.extend(_framework_scratch())  # cached: the answer is fixed for this user
        shared = tuple(dict.fromkeys(shared_caches))
        for name in shared:
            for variable, default in SHAREABLE_CACHES[name].items():
                path = _real(Path(os.environ.get(variable) or default))
                env[variable] = str(path)
                roots.append(path)
        return cls(
            write_roots=tuple(dict.fromkeys(roots)),
            project_roots=tuple(projects),
            protected=tuple(p / rel for p in projects for rel in PROTECTED_IN_PROJECT),
            deny_read=secret_path_regexes(),
            allow_read=(*public_path_regexes(), *_test_fixtures(private_tmp)),
            allow_localhost=allow_localhost,
            allowed_hosts=tuple(dict.fromkeys(h.lower().rstrip(".") for h in allowed_hosts)),
            cache_dir=cache_dir,
            env=env,
            shared_caches=shared,
        )

    def reopened(self) -> tuple[Path, ...]:
        """Write roots that sit inside a protected path (e.g. an extra write dir under .git)."""
        return tuple(r for r in self.write_roots if any(r.is_relative_to(p) for p in self.protected))


def _real(path: Path) -> Path:
    """Sandboxes match real paths (/tmp is /private/tmp on macOS)."""
    return Path(path).expanduser().resolve()


@functools.cache
def _framework_scratch() -> list[Path]:
    """Scratch folders macOS frameworks write regardless of TMPDIR, each one folder under the
    user's own temp directory (`getconf DARWIN_USER_TEMP_DIR`), never the whole of it.

    Seen live: PyTorch on Apple GPUs printed "You don't have permission to save the file
    mpsgraph-..." on every graph it built: Metal writes per-process files (named for the
    process and time) to com.apple.MetalPerformanceShadersGraph there, ignoring TMPDIR."""
    if sys.platform != "darwin":
        return []
    try:
        temp = subprocess.run(
            ["/usr/bin/getconf", "DARWIN_USER_TEMP_DIR"], capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return []
    return [_real(Path(temp)) / "com.apple.MetalPerformanceShadersGraph"] if temp else []


def _temp_dirs() -> list[Path]:
    dirs = [_real(Path(tempfile.gettempdir()))]
    if Path("/tmp").is_dir():
        dirs.append(_real(Path("/tmp")))
    return list(dict.fromkeys(dirs))


def _test_fixtures(private_tmp: Path | None = None) -> tuple[str, ...]:
    """pytest's temp root, where tests make their own fake auth.json and .env files: the
    project's tests have to read them back. Real secrets don't live there, and none can
    be copied there, since reading the original is still denied."""
    # Only the temp directory created for this sandbox is safe to reopen. A broad
    # pytest temp-root exception can also contain the project and its real .env.
    roots = [private_tmp] if private_tmp is not None else []
    return tuple(f"^{re.escape(d.as_posix())}/pytest-of-[^/]+/" for d in dict.fromkeys(roots))


def _cache_env(cache: Path) -> dict[str, str]:
    """Point common package managers at a cache the sandbox owns. Letting them write
    the user's real cache would let a command plant code that runs later, unsandboxed."""
    return {
        "XDG_CACHE_HOME": str(cache),
        "XDG_CONFIG_HOME": str(cache / "config"),
        "XDG_DATA_HOME": str(cache / "data"),
        "XDG_STATE_HOME": str(cache / "state"),
        "UV_CACHE_DIR": str(cache / "uv"),
        "PIP_CACHE_DIR": str(cache / "pip"),
        "npm_config_cache": str(cache / "npm"),
        "npm_config_store_dir": str(cache / "pnpm"),
        "YARN_CACHE_FOLDER": str(cache / "yarn"),
        "COREPACK_HOME": str(cache / "corepack"),
        "PLAYWRIGHT_BROWSERS_PATH": str(cache / "playwright"),
        "PUPPETEER_CACHE_DIR": str(cache / "puppeteer"),
        "CYPRESS_CACHE_FOLDER": str(cache / "cypress"),
        "HF_HOME": str(cache / "huggingface"),
        "TORCH_HOME": str(cache / "torch"),
    }
