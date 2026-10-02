"""Reading and writing exactly the file that was approved; writing all or nothing.

The permission gate judges a resolved, symlink-free path. Between that check and the
write, another process (a background job the agent started in the sandbox, say) could
turn a directory on that path into a symlink and redirect the write outside the
project. So these functions never resolve the path again: they open it one directory
at a time from `/` without following symlinks, and refuse if one has appeared.

Writes are atomic: the new content goes to a temporary file in the same directory,
which is flushed to disk and then renamed over the target. A crash, a full disk or
Ctrl-C leaves the old file as it was, never half-written; directories created for the
write are removed again if it fails. The rename replaces the directory entry itself,
so a symlink put in place of the file is replaced, not followed. (A file with more
than one hard link gets a new inode, so its other names keep the old content.)
"""

from __future__ import annotations

import contextlib
import errno
import os
import secrets
import stat
from collections.abc import Iterator
from pathlib import Path

_DIR_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_SUPPORTED = os.open in os.supports_dir_fd and os.rename in os.supports_dir_fd


class PathChanged(OSError):
    """A directory on the approved path became a symlink (or vanished) before the write."""


def read_bytes(path: Path, limit: int = -1) -> bytes:
    """The contents of `path` (at most `limit` bytes), which must not have become a symlink
    (see the module docstring)."""
    if not _SUPPORTED:
        with path.open("rb") as f:
            return f.read(limit)
    with _directory(path.parent) as dfd:
        try:
            fd = os.open(path.name, os.O_RDONLY | _NOFOLLOW, dir_fd=dfd)
        except OSError as e:
            if e.errno == errno.ELOOP:
                raise PathChanged(
                    errno.ELOOP, f"{path} is now a symlink: it changed after the path was checked"
                ) from None
            raise
        with os.fdopen(fd, "rb") as f:
            return f.read(limit)


def read_text(path: Path, errors: str = "strict") -> str:
    return read_bytes(path).decode("utf-8", errors)


def write_text(path: Path, text: str, *, make_parents: bool = False) -> None:
    """Write `text` to `path` atomically, keeping an existing file's permissions."""
    data = text.encode("utf-8")
    if not _SUPPORTED:  # Windows: no directory handles; still atomic
        if make_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        return
    created: list[tuple[int, str]] = []  # (parent dir fd, name) of directories made for this write
    try:
        with _directory(path.parent, create=created if make_parents else None) as dfd:
            _replace(dfd, path.name, data)
    except BaseException:
        for parent_fd, name in reversed(created):  # leave no empty directories behind
            with contextlib.suppress(OSError):
                os.rmdir(name, dir_fd=parent_fd)
        raise
    finally:
        for parent_fd, _ in created:
            os.close(parent_fd)


def _replace(dfd: int, name: str, data: bytes) -> None:
    try:
        existing = os.stat(name, dir_fd=dfd, follow_symlinks=False)
        mode = stat.S_IMODE(existing.st_mode) if stat.S_ISREG(existing.st_mode) else None
    except FileNotFoundError:
        mode = None
    tmp = f".{name}.{secrets.token_hex(4)}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW, 0o666, dir_fd=dfd)
    try:
        try:
            with os.fdopen(fd, "wb", closefd=False) as f:
                f.write(data)
                f.flush()
                os.fsync(fd)
            if mode is not None:
                os.fchmod(fd, mode)  # new files get the umask's default, like any other
        finally:
            os.close(fd)
        os.rename(tmp, name, src_dir_fd=dfd, dst_dir_fd=dfd)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp, dir_fd=dfd)
        raise


@contextlib.contextmanager
def _directory(path: Path, create: list[tuple[int, str]] | None = None) -> Iterator[int]:
    """A handle on directory `path`, opened from `/` one level at a time without following
    symlinks. With `create`, missing directories are made (and recorded there)."""
    if not path.is_absolute():
        raise ValueError(f"not an absolute path: {path}")
    fd = os.open("/", _DIR_FLAGS)
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except FileNotFoundError:
                if create is None:
                    raise
                os.mkdir(part, 0o777, dir_fd=fd)
                create.append((os.dup(fd), part))
                child = os.open(part, _DIR_FLAGS, dir_fd=fd)
            except OSError as e:
                if e.errno in (errno.ELOOP, errno.ENOTDIR):  # a symlink, or no longer a directory
                    raise PathChanged(
                        e.errno,
                        f"{part} in {path} is now a symlink or not a directory: "
                        "it changed after the path was checked",
                    ) from None
                raise
            os.close(fd)
            fd = child
        yield fd
    finally:
        os.close(fd)
