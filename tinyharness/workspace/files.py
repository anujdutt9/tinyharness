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
import ctypes
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
        with _fallback_directory(path.parent):
            if os.name == "nt":
                return _windows_read(path, limit)
            _refuse_link(path)
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
    if not _SUPPORTED:  # Windows: lock every directory against rename while using the path.
        fallback_created: list[Path] = []
        try:
            with _fallback_directory(path.parent, create=fallback_created if make_parents else None):
                _replace_path(path, data)
        except BaseException:
            for directory in reversed(fallback_created):
                with contextlib.suppress(OSError):
                    directory.rmdir()
            raise
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


def _replace_path(path: Path, data: bytes) -> None:
    """The fallback atomic replace, used while its parent hierarchy is guarded."""
    try:
        existing = path.lstat()
        mode = stat.S_IMODE(existing.st_mode) if stat.S_ISREG(existing.st_mode) else None
    except FileNotFoundError:
        mode = None
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(tmp, flags, 0o666)
    try:
        try:
            with os.fdopen(fd, "wb", closefd=False) as f:
                f.write(data)
                f.flush()
                os.fsync(fd)
            if mode is not None:
                os.chmod(tmp, mode)
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _refuse_link(path: Path) -> None:
    """Reject symlinks and Windows reparse points without following them."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & reparse:
        raise PathChanged(errno.ELOOP, f"{path} is now a symlink: it changed after the path was checked")


@contextlib.contextmanager
def _fallback_directory(path: Path, create: list[Path] | None = None) -> Iterator[None]:
    """Guard a path on platforms without dir-fd operations.

    Windows directory handles omit delete sharing, so none of the checked directories
    can be renamed or replaced until the operation completes. The lstat checks also make
    the behavior testable on POSIX when the fallback is forced.
    """
    if not path.is_absolute():
        raise ValueError(f"not an absolute path: {path}")
    handles: list[int] = []
    current = Path(path.anchor)
    try:
        if os.name == "nt":
            handles.append(_windows_handle(current, directory=True))
        for part in path.parts[1:]:
            current /= part
            try:
                _refuse_link(current)
                exists = current.exists()
            except PathChanged as error:
                raise PathChanged(
                    error.errno,
                    f"{part} in {path} is now a symlink or not a directory: "
                    "it changed after the path was checked",
                ) from None
            if not exists:
                if create is None:
                    raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(current))
                current.mkdir()
                create.append(current)
            try:
                _refuse_link(current)
            except PathChanged as error:
                raise PathChanged(
                    error.errno,
                    f"{part} in {path} is now a symlink or not a directory: "
                    "it changed after the path was checked",
                ) from None
            if not current.is_dir():
                raise PathChanged(
                    errno.ENOTDIR,
                    f"{part} in {path} is now a symlink or not a directory: "
                    "it changed after the path was checked",
                )
            if os.name == "nt":
                handles.append(_windows_handle(current, directory=True))
        yield
    finally:
        for handle in reversed(handles):
            _windows_close(handle)


def _windows_handle(path: Path, *, directory: bool) -> int:
    """Open one Windows object without following a reparse point or allowing replacement."""
    from ctypes import wintypes

    create_file = ctypes.windll.kernel32.CreateFileW  # type: ignore[attr-defined]
    create_file.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    generic_read = 0x80000000
    share_read_write = 0x00000001 | 0x00000002  # deliberately omit FILE_SHARE_DELETE
    open_existing = 3
    backup_semantics = 0x02000000 if directory else 0
    open_reparse_point = 0x00200000
    handle = create_file(
        str(path),
        0 if directory else generic_read,
        share_read_write,
        None,
        open_existing,
        backup_semantics | open_reparse_point,
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle in (None, invalid):
        raise ctypes.WinError()  # type: ignore[attr-defined]
    value = int(handle)
    if _windows_reparse(value):
        _windows_close(value)
        raise PathChanged(errno.ELOOP, f"{path} is now a reparse point: it changed after approval")
    return value


def _windows_reparse(handle: int) -> bool:
    from ctypes import wintypes

    class FileInformation(ctypes.Structure):
        _fields_ = [
            ("attributes", wintypes.DWORD),
            ("creation", wintypes.FILETIME),
            ("access", wintypes.FILETIME),
            ("write", wintypes.FILETIME),
            ("volume", wintypes.DWORD),
            ("size_high", wintypes.DWORD),
            ("size_low", wintypes.DWORD),
            ("links", wintypes.DWORD),
            ("index_high", wintypes.DWORD),
            ("index_low", wintypes.DWORD),
        ]

    info = FileInformation()
    get_info = ctypes.windll.kernel32.GetFileInformationByHandle  # type: ignore[attr-defined]
    get_info.argtypes = (wintypes.HANDLE, ctypes.POINTER(FileInformation))
    get_info.restype = wintypes.BOOL
    if not get_info(wintypes.HANDLE(handle), ctypes.byref(info)):
        raise ctypes.WinError()  # type: ignore[attr-defined]
    return bool(info.attributes & 0x400)


def _windows_close(handle: int) -> None:
    from ctypes import wintypes

    close = ctypes.windll.kernel32.CloseHandle  # type: ignore[attr-defined]
    close.argtypes = (wintypes.HANDLE,)
    close.restype = wintypes.BOOL
    close(wintypes.HANDLE(handle))


def _windows_read(path: Path, limit: int) -> bytes:
    """Read the already-approved Windows file by its no-follow handle."""
    import msvcrt

    handle = _windows_handle(path, directory=False)
    try:
        fd = msvcrt.open_osfhandle(  # type: ignore[attr-defined]
            handle, os.O_RDONLY | getattr(os, "O_BINARY", 0)
        )
    except BaseException:
        _windows_close(handle)
        raise
    with os.fdopen(fd, "rb") as f:  # the fd owns and closes the Windows handle
        return f.read(limit)


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
