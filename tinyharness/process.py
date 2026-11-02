"""Stopping a command together with everything it started.

Bash commands and shell hooks run in their own process group (`start_new_session`),
so a timeout or Ctrl-C can signal all of it, not just the shell: a child that ignores
SIGTERM would otherwise outlive the command and keep writing files. A process that
detaches itself on purpose (`setsid`, a daemon) leaves the group, and nothing short
of a container can follow it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal

KILL_GRACE = 2.0  # seconds from SIGTERM to SIGKILL
_POLL = 0.05


async def kill_group(proc: asyncio.subprocess.Process, grace: float | None = None) -> None:
    """SIGTERM the process's group, wait up to `grace` for every member to exit, then
    SIGKILL what's left. Returns once the process itself has been reaped."""
    if os.name == "nt":
        # Shell hooks can start children before they time out. taskkill is the native
        # process-tree operation; killing only the shell leaves those children alive.
        killer = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(proc.pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
        with contextlib.suppress(ProcessLookupError):
            await proc.wait()
        return
    if not hasattr(os, "killpg"):
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        await proc.wait()
        return
    _signal(proc.pid, signal.SIGTERM)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + (KILL_GRACE if grace is None else grace)
    # The leader exiting isn't enough: its children may still be running.
    while loop.time() < deadline:
        if proc.returncode is not None and not _group_alive(proc.pid):
            return
        await asyncio.sleep(_POLL)
    _signal(proc.pid, signal.SIGKILL)
    await proc.wait()
    # SIGKILL can't be ignored, but members whose parent already exited are reaped by
    # init a moment later; wait for that, so "killed" is true when we return.
    deadline = loop.time() + 1.0
    while _group_alive(proc.pid) and loop.time() < deadline:
        await asyncio.sleep(_POLL / 5)


def _signal(group: int, sig: signal.Signals) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(group, sig)


def group_alive(group: int) -> bool:
    """Whether any process is left in `group` (always False where there are no groups)."""
    return os.name != "nt" and hasattr(os, "killpg") and _group_alive(group)


def _group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)  # signal 0: only checks that someone is there
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_groups_now(pids: list[int], grace: float = 1.0) -> None:
    """`kill_group` without an event loop, for several processes at once: closing a
    session is synchronous, and its background jobs must not outlive it."""
    if not pids:
        return
    if os.name == "nt":
        import subprocess

        for pid in pids:
            with contextlib.suppress(OSError):
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
        return
    if not hasattr(os, "killpg"):
        for pid in pids:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        return
    import time

    for pid in pids:
        _signal(pid, signal.SIGTERM)
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline and any(_group_alive(pid) for pid in pids):
        time.sleep(_POLL)
    for pid in pids:
        if _group_alive(pid):
            _signal(pid, signal.SIGKILL)
