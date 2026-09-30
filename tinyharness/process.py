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
    if not hasattr(os, "killpg"):  # Windows: no process groups
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


def _group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)  # signal 0: only checks that someone is there
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
