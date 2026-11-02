"""Background jobs: Bash commands that keep running while the agent does other work.

A command becomes a job when the model asks for it (`Bash(background=true)`: servers,
watchers, long builds), when a foreground command is still running at its timeout (it
moves here instead of being killed and losing its work), or when the user detaches it.
Its output goes to a log file in the session's scratch directory, which the model reads
with Read or Grep, or through the Job tool (`wait` returns only what's new).

A job belongs to the agent that started it. When it ends, that agent gets a short note
between steps. The note says how it ended and nothing it printed: command output belongs
in a tool result, not in a user-role message where it would speak with the user's voice.
A subagent's jobs stop when the subagent returns, and every job stops when the session
closes: nothing tinyharness starts outlives it.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from tinyharness.process import group_alive, kill_group, kill_groups_now

if TYPE_CHECKING:
    from tinyharness.agent.sessionlog import LogWriter

MAX_RUNNING = 8  # background jobs at once, across the session and its subagents
LOG_MAX_BYTES = 64 * 1024 * 1024  # a job's log stops growing here (the job keeps running)

JobStatus = Literal["running", "exited", "stopped", "killed"]  # killed: its session or subagent ended
JobReason = Literal["background", "timeout", "detached"]


class Output(Protocol):
    """Describe captured output available to a background job."""

    size: int  # bytes written to the log so far
    capped: bool  # the log reached LOG_MAX_BYTES and stopped growing
    changed: asyncio.Event  # set whenever more output arrives (and when the job ends)


@dataclass
class Job:
    """Represent one background command and its captured output."""

    id: str
    command: str
    owner: str
    log: Path
    proc: asyncio.subprocess.Process
    task: asyncio.Task[None]  # drains the output and waits for the process to exit
    output: Output
    reason: JobReason
    started: float = field(default_factory=time.monotonic)
    status: JobStatus = "running"
    code: int | None = None
    ended: float | None = None
    seen: int = 0
    reported: bool = False  # the agent already knows it ended (no note needed)
    sandboxed: bool = False
    # Hosts it reached or was refused through the sandbox proxy, as "lists so far"
    # callables while it runs; frozen when it ends.
    reached: Callable[[], list[str]] = list
    blocked: Callable[[], list[str]] = list
    on_end: list[Callable[[], None]] = field(default_factory=list)

    @property
    def running(self) -> bool:
        """Return whether the command is still running."""
        return self.status == "running"

    def runtime(self) -> float:
        """Return the command's elapsed runtime in seconds."""
        return (self.ended or time.monotonic()) - self.started

    def describe(self) -> str:
        """Describe the job's status and elapsed time."""
        took = duration(self.runtime())
        match self.status:
            case "running":
                return f"running for {took}"
            case "exited":
                return f"exited with code {self.code} after {took}" if self.code else f"finished after {took}"
            case "stopped":
                return f"stopped after {took}"
            case _:
                return f"killed after {took} (its agent finished)"

    def short_command(self, limit: int = 80) -> str:
        """Return the command collapsed and truncated to ``limit`` characters."""
        command = " ".join(self.command.split())
        return command if len(command) <= limit else command[: limit - 1] + "…"

    def unseen(self) -> int:
        """Return the number of captured bytes not yet shown to the agent."""
        return max(self.output.size - self.seen, 0)


def duration(seconds: float) -> str:
    """Format a duration as a compact whole-second string."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


class JobTable:
    """Every background job in a session, its subagents' included.

    Agents use it through a `JobScope` (`scope(owner)`), which sees only that agent's jobs.
    """

    def __init__(
        self,
        directory: Path,
        *,
        max_running: int = MAX_RUNNING,
        background_on_timeout: bool = True,
        transcript: LogWriter | None = None,
        first_id: int = 1,
    ) -> None:
        self.directory = directory
        self.max_running = max_running
        self.background_on_timeout = background_on_timeout
        self.transcript = transcript
        self.jobs: dict[str, Job] = {}
        self._ids = itertools.count(first_id)
        self._scopes: dict[str, JobScope] = {}
        # Foreground commands running now, each with the event that detaches it.
        self._foreground: set[asyncio.Event] = set()
        self.listeners: list[Callable[[Job], None]] = []  # told when any job ends
        self._cleanups: set[asyncio.Future[None]] = set()

    def scope(self, owner: str, outside: list[str] | None = None) -> JobScope:
        """Create the job view owned by one agent."""
        scope = JobScope(self, owner, outside if outside is not None else [])
        self._scopes[owner] = scope
        return scope

    def new_log(self) -> Path:
        """Create a log path for a command that may become a job."""
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / f"{uuid.uuid4().hex[:12]}.log"

    def next_id(self) -> str:
        """Return the next session-local job identifier."""
        return f"j{next(self._ids)}"

    def running(self) -> list[Job]:
        """Return all running jobs in the session."""
        return [job for job in self.jobs.values() if job.running]

    def has_room(self) -> bool:
        """Return whether another background job may be started."""
        return len(self.running()) < self.max_running

    # -- detaching foreground commands (the user's Ctrl+B) ------------------------

    def foreground(self) -> asyncio.Event:
        """Register a foreground command; setting the event moves it to the background."""
        event = asyncio.Event()
        self._foreground.add(event)
        return event

    def done_in_foreground(self, event: asyncio.Event) -> None:
        """Unregister a command that finished in the foreground."""
        self._foreground.discard(event)

    def has_foreground(self) -> bool:
        """Return whether any foreground command can be detached."""
        return bool(self._foreground)

    def detach_foreground(self) -> int:
        """Move every running foreground command to the background; how many there were."""
        for event in self._foreground:
            event.set()
        return len(self._foreground)

    # -- lifecycle ------------------------------------------------------------------

    def adopt(self, job: Job) -> None:
        """Add a newly detached or backgrounded job to the table."""
        self.jobs[job.id] = job
        if self.transcript is not None:
            self.transcript.write(
                "job_start",
                job=job.id,
                command=job.command,
                owner=job.owner,
                reason=job.reason,
                log=str(job.log),
            )
        job.task.add_done_callback(lambda _task: self._ended(job))

    def _ended(self, job: Job, status: JobStatus = "exited") -> None:
        """Record how `job` ended (once), release what it held, and tell its agent."""
        if job.ended is not None:
            return
        job.ended = time.monotonic()
        if job.status == "running":
            job.status = status
        job.code = job.proc.returncode if job.status == "exited" else None
        reached, blocked = list(dict.fromkeys(job.reached())), list(dict.fromkeys(job.blocked()))
        job.reached, job.blocked = (lambda: reached), (lambda: blocked)
        for release in job.on_end:
            with contextlib.suppress(Exception):
                release()
        if job.status == "exited" and group_alive(job.proc.pid):
            # Its shell is done, but something it started (`server > log &`) is still in
            # its process group: that goes with the job, as it would on stop.
            self._cleanups.add(task := asyncio.ensure_future(kill_group(job.proc)))
            task.add_done_callback(self._cleanups.discard)
        job.output.changed.set()  # wake anyone waiting on it
        if self.transcript is not None:
            self.transcript.write(
                "job_end",
                job=job.id,
                status=job.status,
                code=job.code,
                duration_ms=round(job.runtime() * 1000, 3),
            )
        if (scope := self._scopes.get(job.owner)) is not None:
            scope._ended(job)
        for listener in self.listeners:
            with contextlib.suppress(Exception):
                listener(job)

    async def stop(self, job: Job, status: JobStatus = "stopped") -> None:
        """End a running job and everything it started."""
        if not job.running:
            return
        job.status = status
        await kill_group(job.proc)
        try:
            await asyncio.wait_for(asyncio.shield(job.task), 5)
        except TimeoutError:  # something that left its group still holds the output open
            job.task.cancel()
        except Exception:
            pass
        self._ended(job, status)

    def stop_all_now(self) -> None:
        """Kill every running job, without an event loop to wait on (closing the session)."""
        running = self.running()
        for job in running:
            job.status = "killed"
            job.reported = True  # nobody is left to tell
        kill_groups_now([job.proc.pid for job in running])
        for job in running:
            self._ended(job, "killed")


class JobScope:
    """The jobs one agent started: what its tools and its loop see."""

    def __init__(self, table: JobTable, owner: str, outside: list[str]) -> None:
        self.table = table
        self.owner = owner
        self.outside = outside  # the agent's outside-content list, shared and changed in place
        self._notes: list[str] = []  # job ids that ended and haven't been announced
        self._noted_hosts: set[str] = set()

    def child(self, owner: str) -> JobScope:
        """Create a child agent's scope with the shared outside-content list."""
        return self.table.scope(owner, self.outside)

    def get(self, job_id: str) -> Job | None:
        """Return an owned job by identifier, if it exists."""
        job = self.table.jobs.get(job_id.strip())
        return job if job is not None and job.owner == self.owner else None

    def jobs(self) -> list[Job]:
        """Return every job owned by this agent."""
        return [job for job in self.table.jobs.values() if job.owner == self.owner]

    def running(self) -> list[Job]:
        """Return the running jobs owned by this agent."""
        return [job for job in self.jobs() if job.running]

    def _ended(self, job: Job) -> None:
        if not job.reported:
            self._notes.append(job.id)

    def take_notes(self) -> list[str]:
        """Return newly ended-job notes for the agent's next step.

        Downloads are recorded as outside content before the notes are returned.
        """
        self.mark_outside()
        notes = []
        for job_id in self._notes:
            job = self.table.jobs[job_id]
            if not job.reported:
                job.reported = True
                notes.append(_ended_note(job))
        self._notes = []
        return notes

    def mark_outside(self) -> None:
        """Record hosts reached by owned jobs as outside-content sources."""
        for job in self.jobs():
            for host in job.reached():
                if host not in self._noted_hosts:
                    self._noted_hosts.add(host)
                    source = f"Bash job {job.id} downloads from {host}"
                    if source not in self.outside:
                        self.outside.append(source)

    async def stop_all(self) -> None:
        """Stop this agent's running jobs (a subagent returning)."""
        running = self.running()
        for job in running:
            job.reported = True  # nobody is left to tell
        await asyncio.gather(*(self.table.stop(job, "killed") for job in running))
        self.mark_outside()  # what they downloaded may be in the answer it hands back

    def summary(self) -> str:
        """Return running-job state suitable for a compaction summary."""
        running = self.running()
        if not running:
            return ""
        lines = [f"- {job.id} {job.describe()}: {job.short_command()} (log: {job.log})" for job in running]
        return "[background jobs still running]\n" + "\n".join(lines)


def _ended_note(job: Job) -> str:
    unseen = job.unseen()
    more = (
        f" It printed {unseen:,} bytes you haven't seen: Job(id={job.id!r}, action='wait') shows them, "
        f"or Read/Grep {job.log}."
        if unseen
        else ""
    )
    return f"[tinyharness] Background job {job.id} {job.describe()}: {job.short_command()}.{more}"
