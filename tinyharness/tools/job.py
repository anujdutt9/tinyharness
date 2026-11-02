"""Job: wait for a background Bash command's output, or stop it (see tinyharness/jobs.py)."""

from __future__ import annotations

import asyncio
import contextlib
import re
from typing import Literal

from pydantic import Field

from tinyharness.jobs import Job, JobScope, duration
from tinyharness.tools.base import Tool, ToolContext, ToolError, ToolInput
from tinyharness.tools.bash import sandbox_notes

WAIT_HEAD = 5_000  # new output shown from the start...
WAIT_TAIL = 20_000  # ...and from the end (where a build's errors and a server's latest lines are)


class JobInput(ToolInput):
    """Validate a request to wait for or stop a background job."""

    id: str = Field(description="The job id, e.g. 'j2', from the Bash result that started it")
    action: Literal["wait", "stop"] = Field(
        default="wait",
        description="wait: return its new output once it ends, a line matches `until`, or `timeout` "
        "passes; stop: end it and everything it started",
    )
    until: str = Field(
        default="",
        description="For wait: a regular expression (case-insensitive); return as soon as a new "
        "line matches it (e.g. 'ready|listening on')",
    )
    timeout: int = Field(
        default=60, ge=0, le=600, description="For wait: seconds to wait at most (0: just look)"
    )


class JobTool(Tool[JobInput]):
    """Wait for new background-job output or stop a background job."""

    name = "Job"
    Input = JobInput
    needs_permission = False  # it only waits for or stops a command the agent already ran
    description = (
        "Wait for a background Bash job (started with background=true, or moved to the background "
        "at its timeout) or stop it. wait returns only the output you haven't seen yet, as soon as "
        "the job ends, a line matches `until`, or `timeout` seconds pass. Use it instead of sleep "
        "loops. stop ends the job and everything it started."
    )

    def describe(self, args: JobInput) -> str:
        """Return a compact description of the requested job action."""
        if args.action == "stop":
            return f"Job(stop {args.id})"
        until = f", until {args.until!r}" if args.until else ""
        return f"Job(wait {args.id}{until}, {args.timeout}s)"

    async def run(self, args: JobInput, ctx: ToolContext) -> str:
        """Run a wait or stop action against an agent-owned job."""
        jobs = ctx.jobs
        if jobs is None:
            raise ToolError("Background jobs aren't available here, so there's no job to wait for.")
        job = jobs.get(args.id)
        if job is None:
            raise ToolError(_unknown(args.id, jobs))
        pattern = None
        if args.until:
            try:
                pattern = re.compile(args.until, re.MULTILINE | re.IGNORECASE)
            except re.error as e:
                raise ToolError(
                    f"until isn't a valid regular expression ({e}). Fix it, or leave it out."
                ) from None
        if args.action == "stop":
            job.reported = True  # the result says how it ended: no note as well
            await jobs.table.stop(job)
            jobs.mark_outside()
            return _report(job, _take_new(job), "stopped it")
        matched = await _wait(job, pattern, args.timeout)
        if not job.running:
            job.reported = True
        jobs.mark_outside()
        text = _take_new(job)
        if matched and job.running:  # a server that printed "ready" and then crashed has ended
            outcome = f"a line matched {args.until!r}; it's still running"
        elif job.running:
            waited = f"after waiting {args.timeout}s" if args.timeout else "now"
            outcome = f"still running {waited}" + (f"; nothing matched {args.until!r}" if pattern else "")
            if not text and args.timeout:
                # Seen live: a server was up, but its output sat in a buffer, so the model
                # waited a minute for a line that never came and then stopped it.
                text = (
                    "(no new output. Many programs hold their output back when it isn't going to "
                    "a terminal: if it should be ready by now, check it directly, e.g. curl a "
                    "server, rather than waiting for a line.)"
                )
        else:
            outcome = "it has ended"
        return _report(job, text, outcome)


async def _wait(job: Job, pattern: re.Pattern[str] | None, timeout: float) -> bool:
    """Wait until the job ends, its new output matches `pattern`, or `timeout` passes.

    True if it stopped waiting because of a match.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    scanned = job.seen  # output before this was searched already, up to its last whole line
    while True:
        job.output.changed.clear()  # before looking, so output arriving later wakes us
        if pattern is not None:
            data = _read(job, scanned)
            # As lines, whatever the platform: on Windows `^ready$` never matched "ready\r\n".
            if pattern.search(data.decode("utf-8", errors="replace").replace("\r\n", "\n")):
                return True
            scanned += data.rfind(b"\n") + 1  # a partial last line is searched again, whole
        remaining = deadline - loop.time()
        if not job.running or remaining <= 0:
            return False
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(job.output.changed.wait(), remaining)


def _read(job: Job, start: int) -> bytes:
    try:
        with job.log.open("rb") as f:
            f.seek(start)
            return f.read(max(job.output.size - start, 0))
    except OSError:
        return b""


def _new_bytes(job: Job) -> bytes:
    """Log bytes the agent hasn't been shown, ending at a whole UTF-8 character."""
    data = _read(job, job.seen)
    if job.running:  # the rest of a character split across writes comes later
        for back in range(1, min(4, len(data)) + 1):
            byte = data[-back]
            if byte < 0x80:
                break
            if byte >= 0xC0:  # a lead byte: is its character complete?
                need = 2 if byte < 0xE0 else 3 if byte < 0xF0 else 4
                if back < need:
                    data = data[:-back]
                break
    return data


def _take_new(job: Job) -> str:
    """Return and mark seen the job's unseen output, preserving its start and end."""
    data = _new_bytes(job)
    job.seen += len(data)
    text = data.decode("utf-8", errors="replace")
    if len(text) > WAIT_HEAD + WAIT_TAIL:
        omitted = len(text) - WAIT_HEAD - WAIT_TAIL
        text = (
            f"{text[:WAIT_HEAD]}\n… ({omitted:,} characters omitted; all of it is in {job.log}: "
            f"Read or Grep it) …\n{text[-WAIT_TAIL:]}"
        )
    if job.output.capped and job.seen >= job.output.size:
        text += f"\n(its log stopped at {job.output.size:,} bytes; later output wasn't kept)"
    return text


def _report(job: Job, text: str, outcome: str) -> str:
    body = text.rstrip() or "(no new output)"
    # The same advice Bash gives in the foreground: a server that can't bind localhost in the
    # sandbox needs network_host='localhost', not a different command (seen live: a model gave up).
    hints = sandbox_notes(text, job.command, job.code, job.sandboxed, [] if job.running else job.blocked())
    # The outcome goes last, as with Bash: that's where the CLI and the model look for it.
    return f"{body}{hints}\n(job {job.id} {job.describe()}: {job.short_command(60)} · {outcome})"


def _unknown(job_id: str, jobs: JobScope) -> str:
    mine = jobs.jobs()
    if not mine:
        return f"There's no background job {job_id!r}: you haven't started any. Bash(background=true) starts one."
    listed = ", ".join(
        f"{j.id} ({j.describe() if not j.running else 'running ' + duration(j.runtime())}: {j.short_command(40)})"
        for j in mine[-8:]
    )
    return f"There's no background job {job_id!r}. Yours: {listed}."
