"""Who is making a model call: the session, and which agent in it.

Sent as request headers to providers that opt in (`send_ids = true` in the provider's
config), e.g. a gateway in front of your own models that admits, queues and routes per
session and per agent, or just attributes cost. Never sent otherwise.

    X-Session-ID       one per session (a CLI session, or an SDK Session)
    X-Agent-ID         this agent: "<session>-lead", or "<session>-<agent>-<n>" for a subagent
    X-Parent-Agent-ID  the agent that started this one (subagents only)
    X-Agent-Role       lead | subagent
    X-Agent-Name       the agent type, e.g. "explore" ("lead" for the main agent)
    X-Task             the TH_TASK environment variable, if set: a label for the session's work

A ContextVar, so parallel subagents (each in its own asyncio task) never mix theirs up.
"""

from __future__ import annotations

import itertools
import os
from contextvars import ContextVar
from dataclasses import dataclass

_counter = itertools.count(1)


@dataclass(frozen=True)
class CallIdentity:
    session: str
    agent: str
    parent: str = ""
    role: str = "lead"
    name: str = "lead"
    task: str = ""

    def headers(self) -> dict[str, str]:
        h = {
            "X-Session-ID": self.session,
            "X-Agent-ID": self.agent,
            "X-Agent-Role": self.role,
            "X-Agent-Name": self.name,
        }
        if self.parent:
            h["X-Parent-Agent-ID"] = self.parent
        if self.task:
            h["X-Task"] = self.task
        return h


CURRENT: ContextVar[CallIdentity | None] = ContextVar("tinyharness_call_identity", default=None)


def for_session(session_id: str) -> CallIdentity:
    return CallIdentity(
        session=session_id, agent=f"{session_id}-lead", task=os.environ.get("TH_TASK", "").strip()
    )


def child(parent: CallIdentity, name: str) -> CallIdentity:
    """A subagent started by `parent`."""
    return CallIdentity(
        session=parent.session,
        agent=f"{parent.session}-{name}-{next(_counter)}",
        parent=parent.agent,
        role="subagent",
        name=name,
        task=parent.task,
    )
