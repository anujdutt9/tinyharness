"""Network failures in plain words.

httpx reports a failed connection as `ConnectError`, often with an empty or generic
message ("All connection attempts failed"); the reason (refused, reset, DNS, TLS) is a
few exceptions down its chain. `describe` finds it, so an error reads

    can't connect to openrouter.ai: connection refused (ECONNREFUSED)

with a code to search for or report: the OS's errno name, the DNS error, the TLS
reason, or when there's nothing underneath, the kind of httpx error.
"""

from __future__ import annotations

import errno
import os
import socket
import ssl
from dataclasses import dataclass

import httpx

_EAI = {getattr(socket, name): name for name in dir(socket) if name.startswith("EAI_")}

# What was happening when it failed, by httpx's error type (subclasses before bases).
_PHASES: tuple[tuple[type[httpx.TransportError], str], ...] = (
    (httpx.ConnectTimeout, "can't connect to {host}"),
    (httpx.ReadTimeout, "{host} stopped responding"),
    (httpx.WriteTimeout, "couldn't send the request to {host}"),
    (httpx.PoolTimeout, "no free connection to {host}"),
    (httpx.ConnectError, "can't connect to {host}"),
    (httpx.ReadError, "the connection to {host} dropped mid-response"),
    (httpx.RemoteProtocolError, "the connection to {host} dropped mid-response"),
    (httpx.WriteError, "couldn't send the request to {host}"),
    (httpx.ProxyError, "the proxy couldn't reach {host}"),
)


@dataclass(frozen=True)
class NetworkFailure:
    message: str  # "can't connect to openrouter.ai: connection refused (ECONNREFUSED)"
    host: str
    code: str  # "ECONNREFUSED", "EAI_NONAME", "CERTIFICATE_VERIFY_FAILED", "ConnectTimeout", …
    slow_reply: bool  # the server connected but took too long to answer


def describe(error: httpx.TransportError) -> NetworkFailure:
    host = _host(error)
    phase = next((p for kind, p in _PHASES if isinstance(error, kind)), "network error talking to {host}")
    reason, code = _reason(error)
    if isinstance(error, httpx.TimeoutException):
        seconds = _timeout(error)
        reason = f"timed out after {seconds:g}s" if seconds else "timed out"
    message = f"{phase.format(host=host)}: {reason} ({code})"
    return NetworkFailure(message, host, code, slow_reply=isinstance(error, httpx.ReadTimeout))


def _reason(error: BaseException) -> tuple[str, str]:
    """The most specific cause in the chain, as (reason, code)."""
    for cause in reversed(_chain(error)):  # the deepest cause is the most specific
        if isinstance(cause, ssl.SSLCertVerificationError):
            return (
                f"TLS certificate check failed: {str(cause.verify_message).rstrip('.')}",
                cause.reason or "SSL",
            )
        if isinstance(cause, ssl.SSLError):
            return f"TLS error: {cause.reason or cause}", cause.reason or "SSL"
        if isinstance(cause, socket.gaierror):
            return "DNS lookup failed", _EAI.get(cause.errno, f"EAI {cause.errno}")
        if isinstance(cause, OSError) and cause.errno:
            what = os.strerror(cause.errno)  # the OS's wording, not asyncio's "Connect call failed"
            return what[:1].lower() + what[1:], errno.errorcode.get(cause.errno, f"errno {cause.errno}")
    text = next((str(c) for c in _chain(error) if str(c)), "")
    return text or "no details from the network library", type(error).__name__


def _chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    current: BaseException | None = error
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    return chain


def _host(error: httpx.TransportError) -> str:
    try:
        request = error.request
    except RuntimeError:  # raised before a request was made
        return "the server"
    # The Host header names the site even when the connection was pinned to an IP.
    return request.headers.get("host") or request.url.host or "the server"


def _timeout(error: httpx.TimeoutException) -> float | None:
    try:
        limits = error.request.extensions.get("timeout") or {}
    except RuntimeError:
        return None
    key = {
        httpx.ConnectTimeout: "connect",
        httpx.ReadTimeout: "read",
        httpx.WriteTimeout: "write",
        httpx.PoolTimeout: "pool",
    }.get(type(error))
    value = limits.get(key) if key else None
    return float(value) if isinstance(value, int | float) else None
