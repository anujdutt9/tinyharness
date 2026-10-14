"""Bash: run a shell command in the working directory, inside the OS sandbox.

Each call is a fresh process (no shell state carries over), started in its own
process group so a timeout or Ctrl-C kills everything it spawned. The environment
has credentials stripped, and output is capped (head and tail kept) before the
loop masks any remaining secrets.

With a sandbox, the command can write only to the project and private scratch/cache dirs,
can't read credential files and reaches only the allowlist proxy's hosts, so the gate
lets it run without asking in default and auto modes. Without one, every command asks.
Containment is chosen by the harness, never by a model tool argument.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shlex
import shutil
import sys
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError, field_validator

from tinyharness.process import kill_group
from tinyharness.sandbox import Sandbox, SandboxPolicy
from tinyharness.sandbox.proxy import AllowlistProxy, host_allowed
from tinyharness.security import sanitized_env
from tinyharness.tools.base import ApprovalBoundary, Tool, ToolContext, ToolInput, note_outside

HEAD_BYTES = 15_000  # output kept from the start...
TAIL_BYTES = 15_000  # ...and from the end
# Nothing waits for a person: no pagers, colors, terminal UIs or password prompts.
# A command that would ask hangs until it times out.
NONINTERACTIVE_ENV = {
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "GH_PAGER": "cat",
    "NO_COLOR": "1",
    "TERM": "dumb",
    "GIT_TERMINAL_PROMPT": "0",
}

# Inherited settings that force color: output goes to the model, where color codes are
# noise (and Node warns on every command when FORCE_COLOR and NO_COLOR are both set).
_FORCES_COLOR = ("FORCE_COLOR", "CLICOLOR_FORCE", "PY_COLORS")


def command_env(extra: dict[str, str]) -> dict[str, str]:
    """The environment a command runs with: no credentials, nothing interactive, no color."""
    env = {**sanitized_env(), **NONINTERACTIVE_ENV, **extra}
    for name in _FORCES_COLOR:
        env.pop(name, None)
    return env


# Commands that only read, so subagents running side by side needn't take turns on them
# (Tool.writes). Deliberately short: anything else counts as writing.
_READERS = frozenset(
    {"ls", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "cut", "tr", "nl", "stat",
     "du", "df", "pwd", "which", "echo", "printf", "basename", "dirname", "realpath", "jq",
     "column", "diff", "cmp", "tree"}
)  # fmt: skip
_GIT_READS = frozenset({"status", "log", "diff", "show", "ls-files", "rev-parse", "blame"})
_FIND_WRITES = frozenset({"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprintf", "-fls"})
_SQLITE_WRITES = (".output", ".once", ".save", ".backup", ".import", ".shell", ".system", ".log", ".excel")

# Keep this deterministic: a small model must not decide whether an irreversible shell
# command is routine. These recognizable destructive operations always ask in auto mode.
_DESTRUCTIVE_COMMANDS = (
    (re.compile(r"(?:^|[;&|()\s])(?:/[^\s;&|()]+/)?(?:rm|rmdir|unlink|shred)(?=\s|$)"), "deletes files"),
    (re.compile(r"(?:^|[;&|()\s])find\s+[^\n;&|]*\s-delete(?:\s|$)"), "deletes files with find"),
    (
        re.compile(r"(?:^|[;&|()\s])git\s+[^\n;&|]*\breset\s+--hard(?:\s|$)"),
        "discards changes with git reset --hard",
    ),
    (
        re.compile(r"(?:^|[;&|()\s])git\s+[^\n;&|]*\bclean\s+-[^\s;&|]*f"),
        "deletes untracked files with git clean",
    ),
)

_SHELL_OPERATORS = frozenset(";&|()\n")
_WRAPPERS = {"command", "env", "nohup", "sudo", "time"}
_SHELL_PREFIXES = {"!", "if", "then", "elif", "else", "while", "until", "do"}
_WRAPPER_OPTIONS_WITH_VALUES = {
    "env": {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"},
    "sudo": {"-C", "-D", "-g", "-h", "-p", "-R", "-T", "-u"},
    "time": {"-f", "--format", "-o", "--output"},
}
_DOWNLOAD_AND_EXECUTE = re.compile(
    r"\b(?:curl|wget)\b[^\n]*(?:\||&&|;)\s*(?:sudo\s+)?"
    r"(?:bash|sh|zsh|python\d*(?:\.\d+)*|node|chmod\s+\+x|\./)\b"
    r"|\b(?:iwr|irm|invoke-webrequest|invoke-restmethod)\b[^\n]*\|\s*"
    r"(?:iex|invoke-expression)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _SoftwareAcquisition:
    scope: str
    kind: str
    needs_python_environment: bool = False
    python_launchers: tuple[str, ...] = ()


def _software_acquisition(command: str, cwd: Path | None = None) -> _SoftwareAcquisition | None:
    """Return one boundary covering every acquisition in the complete command."""
    found: list[_SoftwareAcquisition] = []
    if _DOWNLOAD_AND_EXECUTE.search(command):
        found.append(_SoftwareAcquisition("software:bootstrap", "downloaded executable"))
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|()\n")
        lexer.whitespace = " \t\r"  # an unquoted newline is a command boundary, not a space
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        tokens = []

    segment: list[str] = []
    segment_cwd = cwd
    for token in [*tokens, ";"]:
        if token and set(token) <= _SHELL_OPERATORS:
            if acquisition := _acquisition_in_segment(segment, segment_cwd):
                found.append(acquisition)
            segment_cwd = _directory_after(segment, segment_cwd)
            segment = []
        else:
            segment.append(token)
    if not found:
        return None

    # A remembered decision must authorize exactly the command the user saw, not only
    # its first install segment. Include nested/script scopes so changing a PEP 723
    # dependency also invalidates the earlier approval.
    material = command + "\0" + "\0".join(item.scope for item in found)
    digest = hashlib.sha256(material.encode()).hexdigest()[:16]
    kinds = tuple(dict.fromkeys(item.kind for item in found))
    kind = kinds[0] if len(kinds) == 1 else "software"
    return _SoftwareAcquisition(
        f"software:command:{digest}",
        kind,
        any(item.needs_python_environment for item in found),
        tuple(launcher for item in found for launcher in item.python_launchers),
    )


def _directory_after(tokens: list[str], cwd: Path | None) -> Path | None:
    """Track a simple shell `cd`; an expression we cannot resolve makes later paths uncertain."""
    if not tokens or Path(tokens[0]).name.lower() != "cd":
        return cwd
    values = [value for value in tokens[1:] if value != "--"]
    if len(values) != 1 or any(char in values[0] for char in "$`*"):
        return None
    path = Path(values[0]).expanduser()
    if not path.is_absolute():
        if cwd is None:
            return None
        path = cwd / path
    return path.resolve()


def _acquisition_in_segment(tokens: list[str], cwd: Path | None = None) -> _SoftwareAcquisition | None:
    if not tokens:
        return None
    index = 0
    while index < len(tokens):
        name = Path(tokens[index]).name.lower().removesuffix(".exe").removesuffix(".cmd")
        if name in _SHELL_PREFIXES:
            index += 1
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", tokens[index]):
            index += 1
            continue
        if name in _WRAPPERS:
            index += 1
            options_with_values = _WRAPPER_OPTIONS_WITH_VALUES.get(name, set())
            while index < len(tokens):
                value = tokens[index]
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", value):
                    index += 1
                elif value.partition("=")[0] in options_with_values:
                    index += 1 if "=" in value else 2
                elif value.startswith("-"):
                    index += 1
                else:
                    break
            continue
        break
    if index >= len(tokens):
        return None

    executable_token = tokens[index]
    executable = Path(executable_token).name.lower().removesuffix(".exe").removesuffix(".cmd")
    args = tokens[index + 1 :]
    if executable in {"bash", "sh", "zsh"}:
        command_index = next(
            (
                index
                for index, arg in enumerate(args)
                if arg == "-c" or (arg.startswith("-") and not arg.startswith("--") and "c" in arg[1:])
            ),
            None,
        )
        if command_index is not None and command_index + 1 < len(args):
            return _software_acquisition(args[command_index + 1], cwd)
    python_invocation = bool(re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable))
    if python_invocation:
        try:
            module = args.index("-m")
        except ValueError:
            module = -1
        if module >= 0 and module + 1 < len(args):
            executable, args = args[module + 1].lower(), args[module + 2 :]

    pip_executable = bool(re.fullmatch(r"pip(?:\d+(?:\.\d+)*)?", executable))
    positional = _positional_args("pip" if pip_executable else executable, args)
    operation = positional[0] if positional else ""
    subcommand = positional[1] if len(positional) > 1 else ""
    explicit_python_destination = any(
        arg in {"--prefix", "--root", "--system", "--target", "--user", "-t"}
        or arg.startswith(("--prefix=", "--root=", "--target="))
        for arg in args
    )
    javascript_acquisition = (
        executable in {"npx", "pnpx", "bunx"} and not {"--no-install", "--offline"} & set(args)
    ) or (
        (executable == "npm" and operation in {"add", "ci", "exec", "i", "install", "update", "upgrade"})
        or (executable == "pnpm" and operation in {"add", "dlx", "i", "install", "up", "update"})
        or (executable == "yarn" and operation in {"add", "dlx", "install", "up", "upgrade"})
        or (executable == "bun" and operation in {"add", "install", "update", "x"})
    )
    kind = ""
    python_environment = False
    uv_script_scope = _uv_script_dependency_scope(args, cwd) if executable == "uv" else None
    uv_inline_dependencies = (
        executable == "uv"
        and operation == "run"
        and any(
            arg in {"--with", "--with-editable", "--with-requirements"}
            or arg.startswith(("--with=", "--with-editable=", "--with-requirements="))
            for arg in args
        )
    )
    if pip_executable and operation in {"install", "download", "wheel"}:
        kind = "Python package"
        python_environment = operation == "install" and not explicit_python_destination
    elif executable == "uv" and (
        operation in {"add", "sync"}
        or (operation == "pip" and subcommand in {"install", "sync"})
        or (operation == "tool" and subcommand in {"install", "upgrade"})
        or uv_script_scope is not None
        or uv_inline_dependencies
    ):
        kind = "Python package" if operation != "tool" else "Python tool"
        python_environment = operation not in {"tool", "run"} and not explicit_python_destination
    elif executable == "uvx" and "--offline" not in args:
        kind = "Python tool"
    elif javascript_acquisition:
        kind = "JavaScript package"
    elif executable == "playwright" and operation == "install":
        kind = "browser runtime"
    elif executable == "cargo" and operation == "install":
        kind = "Rust package"
    elif executable == "go" and operation == "install":
        kind = "Go package"
    elif executable in {"gem", "bundle"} and operation in {"install", "update"}:
        kind = "Ruby package"
    elif executable == "composer" and operation in {"install", "require", "update"}:
        kind = "PHP package"
    elif executable == "dotnet" and operation in {"add", "restore", "tool"}:
        kind = ".NET package"
    elif executable in {
        "brew",
        "apt",
        "apt-get",
        "dnf",
        "yum",
        "pacman",
        "winget",
        "choco",
    } and operation in {
        "add",
        "install",
        "upgrade",
    }:
        kind = "system package"
    elif executable in {"hf", "huggingface-cli"} and operation == "download":
        kind = "model or dataset"
    elif executable in {"ollama", "docker", "podman"} and operation == "pull":
        kind = "model or container image"
    if not kind:
        return None

    normalized = " ".join(tokens)
    scope = uv_script_scope or f"software:{executable}:{hashlib.sha256(normalized.encode()).hexdigest()[:16]}"
    launcher = (
        (executable_token,)
        if python_environment and python_invocation and Path(executable_token).is_absolute()
        else ()
    )
    return _SoftwareAcquisition(scope, kind, python_environment, launcher)


_OPTIONS_WITH_VALUES = {
    "pip": {"--cache-dir", "--cert", "--client-cert", "--proxy", "--python", "--retries", "--timeout"},
    "uv": {"--cache-dir", "--color", "--config-file", "--directory", "--project", "--python"},
    "npm": {"--cache", "--prefix", "--registry", "--userconfig", "--workspace"},
    "pnpm": {"--config-dir", "--dir", "--global-dir", "--store-dir", "--workspace-dir"},
    "yarn": {"--cwd"},
    "bun": {"--cwd"},
}


def _positional_args(executable: str, args: list[str]) -> list[str]:
    """Command words after global options, without mistaking an option value for an operation."""
    out: list[str] = []
    values = _OPTIONS_WITH_VALUES.get(executable, set())
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            out.extend(value.lower() for value in args[index + 1 :])
            break
        name = arg.partition("=")[0]
        if name in values:
            index += 1 if "=" in arg else 2
            continue
        if arg.startswith("-"):
            index += 1
            continue
        out.append(arg.lower())
        index += 1
    return out


def _uv_script_dependency_scope(args: list[str], cwd: Path | None = None) -> str | None:
    """Stable approval scope when `uv run` may resolve a script's declared packages."""
    if "run" not in args or {"--offline", "--no-sync"} & set(args):
        return None
    run = args.index("run")
    script_cwd = cwd
    for index, arg in enumerate(args[:run]):
        directory = arg.partition("=")[2] if arg.startswith("--directory=") else None
        if arg == "--directory" and index + 1 < run:
            directory = args[index + 1]
        if directory is None:
            continue
        if any(char in directory for char in "$`*"):
            script_cwd = None
            break
        selected = Path(directory).expanduser()
        if not selected.is_absolute():
            if script_cwd is None:
                break
            selected = script_cwd / selected
        script_cwd = selected.resolve()
    for value in args[run + 1 :]:
        if not value.lower().endswith(".py"):
            continue
        try:
            path = Path(value).expanduser()
            if not path.is_absolute():
                if script_cwd is None:
                    raise OSError("the script's working directory is unknown")
                path = script_cwd / path
            header = path.read_text(errors="replace")[:65_536]
        except OSError:
            # The child shell may resolve a variable or change directory before running
            # uv. If the harness cannot inspect a Python script, fail closed: otherwise
            # `cd sub && uv run tool.py` can install PEP 723 dependencies without the
            # software-acquisition approval. The scope stays stable for this expression.
            digest = hashlib.sha256(value.encode()).hexdigest()[:16]
            return f"software:uv-script:{digest}"
        if not (
            re.search(r"(?m)^#\s*///\s*script\s*$", header)
            and re.search(r"(?m)^#\s*dependencies\s*=", header)
        ):
            return None
        fingerprint = f"{path.resolve()}\0{header}"
        digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
        return f"software:uv-script:{digest}"
    return None


def _python_install_environment(
    command: str, policy: SandboxPolicy
) -> tuple[str, dict[str, str], Path | None]:
    """Prepare an approved Python install for a writable environment, creating one if needed."""
    project = (policy.project_roots or policy.write_roots)[0].resolve()
    acquisition = _software_acquisition(command, project)
    if acquisition is None or not acquisition.needs_python_environment:
        return command, {}, None
    roots = tuple(path.resolve() for path in policy.write_roots)
    project = (policy.project_roots or policy.write_roots)[0].resolve()
    candidates = [
        *(
            Path(value).expanduser()
            for name in ("VIRTUAL_ENV", "CONDA_PREFIX")
            if (value := os.environ.get(name))
        ),
        project / ".venv",
        project / "venv",
        project / ".conda",
    ]
    environment = next(
        (path.resolve() for path in candidates if _environment_python(path) and _inside(path, roots)),
        None,
    )
    if environment is None:
        environment = next(
            (
                candidate
                for candidate in (project / ".venv", project / ".tinyharness-venv")
                if _inside(candidate, roots)
            ),
            None,
        )
    if environment is None:
        raise RuntimeError("no safe project-local path is available for a Python environment")
    python = _environment_python(environment)
    create = python is None
    scripts = environment / ("Scripts" if sys.platform == "win32" else "bin")
    selected_python = python or scripts / ("python.exe" if sys.platform == "win32" else "python")
    env = {
        "VIRTUAL_ENV": str(environment),
        "PATH": str(scripts) + os.pathsep + os.environ.get("PATH", ""),
    }
    if create:
        launcher = "python" if sys.platform == "win32" else shlex.quote(sys.executable)
        command = f"{launcher} -m venv {shlex.quote(str(environment))} && {command}"
    for launcher in acquisition.python_launchers:
        if Path(launcher).resolve() == selected_python.resolve():
            continue
        # The approval promises the selected writable environment. An absolute Python
        # ignores PATH, so bind that invocation to the environment's interpreter too.
        word = re.compile(rf"(?<!\S)(?P<quote>['\"]?){re.escape(launcher)}(?P=quote)(?=\s|$)")
        command, changed = word.subn(shlex.quote(str(selected_python)), command, count=1)
        if not changed:
            raise RuntimeError(
                f"cannot safely redirect the explicit Python interpreter {launcher!r} "
                f"to the approved environment at {environment}"
            )
    return command, env, environment


def _environment_python(environment: Path) -> Path | None:
    for relative in ("bin/python", "Scripts/python.exe", "python.exe"):
        candidate = environment / relative
        if candidate.is_file():
            return candidate
    return None


def _inside(path: Path, roots: tuple[Path, ...]) -> bool:
    resolved = path.resolve()
    return any(resolved == root or resolved.is_relative_to(root) for root in roots)


class BashInput(ToolInput):
    command: str = Field(description="The shell command to run")
    timeout: int = Field(default=120, ge=1, le=600, description="Seconds before the command is killed")
    description: str = Field(default="", description="What the command does, in 5-10 words")
    network_host: str = Field(
        default="",
        description="One exact hostname this sandboxed command needs. Use 'localhost' when a "
        "test, preview, or development server needs loopback; use other hosts after the sandbox "
        "reports that destination as blocked. Approval never removes containment.",
    )

    @field_validator("network_host")
    @classmethod
    def _exact_hostname(cls, value: str) -> str:
        host = value.lower().rstrip(".")
        if not host:
            return ""
        if (
            len(host) > 253
            or "://" in host
            or any(char in host for char in "/*:@[]")
            or any(char.isspace() for char in host)
            or not all(
                len(part) <= 63 and re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", part)
                for part in host.split(".")
            )
        ):
            raise ValueError("network_host must be one exact hostname, such as cdn.example.com")
        return host


@dataclass
class Execution:
    output: str
    code: int | None  # None: it timed out and was killed
    timed_out: bool
    sandboxed: bool
    blocked: list[str]  # hosts the sandbox proxy refused while it ran
    reached: list[str]  # hosts it downloaded from, through the proxy


class BashTool(Tool[BashInput]):
    name = "Bash"
    Input = BashInput

    def __init__(
        self,
        sandbox: Sandbox | None,
        policy: SandboxPolicy,
        shell: str | None = None,
        proxy: AllowlistProxy | None = None,
        session_hosts: set[str] | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.policy = policy
        self.shell = shell or default_shell()
        self.proxy = proxy if sandbox is not None else None
        self.session_hosts = session_hosts if session_hosts is not None else set()
        network = (
            "network only to package registries (PyPI, npm), GitHub and common CDNs, through a "
            "proxy (/sandbox lists them)"
            if self.proxy or (sandbox and getattr(sandbox, "manages_network", False) and policy.allowed_hosts)
            else "no network access"
        )
        limits = (
            "It runs in a sandbox: it can write only inside the working directory and private "
            f"scratch/cache directories, can't read credential files, and has {network}"
            + (" (and localhost)" if policy.allow_localhost else "")
            + ". API key variables are stripped, and apps (`open`, browsers like Chrome.app) can't "
            "start. When a download names a blocked destination, retry once with network_host set "
            "to that exact hostname; approval keeps the command sandboxed. Use structured file/Open "
            "tools for host actions; Bash never drops containment. Explicit software acquisition "
            "asks first; run an already-installed executable directly to reuse it without another prompt. "
            "A bare Python install uses a writable active environment or a project-local .venv."
            if sandbox
            else "It runs without a sandbox, so the user approves every command."
        )
        self.description = (
            "Run a shell command in the working directory and return its combined stdout/stderr "
            "and exit code. Use it to run tests, builds, linters, git and other CLI tools. Each call "
            "starts fresh in the working directory (use `cd dir && ...`); nothing interactive. "
            "Prefer Read/Grep/Glob/Edit over cat/grep/find/sed for files. "
            f"{limits}"
        )

    def describe(self, args: BashInput) -> str:
        command = args.command if len(args.command) <= 80 else args.command[:79] + "…"
        network = f"; network: {args.network_host}" if args.network_host else ""
        return f"Bash({command}{network})"

    def target_host(self, args: BashInput) -> str | None:
        return args.network_host or None

    def approval_boundary(self, args: BashInput) -> ApprovalBoundary | None:
        project = (self.policy.project_roots or self.policy.write_roots)[0].resolve()
        acquisition = _software_acquisition(args.command, project)
        if acquisition is None:
            return None
        if acquisition.needs_python_environment:
            _, _, environment = _python_install_environment(args.command, self.policy)
            location = f"the writable environment at {environment}"
        else:
            location = (
                "the private tinyharness cache or another permitted root" if self.sandbox else "the host"
            )
        return ApprovalBoundary(
            acquisition.scope,
            f"this command can download and execute new {acquisition.kind} code; "
            f"it will write only to {location}",
            f"allow this {acquisition.kind} acquisition for this session",
        )

    def writes(self, args: BashInput) -> bool:
        return not only_reads(args.command)

    def contained(self, args: dict[str, Any]) -> bool:
        return self.sandbox is not None

    def can_send_out(self, args: dict[str, Any]) -> bool:
        """The sandbox's allowlist is still an outbound channel for readable project data."""
        return self.sandbox is None or bool(
            self.policy.allowed_hosts or self.session_hosts or self._network_host(args)
        )

    def escapes(self, args: dict[str, Any]) -> str | None:
        if self.sandbox is None:
            return (
                "no sandbox backend is active: this command can use the network and change any of your files"
            )
        host = self._network_host(args)
        if host and not self._host_is_allowed(host):
            return (
                f"this keeps the command sandboxed but lets it connect to {host}; "
                "that destination can receive any non-secret project data the command can read"
            )
        return None

    def rememberable_escape(self, args: dict[str, Any]) -> bool:
        host = self._network_host(args)
        return bool(self.sandbox is not None and host and not self._host_is_allowed(host))

    def _host_is_allowed(self, host: str) -> bool:
        if host == "localhost":
            return self.policy.allow_localhost or host in self.session_hosts
        return host_allowed(host, (*self.policy.allowed_hosts, *self.session_hosts))

    def _network_host(self, args: BashInput | dict[str, Any]) -> str:
        if isinstance(args, dict):
            try:
                args = BashInput.model_validate(args)
            except ValidationError:
                return ""
        return args.network_host

    def risk(self, args: dict[str, Any], path: Path | None) -> str | None:
        command = str(args.get("command", ""))
        for pattern, reason in _DESTRUCTIVE_COMMANDS:
            if pattern.search(command):
                return reason
        return None if self.sandbox else "runs a shell command without a sandbox"

    async def run(self, args: BashInput, ctx: ToolContext) -> str:
        run = await self.execute(args, ctx)
        if run.reached:  # what it downloaded is outside content (a README can carry instructions)
            note_outside(ctx, f"Bash downloads from {', '.join(run.reached[:3])}")
        return self._report(run.output, run.code, run.timed_out, args, run.sandboxed, run.blocked)

    async def execute(self, args: BashInput, ctx: ToolContext) -> Execution:
        """Run the command; what it printed and how it ended, before any notes are added."""
        command, install_env, _ = _python_install_environment(args.command, self.policy)
        argv = shell_argv(self.shell, command)
        sandboxed = self.sandbox is not None
        policy, denied_before, reached_before = self.policy, 0, 0
        active_proxy = self.proxy
        temporary_proxy = False
        added_hosts = (*sorted(self.session_hosts), *((args.network_host,) if args.network_host else ()))
        if sandboxed and added_hosts:
            localhost = "localhost" in added_hosts
            remote_hosts = tuple(host for host in added_hosts if host != "localhost")
            policy = replace(
                policy,
                allow_localhost=policy.allow_localhost or localhost,
                allowed_hosts=tuple(dict.fromkeys((*policy.allowed_hosts, *remote_hosts))),
            )
            if (
                remote_hosts
                and not getattr(self.sandbox, "manages_network", False)
                and (
                    active_proxy is None
                    or any(not host_allowed(host, active_proxy.allowed) for host in remote_hosts)
                )
            ):
                active_proxy = AllowlistProxy(policy.allowed_hosts)
                temporary_proxy = True
        try:
            if sandboxed and active_proxy is not None:
                port = await active_proxy.start()
                policy = replace(policy, proxy_port=port, env={**policy.env, **active_proxy.env()})
                denied_before, reached_before = len(active_proxy.denied), len(active_proxy.reached)
            if sandboxed:
                assert self.sandbox is not None
                argv = self.sandbox.wrap(argv, policy)
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=ctx.workspace.root,
                env=command_env({**(policy.env if sandboxed else {}), **install_env}),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                # SRT owns the child job on Windows; elsewhere tinyharness owns a POSIX group.
                start_new_session=sys.platform != "win32",
            )
        except BaseException:
            if temporary_proxy and active_proxy is not None:
                active_proxy.close()
            raise
        spill = ctx.scratch.dir / f"bash-{uuid.uuid4().hex[:8]}.log" if ctx.scratch else None
        output = _Capture(spill)
        timed_out = False

        async def finish() -> None:
            assert proc.stdout is not None
            await output.drain(proc.stdout)
            await proc.wait()  # inside the timeout: it may have closed its output and kept running

        try:
            try:
                await asyncio.wait_for(finish(), args.timeout)
            except TimeoutError:
                timed_out = True
                await kill_group(proc)
            except BaseException:  # Ctrl-C: don't leave the command running
                await kill_group(proc)
                raise
            blocked = (
                list(dict.fromkeys(active_proxy.denied[denied_before:])) if sandboxed and active_proxy else []
            )
            reached = (
                list(dict.fromkeys(active_proxy.reached[reached_before:]))
                if sandboxed and active_proxy
                else []
            )
        finally:
            if temporary_proxy and active_proxy is not None:
                active_proxy.close()
        code = None if timed_out else proc.returncode
        return Execution(output.text(), code, timed_out, sandboxed, blocked, reached)

    def _report(
        self,
        text: str,
        code: int | None,
        timed_out: bool,
        args: BashInput,
        sandboxed: bool,
        blocked: list[str] | None = None,
    ) -> str:
        text = text.rstrip() or "(no output)"
        if blocked:
            text += (
                f"\n[tinyharness:network_destination_denied host={blocked[0]}] "
                "The command stayed sandboxed. If this destination is required and trusted, "
                f"retry with network_host={blocked[0]!r}; do not change the command's containment."
            )
        # Sandbox failures don't say "sandbox": a blocked write says "Operation not permitted",
        # and a macOS app (which needs LaunchServices) dies with "Abort trap: 6".
        denied = "Operation not permitted" in text or "Permission denied" in text
        if sandboxed and code and ".app/" in args.command:
            text += (
                "\n[tinyharness:host_application_blocked] This command stayed sandboxed. "
                "Use a structured host-application tool if the user needs to see the result."
            )
        elif sandboxed and denied and _looks_like_localhost_denial(text):
            text += (
                "\n[tinyharness:localhost_denied] This command stayed sandboxed. If it needs a "
                "local test, preview, or development server, retry with network_host='localhost'; "
                "do not change the command's containment."
            )
        elif sandboxed and denied:
            text += (
                "\n[tinyharness:filesystem_or_process_denied] This command stayed sandboxed. "
                "Use the working directory, $TMPDIR, or an explicitly configured writable root; "
                "do not retry outside containment."
            )
        # The outcome goes last: the CLI reads it from the final line (a failed test run
        # showed as passing when this note came after it).
        if timed_out:
            text += f"\n(timed out after {args.timeout}s; the command was killed)"
        elif code:
            text += f"\n(exit code {code})"
        return text


def _looks_like_localhost_denial(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in ("127.0.0.1", "localhost", "bind on address"))


def only_reads(command: str) -> bool:
    """Whether a command certainly changes nothing: known readers, joined only by pipes,
    with no redirection, substitution or chaining. Unsure means no."""
    if any(c in command for c in "><;&`$(){}\n"):
        return False
    for segment in command.split("|"):
        try:
            words = shlex.split(segment)
        except ValueError:
            return False
        if not words:
            return False
        name, rest = words[0], words[1:]
        if name in _READERS:
            continue
        if name == "find" and not _FIND_WRITES.intersection(rest):
            continue
        if name == "git" and rest[:1] and rest[0] in _GIT_READS:
            continue
        if name == "sqlite3" and "-readonly" in rest and not any(w in segment for w in _SQLITE_WRITES):
            continue
        return False
    return True


def default_shell() -> str:
    return shutil.which("bash") or shutil.which("sh") or "/bin/sh"


def shell_argv(shell: str, command: str) -> list[str]:
    """Run command chains fail-fast, including failures hidden in pipelines."""
    name = Path(shell).name.lower()
    if name in {"bash", "zsh"} or name.startswith("bash"):
        return [shell, "-e", "-o", "pipefail", "-c", command]
    return [shell, "-e", "-c", command]


class _Capture:
    """Keeps the first HEAD_BYTES and last TAIL_BYTES of a stream, whatever its size. Once
    output is dropped, all of it also goes to `spill` (a file in the session's scratch
    directory), so the model can read the middle instead of running a slow command again."""

    def __init__(self, spill: Path | None = None) -> None:
        self.head = bytearray()
        self.tail = bytearray()
        self.dropped = 0
        self.spill = spill
        self._file: Any = None

    async def drain(self, stream: asyncio.StreamReader) -> None:
        try:
            while chunk := await stream.read(65536):
                if self._file is not None:
                    self._file.write(chunk)
                room = HEAD_BYTES - len(self.head)
                self.head += chunk[:room]
                self.tail += chunk[room:] if room > 0 else chunk
                if len(self.tail) > TAIL_BYTES:
                    if self._file is None and self.spill is not None:
                        try:
                            self._file = self.spill.open("wb")
                            self._file.write(self.head + self.tail)  # everything so far
                        except OSError:
                            self.spill = None
                    self.dropped += len(self.tail) - TAIL_BYTES
                    del self.tail[:-TAIL_BYTES]
        finally:
            if self._file is not None:
                self._file.close()

    def text(self) -> str:
        head = self.head.decode("utf-8", errors="replace")
        tail = self.tail.decode("utf-8", errors="replace")
        if self.dropped:
            where = (
                f"; all of it is in {self.spill}: Read or Grep that file instead of running the command again"
                if self._file is not None and self.spill is not None
                else ""
            )
            return f"{head}\n… ({self.dropped:,} bytes of output omitted{where}) …\n{tail}"
        return head + tail
