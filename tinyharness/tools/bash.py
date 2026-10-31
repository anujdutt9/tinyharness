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
import fnmatch
import glob
import hashlib
import itertools
import os
import re
import shlex
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError, field_validator

from tinyharness.process import kill_group
from tinyharness.sandbox import Sandbox, SandboxPolicy
from tinyharness.sandbox.policy import PROTECTED_IN_PROJECT
from tinyharness.sandbox.proxy import PACKAGE_REGISTRY_HOSTS, AllowlistProxy, host_allowed, service_hosts
from tinyharness.security import is_secret_path, sanitized_env
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
_FIND_WRITES = frozenset(
    {"-exec", "-execdir", "-ok", "-okdir", "-delete", "-fprint", "-fprint0", "-fprintf", "-fls"}
)
# Options that make a reader run a program or write a file: `rg --pre=sh`, `tree -o FILE`,
# `git diff --output=FILE`. git and GNU tools take any unambiguous prefix of a long option
# (`--outp=FILE`), so a prefix counts too.
_UNSAFE_OPTIONS = {
    "rg": ("--pre", "--hostname-bin"),
    "tree": ("-o",),
    "git": ("--output", "--ext-diff", "--textconv"),
    "sort": ("--compress-program",),
}

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

# With the sandbox off, these ask the user even in auto mode: the sandbox no longer stops
# them, and a review model must not be the only thing between them and your machine.
_AS_ANOTHER_USER = re.compile(r"(?:^|[;&|()\s])(?:sudo|doas|su)(?=\s|$)")
_OTHER_USER = frozenset({"sudo", "doas", "su", "pkexec", "run0"})  # also as /usr/bin/sudo, "sudo"
_CREDENTIAL_DIRS = re.compile(r"(^|/)\.(ssh|aws|gnupg|kube|azure|docker)(/|$)|(^|/)\.config/(gh|gcloud)(/|$)")
_WORDS = re.compile(r"[^\s;&|()<>'\"`]+")
# What makes the shell turn one word into other paths (`cat ~/.ss?/id_*`), so the
# credential check looks at what the word expands to, not only how it's spelled.
_EXPANDS = re.compile(r"[*?\[{$~]")
_BRACE = re.compile(r"\{([^{}]*,[^{}]*)\}")
# The directories _CREDENTIAL_DIRS names, for patterns that could spell one.
_CREDENTIAL_NAMES = (".ssh", ".aws", ".gnupg", ".kube", ".azure", ".docker", "gh", "gcloud")
_MAX_EXPANSIONS = 200
# Programs that fetch from the network: without the proxy's record, their output counts
# as outside content.
_MAY_DOWNLOAD = re.compile(
    r"https?://|(?:^|[;&|()\s])(?:curl|wget|gh|git\s+(?:clone|fetch|pull)|pip3?\s+install|"
    r"uv\s+(?:pip|add|sync|tool|run)|npm|npx|pnpm|yarn|bun|brew|ssh|scp|rsync)(?=\s|$)"
)
_SHELL_OPERATORS = frozenset(";&|()\n")
_WRAPPERS = {"builtin", "command", "env", "exec", "nohup", "sudo", "time"}
_SHELL_PREFIXES = {"!", "if", "then", "elif", "else", "while", "until", "do"}
_WRAPPER_OPTIONS_WITH_VALUES = {
    "env": {"-u", "--unset", "-C", "--chdir", "-S", "--split-string"},
    "exec": {"-a"},
    "sudo": {"-C", "-D", "-g", "-h", "-p", "-R", "-T", "-u"},
    "time": {"-f", "--format", "-o", "--output"},
}
# The command makes its own environment first (`uv venv && uv pip install -e .`):
# creating one ahead of it made `uv venv` fail with "already exists".
_MAKES_VENV = re.compile(r"(?:^|[;&|()\s])(?:uv\s+venv|\S*python[\d.]*\s+-m\s+venv|virtualenv)(?=\s|$)")
_DOWNLOAD_AND_EXECUTE = re.compile(
    r"\b(?:curl|wget)\b[^\n]*(?:\||&&|;)\s*(?:sudo\s+)?"
    r"(?:bash|sh|zsh|python\d*(?:\.\d+)*(?!\s+(?:-\S+\s+)*-c(?:\s|$))|node|chmod\s+\+x|\./)\b"
    r"|\b(?:iwr|irm|invoke-webrequest|invoke-restmethod)\b[^\n]*\|\s*"
    r"(?:iex|invoke-expression)\b",
    re.IGNORECASE,
)
_PYTHON_PIPE_INLINE = re.compile(
    r"\b(?:curl|wget)\b[^\n]*\|\s*(?:sudo\s+)?python\d*(?:\.\d+)*\s+"
    r"(?:-\S+\s+)*-c\s+(?P<quote>['\"])(?P<code>.*?)(?P=quote)",
    re.IGNORECASE,
)
_EXECUTES_STDIN = re.compile(r"\b(?:exec|eval|compile)\s*\(", re.IGNORECASE)
_SHELL_REDIRECTION = re.compile(r"^\d*(?:>>?|<<?)(?:&\d+|\S+)$")
_FD_DUPLICATION = re.compile(r"(?<!\S)\d*[<>]&\d+(?!\S)")
# Throwing a known local reader's output away cannot send project data anywhere.
# Keep other redirections conservative: they may read or change a real file.
_DEVNULL_OUTPUT = re.compile(r"(?<!\S)\d*>>?/dev/null(?!\S)")


@dataclass(frozen=True)
class _SoftwareAcquisition:
    scope: str
    kind: str
    needs_python_environment: bool = False
    # Token-index paths to absolute Python launchers. A nested path starts with the
    # outer shell/env argument token, then identifies the executable inside it.
    python_launchers: tuple[tuple[int, ...], ...] = ()
    # Installs only what the project already declares (`uv sync`, `pip install -e .`,
    # `npm ci`), so there is no new package name to approve.
    declared: bool = False


def _software_acquisition(command: str, cwd: Path | None = None) -> _SoftwareAcquisition | None:
    """Return one boundary covering every acquisition in the complete command."""
    found: list[_SoftwareAcquisition] = []
    inline_exec = any(
        _EXECUTES_STDIN.search(match.group("code")) for match in _PYTHON_PIPE_INLINE.finditer(command)
    )
    if _DOWNLOAD_AND_EXECUTE.search(command) or inline_exec:
        found.append(_SoftwareAcquisition("software:bootstrap", "downloaded executable"))
    try:
        # `shlex` otherwise splits `2>&1` into three words and mistakes `&` for a
        # command boundary. Keep three inert words so launcher token offsets still line
        # up with the original command when an absolute Python path must be rewritten.
        parsed_command = _FD_DUPLICATION.sub("--tinyharness-fd --tinyharness-fd --tinyharness-fd", command)
        lexer = shlex.shlex(parsed_command, posix=True, punctuation_chars=";&|()\n")
        lexer.whitespace = " \t\r"  # an unquoted newline is a command boundary, not a space
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        tokens = []

    segment: list[str] = []
    segment_start = 0
    segment_cwd = cwd
    for token_index, token in enumerate([*tokens, ";"]):
        if token and set(token) <= _SHELL_OPERATORS:
            if acquisition := _acquisition_in_segment(segment, segment_cwd, segment_start):
                found.append(acquisition)
            segment_cwd = _directory_after(segment, segment_cwd)
            segment = []
            segment_start = token_index + 1
        else:
            segment.append(token)
    if not found:
        return None

    pending = [item for item in found if not item.declared] or found
    kinds = tuple(dict.fromkeys(item.kind for item in pending))
    kind = kinds[0] if len(kinds) == 1 else "software"
    if any(item.scope.startswith(("software:bootstrap", "software:uv-script")) for item in pending):
        # Piped installers and PEP 723 scripts are approved per command: changing the
        # script's dependencies invalidates the earlier approval.
        material = command + "\0" + "\0".join(item.scope for item in pending)
        scope = f"software:command:{hashlib.sha256(material.encode()).hexdigest()[:16]}"
    else:
        # One yes covers this kind of install for the session. Asking again for every
        # spelling of `pip install` taught people to click through; the boundary that
        # matters once outside text can steer the model is the outside-content check.
        scope = "software:" + "+".join(sorted(kinds))
    return _SoftwareAcquisition(
        scope,
        kind,
        any(item.needs_python_environment for item in found),
        tuple(launcher for item in found for launcher in item.python_launchers),
        declared=all(item.declared for item in found),
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


def _acquisition_in_segment(
    tokens: list[str], cwd: Path | None = None, token_offset: int = 0
) -> _SoftwareAcquisition | None:
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
                option, equals, inline = value.partition("=")
                if name == "env" and option in {"-S", "--split-string"}:
                    if equals:
                        nested, argument_index = inline, index
                    elif index + 1 < len(tokens):
                        nested, argument_index = tokens[index + 1], index + 1
                    else:
                        return None
                    acquisition = _software_acquisition(nested, cwd)
                    if acquisition is None:
                        return None
                    launchers = (
                        tuple((token_offset + argument_index, *path) for path in acquisition.python_launchers)
                        if not equals
                        else tuple(
                            (-(token_offset + argument_index + 1), *path)
                            for path in acquisition.python_launchers
                        )
                    )
                    return replace(acquisition, python_launchers=launchers)
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
            acquisition = _software_acquisition(args[command_index + 1], cwd)
            if acquisition is None:
                return None
            argument_index = token_offset + index + command_index + 2
            return replace(
                acquisition,
                python_launchers=tuple((argument_index, *path) for path in acquisition.python_launchers),
            )
    if executable == "eval" and args:
        # `eval` runs its joined arguments as a command, like `sh -c`. Launchers are
        # only rewritable when that command is a single word of the original.
        acquisition = _software_acquisition(" ".join(args), cwd)
        if acquisition is None:
            return None
        argument_index = token_offset + index + 1
        launchers = (
            tuple((argument_index, *path) for path in acquisition.python_launchers) if len(args) == 1 else ()
        )
        return replace(acquisition, python_launchers=launchers)
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

    declared = False
    if kind == "Python package" and uv_script_scope is None and not uv_inline_dependencies:
        if executable == "uv" and operation == "sync":
            declared = True
        elif pip_executable and operation == "install":
            declared = _declared_python_install(_after_words(args, "install"), cwd)
        elif executable == "uv" and operation == "pip":
            files_only = subcommand == "sync"
            declared = _declared_python_install(_after_words(args, "pip", subcommand), cwd, files_only)
    elif kind == "JavaScript package":
        declared = (
            executable in {"npm", "pnpm", "yarn", "bun"}
            and operation in {"ci", "i", "install"}
            and len(positional) == 1
        )
    elif kind in {"Ruby package", "PHP package", ".NET package"}:
        declared = (executable, operation) in {
            ("bundle", "install"),
            ("composer", "install"),
            ("dotnet", "restore"),
        }

    normalized = " ".join(tokens)
    scope = uv_script_scope or f"software:{executable}:{hashlib.sha256(normalized.encode()).hexdigest()[:16]}"
    launcher = (
        ((token_offset + index,),)
        if python_environment and python_invocation and Path(executable_token).is_absolute()
        else ()
    )
    return _SoftwareAcquisition(scope, kind, python_environment, launcher, declared)


# pip / uv pip options whose value is a requirements file, an editable path, another
# package source, or something else that is not a requirement.
_PIP_FILES = frozenset(
    {"-r", "--requirement", "-c", "--constraint", "-b", "--build-constraint", "--overrides"}
)
_PIP_EDITABLE = frozenset({"-e", "--editable"})
# Another index is a different source than the one the project declared against.
_PIP_OTHER_SOURCES = frozenset(
    {"-i", "--index-url", "--extra-index-url", "--index", "--default-index", "-f", "--find-links"}
)
_PIP_VALUES = frozenset(
    {"-p", "--python", "-t", "--target", "--prefix", "--root", "--cache-dir", "--config-file",
     "--python-version", "--python-platform", "--platform", "--group", "--extra", "--src",
     "--no-binary", "--only-binary", "--proxy", "--timeout", "--retries", "--cert",
     "--client-cert", "--color", "--directory", "--project", "--upgrade-package", "-P"}
)  # fmt: skip
_PIP_TAKES_VALUE = _PIP_FILES | _PIP_EDITABLE | _PIP_OTHER_SOURCES | _PIP_VALUES


def _after_words(args: list[str], *words: str) -> list[str]:
    """The arguments after the command words (`install`, or `pip install`)."""
    lowered = [arg.lower() for arg in args]
    index = 0
    for word in words:
        index = lowered.index(word, index) + 1 if word in lowered[index:] else len(args)
    return args[index:]


def _declared_python_install(args: list[str], cwd: Path | None, files_only: bool = False) -> bool:
    """Whether a pip install names only local paths and requirement files, no new package.

    An option it doesn't know that takes a value makes that value look like a package
    name, so the install asks: unknown means not declared.
    """
    index = 0
    while index < len(args):
        arg = args[index]
        if _SHELL_REDIRECTION.fullmatch(arg):
            index += 1
            continue
        option, equals, value = arg.partition("=")
        if not equals and arg[:2] in _PIP_TAKES_VALUE and len(arg) > 2 and not arg.startswith("--"):
            option, equals, value = arg[:2], "=", arg[2:]  # `-rrequirements.txt`, `-e.`
        takes_value = option in _PIP_TAKES_VALUE
        if takes_value and not equals:
            value = args[index + 1] if index + 1 < len(args) else ""
        index += 2 if takes_value and not equals else 1
        if option in _PIP_OTHER_SOURCES:
            return False
        if option in _PIP_FILES or option in _PIP_EDITABLE:
            if not _local_requirement(value, cwd, named=False):
                return False
        elif not arg.startswith("-") and not _local_requirement(arg, cwd, named=not files_only):
            return False
    return True


def _local_requirement(value: str, cwd: Path | None, *, named: bool) -> bool:
    """A requirement that is a path on this machine, not a package on an index.

    With `named`, a bare word is a package name to pip even when a folder of that name
    exists, so the value must look like a path.
    """
    if cwd is None or not value or "://" in value or "@" in value or any(c in value for c in "$`*"):
        return False
    text = re.sub(r"\[[^\]]*\]$", "", value)  # extras: `python[all_mps]`
    if named and not ("/" in text or text.startswith(".") or text.endswith((".whl", ".tar.gz", ".zip"))):
        return False
    path = Path(text).expanduser()
    return (path if path.is_absolute() else cwd / path).exists()


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


def _shell_token_spans(command: str) -> list[tuple[str, int, int]]:
    """Shell words and operators with their source spans, matching the detector's lexer."""
    out: list[tuple[str, int, int]] = []
    index = 0
    while index < len(command):
        if command[index] in " \t\r":
            index += 1
            continue
        if command[index] == "#":
            newline = command.find("\n", index)
            index = len(command) if newline < 0 else newline
            continue
        start = index
        if command[index] in _SHELL_OPERATORS:
            while index < len(command) and command[index] in _SHELL_OPERATORS:
                index += 1
            out.append((command[start:index], start, index))
            continue
        quote = ""
        while index < len(command):
            char = command[index]
            if quote:
                if char == quote:
                    quote = ""
                elif char == "\\" and quote == '"' and index + 1 < len(command):
                    index += 1
                index += 1
                continue
            if char in "'\"":
                quote = char
                index += 1
            elif char == "\\" and index + 1 < len(command):
                index += 2
            elif char in " \t\r" or char in _SHELL_OPERATORS:
                break
            else:
                index += 1
        raw = command[start:index]
        parsed = shlex.split(raw, comments=False, posix=True)
        if quote or len(parsed) != 1:
            raise ValueError("cannot safely locate the parsed shell token")
        out.append((parsed[0], start, index))
    return out


def _rewrite_python_launchers(command: str, locations: tuple[tuple[int, ...], ...], replacement: Path) -> str:
    """Replace only parsed installer executable tokens, recursively through shell arguments."""
    if not locations:
        return command
    spans = _shell_token_spans(command)
    by_token: dict[int, list[tuple[int, ...]]] = {}
    for location in locations:
        by_token.setdefault(location[0], []).append(location[1:])
    changes: list[tuple[int, int, str]] = []
    for located_index, tails in by_token.items():
        inline_env = located_index < 0
        token_index = -located_index - 1 if inline_env else located_index
        if token_index >= len(spans):
            raise RuntimeError("cannot safely locate the approved Python installation invocation")
        value, start, end = spans[token_index]
        if inline_env:
            option, equals, nested = value.partition("=")
            if not equals:
                raise RuntimeError("cannot safely locate env's split-string command")
            changed = f"{option}={_rewrite_python_launchers(nested, tuple(tails), replacement)}"
        elif all(not tail for tail in tails):
            changed = str(replacement)
        elif any(not tail for tail in tails):
            raise RuntimeError("ambiguous Python installation invocation")
        else:
            changed = _rewrite_python_launchers(value, tuple(tails), replacement)
        changes.append((start, end, shlex.quote(changed)))
    for start, end, changed in reversed(changes):
        command = command[:start] + changed + command[end:]
    return command


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
    command = _rewrite_python_launchers(command, acquisition.python_launchers, selected_python)
    if create and not _MAKES_VENV.search(command):
        launcher = "python" if sys.platform == "win32" else shlex.quote(sys.executable)
        command = f"{launcher} -m venv {shlex.quote(str(environment))} && {command}"
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
        self.policy = policy
        self.shell = shell or default_shell()
        self.session_hosts = session_hosts if session_hosts is not None else set()
        self.set_sandbox(sandbox, proxy)

    def set_sandbox(self, sandbox: Sandbox | None, proxy: AllowlistProxy | None = None) -> None:
        """Run later commands in `sandbox`, or without one. The description changes with it,
        so the model knows what its commands can reach."""
        self.sandbox = sandbox
        self.proxy = proxy if sandbox is not None else None
        network = (
            "network only to package registries (PyPI, npm), GitHub and common CDNs, through a "
            "proxy (/sandbox lists them)"
            if self.proxy
            or (sandbox and getattr(sandbox, "manages_network", False) and self.policy.allowed_hosts)
            else "no network access"
        )
        limits = (
            "It runs in a sandbox: it can write only inside the working directory and private "
            f"scratch/cache directories, can't read credential files, and has {network}"
            + (" (and localhost)" if self.policy.allow_localhost else "")
            + ". API key variables are stripped, and apps (`open`, browsers like Chrome.app) can't "
            "start. When a download names a blocked destination, retry once with network_host set "
            "to that exact hostname; approval keeps the command sandboxed. Use structured file/Open "
            "tools for host actions; Bash never drops containment. Installing what the project declares "
            "(`uv sync`, `pip install -e .`, `npm ci`) runs freely when it is its own command, apart from "
            "tests and other steps; naming a new package is reviewed first; run "
            "an already-installed executable directly to reuse it without another prompt. "
            "A bare Python install uses a writable active environment or a project-local .venv."
            if sandbox
            else "The user turned the sandbox off: commands run as the user, can read and write any "
            "folder and reach any host, so network_host isn't needed. Commands that delete files, "
            "use sudo, touch credential files or change files that run code later (git hooks, "
            "editor settings, .envrc) ask the user first. API key variables are stripped. "
            "A bare Python install uses a writable active environment or a project-local .venv."
        )
        self.description = (
            "Run a shell command in the working directory and return its combined stdout/stderr "
            "and exit code. Use it to run tests, builds, linters, git and other CLI tools. Each call "
            "starts fresh in the working directory (use `cd dir && ...`); nothing interactive. "
            "Commands use ordinary shell semantics: join dependent steps with `&&`, and independent "
            "probes with `;`. Bash already keeps the beginning and end of large output, so prefer a "
            "command's own limit flag (`git log -n 3`) over piping into `head` or `tail`. "
            "To look at files use Read/Grep/Glob, not cat/head/ls/find/grep: they run in parallel, "
            "never ask, and take paths in any subdirectory. Use Edit over sed. "
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
        if acquisition is None or acquisition.declared:
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
            f"allow {acquisition.kind} installs for this session",
        )

    def writes(self, args: BashInput | dict[str, Any]) -> bool:
        command = (
            args.command if isinstance(args, BashInput) else str(args.get("command", ""))
        )  # the gate's raw args
        return not (only_reads(command) or certainly_offline(command))

    def contained(self, args: dict[str, Any]) -> bool:
        return self.sandbox is not None

    def runs_unsandboxed(self, args: dict[str, Any]) -> bool:
        return self.sandbox is None

    def can_send_out(self, args: dict[str, Any]) -> bool:
        """The sandbox's allowlist is still an outbound channel for readable project data,
        except for plain downloads: they send an allowed host nothing but the URL."""
        if self.sandbox is None:  # any host: only a command that can't use the network is safe
            return not certainly_offline(str(args.get("command", "")))
        if not (self.policy.allowed_hosts or self.session_hosts or self._network_host(args)):
            return False
        return bool(self._network_host(args)) or not network_reads_only(str(args.get("command", "")))

    def escapes(self, args: dict[str, Any]) -> str | None:
        if self.sandbox is None:  # off is the user's choice (/sandbox): there's nothing to escape
            return None
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
        if self.sandbox:
            return None
        return _unsandboxed_risk(
            command, next(iter(self.policy.project_roots or self.policy.write_roots), None)
        )

    async def run(self, args: BashInput, ctx: ToolContext) -> str:
        started = time.time()
        run = await self.execute(args, ctx)
        reached = run.reached
        if _installs_declared_only(args.command, self.policy):
            # Installing what the project declares is authorized project setup inside the
            # sandbox, not new outside content: the repo's authors chose those packages. A
            # new package name still counts: its build and its code can print or plant text.
            reached = [host for host in reached if not host_allowed(host, PACKAGE_REGISTRY_HOSTS)]
        if reached:  # what it downloaded is outside content (a README can carry instructions)
            note_outside(ctx, f"Bash downloads from {', '.join(reached[:3])}")
        elif (
            not run.sandboxed
            and _MAY_DOWNLOAD.search(args.command)
            and not _installs_declared_only(args.command, self.policy)
        ):
            # No proxy says which hosts it reached: anything that looks like a download counts.
            note_outside(ctx, "Bash network use (no sandbox)")
        notes = self._cached_python_notes(args.command, started) if run.sandboxed else ()
        return self._report(run.output, run.code, run.timed_out, args, run.sandboxed, run.blocked, notes)

    def _cached_python_notes(self, command: str, since: float) -> tuple[str, ...]:
        """Say so when a new virtualenv's Python lives in the private sandbox cache.

        Seen live: `uv venv -p 3.12` downloaded Python into the sandbox cache, so the
        project's .venv worked from the user's terminal and IDE only while that cache lasted.
        """
        cache = self.policy.cache_dir
        if cache is None or "uv" in self.policy.shared_caches or not _MAKES_VENV.search(command):
            return ()
        notes = []
        for root in self.policy.project_roots:
            for config in (*root.glob("*/pyvenv.cfg"), *root.glob("*/*/pyvenv.cfg")):
                try:
                    if config.stat().st_mtime < since - 1:
                        continue
                    home = next(
                        (line.partition("=")[2].strip() for line in config.read_text().splitlines()
                         if line.partition("=")[0].strip() == "home"),
                        "",
                    )  # fmt: skip
                except OSError:
                    continue
                if home and Path(home).resolve().is_relative_to(cache):
                    notes.append(
                        f"[tinyharness:venv_uses_sandbox_python path={config.parent}] Tell the user: this "
                        "virtualenv works from their own terminal and IDE, but its Python is stored in "
                        f"tinyharness's private sandbox cache ({cache}), so clearing that cache breaks it. "
                        'To keep uv\'s Pythons where uv normally keeps them, set `share_caches = ["uv"]` '
                        "under [sandbox] in ~/.tinyharness/config.toml and recreate the venv."
                    )
        return tuple(notes)

    async def execute(self, args: BashInput, ctx: ToolContext) -> Execution:
        """Run the command; what it printed and how it ended, before any notes are added."""
        command, install_env, _ = _python_install_environment(args.command, self.policy)
        argv = shell_argv(self.shell, command)
        sandboxed = self.sandbox is not None
        policy, denied_before, reached_before = self.policy, 0, 0
        active_proxy = self.proxy
        temporary_proxy = False
        added_hosts = (
            *sorted(self.session_hosts),
            *(service_hosts(args.network_host) if args.network_host else ()),
        )
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
                env=command_env(
                    {
                        **(self.sandbox.environment(policy) if sandboxed and self.sandbox else {}),
                        **install_env,
                    }
                ),
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

    def _blocked_path(self, text: str) -> tuple[str, bool]:
        """The path a denied write named, and whether it is a protected one (.git/hooks).

        A path outside every writable root counts, and so does a protected path inside
        one, even relative (`pre-commit install` reports `.git/hooks/pre-commit`).
        """
        roots = self.policy.write_roots
        for match in _DENIED_PATH.finditer(text):
            path = Path(next(group for group in match.groups() if group))
            if self._protected(path):
                return str(path), True
            if path.is_absolute() and not any(path == root or path.is_relative_to(root) for root in roots):
                return str(path), False
        return "", False

    def _protected(self, path: Path) -> bool:
        if path.is_absolute() and any(path == p or path.is_relative_to(p) for p in self.policy.protected):
            return True
        parts = path.parts
        for rel in PROTECTED_IN_PROJECT:
            want = Path(rel).parts
            if any(parts[i : i + len(want)] == want for i in range(len(parts) - len(want) + 1)):
                return True
        return False

    def _report(
        self,
        text: str,
        code: int | None,
        timed_out: bool,
        args: BashInput,
        sandboxed: bool,
        blocked: list[str] | None = None,
        notes: tuple[str, ...] = (),
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
        elif sandboxed and denied and (protected := self._blocked_path(text))[1]:
            # Seen live: `pre-commit install` hit .git/hooks; nothing said only the user can.
            text += (
                f"\n[tinyharness:protected_path_denied path={protected[0]}] This command stayed "
                f"sandboxed. {protected[0]} is protected: git hooks and config and tinyharness's own "
                "settings run code later, outside the sandbox, so no sandboxed command may change them. "
                "Don't look for a way around it: tell the user the exact command to run in their own "
                "terminal for this step (for example `pre-commit install`), and carry on without it."
            )
        elif sandboxed and denied:
            path, _ = self._blocked_path(text)
            text += (
                f"\n[tinyharness:filesystem_or_process_denied{f' path={path}' if path else ''}] "
                "This command stayed sandboxed. Use the working directory, $TMPDIR, or an explicitly "
                "configured writable root; do not retry outside containment."
            )
            if path and _is_home_dot_dir(path):
                # Seen live: SGLang hard-codes ~/.cache/sglang instead of following XDG.
                text += (
                    f" The tool writes {path} directly instead of following $XDG_CACHE_HOME, which "
                    "already points at a writable sandbox cache. Find the variable that sets this "
                    "folder (most tools have one, like FOO_CACHE_DIR or FOO_HOME) and point it beneath "
                    "$XDG_CACHE_HOME. If you have a Memory tool, save that variable there so later "
                    "sessions set it from the start."
                )
        for note in notes:
            text += f"\n{note}"
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
        if _unsafe_options(name, rest):
            return False
        if name in _READERS:
            continue
        if name == "find" and not _FIND_WRITES.intersection(rest):
            continue
        if name == "git" and rest[:1] and rest[0] in _GIT_READS:
            continue
        return False
    return True


def _unsafe_options(name: str, args: list[str]) -> bool:
    """Whether `args` include an option that makes `name` run a program or write a file."""
    for option in _UNSAFE_OPTIONS.get(name, ()):
        for arg in args:
            given = arg.split("=", 1)[0]
            if option.startswith("--"):
                if len(given) > 2 and given.startswith("--") and option.startswith(given):
                    return True
            elif arg.startswith("-") and not arg.startswith("--") and option[1] in arg[1:]:
                return True  # a short option, perhaps bundled (`tree -ao FILE`)
    return False


# Shell builtins that may sit beside a declared install without running another program.
# `source` is deliberately absent: an activation-looking script can execute anything.
_SETUP_STEPS = frozenset({"cd", "export", "true"})
# Local file and text tools that have no way to reach the network themselves.
_LOCAL_STEPS = frozenset(
    {"cp", "mv", "mkdir", "touch", "ln", "sed", "awk", "sort", "uniq", "uname", "date", "sleep", "false"}
)
# git subcommands that only touch the local repository (not fetch, pull, clone, push).
_GIT_LOCAL = _GIT_READS | {"checkout", "switch", "restore", "add", "branch", "reset", "stash"}


def _installs_declared_only(command: str, policy: SandboxPolicy) -> bool:
    """Whether the command is project setup: declared installs, plus steps that can't fetch.

    Traffic is attributed to the whole command, so any other step (`npm view x`, a
    `python -c` that makes a request) keeps the registry hosts as outside content.
    """
    # shlex preserves legacy command substitutions as ordinary text. Reject them before
    # deciding that a builtin or local reader cannot have fetched the registry traffic.
    if "`" in command:
        return False
    cwd: Path | None = (policy.project_roots or policy.write_roots)[0].resolve()
    try:
        lexer = shlex.shlex(_FD_DUPLICATION.sub("", command), posix=True, punctuation_chars=";&|()\n")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return False
    # A subshell's steps are judged like any others: `(cd x && uv sync)` is setup, and
    # `uv sync && (curl ...)` still has a step that fetches.
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token and set(token) <= _SHELL_OPERATORS:
            segments.append([])
        else:
            segments[-1].append(token)
    installs = 0
    for segment in (item for item in segments if item):
        words = [word for word in segment if not _SHELL_REDIRECTION.fullmatch(word)]
        name = Path(words[0]).name if words else ""
        if (
            name in _SETUP_STEPS
            or name in _READERS
            or name in _LOCAL_STEPS
            or (name == "git" and len(words) > 1 and words[1] in _GIT_LOCAL)
            or (name in {"source", "."} and len(words) == 2 and _project_activation(words[1], cwd, policy))
            or _MAKES_VENV.search(" ".join(words))
        ):
            pass
        elif (acquisition := _acquisition_in_segment(segment, cwd)) is not None and acquisition.declared:
            installs += 1
        else:
            return False
        cwd = _directory_after(segment, cwd)
    return installs > 0


def _project_activation(path: str, cwd: Path | None, policy: SandboxPolicy) -> bool:
    """Whether a sourced script is a virtualenv's `bin/activate` inside the project.

    It runs like any script, so it is trusted only as far as the project's own files are
    (a declared install runs the project's build code too). One elsewhere, such as under
    /tmp, could download anything.
    """
    if _EXPANDS.search(path) or cwd is None:
        return False
    script = cwd / path
    environment = script.parent.parent
    return (
        script.name == "activate"
        and script.parent.name == "bin"
        and script.is_file()
        and (environment / "pyvenv.cfg").is_file()
        and _inside(script, tuple(root.resolve() for root in policy.project_roots))
    )


def certainly_offline(command: str) -> bool:
    """Whether every command is a known local reader that cannot use the network.

    This is narrower than "read-only": arbitrary programs may read files and send them out.
    It lets outside content coexist with local inspection such as `which cargo; ls bin | head`.
    """
    command = _DEVNULL_OUTPUT.sub("", _FD_DUPLICATION.sub("", command))
    if any(marker in command for marker in ("`", "$(", "<", ">", "(", ")", "{")):
        return False
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|\n")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return False
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token and set(token) <= _SHELL_OPERATORS:
            if segments[-1]:
                segments.append([])
        else:
            segments[-1].append(token)
    return bool(segments and any(segments)) and all(_offline_step(words) for words in segments if words)


def _offline_step(words: list[str]) -> bool:
    """One step of `certainly_offline`: a known local reader."""
    name, rest = words[0], words[1:]
    if _unsafe_options(name, rest):
        return False
    if name in _READERS:
        return True
    if name == "find" and not _FIND_WRITES.intersection(rest):
        return True
    if (
        name == "git"
        and rest[:1]
        and (rest[0] in _GIT_READS or (rest[0] == "remote" and rest[1:] in (["-v"], ["--verbose"])))
    ):
        return True
    if name in {"cd", "test", "[", "[["}:
        return True
    return name in {"python", "python3", "uv", "conda", "cargo", "rustc"} and rest in (["--version"], ["-V"])


# Variables a plain download may name: loop variables, and these, which hold paths the
# harness set rather than anything read from a file.
_PATH_VARIABLES = frozenset({"TMPDIR", "HOME", "PWD", "XDG_CACHE_HOME"})
_VARIABLE = re.compile(r"\$(?:\{(\w+)\}|(\w+))")
# Text tools that cannot open a connection (awk is absent: gawk reads /inet/tcp files).
# sed is judged by its script (`_plain_sed`), sort by its options (`_UNSAFE_OPTIONS`).
_TEXT_STEPS = frozenset({"sort", "uniq", "tee", "base64", "shasum", "md5", "mkdir", "touch", "true", "sleep"})
# One sed command that only prints, deletes or substitutes, with an optional address or
# range. Anything else is refused: GNU sed's `e` command and `s///e` flag run programs.
_SED_ADDRESS = r"(?:\d+|\$|/(?:[^/\\\n]|\\.)*/)"
_SED_COMMAND = re.compile(
    rf"\s*(?:{_SED_ADDRESS}(?:\s*,\s*{_SED_ADDRESS})?\s*!?\s*)?"
    r"(?:[pdqP=]|s(?P<d>[^\w\s\\])(?:(?!(?P=d))[^\\]|\\.)*(?P=d)(?:(?!(?P=d))[^\\]|\\.)*(?P=d)[gpiImM0-9]*)\s*"
)
_SED_FLAGS = frozenset(
    {"-n", "--quiet", "--silent", "-E", "-r", "--regexp-extended", "-u", "--unbuffered", "--sandbox"}
)
_CURL_FLAGS = frozenset("sSfLiIvO")
_CURL_LONG = frozenset(
    {"--silent", "--show-error", "--fail", "--fail-with-body", "--location", "--include", "--head",
     "--compressed", "--verbose", "--remote-name", "--globoff"}
)  # fmt: skip
_CURL_VALUES = frozenset(
    {
        "-o",
        "--output",
        "-m",
        "--max-time",
        "--connect-timeout",
        "--retry",
        "--retry-delay",
        "-w",
        "--write-out",
    }
)
_WGET_FLAGS = frozenset("qv")
_WGET_VALUES = frozenset({"-O", "--output-document", "-T", "--timeout", "-t", "--tries"})
_GH_READS = {
    "issue": {"list", "view", "status"},
    "pr": {"list", "view", "diff", "checks", "status"},
    "repo": {"view"},
    "release": {"list", "view"},
    "run": {"list", "view"},
}
_GH_API_OPTIONS = frozenset({"--jq", "-q", "--paginate", "--slurp", "-R", "--repo"})
# What a `python -c` text filter may import. With no other module, no dunder names and no
# dynamic imports or evaluation it has no way to open a connection.
_FILTER_MODULES = frozenset(
    {"json", "sys", "re", "textwrap", "collections", "itertools", "datetime", "math", "csv", "html", "pprint"}
)
_FILTER_FORBIDDEN = ("__", "importlib", "exec", "eval", "compile", "getattr", "globals", "vars(", "modules")
_IMPORTS = re.compile(r"\b(?:from\s+([\w.]+)\s+import|import\s+([\w., ]+))")


def network_reads_only(command: str) -> bool:
    """Whether every step reads: a local reader, or a plain download of literal URLs.

    Once outside content is in the conversation, a call that could send project data out
    is reviewed (permissions/gate.py). A plain download sends nothing but the URL it
    names, and only to a host the sandbox already allows; what it fetches is outside
    content all the same. Seen live: 34 of 65 auto-mode reviews were `curl` reads of
    GitHub issues and docs, each one waiting on a model.
    """
    command = _DEVNULL_OUTPUT.sub("", _FD_DUPLICATION.sub("", command))
    if "`" in command or "$(" in command:
        return False
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|\n<>()")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return False
    segments: list[list[str]] = [[]]
    expect_target = False
    for token in tokens:
        if expect_target:  # the file after `>`: writing a file sends nothing anywhere
            expect_target = False
            continue
        if token in (">", ">>"):
            expect_target = True
        elif token in (";", "&&", "||", "|", "\n"):
            if segments[-1]:
                segments.append([])
        elif token and set(token) <= set(";&|\n<>()"):
            return False  # input redirection, subshells, background jobs
        else:
            segments[-1].append(token)
    if expect_target or not any(segments):
        return False
    variables = set(_PATH_VARIABLES)
    for words in (segment for segment in segments if segment):
        if words[0] == "do":
            words = words[1:]
        if not words or words == ["done"]:
            continue
        if words[0] == "for":
            if len(words) < 3 or words[2] != "in" or not words[1].isidentifier():
                return False
            variables.add(words[1])
            if any("$" in word for word in words[3:]):
                return False
            continue
        if any(name not in variables for word in words for name in _variables(word)):
            return False
        if not _reading_step(words):
            return False
    return True


def _variables(word: str) -> list[str]:
    return [braced or bare for braced, bare in _VARIABLE.findall(word)]


def _reading_step(words: list[str]) -> bool:
    name, rest = words[0], words[1:]
    if _unsafe_options(name, rest):
        return False
    if _offline_step(words) or name in _TEXT_STEPS:
        return True
    if name == "sed":
        return _plain_sed(rest)
    if name == "curl":
        return _plain_options(rest, _CURL_FLAGS, _CURL_LONG, _CURL_VALUES)
    if name == "wget":
        return _plain_options(rest, _WGET_FLAGS, frozenset({"--quiet", "-nv"}), _WGET_VALUES)
    if name == "gh":
        return _gh_reads(rest)
    if Path(name).name in {"python", "python3"} or re.fullmatch(r"python3\.\d+", Path(name).name):
        return _text_filter(rest)
    return False


def _plain_options(
    args: list[str], flags: frozenset[str], long: frozenset[str], values: frozenset[str]
) -> bool:
    """Only output, timing and redirect options, and at least one http(s) URL."""
    urls = 0
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            if arg.split("=", 1)[0] in values:
                i += 1 if "=" in arg else 2
                continue
            if arg not in long:
                return False
        elif arg.startswith("-") and len(arg) > 1:
            letters = arg[1:]
            for k, letter in enumerate(letters):
                if f"-{letter}" in values:  # its value is the rest of the word, or the next word
                    i += k == len(letters) - 1
                    break
                if letter not in flags:
                    return False
        elif re.match(r"https?://", arg):
            urls += 1
        else:
            return False
        i += 1
    return urls > 0


def _plain_sed(args: list[str]) -> bool:
    """`sed -n '1,20p'`, `sed 's/a/b/g' file`: scripts that only print, delete or substitute."""
    scripts, operands = [], []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("-e", "--expression"):
            if i + 1 == len(args):
                return False
            scripts.append(args[i + 1])
            i += 1
        elif arg.startswith("--expression="):
            scripts.append(arg.split("=", 1)[1])
        elif len(arg) > 2 and arg[0] == "-" and arg[-1] == "e" and set(arg[1:-1]) <= set("nEru"):
            if i + 1 == len(args):  # bundled flags ending in -e (`sed -ne 's/x/y/p'`)
                return False
            scripts.append(args[i + 1])
            i += 1
        elif arg in _SED_FLAGS or (len(arg) > 1 and arg[0] == "-" and set(arg[1:]) <= set("nEru")):
            pass
        elif arg.startswith(("-i", "--in-place")):
            pass  # editing in place writes a file, which sends nothing anywhere
        elif arg.startswith("-") and arg != "-":
            return False  # -f reads a script we can't see; anything unknown
        else:
            operands.append(arg)
        i += 1
    if not scripts:
        if not operands:
            return False
        scripts.append(operands[0])
    return all(_sed_script(script) for script in scripts)


def _sed_script(script: str) -> bool:
    position = 0
    while match := _SED_COMMAND.match(script, position):
        position = match.end()
        if position == len(script):
            return True
        if script[position] not in ";\n":
            return False
        position += 1
        if position == len(script):
            return True
    return False


def _gh_reads(args: list[str]) -> bool:
    """`gh issue view`, `gh pr list`, `gh api <path>` and the like: no method, field or input."""
    if not args:
        return False
    if args[0] == "search":
        return True
    if args[0] != "api":
        return len(args) > 1 and args[1] in _GH_READS.get(args[0], set())
    positional = []
    i = 1
    while i < len(args):
        arg = args[i]
        if arg.startswith("-"):
            option = arg.split("=", 1)[0]
            if option not in _GH_API_OPTIONS:
                return False
            i += option not in {"--paginate", "--slurp"} and "=" not in arg
        else:
            positional.append(arg)
        i += 1
    return len(positional) == 1


def _text_filter(args: list[str]) -> bool:
    """`python3 -c '...'` that only reshapes text: standard modules that can't connect."""
    if len(args) < 2 or args[0] != "-c":
        return False
    code = args[1]
    if any(word in code for word in _FILTER_FORBIDDEN):
        return False
    for source, names in _IMPORTS.findall(code):
        modules = [source] if source else [name.strip().split(" ")[0] for name in names.split(",")]
        if any(module.split(".")[0] not in _FILTER_MODULES for module in modules if module):
            return False
    return True


def _unsandboxed_risk(command: str, cwd: Path | None = None) -> str | None:
    """Why an unsandboxed command must ask the user, or None for the review to judge."""
    shell_words = _shell_words(command)
    if _AS_ANOTHER_USER.search(command) or any(Path(word).name in _OTHER_USER for word, _ in shell_words):
        return "runs a command as another user (sudo)"
    words = [word.removeprefix("~") for word in _WORDS.findall(command)]
    for word, expands in shell_words:
        words.append(word.removeprefix("~"))
        if expands:
            paths = _expansions(word, cwd or Path.cwd())
            if paths is None:
                return "touches credential files (its pattern names too many paths to check)"
            words += paths
    if any(is_secret_path(word) or _CREDENTIAL_DIRS.search(word) for word in words):
        return "touches credential files"
    if not only_reads(command):
        for rel in PROTECTED_IN_PROJECT:
            if any(re.search(rf"(^|/){re.escape(rel)}(/|$)", word) for word in words):
                return f"changes {rel}, which runs code later (hooks, editor or agent settings)"
    return None


def _shell_words(command: str) -> list[tuple[str, bool]]:
    """Each word as the shell reads it, quotes and escapes removed (`~/.s"s"h` is
    `~/.ssh`, `\\sudo` is `sudo`), and whether the shell expands it: a `~`, variable,
    brace or glob outside quotes. A command that doesn't parse has no words."""
    try:
        lexer = shlex.shlex(command, posix=False, punctuation_chars=";&|()<>")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return []
    words = []
    for token in tokens:
        try:
            word = " ".join(shlex.split(token))
        except ValueError:
            word = token
        unquoted = re.sub(r"'[^']*'|\"(?:[^\"\\]|\\.)*\"", "", token)
        words.append((word.strip("`"), bool(_EXPANDS.search(unquoted))))
    return words


def _expansions(word: str, cwd: Path) -> list[str] | None:
    """The paths a word stands for once the shell expands `~`, variables, braces and
    globs, as far as they exist now. None when there are too many to check."""
    found = []
    patterns = _braces(os.path.expandvars(os.path.expanduser(word)))
    if patterns is None:
        return None
    for pattern in patterns:
        found.append(pattern)
        if glob.has_magic(pattern) and os.path.isabs(pattern):
            found += _credential_spellings(pattern)
        if glob.has_magic(pattern):
            where = pattern if os.path.isabs(pattern) else os.path.join(glob.escape(str(cwd)), pattern)
            matches = list(itertools.islice(glob.iglob(where), _MAX_EXPANSIONS + 1))
            if len(matches) > _MAX_EXPANSIONS:
                return None
            found += matches
    return found


def _credential_spellings(pattern: str) -> list[str]:
    """`~/.a?s/credentials` read as `~/.aws/credentials`: what a pattern could name even
    where nothing matches yet. Like the shell, `*` alone doesn't match a dot directory."""
    parts = pattern.split("/")
    return [
        "/".join([*parts[:i], name, *parts[i + 1 :]])
        for i, part in enumerate(parts)
        if glob.has_magic(part)
        for name in _CREDENTIAL_NAMES
        if (part.startswith(".") or not name.startswith(".")) and fnmatch.fnmatchcase(name, part)
    ]


def _braces(word: str) -> list[str] | None:
    """`~/.{ssh,aws}/x` as the shell spells it out; None past `_MAX_EXPANSIONS`."""
    words = [word]
    while any(_BRACE.search(w) for w in words):
        spelled = []
        for w in words:
            if match := _BRACE.search(w):
                spelled += [
                    w[: match.start()] + part + w[match.end() :] for part in match.group(1).split(",")
                ]
            else:
                spelled.append(w)
        if len(spelled) > _MAX_EXPANSIONS:
            return None
        words = spelled
    return words


def default_shell() -> str:
    return shutil.which("bash") or shutil.which("sh") or "/bin/sh"


def shell_argv(shell: str, command: str) -> list[str]:
    """Run with the shell's ordinary command-list and pipeline semantics."""
    return [shell, "-c", command]


# Where a denied write says what it tried: Python's `Operation not permitted: '/x'` and
# BSD tools' `mkdir: /x: Operation not permitted`.
_DENIED_PATH = re.compile(
    r"(?:Operation not permitted|Permission denied): '([^']+)'"
    r"|(/[^\s:'\"]+): (?:Operation not permitted|Permission denied)"
)


def _is_home_dot_dir(path: str) -> bool:
    """Under a hidden folder in the home directory (~/.cache/x, ~/.triton/x), where tools keep caches."""
    try:
        parts = Path(path).relative_to(Path.home()).parts
    except ValueError:
        return False
    return len(parts) >= 2 and parts[0].startswith(".")


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
