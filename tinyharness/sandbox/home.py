"""A private, per-session home directory for sandboxed commands.

Tools write their own settings and state under `$HOME` all the time (IPython, matplotlib,
Jupyter, `pip --user`, `git config --global`). The real home stays closed to sandboxed
writes, because files there (~/.zshrc, ~/.ipython/startup, ~/.config/...) run code later,
outside the sandbox. So sandboxed commands get `HOME` set to an empty folder in the
session's scratch directory instead: their writes succeed, and are gone when the session
ends, so one compromised run can't plant something a later one picks up.

The folder is set up once, before any command runs, and never changed in reaction to
command output. It holds only what's projected into it here:

- a git global config with just your name, email and default branch (no includes, URL
  rewrites, hooks, helpers or signing), so commits keep your identity;
- links to the real-home paths you list in `[sandbox] home_read`, for tools that need
  more. Links only make a path reachable as ~/x: writes through them still land outside
  the write roots and are refused, and credential files stay unreadable.

Programs that find home without `$HOME` (`getpwuid`, `NSHomeDirectory`) still see the real
one and are refused writes there, as before. Windows isn't covered: there `HOME` alone
isn't the home (`USERPROFILE`, `APPDATA`, the registry), so commands keep today's behavior.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Iterable
from pathlib import Path

from tinyharness.paths import home_dir
from tinyharness.security import is_secret_path

# The only git settings carried over: who you are and what a new branch is called.
# Everything else in a global git config can run code or redirect traffic.
GIT_IDENTITY = ("user.name", "user.email", "init.defaultBranch")


def private_home(scratch: Path, home_read: Iterable[str] = ()) -> Path:
    """Create the session's private home in `scratch` and project what it needs.

    Parameters
    ----------
    scratch : Path
        The session's scratch directory (already a sandbox write root).
    home_read : iterable of str
        Real-home paths (`~/x` or absolute) to link in, from the user's own config.

    Returns
    -------
    Path
        The new home, mode 0700.

    Raises
    ------
    ValueError
        When a `home_read` entry is missing, outside the home, a credential or
        tinyharness path, or a symlink loop. The message names the entry.
    """
    links = [_checked(entry) for entry in home_read]  # before creating anything
    home = scratch / "home"
    scratch.mkdir(parents=True, exist_ok=True)
    home.mkdir(mode=0o700, exist_ok=True)
    home.chmod(0o700)  # mkdir's mode is masked by the umask
    identity = git_identity()
    if identity:
        (home / ".gitconfig").write_text(_git_config(identity))
    for relative, target in links:
        link = home / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.is_symlink() and not link.exists():
            link.symlink_to(target)
    return home


def home_env(home: Path) -> dict[str, str]:
    """The environment that points a sandboxed command's home at `home`."""
    local = home / ".local"
    return {
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "XDG_DATA_HOME": str(local / "share"),
        "XDG_STATE_HOME": str(local / "state"),
        "XDG_CACHE_HOME": str(home / ".cache"),
        # Named, not just found through HOME: an inherited GIT_CONFIG_GLOBAL would point git
        # back at your full real config (helpers, rewrites) and `git config --global` at it.
        "GIT_CONFIG_GLOBAL": str(home / ".gitconfig"),
        # Do not layer /etc/gitconfig on top: GIT_CONFIG_SYSTEM can redirect that lookup to
        # an arbitrary file from the host environment.
        "GIT_CONFIG_NOSYSTEM": "1",
        # `pip install --user` puts commands in ~/.local/bin on every platform, not
        # ~/Library/Python/X.Y/bin as macOS framework Pythons do (seen in the prototype).
        "PYTHONUSERBASE": str(local),
        # Last, so something installed there runs but never shadows an existing command.
        "PATH": os.environ.get("PATH", "") + os.pathsep + str(local / "bin"),
    }


def without_git_overrides(env: dict[str, str]) -> dict[str, str]:
    """`env` minus the variables that inject git config from outside the private home.

    GIT_CONFIG_COUNT with GIT_CONFIG_KEY_n/GIT_CONFIG_VALUE_n, GIT_CONFIG_PARAMETERS
    (`git -c`), GIT_CONFIG_SYSTEM and the old GIT_CONFIG each add or replace settings,
    so an inherited one would bring back what the generated config leaves out.
    """
    return {
        key: value
        for key, value in env.items()
        if key not in _GIT_OVERRIDES and not key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
    }


_GIT_OVERRIDES = frozenset({"GIT_CONFIG", "GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_SYSTEM"})


def git_identity() -> dict[str, str]:
    """Your global git name, email and default branch, ignoring included files."""
    git = shutil.which("git")
    if git is None:
        return {}
    try:
        out = subprocess.run(
            [
                git,
                "config",
                "--global",
                "--no-includes",
                "-z",
                "--get-regexp",
                r"^(user\.(name|email)|init\.defaultbranch)$",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            stdin=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    wanted = {key.lower(): key for key in GIT_IDENTITY}
    identity: dict[str, str] = {}
    for record in out.split("\0"):
        key, _, value = record.partition("\n")
        if key.lower() in wanted and value and "\n" not in value:
            identity[wanted[key.lower()]] = value  # the last one wins, as in git
    return identity


def _git_config(identity: dict[str, str]) -> str:
    sections: dict[str, list[str]] = {}
    for key, value in identity.items():
        section, _, name = key.partition(".")
        quoted = value.replace("\\", "\\\\").replace('"', '\\"')
        sections.setdefault(section, []).append(f'\t{name} = "{quoted}"')
    return "".join(f"[{section}]\n" + "\n".join(lines) + "\n" for section, lines in sections.items())


def _checked(entry: str) -> tuple[Path, Path]:
    """(path relative to the home, resolved target) for one `home_read` entry."""
    real_home = Path.home().resolve()
    given = Path(entry).expanduser()
    if not given.is_absolute():
        given = real_home / given
    try:
        target = given.resolve(strict=True)
    except FileNotFoundError:
        raise ValueError(f"sandbox.home_read: {entry!r} does not exist") from None
    except (OSError, RuntimeError):
        raise ValueError(f"sandbox.home_read: {entry!r} can't be resolved (a symlink loop?)") from None
    if target == real_home or not target.is_relative_to(real_home):
        raise ValueError(f"sandbox.home_read: {entry!r} is not inside your home directory")
    tinyharness = home_dir().expanduser().resolve()
    secret = is_secret_path(target) or is_secret_path(target / "x")  # ~/.ssh as well as ~/.ssh/x
    if secret or target.is_relative_to(tinyharness) or tinyharness.is_relative_to(target):
        raise ValueError(f"sandbox.home_read: {entry!r} holds credentials or tinyharness's own files")
    # Linked where it appears under the home, so `~/x` means the same thing inside.
    lexical = Path(os.path.normpath(given))
    relative = (
        lexical.relative_to(real_home) if lexical.is_relative_to(real_home) else target.relative_to(real_home)
    )
    return relative, target
