from __future__ import annotations

from pathlib import Path


class Workspace:
    """Resolves the paths tools operate on.

    `root` is the user's working directory. Tools may target any path; the
    permission gate decides whether to allow it. This class only answers
    "where does this path really point?" and "is it inside the workspace?".

    Symlinks are always resolved first, so `link -> /etc/passwd` is judged as
    /etc/passwd. `read_roots` are extra directories (e.g. skill bundles) the
    agent may read without prompting.
    """

    def __init__(self, root: Path, read_roots: list[Path] | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.read_roots: list[Path] = []
        for r in read_roots or []:
            self.add_read_root(r)

    def add_read_root(self, path: Path) -> None:
        try:
            resolved = Path(path).expanduser().resolve()
        except OSError:
            return
        if resolved not in self.read_roots:
            self.read_roots.append(resolved)

    def resolve(self, user_path: str | Path) -> Path | None:
        """Absolute, symlink-resolved path, or None if it can't be resolved."""
        p = Path(user_path).expanduser()
        if not p.is_absolute():
            p = self.root / p
        try:
            return p.resolve(strict=False)
        except (OSError, RuntimeError):
            return None

    def contains(self, path: str | Path) -> bool:
        """True if the path is inside the working directory."""
        resolved = self.resolve(path)
        return resolved is not None and resolved.is_relative_to(self.root)

    def in_read_root(self, path: str | Path) -> bool:
        """True if the path is inside a registered read root (not the working dir)."""
        resolved = self.resolve(path)
        return resolved is not None and any(resolved.is_relative_to(r) for r in self.read_roots)

    def is_trusted(self, path: str | Path) -> bool:
        """Inside the working directory or a read root — i.e. no extra prompt needed."""
        return self.contains(path) or self.in_read_root(path)

    def relative(self, path: Path) -> str:
        return str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)
