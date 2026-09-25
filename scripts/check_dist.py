"""Check a built wheel before it ships: data files present, installs cleanly, runs.

uv build && uv run python scripts/check_dist.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REQUIRED = [
    "tinyharness/py.typed",
    "tinyharness/llm/providers.toml",
    "tinyharness/prompts/system.md",
    "tinyharness/prompts/compaction.md",
    "tinyharness/prompts/classifier.md",
    "tinyharness/prompts/agents/explore.md",
]


def main() -> None:
    wheels = sorted(Path("dist").glob("tinyharness-*.whl"))
    if not wheels:
        sys.exit("no wheel in dist/; run `uv build` first")
    wheel = wheels[-1]
    names = set(zipfile.ZipFile(wheel).namelist())
    if missing := [f for f in REQUIRED if f not in names]:
        sys.exit(f"{wheel.name} is missing: {', '.join(missing)}")
    if leaked := [n for n in names if n.startswith(("tests/", "examples/")) or n.endswith(".env")]:
        sys.exit(f"{wheel.name} contains files it shouldn't: {', '.join(leaked)}")

    with tempfile.TemporaryDirectory() as tmp:
        venv = Path(tmp) / "venv"
        subprocess.run(["uv", "venv", "-q", str(venv)], check=True)
        python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run(["uv", "pip", "install", "-q", "--python", str(python), str(wheel)], check=True)
        cli = venv / ("Scripts/tinyharness.exe" if sys.platform == "win32" else "bin/tinyharness")
        version = subprocess.run([str(cli), "--version"], check=True, capture_output=True, text=True).stdout
        sdk = "import tinyharness as t; print(t.__version__, t.Session.__name__, t.load_config().provider)"
        out = subprocess.run(
            [str(python), "-c", sdk], check=True, capture_output=True, text=True, cwd=tmp
        ).stdout
    print(f"ok: {wheel.name} · cli: {version.strip()} · sdk: {out.strip()}")


if __name__ == "__main__":
    main()
