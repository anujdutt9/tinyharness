"""Make an HTML page self-contained: replace each <script src="..."> that points at a
local file (relative to the page) with the file's contents, inline.

    python3 inline_scripts.py report.html            # rewrites report.html in place
    python3 inline_scripts.py report.html -o out.html

A Chart.js link to a CDN is swapped for the copy bundled with tinyharness, so a page
written with the usual CDN tag still ends up working offline. Runs on the standard
library only, so it works inside the sandbox.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

BUNDLED_CHART_JS = Path(__file__).with_name("chart.umd.min.js")
_SCRIPT = re.compile(r"<script\b([^>]*?)\bsrc\s*=\s*([\"'])([^\"']+)\2([^>]*)>\s*</script>", re.IGNORECASE)
_CHART_CDN = re.compile(
    r"^(https?:)?//.*\bchart(\.js)?\b.*\.js(\?.*)?$|^(https?:)?//.*/npm/chart\.js", re.IGNORECASE
)


def inline(html: str, base: Path) -> tuple[str, list[str], list[str]]:
    """The page with local scripts inlined; which sources were inlined; which were left."""
    done: list[str] = []
    left: list[str] = []

    def replace(match: re.Match[str]) -> str:
        before, src, after = match.group(1), match.group(3), match.group(4)
        if _CHART_CDN.match(src):
            path = BUNDLED_CHART_JS
        elif re.match(r"^[a-z][a-z0-9+.-]*:|^//", src, re.IGNORECASE):
            left.append(src)  # some other remote script: leave it to the author
            return match.group(0)
        else:
            path = (base / src).resolve()
        if not path.is_file():
            left.append(src)
            return match.group(0)
        code = path.read_text(encoding="utf-8").replace("</script", "<\\/script")
        done.append(src)
        attrs = (before + after).strip()
        return f"<script{' ' + attrs if attrs else ''}>\n{code}\n</script>"

    return _SCRIPT.sub(replace, html), done, left


def main() -> int:
    parser = argparse.ArgumentParser(description="Inline a page's local <script src> files")
    parser.add_argument("page", type=Path)
    parser.add_argument("-o", "--out", type=Path, help="write here instead of in place")
    args = parser.parse_args()
    page = args.page.resolve()
    html, done, left = inline(page.read_text(encoding="utf-8"), page.parent)
    out = args.out or page
    out.write_text(html, encoding="utf-8")
    print(f"inlined {len(done)} script(s) into {out}" + (f": {', '.join(done)}" if done else ""))
    if left:
        print(f"left as links (not local files): {', '.join(left)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
