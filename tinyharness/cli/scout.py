"""Scout, the tinyharness pony, as terminal pixel art.

The README draws Scout in full; here he's a 20 × 20 head-and-shoulders portrait (his face
is what carries him at this size), drawn with half-block characters: each character
cell shows two pixels, the top one as the foreground of "▀" and the bottom one as its
background. That needs a terminal with real colors, so `fits` says when to show him.
"""

from __future__ import annotations

from rich.console import Console
from rich.text import Text

# One letter per pixel; "." is transparent.
PIXELS = (
    "...kk..........kk...",
    "..kbwk..kkkk..kwbk..",
    "..kbwkkkBBBBkkkwbk..",
    "..kbkBBBBBBBBBBkbk..",
    ".kkBBBBBBBBBBBBBBkk.",
    ".kBBBBBBBBBBBBBBBBk.",
    "kBBBBBBBBBwwBBwBBBBk",
    "kBBBwBBBwwwwwwwwBBBk",
    "kBBwwBBwwwwwwwwwwBBk",
    "kBwwwkkwwwwwwkkwwwBk",
    "kBwwwkkwwwwwwkkwwwBk",
    ".kwwwwwwwwwwwwwwwwk.",
    ".kwpwwwwbbbbwwwwpwk.",
    "..kwwwwbbbbbbwwwwk..",
    "..kwwwbbkbbkbbwwwk..",
    "...kwwbbbkkbbbwwk...",
    "....kkwwwwwwwwkk....",
    "...kwwwwwwwwwwwwk...",
    ".kyyywwwwwwwwwwyyyk.",
    "kwwwyyyyyyyyyyyywwwk",
)
COLORS = {
    "k": "#17212c",  # outline (on a dark terminal it recedes, and the fills carry him)
    "B": "#1f5f99",  # mane
    "b": "#8ec2ec",  # muzzle, inner ears
    "w": "#fafcfd",  # coat
    "p": "#f08c82",  # cheeks
    "y": "#ffd640",  # harness
}
WIDTH, HEIGHT = len(PIXELS[0]), (len(PIXELS) + 1) // 2  # in character cells


def fits(console: Console) -> bool:
    """Whether to draw Scout: a color terminal (truecolor or 256 colors) with room beside him."""
    return console.color_system in ("truecolor", "256") and console.width >= 72


def portrait() -> Text:
    """Scout, one line of half-blocks per two rows of pixels."""
    out = Text()
    rows = list(PIXELS) + (["." * WIDTH] if len(PIXELS) % 2 else [])
    for y in range(0, len(rows), 2):
        if y:
            out.append("\n")
        for top, bottom in zip(rows[y], rows[y + 1], strict=True):
            if top == "." and bottom == ".":
                out.append(" ")
            elif top == ".":
                out.append("▄", style=COLORS[bottom])
            elif bottom == ".":
                out.append("▀", style=COLORS[top])
            else:
                out.append("▀", style=f"{COLORS[top]} on {COLORS[bottom]}")
    return out
