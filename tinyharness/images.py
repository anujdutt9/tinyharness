"""Keeping images within what every provider accepts.

Anthropic caps each image at 2576 px a side once a request carries many of them (and at
5 MB); a Retina screenshot is often 3000+ px wide. A rejected image stays in the history
and fails every later request, so images are shrunk before they go in: at most MAX_SIDE
on their longest side and MAX_BYTES in size. One already within both is left as it is.
"""

from __future__ import annotations

import base64
import io

from PIL import Image as PILImage
from PIL import UnidentifiedImageError

from tinyharness.agent.history import Image

MAX_SIDE = 2000  # px; under every supported provider's limit
MAX_BYTES = 5_000_000  # the strictest provider's per-image limit (Anthropic)
MAX_SOURCE_BYTES = 50_000_000  # bigger files aren't decoded at all

_FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP", "image/gif": "PNG"}


def fit(data: bytes, media_type: str, max_side: int = MAX_SIDE) -> tuple[bytes, str]:
    """`data` shrunk to at most `max_side` px a side and MAX_BYTES, with its media type
    (a GIF comes back as its first frame, in PNG). Unchanged if it already fits.
    Raises ValueError if it can't be decoded or is far too big to try."""
    if len(data) > MAX_SOURCE_BYTES:
        raise ValueError(f"{len(data) / 1_000_000:.0f} MB is too large for an image to look at")
    try:
        with PILImage.open(io.BytesIO(data)) as picture:
            width, height = picture.size
            if max(width, height) <= max_side and len(data) <= MAX_BYTES and media_type != "image/gif":
                return data, media_type
            picture.seek(0)  # an animation: its first frame
            scale = min(1.0, max_side / max(width, height))
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            shrunk = picture.convert("RGBA" if _has_alpha(picture) else "RGB").resize(
                size, PILImage.Resampling.LANCZOS
            )
    except (UnidentifiedImageError, OSError) as e:
        raise ValueError(f"not a readable image ({e})") from None

    fmt = _FORMATS.get(media_type, "PNG")
    out = _save(shrunk, fmt)
    if len(out) > MAX_BYTES:  # a big photo saved as PNG: JPEG is far smaller
        fmt, out = "JPEG", _save(shrunk.convert("RGB"), "JPEG")
    return out, f"image/{fmt.lower()}"


def fit_image(image: Image, max_side: int = MAX_SIDE) -> Image:
    """An Image (base64) shrunk like `fit`; the same object if it already fits."""
    raw = base64.b64decode(image.data)
    data, media_type = fit(raw, image.media_type, max_side)
    if data is raw:
        return image
    return Image(media_type, base64.b64encode(data).decode("ascii"), image.name)


def _has_alpha(picture: PILImage.Image) -> bool:
    return picture.mode in ("RGBA", "LA") or (picture.mode == "P" and "transparency" in picture.info)


def _save(picture: PILImage.Image, fmt: str) -> bytes:
    buffer = io.BytesIO()
    if fmt == "JPEG":
        picture.convert("RGB").save(buffer, "JPEG", quality=85, optimize=True)
    elif fmt == "WEBP":
        picture.save(buffer, "WEBP", quality=85)
    else:
        picture.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()
