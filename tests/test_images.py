"""Images are kept within every provider's limits, and a rejected one doesn't break the session."""

import base64
import io

import pytest
from PIL import Image as PILImage

from conftest import ScriptedLLM, make_ctx, png
from tinyharness.agent.compaction import Compactor
from tinyharness.agent.history import History, Image, Message
from tinyharness.agent.loop import LoopDeps, run_agent_loop
from tinyharness.agent.transcript import Transcript
from tinyharness.images import MAX_BYTES, fit, fit_image
from tinyharness.llm.client import _http_error
from tinyharness.llm.types import ImageTooLargeError
from tinyharness.permissions import PermissionGate, PermissionMode
from tinyharness.tools.registry import ToolRegistry
from tinyharness.workspace import Workspace


def size(data: bytes) -> tuple[int, int]:
    return PILImage.open(io.BytesIO(data)).size


def as_image(data: bytes, name: str = "shot.png") -> Image:
    return Image("image/png", base64.b64encode(data).decode("ascii"), name)


def test_an_image_that_fits_is_left_alone():
    data = png(800, 600)
    assert fit(data, "image/png") == (data, "image/png")
    image = as_image(data)
    assert fit_image(image) is image


def test_the_longest_side_is_shrunk_to_2000_keeping_the_shape():
    out, kind = fit(png(3024, 1964), "image/png")
    assert kind == "image/png" and size(out) == (2000, 1299)
    out, _ = fit(png(1000, 4000), "image/png")
    assert size(out) == (500, 2000)


def test_a_photo_too_big_as_png_becomes_jpeg():
    out, kind = fit(png(1900, 1900, noise=True), "image/png")  # ~10 MB of noise as PNG
    assert kind == "image/jpeg" and len(out) <= MAX_BYTES


def test_a_gif_is_sent_as_its_first_frame():
    frames = [PILImage.new("RGB", (40, 40), color) for color in ((255, 0, 0), (0, 0, 255))]
    buffer = io.BytesIO()
    frames[0].save(buffer, "GIF", save_all=True, append_images=frames[1:])
    out, kind = fit(buffer.getvalue(), "image/gif")
    assert kind == "image/png" and PILImage.open(io.BytesIO(out)).convert("RGB").getpixel((0, 0)) == (
        255,
        0,
        0,
    )


def test_a_broken_image_is_an_error():
    with pytest.raises(ValueError, match="not a readable image"):
        fit(b"\x89PNG\r\n\x1a\n" + bytes(64), "image/png")


def test_the_providers_image_error_is_recognized():
    """A routed provider response containing an oversized image error."""
    body = (
        '{"error":{"message":"Provider returned error","code":400,"metadata":{"raw":"{\\"type\\":\\"error\\",'
        '\\"error\\":{\\"type\\":\\"invalid_request_error\\",\\"message\\":\\"messages.12.content.1.image.source.'
        "base64.data: At least one of the image dimensions exceed max allowed size for many-image requests: "
        '2576 pixels\\"}}","provider_name":"Google"}}}'
    )
    assert isinstance(
        _http_error(400, body, "https://openrouter.ai/api/v1/chat/completions"), ImageTooLargeError
    )


async def run_with(tmp_path, llm, history):
    with Transcript(tmp_path, "test") as transcript:
        deps = LoopDeps(
            llm=llm,
            tools=ToolRegistry([]),
            gate=PermissionGate(workspace=Workspace(tmp_path), mode=PermissionMode.BYPASS),
            compactor=Compactor(llm),
            transcript=transcript,
            ctx=make_ctx(tmp_path),
        )
        return [e async for e in run_agent_loop("what changed?", history, deps)]


class RejectsBigImages(ScriptedLLM):
    """Like the provider: any image over 1600 px a side fails the whole request."""

    async def stream(self, messages, tools=None, **kw):
        self.requests.append(list(messages))
        if any(max(size(base64.b64decode(i.data))) > 1600 for m in messages for i in m.images):
            raise ImageTooLargeError("image dimensions exceed max allowed size")
        self.requests.pop()
        async for chunk in super().stream(messages, tools, **kw):
            yield chunk


async def test_a_rejected_image_is_shrunk_and_the_request_retried(tmp_path):
    big = as_image(png(2000, 1000))
    history = History(
        [Message(role="system", content="s"), Message(role="user", content="slide 1", images=[big])]
    )
    llm = RejectsBigImages(["Nothing broke."])
    events = await run_with(tmp_path, llm, history)
    assert any(getattr(e, "text", "") == "Nothing broke." for e in events)
    assert size(base64.b64decode(history.messages[1].images[0].data)) == (1568, 784)


async def test_images_are_dropped_when_shrinking_cant_help(tmp_path):
    """If the images are already small, the limit is something else: drop them, with a note,
    rather than fail every request from now on."""

    class RejectsAnyImage(ScriptedLLM):
        async def stream(self, messages, tools=None, **kw):
            if any(m.images for m in messages):
                raise ImageTooLargeError("image exceeds the limit")
            async for chunk in super().stream(messages, tools, **kw):
                yield chunk

    history = History(
        [
            Message(role="system", content="s"),
            Message(role="user", content="look", images=[as_image(png(100, 100))]),
        ]
    )
    await run_with(tmp_path, RejectsAnyImage(["ok"]), history)
    assert history.messages[1].images == []
    assert "[removed: shot.png; the model rejected it as too large]" in history.messages[1].content


async def test_a_model_that_takes_no_images_is_asked_again_with_notes(tmp_path):
    """vLLM serving a model text-only refuses any image ("At most 0 image(s)"): the loop marks the
    model, and from then on its requests carry a note in place of each image (Exp-1A attempt 3:
    the lead's screenshot of its own HTML report ended the turn)."""
    from tinyharness.agent.history import without_images
    from tinyharness.llm.types import ImagesUnsupportedError

    class TextOnly(ScriptedLLM):
        def __init__(self, turns):
            super().__init__(turns)
            self.no_images = set()

        async def stream(self, messages, tools=None, **kw):
            if self.model in self.no_images:
                messages = without_images(messages)
            if any(m.images for m in messages):
                raise ImagesUnsupportedError("At most 0 image(s) may be provided in one prompt.")
            async for chunk in super().stream(messages, tools, **kw):
                yield chunk

    history = History(
        [
            Message(role="system", content="s"),
            Message(role="user", content="check it", images=[as_image(png(10, 10))]),
        ]
    )
    llm = TextOnly(["I can't see images; the HTML validates."])
    events = await run_with(tmp_path, llm, history)
    assert any(getattr(e, "text", "") == "I can't see images; the HTML validates." for e in events)
    assert llm.no_images == {"fake"}
    assert "Cannot read shot.png" in llm.requests[-1][1].content
