"""Image format and per-request parameter tests for the full-format work."""

from __future__ import annotations

import base64
import io

import pytest

from core.image_format import validate_image_bytes, validate_svg_bytes
from core.image_params import build_image_params
from core.providers.base import FormatMismatchError, ProviderSettings
from core.providers.openai_compat import OpenAICompatBackend


def _backend() -> OpenAICompatBackend:
    return OpenAICompatBackend(
        ProviderSettings(
            id="mock",
            template_key="openai_images",
            base_url="http://127.0.0.1:9/v1",
            api_keys=["sk-test"],
            model="gpt-image-2.6",
            max_retries=0,
            timeout=1,
        )
    )


def _png() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    image = Image.effect_noise((64, 64), 100).convert("RGB")
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _gif() -> bytes:
    from PIL import Image

    frames = [
        Image.effect_noise((64, 64), 100 + index).convert("P") for index in range(3)
    ]
    buffer = io.BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
    )
    return buffer.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_image_params_validate_and_default_gif_frames():
    params = build_image_params(output_format="GIF", n=2)
    assert params["output_format"] == "gif"
    assert params["n"] == 2
    assert params["frame_count"] == 9
    assert params["frame_duration_ms"] == 200

    svg = build_image_params(output_format="svg", n=4, frame_count=4)
    assert svg == {"output_format": "svg"}

    with pytest.raises(ValueError):
        build_image_params(output_format="bmp")
    with pytest.raises(ValueError):
        build_image_params(n=11)
    with pytest.raises(ValueError):
        build_image_params(frame_count=1, output_format="gif")
    with pytest.raises(ValueError):
        build_image_params(
            output_format="gif", formats=("png", "jpeg", "webp")
        )


def test_image_params_keep_zero_compression_and_markup():
    params = build_image_params(
        output_format="webp", output_compression=0, markup="把左边改成红色"
    )
    assert params["output_compression"] == 0
    assert params["markup"] == "把左边改成红色"


@pytest.mark.asyncio
async def test_extract_images_walks_every_item():
    backend = _backend()
    response = {
        "data": [
            {"b64_json": _b64(_png())},
            {"b64_json": _b64(_png())},
        ]
    }
    results = await backend._extract_images(response, "gpt-image-2.6", "1024x1024")
    assert len(results) == 2
    assert all(item.mime == "image/png" for item in results)


@pytest.mark.asyncio
async def test_gif_is_accepted_and_png_is_rejected_for_gif():
    backend = _backend()
    gif = await backend._extract_images(
        {"data": [{"b64_json": _b64(_gif())}]},
        "gpt-image-2.6",
        "1024x1024",
        "gif",
    )
    assert gif[0].mime == "image/gif"
    assert gif[0].ext == "gif"

    with pytest.raises(FormatMismatchError):
        await backend._extract_images(
            {"data": [{"b64_json": _b64(_png())}]},
            "gpt-image-2.6",
            "1024x1024",
            "gif",
        )


@pytest.mark.asyncio
async def test_svg_is_accepted_only_when_expected():
    backend = _backend()
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>'
    results = await backend._extract_images(
        {"data": [{"b64_json": _b64(svg)}]},
        "gpt-image-2.6",
        "1024x1024",
        "svg",
    )
    assert results[0].mime == "image/svg+xml"
    assert results[0].ext == "svg"

    with pytest.raises(FormatMismatchError):
        await backend._extract_images(
            {"data": [{"b64_json": _b64(_png())}]},
            "gpt-image-2.6",
            "1024x1024",
            "svg",
        )


def test_svg_scripts_are_refused():
    with pytest.raises(ValueError):
        validate_svg_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>')
    with pytest.raises(ValueError):
        validate_image_bytes(
            b'<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>',
            allow_svg=False,
        )
