"""Image format and per-request parameter tests for the full-format work."""

from __future__ import annotations

import asyncio
import base64
import io

import pytest

from core.image_format import validate_image_bytes, validate_svg_bytes
from core.image_params import build_image_params
from core.provider_stats import ProviderStats
from core.providers.base import FormatMismatchError, ImageResult, ProviderSettings
from core.providers.openai_compat import OpenAICompatBackend
from core.task_store import TaskRecord
from test_plugin_lifecycle import _Ctx, _config, plugin_module  # noqa: F401


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


def _svg() -> bytes:
    return (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
        b'<rect width="10" height="10"/></svg>'
    )


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
    with pytest.raises(ValueError, match="红笔标注"):
        build_image_params(output_format="svg", markup="把左边改成红色")


def test_image_params_keep_quality_optional():
    assert "quality" not in build_image_params()
    assert build_image_params(quality="high") == {"quality": "high"}
    with pytest.raises(ValueError):
        build_image_params(quality="highest")


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
async def test_requested_jpeg_keeps_png_bytes():
    """No PNG-to-JPEG conversion: the provider's bytes are what we keep."""
    backend = _backend()
    png = _png()

    results = await backend._extract_images(
        {"data": [{"b64_json": _b64(png)}]},
        "gpt-image-2.6",
        "1024x1024",
        "jpeg",
    )

    assert results[0].data == png
    assert results[0].mime == "image/png"
    assert results[0].ext == "png"


@pytest.mark.asyncio
async def test_svg_request_sends_only_the_documented_switches():
    backend = _backend()
    seen: dict = {}

    async def fake_post_json(url, key, payload, *, retries=0):
        seen["url"] = url
        seen["payload"] = payload
        return {"data": [{"b64_json": _b64(_svg())}]}

    backend._post_json = fake_post_json
    params = build_image_params(
        output_format="svg",
        n=4,
        output_compression=80,
        frame_count=4,
    )
    results = await backend.generate("把这张照片画成矢量图", extra=params)

    assert seen["payload"]["output_format"] == "svg"
    assert "n" not in seen["payload"]
    assert "output_compression" not in seen["payload"]
    assert "svg" not in seen["payload"]
    assert results[0].mime == "image/svg+xml"


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


# ----------------------------------------------------------------------
# gif/svg chains: no provider may be dropped before the request
# ----------------------------------------------------------------------

class _ChainSettings:
    """Provider settings without any gif/svg capability switch."""

    def __init__(self) -> None:
        self.max_concurrent = 0
        self.default_size = "1024x1024"


class _ChainRegistry:
    def __init__(self, backends: dict) -> None:
        self._backends = backends
        self._settings = {provider_id: _ChainSettings() for provider_id in backends}

    def get_backend(self, provider_id):
        if provider_id not in self._backends:
            raise KeyError(provider_id)
        return self._backends[provider_id]

    def get_settings(self, provider_id):
        if provider_id not in self._settings:
            raise KeyError(provider_id)
        return self._settings[provider_id]


class _ChainGallery:
    def __init__(self) -> None:
        self.saved: list[bytes] = []

    async def save_image(self, data, **kwargs):
        self.saved.append(data)
        return f"media/fake-{len(self.saved)}.gif"


class _ChainStore:
    def set_audit(self, *args, **kwargs):
        return None


class _ChainBackend:
    """Records the extra payload each chain step handed to the provider."""

    supports_edit = True

    def __init__(self, *, result=None, error=None) -> None:
        self.result = list(result or [])
        self.error = error
        self.calls: list[dict] = []

    async def generate(self, prompt, *, intent=None, size_hint="", model="", extra=None):
        self.calls.append(dict(extra or {}))
        if self.error is not None:
            raise self.error
        return list(self.result)

    async def edit(self, prompt, images, *, intent=None, size_hint="", model="", extra=None):
        return await self.generate(prompt, extra=extra)


def _chain_plugin(plugin_module, tmp_path, backends):
    plugin = plugin_module.AIImagePlugin(_Ctx(tmp_path), _config())
    plugin._data_dir = tmp_path
    plugin._registry = _ChainRegistry(backends)
    plugin._stats = ProviderStats(tmp_path / "stats", threshold=3, cooldown=60)
    plugin._gallery = _ChainGallery()
    plugin._store = _ChainStore()
    return plugin


def _gif_record(chain):
    return TaskRecord(
        task_id="t-gif",
        kind="generate_image",
        prompt="用 aiimg 做一张循环动画 GIF，9 帧",
        chain=list(chain),
        extra={
            "chain_items": [{"provider": p, "output": ""} for p in chain],
            "image_params": {
                "output_format": "gif",
                "frame_count": 9,
                "frame_duration_ms": 200,
            },
        },
    )


def test_provider_settings_expose_no_gif_svg_capability_switches():
    """0.5.2 deleted these switches; a chain filter reading them can only fail."""
    settings = ProviderSettings(id="mock", template_key="openai_images")
    assert not hasattr(settings, "supports_gif")
    assert not hasattr(settings, "supports_svg")


def test_gif_request_tries_every_provider_instead_of_pre_filtering(plugin_module, tmp_path):
    first = _ChainBackend(error=FormatMismatchError("服务商 first 返回 image/png，请求的是 gif"))
    second = _ChainBackend(error=FormatMismatchError("服务商 second 返回 image/png，请求的是 gif"))
    plugin = _chain_plugin(plugin_module, tmp_path, {"first": first, "second": second})

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(
            plugin._run_image_chain(_gif_record(["first", "second"]), kind="generate")
        )

    # Both providers were asked for a GIF, with the request parameters intact.
    assert len(first.calls) == 1
    assert len(second.calls) == 1
    assert first.calls[0]["output_format"] == "gif"
    assert second.calls[0]["output_format"] == "gif"
    assert second.calls[0]["frame_count"] == 9
    # The failure is the chain summary, not a missing-capability refusal.
    message = str(excinfo.value)
    assert "所有服务商均失败" in message
    assert "没有服务商支持" not in message


def test_gif_request_reaches_the_provider_that_returns_a_gif(plugin_module, tmp_path):
    plain = _ChainBackend(error=FormatMismatchError("服务商 plain 返回 image/png，请求的是 gif"))
    animated = _ChainBackend(
        result=[
            ImageResult(
                data=_gif(), mime="image/gif", ext="gif", size="512x512", model="mock"
            )
        ]
    )
    plugin = _chain_plugin(plugin_module, tmp_path, {"plain": plain, "animated": animated})

    result = asyncio.run(
        plugin._run_image_chain(_gif_record(["plain", "animated"]), kind="generate")
    )

    assert result["provider"] == "animated"
    assert len(result["media"]) == 1
    assert plain.calls[0]["output_format"] == "gif"
    assert animated.calls[0]["output_format"] == "gif"
