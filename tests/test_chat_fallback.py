"""The chat backend hands over to the images API when the model refuses
chat-completions style generation."""

import io

import pytest
from PIL import Image as PILImage

from core.providers import openai_compat
from core.providers.base import ImageResult, ProviderSettings
from core.providers.openai_chat import OpenAIChatImageBackend


def _settings(**overrides) -> ProviderSettings:
    values = dict(
        id="chat-provider",
        template_key="openai_chat",
        base_url="https://example.test/v1",
        api_keys=["key-1", "key-2"],
        model="gemini-3-pro-image",
        max_retries=0,
        key_cooldown_seconds=300,
    )
    values.update(overrides)
    return ProviderSettings(**values)


def _png(width: int = 32, height: int = 32) -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", (width, height), (1, 2, 3)).save(buffer, format="PNG")
    return buffer.getvalue()


def _result() -> ImageResult:
    return ImageResult(data=_png(), mime="image/png", ext="png", size="1024x1024", model="m")


@pytest.mark.parametrize(
    "message",
    [
        "HTTP 400: this model does not support chat completions",
        "HTTP 400: image generation model, please use images/generations",
        "HTTP 500: get file base64 from url failed",
        "HTTP 400: image_url is required for image edits",
        "HTTP 400: unsupported protocol scheme",
    ],
)
def test_marker_errors_are_recognised(message):
    backend = OpenAIChatImageBackend(_settings())

    assert backend._should_fallback(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "HTTP 401: invalid api key",
        "HTTP 429: rate limited",
        "chat response contained no image",
        "connection reset by peer",
    ],
)
def test_other_errors_are_not_markers(message):
    backend = OpenAIChatImageBackend(_settings())

    assert backend._should_fallback(message) is False


def test_generate_falls_back_to_the_images_api(monkeypatch):
    backend = OpenAIChatImageBackend(_settings())
    calls = []

    async def failing_post_chat(**kwargs):
        raise RuntimeError("HTTP 400: does not support chat completions")

    async def fake_generate(self, prompt, *, intent=None, size_hint="", model="", extra=None):
        calls.append(prompt)
        return [_result()]

    monkeypatch.setattr(backend, "_post_chat", failing_post_chat)
    monkeypatch.setattr(openai_compat.OpenAICompatBackend, "generate", fake_generate)

    import asyncio

    results = asyncio.run(backend.generate("一只猫"))

    assert calls == ["一只猫"]
    assert len(results) == 1
    assert results[0].ext == "png"


def test_fallback_backend_is_created_once(monkeypatch):
    backend = OpenAIChatImageBackend(_settings())

    import asyncio

    first = asyncio.run(backend._fallback_backend())
    second = asyncio.run(backend._fallback_backend())

    assert first is second
    assert first.api_keys == ["key-1", "key-2"]


def test_non_marker_failure_is_raised(monkeypatch):
    backend = OpenAIChatImageBackend(_settings())

    async def failing_post_chat(**kwargs):
        raise RuntimeError("HTTP 401: invalid api key")

    monkeypatch.setattr(backend, "_post_chat", failing_post_chat)

    import asyncio

    with pytest.raises(RuntimeError, match="invalid api key"):
        asyncio.run(backend.generate("一只猫"))


def test_edit_falls_back_to_the_images_api(monkeypatch, tmp_path):
    backend = OpenAIChatImageBackend(_settings())
    calls = []
    image = tmp_path / "input.png"
    image.write_bytes(_png())

    async def failing_post_chat(**kwargs):
        raise RuntimeError("HTTP 400: image generation model, use images/generations")

    async def fake_edit(self, prompt, images, *, intent=None, size_hint="", model="", extra=None):
        calls.append((prompt, list(images)))
        return [_result()]

    monkeypatch.setattr(backend, "_post_chat", failing_post_chat)
    monkeypatch.setattr(openai_compat.OpenAICompatBackend, "edit", fake_edit)

    import asyncio

    results = asyncio.run(backend.edit("加点阳光", [image]))

    assert calls == [("加点阳光", [image])]
    assert len(results) == 1
