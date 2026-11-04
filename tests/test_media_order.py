"""媒体顺序回归：一次请求同时带图片和文字时，图片必须排在文字之前。

三种传输形式分别覆盖：
- 对话消息数组（openai_chat）：内容块中所有图片在前、文字最后；
- 完整地址接口（openai_full_url）：表单字段与 JSON 请求体都是先图后文；
- 图片编辑接口（openai_compat）：图片与提示词是独立字段，只校验两者都在场。
"""

import asyncio
import io

import pytest
from PIL import Image as PILImage

from core.providers import openai_chat
from core.providers.base import ProviderSettings
from core.providers.openai_chat import OpenAIChatImageBackend, build_media_parts
from core.providers.openai_full_url import (
    OpenAIFullURLBackend,
    build_edit_multipart,
    build_edit_multipart_fields,
)

PNG_MIME = "image/png"


def _png(color=(10, 20, 30), size=(8, 8)) -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _settings(**overrides) -> ProviderSettings:
    values = dict(
        id="media-order",
        template_key="openai_chat",
        base_url="https://example.test/v1",
        api_keys=["key-1"],
        model="image-model",
        max_retries=0,
    )
    values.update(overrides)
    return ProviderSettings(**values)


def _form_field_names(form) -> list[str]:
    """读出 multipart 表单的字段顺序（aiohttp 把 name 放在字段选项里）。"""
    names: list[str] = []
    for entry in getattr(form, "_fields", []):
        options = entry[0]
        names.append(str(options.get("name") or ""))
    return names


def _write_images(tmp_path, count=2):
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
    paths = []
    for index in range(count):
        path = tmp_path / f"input_{index + 1}.png"
        path.write_bytes(_png(colors[index % len(colors)]))
        paths.append(path)
    return paths


class _FakeResponse:
    def __init__(self, body: bytes, headers: dict | None = None):
        self._body = body
        self.status = 200
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def read(self):
        return self._body


class _FakeSession:
    def __init__(self, response: _FakeResponse):
        self.response = response
        self.forms = []
        self.requests = []

    def post(self, url, **kwargs):
        self.requests.append(url)
        self.forms.append(kwargs.get("data"))
        return self.response


# -- 对话消息数组 -----------------------------------------------------------


def test_chat_parts_put_every_image_before_the_text():
    parts = build_media_parts("把这两张图合成一张", [_png((255, 0, 0)), _png((0, 0, 255))])

    types = [part["type"] for part in parts]
    text_indexes = [index for index, kind in enumerate(types) if kind == "text"]
    image_indexes = [index for index, kind in enumerate(types) if kind == "image_url"]

    assert text_indexes == [len(parts) - 1]  # 文字只有一个，且在最后
    assert image_indexes == [0, 1]  # 两张图都在最前面
    assert all(index < text_indexes[0] for index in image_indexes)
    assert parts[-1]["text"] == "把这两张图合成一张"
    assert parts[0]["image_url"]["url"].startswith(f"data:{PNG_MIME};base64,")


@pytest.mark.parametrize("model", ["gemini-3-pro-image", "gpt-image-1", "z-image-turbo"])
def test_chat_edit_request_sends_images_before_text(model, tmp_path, monkeypatch):
    """实际请求里也是先图后文；Gemini 模型同样走这套消息数组。"""
    backend = OpenAIChatImageBackend(_settings(model=model))
    images = _write_images(tmp_path)
    captured = {}

    async def capture(**kwargs):
        captured["messages"] = kwargs["messages"]
        raise RuntimeError("HTTP 401: invalid api key")

    monkeypatch.setattr(backend, "_post_chat", capture)

    with pytest.raises(RuntimeError, match="invalid api key"):
        asyncio.run(backend.edit("加点阳光", images))

    content = captured["messages"][0]["content"]
    types = [part["type"] for part in content]
    assert types == ["image_url", "image_url", "text"]
    assert content[-1]["text"] == "加点阳光"


def test_gemini_native_shape_keeps_images_first_when_present():
    """当前没有 Gemini 原生请求体；一旦加上，这条用例自动接管并检查顺序。"""
    builder = getattr(openai_chat, "build_gemini_parts", None)
    if builder is None:
        pytest.skip("当前实现没有 Gemini 原生请求体（contents/parts），Gemini 复用对话消息数组")

    body = builder("描述这张图", [_png((255, 0, 0))])
    parts = body["parts"] if isinstance(body, dict) else body
    image_indexes = [
        index
        for index, part in enumerate(parts)
        if isinstance(part, dict) and ("inlineData" in part or "inline_data" in part)
    ]
    text_indexes = [
        index
        for index, part in enumerate(parts)
        if isinstance(part, dict) and "text" in part
    ]

    assert text_indexes and image_indexes
    assert max(image_indexes) < min(text_indexes)


# -- 完整地址接口：表单 -----------------------------------------------------


def test_full_url_multipart_fields_put_images_before_the_prompt(tmp_path):
    images = _write_images(tmp_path)
    payload = {"model": "image-model", "prompt": "加点阳光", "size": "1024x1024"}

    fields = build_edit_multipart_fields(payload, images)
    names = [name for name, _value, _options in fields]

    assert names.count("image") == len(images)
    assert "prompt" in names
    prompt_index = names.index("prompt")
    image_indexes = [index for index, name in enumerate(names) if name == "image"]
    assert image_indexes and max(image_indexes) < prompt_index
    # 图片字段带文件数据，文字字段保持文本
    assert all(isinstance(fields[index][1], bytes) for index in image_indexes)
    assert fields[prompt_index][1] == "加点阳光"


def test_full_url_multipart_form_keeps_images_first(tmp_path):
    images = _write_images(tmp_path, count=1)

    form = build_edit_multipart({"model": "image-model", "prompt": "再亮一点"}, images)
    names = _form_field_names(form)

    assert names.count("image") == 1
    assert names[0] == "image"
    assert names.index("prompt") > names.index("image")


def test_full_url_edit_multipart_really_builds_images_first(tmp_path, monkeypatch):
    """真实编辑流程发出的表单同样是先图后文。"""
    endpoint = "https://example.test/v1/edit"
    backend = OpenAIFullURLBackend(_settings(full_edit_url=endpoint))
    images = _write_images(tmp_path, count=1)
    session = _FakeSession(
        _FakeResponse(_png((1, 2, 3), size=(16, 16)), {"Content-Type": PNG_MIME})
    )

    async def session_get():
        return session

    monkeypatch.setattr(backend, "_session_get", session_get)

    results = asyncio.run(
        backend._edit_multipart(
            endpoint,
            {"model": "image-model", "prompt": "加点阳光", "size": "1024x1024"},
            images,
            "image-model",
            "1024x1024",
        )
    )

    names = _form_field_names(session.forms[0])
    assert names[0] == "image"
    assert names.index("prompt") > 0
    assert [result.mime for result in results] == [PNG_MIME]


# -- 完整地址接口：JSON 请求体 ----------------------------------------------


def test_full_url_json_body_puts_image_keys_before_the_prompt(tmp_path, monkeypatch):
    """JSON 分支按字段出现顺序解析，图片键必须排在 prompt 之前。"""
    backend = OpenAIFullURLBackend(_settings(full_edit_url="https://example.test/v1/edit"))
    images = _write_images(tmp_path, count=1)
    payloads = []

    async def fake_post_json(url, payload):
        payloads.append(payload)
        return {"data": [{"url": "https://example.test/out.png"}]}

    async def fake_save_ref(raw, endpoint, model, size):
        from core.providers.base import ImageResult

        return ImageResult(data=_png(), mime=PNG_MIME, ext="png", size=size, model=model)

    monkeypatch.setattr(backend, "_post_json", fake_post_json)
    monkeypatch.setattr(backend, "_save_ref", fake_save_ref)

    asyncio.run(backend.edit("加点阳光", images))

    keys = list(payloads[0].keys())
    image_keys = [key for key in keys if "image" in key or key == "object_refs"]
    assert image_keys, keys
    assert "prompt" in keys
    assert max(keys.index(key) for key in image_keys) < keys.index("prompt")


# -- 图片编辑接口：独立字段 -------------------------------------------------


def test_compat_edit_always_carries_image_and_prompt(tmp_path, monkeypatch):
    """images/edits 的图片与提示词是分开的字段，顺序无关，但在场是硬要求。"""
    from core.providers.openai_compat import OpenAICompatBackend

    backend = OpenAICompatBackend(_settings(model="z-image-turbo"))
    images = _write_images(tmp_path, count=1)
    captured = {}

    async def fake_post_form(url, key, fields, files, *, retries=0):
        captured["fields"] = dict(fields)
        captured["files"] = list(files)
        raise RuntimeError("HTTP 401: invalid api key")

    monkeypatch.setattr(backend, "_post_form", fake_post_form)

    with pytest.raises(RuntimeError, match="invalid api key"):
        asyncio.run(backend.edit("加点阳光", images))

    assert captured["fields"]["prompt"] == "加点阳光"
    assert captured["files"]
    assert all(entry[0] == "image" or entry[0].startswith("reference_") for entry in captured["files"])
