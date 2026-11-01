"""ProviderRegistry config loading, validation and backend construction tests."""

import asyncio
from pathlib import Path

from core.provider_registry import ProviderRegistry, normalize_template_key


def _registry(tmp_path: Path, providers: list[dict]) -> ProviderRegistry:
    return ProviderRegistry({"providers": providers}, data_dir=tmp_path)


def test_load_image_provider(tmp_path):
    registry = _registry(
        tmp_path,
        [
            {
                "id": "p1",
                "__template_key": "openai_images",
                "base_url": "https://ai.example.com/v1",
                "api_keys": ["sk-1"],
                "model": "z-image-turbo",
            }
        ],
    )
    assert registry.validate() == []
    assert registry.image_provider_ids() == ["p1"]
    settings = registry.get_settings("p1")
    assert settings.template_key == "openai_images"
    assert settings.api_keys == ["sk-1"]
    assert settings.model == "z-image-turbo"
    assert "1024x1024" in settings.allowed_sizes
    asyncio.run(registry.close())


def test_alias_openai(tmp_path):
    registry = _registry(
        tmp_path,
        [{"id": "p", "__template_key": "openai", "base_url": "https://x/v1", "model": "m"}],
    )
    assert registry.get_settings("p").template_key == "openai_images"


def test_template_key_from_id_fallback(tmp_path):
    registry = _registry(
        tmp_path,
        [{"id": "openai_full_url", "full_generate_url": "https://x/v1/generate", "model": "m"}],
    )
    assert registry.get_settings("openai_full_url").template_key == "openai_full_url_images"


def test_normalize_template_key():
    assert normalize_template_key("openai_compat") == "openai_images"
    assert normalize_template_key("openai_full_url") == "openai_full_url_images"
    assert normalize_template_key("") == ""


def test_missing_required(tmp_path):
    registry = _registry(tmp_path, [{"id": "p", "__template_key": "openai_images"}])
    assert registry.has("p") is False
    assert any("缺少必填" in e for e in registry.validate())


def test_unknown_template(tmp_path):
    registry = _registry(tmp_path, [{"id": "p", "__template_key": "weird", "base_url": "https://x"}])
    assert any("未知模板" in e for e in registry.validate())


def test_providers_not_list(tmp_path):
    registry = ProviderRegistry({"providers": "nope"}, data_dir=tmp_path)
    assert any("不是数组" in e for e in registry.validate())


def test_video_provider_backend(tmp_path):
    registry = _registry(
        tmp_path,
        [
            {
                "id": "v",
                "__template_key": "agnes_video",
                "api_keys": ["k"],
                "base_url": "https://api.agnes.test/v1",
                "model": "agnes-video-2.5-flash",
            }
        ],
    )
    assert registry.video_provider_ids() == ["v"]
    backend = registry.get_video_backend("v")
    assert backend.provider_id == "v"
    asyncio.run(registry.close())


def test_list_providers(tmp_path):
    registry = _registry(
        tmp_path,
        [
            {
                "id": "p",
                "__template_key": "openai_images",
                "base_url": "https://x/v1",
                "model": "m",
                "supports_edit": False,
            }
        ],
    )
    item = registry.list_providers()[0]
    assert item["id"] == "p"
    assert item["capabilities"] == ["generate"]
    assert item["has_keys"] is False


def test_list_settings_roundtrip(tmp_path):
    registry = _registry(
        tmp_path,
        [
            {
                "id": "p",
                "__template_key": "openai_images",
                "base_url": "https://x/v1",
                "api_keys": ["sk-1"],
                "model": "m",
            }
        ],
    )
    settings = registry.list_settings()[0]
    assert settings["id"] == "p"
    assert settings["template_key"] == "openai_images"
    assert settings["api_keys"] == ["sk-1"]
    assert settings["model"] == "m"


def test_get_backend_unknown_raises(tmp_path):
    registry = _registry(tmp_path, [])
    try:
        registry.get_backend("nope")
        assert False, "expected KeyError"
    except KeyError:
        pass
