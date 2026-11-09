"""Encoding knobs live in the gallery section and drive the stored derivatives."""

from __future__ import annotations

import io

import pytest
from PIL import Image as PILImage

from core.image_format import preview_jpeg, thumbnail_webp
from core.image_store import ImageStore


def _png(width: int = 320, height: int = 200) -> bytes:
    out = io.BytesIO()
    PILImage.new("RGB", (width, height), (40, 90, 160)).save(out, format="PNG")
    return out.getvalue()


def _size(data: bytes) -> tuple[int, int]:
    with PILImage.open(io.BytesIO(data)) as image:
        return image.size


def _format(data: bytes) -> str:
    with PILImage.open(io.BytesIO(data)) as image:
        return image.format


def test_thumbnail_honours_size_quality_and_format():
    source = _png(800, 600)

    webp = thumbnail_webp(source, size=200, quality=50, fmt="webp")
    assert _format(webp) == "WEBP"
    assert max(_size(webp)) == 200

    jpeg = thumbnail_webp(source, size=120, quality=50, fmt="jpeg")
    assert _format(jpeg) == "JPEG"
    assert max(_size(jpeg)) == 120


def test_preview_honours_size_and_format():
    source = _png(1200, 900)

    jpeg = preview_jpeg(source, size=400, quality=70, fmt="jpeg")
    assert _format(jpeg) == "JPEG"
    assert max(_size(jpeg)) == 400

    webp = preview_jpeg(source, size=300, quality=70, fmt="webp")
    assert _format(webp) == "WEBP"
    assert max(_size(webp)) == 300


def test_store_uses_configured_suffixes_and_locked_widths(tmp_path):
    store = ImageStore(
        tmp_path,
        {
            "gallery": {
                "thumb_format": "jpeg",
                "preview_format": "webp",
                # These two are no longer settings; stale values must not win.
                "thumb_width": 160,
                "preview_size": 320,
            },
        },
    )

    assert store.thumb_path("abc").suffix == ".jpeg"
    assert store.preview_path("abc").suffix == ".webp"

    params = store.encoding_params()
    assert params["thumb_size"] == 480
    assert params["preview_size"] == 2048


def test_store_writes_derivatives_in_the_configured_format(tmp_path):
    store = ImageStore(
        tmp_path,
        {"gallery": {"thumb_format": "jpeg", "preview_format": "webp"}},
    )
    data = _png(600, 400)
    store.write_derivatives("img1", data)

    thumb = store.thumb_path("img1")
    preview = store.preview_path("img1")
    assert thumb.is_file() and preview.is_file()
    assert _format(thumb.read_bytes()) == "JPEG"
    assert _format(preview.read_bytes()) == "WEBP"


def test_encoding_version_changes_and_clears_derivatives(tmp_path):
    store = ImageStore(tmp_path, {"gallery": {}})
    before = store.encoding_version()
    store.write_derivatives("img1", _png(600, 400))
    thumb = store.thumb_path("img1")
    preview = store.preview_path("img1")
    assert thumb.is_file() and preview.is_file()

    assert store.update_config({"gallery": {"thumb_quality": 50}}) is True
    assert store.encoding_version() != before
    assert store.clear_derivatives() >= 2
    assert not thumb.exists() and not preview.exists()


def test_orphan_derivatives_are_removed(tmp_path):
    store = ImageStore(tmp_path, {"gallery": {}})
    store.write_derivatives("orphan", _png(600, 400))
    assert store.thumb_path("orphan").is_file()
    assert store.preview_path("orphan").is_file()

    assert store.prune_orphan_derivatives() >= 2
    assert not store.thumb_path("orphan").exists()
    assert not store.preview_path("orphan").exists()


def test_encoding_params_fall_back_to_defaults(tmp_path):
    store = ImageStore(tmp_path, {"gallery": {}})
    params = store.encoding_params()

    assert params == {
        "thumb_size": 480,
        "thumb_quality": 82,
        "thumb_format": "webp",
        "preview_size": 2048,
        "preview_quality": 90,
        "preview_format": "jpeg",
    }


def test_validate_image_bytes_respects_the_custom_cap():
    from core.image_format import validate_image_bytes

    data = _png(64, 64)
    validate_image_bytes(data, max_bytes=len(data) + 10)

    with pytest.raises(ValueError, match="MB"):
        validate_image_bytes(data, max_bytes=len(data) - 1)


def test_validate_image_bytes_rejects_a_pixel_bomb():
    from core.image_format import validate_image_bytes

    data = _png(64, 64)
    validate_image_bytes(data, max_pixels=64 * 64)

    with pytest.raises(ValueError, match="尺寸太大"):
        validate_image_bytes(data, max_pixels=64 * 64 - 1)
