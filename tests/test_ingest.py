"""Ingest safety: EXIF orientation, duplicate detection and placeholder rules."""

from __future__ import annotations

import asyncio
import io

from PIL import Image as PILImage

from core.image_format import (
    content_digest,
    is_placeholder_image,
    normalize_orientation,
)
from core.image_store import ImageStore


def _jpeg(width: int, height: int, orientation: int | None = None) -> bytes:
    out = io.BytesIO()
    image = PILImage.new("RGB", (width, height), (30, 60, 90))
    if orientation is None:
        image.save(out, format="JPEG")
    else:
        exif = image.getexif()
        exif[0x0112] = orientation
        image.save(out, format="JPEG", exif=exif)
    return out.getvalue()


def _png(width: int = 24, height: int = 24) -> bytes:
    out = io.BytesIO()
    PILImage.new("RGB", (width, height), (200, 120, 40)).save(out, format="PNG")
    return out.getvalue()


def _size(data: bytes) -> tuple[int, int]:
    with PILImage.open(io.BytesIO(data)) as image:
        return image.size


def test_orientation_is_baked_in_and_tag_dropped():
    data = _jpeg(40, 20, orientation=6)

    fixed = normalize_orientation(data)

    assert fixed is not data
    assert _size(fixed) == (20, 40)
    with PILImage.open(io.BytesIO(fixed)) as image:
        assert not image.getexif().get(0x0112)


def test_upright_images_pass_through_untouched():
    data = _jpeg(40, 20)

    assert normalize_orientation(data) is data


def test_digest_is_stable_and_content_addressed():
    data = _png()
    assert content_digest(data) == content_digest(bytes(data))
    assert content_digest(data) != content_digest(_png(25, 24))


def test_placeholder_detection_covers_tiny_and_one_pixel():
    assert is_placeholder_image(b"")
    assert is_placeholder_image(b"\x89PNG\r\n\x1a\n" + b"\x00" * 40)
    assert is_placeholder_image(_png(1, 1)) is True
    assert is_placeholder_image(_png(64, 64)) is False


def test_store_reuses_the_same_file_for_identical_bytes(tmp_path):
    store = ImageStore(tmp_path, {"gallery": {}})
    data = _png()

    async def run():
        first = await store.save_image(
            data, provider_id="p", model="m", size="24x24", sid="u", task_id="t1",
            dedupe=True,
        )
        second = await store.save_image(
            data, provider_id="p", model="m", size="24x24", sid="u", task_id="t2",
            dedupe=True,
        )
        return first, second

    first, second = asyncio.run(run())

    assert first == second
    assert len(store.list_gallery()) == 1
    row = store.find_by_digest(content_digest(data))
    assert row is not None and row["path"] == str(first)


def test_store_keeps_separate_rows_without_dedupe(tmp_path):
    store = ImageStore(tmp_path, {"gallery": {}})
    data = _png()

    async def run():
        for index in range(2):
            await store.save_image(
                data, provider_id="p", model="m", size="24x24", sid="u",
                task_id=f"t{index}",
            )

    asyncio.run(run())

    assert len(store.list_gallery()) == 2


def test_the_same_rotated_source_collapses_after_normalising(tmp_path):
    """Two copies of one phone photo share bytes only once orientation is baked in."""
    store = ImageStore(tmp_path, {"gallery": {}})
    rotated = _jpeg(40, 20, orientation=6)

    async def run():
        first = await store.save_image(
            rotated, provider_id="p", model="m", size="20x40", sid="u",
            task_id="a", dedupe=True,
        )
        second = await store.save_image(
            rotated, provider_id="p", model="m", size="20x40", sid="u",
            task_id="b", dedupe=True,
        )
        return first, second

    first, second = asyncio.run(run())

    assert first == second
    assert _size(first.read_bytes()) == (20, 40)
    assert len(store.list_gallery()) == 1
