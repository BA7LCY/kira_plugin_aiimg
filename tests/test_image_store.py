"""ImageStore (gallery) tests: save, query, favorite, cleanup, retention."""

import asyncio
import base64
import time

from core.image_store import ImageStore

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _save(store: ImageStore, **kw):
    defaults = dict(provider_id="p1", model="m1", size="1x1", sid="s1", task_id="t1")
    defaults.update(kw)
    return asyncio.run(store.save_image(PNG_1PX, **defaults))


def _id(path) -> str:
    return path.stem.split("_", 1)[1]


def _close(store: ImageStore) -> None:
    asyncio.run(store.close())


def test_save_and_get(tmp_path):
    store = ImageStore(tmp_path, {})
    path = _save(store)
    assert path.exists()
    assert path.read_bytes() == PNG_1PX
    record = store.get(_id(path))
    assert record is not None
    assert record["provider"] == "p1"
    assert record["kind"] == "image"
    _close(store)


def test_video_budget_is_separate_from_images(tmp_path):
    store = ImageStore(tmp_path, {"gallery": {"max_count": 1, "video_max_count": 1}})
    image = _save(store)
    first = store.gallery_dir / "clip-1.mp4"
    first.write_bytes(b"already validated video")
    second = store.gallery_dir / "clip-2.mp4"
    second.write_bytes(b"already validated video")
    try:
        asyncio.run(store.register_video(
            first, provider_id="p1", model="m", size="720p", sid="s1", task_id="v1",
        ))
        asyncio.run(store.register_video(
            second, provider_id="p1", model="m", size="720p", sid="s1", task_id="v2",
        ))
        rows = store.list_gallery()
        kinds = sorted(row["kind"] for row in rows)
        assert kinds == ["image", "video"]
        assert image.exists()
        assert not first.exists() and second.exists()
        video_row = next(row for row in rows if row["kind"] == "video")
        assert store.delete(video_row["id"])
        assert not second.exists()
    finally:
        _close(store)


def test_selfie_budget_is_separate_from_images(tmp_path):
    store = ImageStore(tmp_path, {
        "gallery": {"max_count": 1, "selfie_max_count": 1},
    })
    _save(store, provider_id="p1")
    _save(store, provider_id="p2")
    _save(store, provider_id="p3", kind="selfie")
    _save(store, provider_id="p4", kind="selfie")

    rows = store.list_gallery()
    kinds = sorted(row["kind"] for row in rows)
    assert kinds == ["image", "selfie"]
    assert {row["provider"] for row in rows} == {"p2", "p4"}
    _close(store)


def test_selfie_retention_is_separate_from_images(tmp_path):
    store = ImageStore(tmp_path, {
        "gallery": {"retention_days": 0, "selfie_retention_days": 1},
    })
    image = _save(store, provider_id="p1")
    selfie = _save(store, provider_id="p2", kind="selfie")
    cutoff = time.time() - 2 * 86400
    for path in (image, selfie):
        store._execute(
            "UPDATE images SET created_at = ? WHERE id = ?", (cutoff, _id(path))
        )

    asyncio.run(store.cleanup())

    ids = [r["id"] for r in store.list_gallery()]
    assert ids == [_id(image)]
    _close(store)


def test_list_gallery(tmp_path):
    store = ImageStore(tmp_path, {})
    a = _save(store)
    b = _save(store, provider_id="p2")
    ids = [r["id"] for r in store.list_gallery()]
    assert ids == [_id(b), _id(a)]
    _close(store)


def test_gallery_paging(tmp_path):
    store = ImageStore(tmp_path, {})
    saved = [_save(store, provider_id=f"p{i}") for i in range(3)]
    first = store.list_gallery(limit=2)
    second = store.list_gallery(offset=2, limit=2)
    assert [row["id"] for row in first] == [_id(saved[2]), _id(saved[1])]
    assert [row["id"] for row in second] == [_id(saved[0])]
    _close(store)


def test_delete_removes_thumbnail(tmp_path):
    store = ImageStore(tmp_path, {})
    path = _save(store)
    image_id = _id(path)
    thumb = store.thumb_path(image_id)
    thumb.write_bytes(PNG_1PX)
    assert thumb.is_file()
    store.delete(image_id)
    assert not thumb.exists()
    _close(store)


def test_save_writes_thumb_and_preview(tmp_path):
    store = ImageStore(tmp_path, {})
    path = _save(store)
    image_id = _id(path)
    thumb = store.thumb_path(image_id)
    preview = store.preview_path(image_id)
    assert thumb.is_file() and thumb.read_bytes()[:4] == b"RIFF"
    assert preview.is_file() and preview.read_bytes()[:2] == b"\xff\xd8"
    store.delete(image_id)
    assert not preview.exists()
    _close(store)


def test_favorite_and_stats(tmp_path):
    store = ImageStore(tmp_path, {})
    a = _save(store, prompt="a cat on a roof")
    b = _save(store, provider_id="p2")
    store.set_favorite(_id(a), True)
    stats = store.stats()
    assert stats["count"] == 2
    assert stats["favorites"] == 1
    assert stats["bytes"] == 2 * len(PNG_1PX)
    record_a = store.get(_id(a))
    assert record_a["favorite"] == 1
    assert record_a["prompt"] == "a cat on a roof"
    assert store.get(_id(b))["prompt"] == ""
    _close(store)


def test_delete_refuses_favorite(tmp_path):
    store = ImageStore(tmp_path, {})
    a = _save(store)
    store.set_favorite(_id(a), True)
    assert store.delete(_id(a)) is False
    assert store.get(_id(a)) is not None
    store.set_favorite(_id(a), False)
    assert store.delete(_id(a)) is True
    assert store.get(_id(a)) is None
    _close(store)


def test_cleanup_keeps_favorites(tmp_path):
    store = ImageStore(tmp_path, {"gallery": {"max_count": 1, "retention_days": 0}})
    a = _save(store, provider_id="p1")
    store.set_favorite(_id(a), True)
    b = _save(store, provider_id="p2")
    ids = [r["id"] for r in store.list_gallery()]
    assert _id(a) in ids
    assert _id(b) in ids
    _close(store)


def test_cleanup_respects_max_count(tmp_path):
    store = ImageStore(tmp_path, {"gallery": {"max_count": 1, "retention_days": 0}})
    a = _save(store, provider_id="p1")
    _save(store, provider_id="p2")
    ids = [r["id"] for r in store.list_gallery()]
    assert len(ids) == 1
    assert _id(a) not in ids
    _close(store)


def test_cleanup_drops_expired_before_trimming_capacity(tmp_path):
    store = ImageStore(tmp_path, {"gallery": {"max_count": 3, "retention_days": 1}})
    expired = [_save(store, provider_id=f"old{i}") for i in range(4)]
    fresh = [_save(store, provider_id=f"new{i}") for i in range(3)]
    cutoff = time.time() - 2 * 86400
    for path in expired:
        store._execute(
            "UPDATE images SET created_at = ? WHERE id = ?", (cutoff, _id(path))
        )

    asyncio.run(store.cleanup())

    ids = [r["id"] for r in store.list_gallery()]
    assert sorted(ids) == sorted(_id(path) for path in fresh)
    _close(store)


def test_max_count_ignores_favorites(tmp_path):
    store = ImageStore(tmp_path, {"gallery": {"max_count": 2, "retention_days": 0}})
    favorite = _save(store, provider_id="fav")
    store.set_favorite(_id(favorite), True)
    fresh = [_save(store, provider_id=f"p{i}") for i in range(4)]

    ids = [r["id"] for r in store.list_gallery()]

    assert _id(favorite) in ids
    assert len(ids) == 3
    assert sorted(ids) == sorted([_id(favorite), _id(fresh[-1]), _id(fresh[-2])])
    _close(store)


def test_recent_images(tmp_path):
    store = ImageStore(tmp_path, {})
    for i in range(3):
        _save(store, provider_id=f"p{i}", sid="s1")
    recent = store.recent_images("s1")
    assert len(recent) == 3
    assert store.recent_images("nobody") == []
    _close(store)
