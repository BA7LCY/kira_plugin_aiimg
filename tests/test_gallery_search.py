"""Gallery search, filter, bulk and detail behaviour on ImageStore."""

import asyncio
import base64

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


def _set_created(store: ImageStore, image_id: str, stamp: float) -> None:
    store._execute(
        "UPDATE images SET created_at = ? WHERE id = ?", (stamp, image_id)
    )


def _close(store: ImageStore) -> None:
    asyncio.run(store.close())


def test_search_matches_prompt_literally(tmp_path):
    store = ImageStore(tmp_path, {})
    fox = _save(store, prompt="a red fox")
    percent = _save(store, prompt="100% fox")
    underscore = _save(store, prompt="under_score")
    _save(store, prompt="underXscore")

    rows, total = store.search_gallery(query="fox")
    assert total == 2
    assert {row["id"] for row in rows} == {_id(fox), _id(percent)}

    rows, total = store.search_gallery(query="100%")
    assert total == 1
    assert rows[0]["id"] == _id(percent)

    rows, total = store.search_gallery(query="_")
    assert total == 1
    assert rows[0]["id"] == _id(underscore)

    assert store.search_gallery(query="under_score")[1] == 1
    assert store.search_gallery(query="underXscore")[1] == 1
    assert store.search_gallery(query="missing")[1] == 0
    _close(store)


def test_search_filters_kind_and_favorite(tmp_path):
    store = ImageStore(tmp_path, {})
    first = _save(store, prompt="cat")
    second = _save(store, prompt="dog")
    video = _save(store, kind="video", prompt="clip")

    rows, total = store.search_gallery(kind="video")
    assert total == 1
    assert rows[0]["id"] == _id(video)

    assert store.favorite_many([_id(first)], True) == 1
    rows, total = store.search_gallery(favorite=True)
    assert total == 1
    assert rows[0]["id"] == _id(first)

    rows, total = store.search_gallery(favorite=False)
    assert total == 2
    assert _id(first) not in {row["id"] for row in rows}

    rows, total = store.search_gallery(kind="image", favorite=False)
    assert total == 1
    assert rows[0]["id"] == _id(second)

    rows, total = store.search_gallery(kind="image", favorite=True, query="cat")
    assert total == 1
    assert rows[0]["id"] == _id(first)
    _close(store)


def test_search_sort_and_pagination(tmp_path):
    store = ImageStore(tmp_path, {})
    saved = [_save(store, prompt=f"pic {i}") for i in range(5)]
    for index, path in enumerate(saved):
        _set_created(store, _id(path), 1_000 + index)

    rows, total = store.search_gallery(sort="new")
    assert total == 5
    assert [row["id"] for row in rows] == [_id(p) for p in reversed(saved)]

    rows, _ = store.search_gallery(sort="old")
    assert [row["id"] for row in rows] == [_id(p) for p in saved]

    page, total = store.search_gallery(sort="new", offset=1, limit=2)
    assert total == 5
    assert [row["id"] for row in page] == [_id(saved[3]), _id(saved[2])]

    page, total = store.search_gallery(sort="old", offset=3, limit=2)
    assert total == 5
    assert [row["id"] for row in page] == [_id(saved[3]), _id(saved[4])]

    page, total = store.search_gallery(query="pic", offset=10, limit=2)
    assert total == 5
    assert page == []
    _close(store)


def test_search_skips_archived(tmp_path):
    store = ImageStore(tmp_path, {})
    path = _save(store, prompt="hidden fox")
    store._execute(
        "UPDATE images SET archived = 1 WHERE id = ?", (_id(path),)
    )

    assert store.search_gallery(query="fox") == ([], 0)
    rows, total = store.search_gallery(query="fox", include_archived=True)
    assert total == 1
    assert rows[0]["id"] == _id(path)
    _close(store)


def test_search_indexes_exist(tmp_path):
    store = ImageStore(tmp_path, {})
    names = {
        row["name"]
        for row in store._execute("PRAGMA index_list(images)").fetchall()
    }
    assert {"idx_images_created_at", "idx_images_favorite", "idx_images_kind"} <= names
    _close(store)


def test_bulk_delete_removes_files_and_skips_favorites(tmp_path):
    store = ImageStore(tmp_path, {})
    keep = _save(store, prompt="keep")
    drop_a = _save(store)
    drop_b = _save(store)
    keep_id = _id(keep)
    store.favorite_many([keep_id], True)

    thumbs = {pid: store.thumb_path(pid) for pid in map(_id, (keep, drop_a, drop_b))}
    previews = {pid: store.preview_path(pid) for pid in thumbs}

    result = store.delete_many([_id(drop_a), _id(drop_b), keep_id, "missing"])
    assert result == {"deleted": 2, "skipped": 1}

    assert store.get(keep_id) is not None
    assert store.get(_id(drop_a)) is None
    assert store.get(_id(drop_b)) is None
    assert keep.is_file() and thumbs[keep_id].is_file()

    for path in (drop_a, drop_b):
        assert not path.exists()
    for pid in (_id(drop_a), _id(drop_b)):
        assert not thumbs[pid].exists()
        assert not previews[pid].exists()

    assert store.delete_many([]) == {"deleted": 0, "skipped": 0}
    _close(store)


def test_bulk_favorite_stays_out_of_capacity(tmp_path):
    store = ImageStore(tmp_path, {"gallery": {"max_count": 0, "retention_days": 0}})
    first = _save(store)
    second = _save(store)
    third = _save(store)

    assert store.favorite_many([_id(first), _id(second)], True) == 2
    store.update_config({"gallery": {"max_count": 1, "retention_days": 0}})
    asyncio.run(store.cleanup())
    assert len(store.list_gallery()) == 3

    assert store.favorite_many([_id(first), _id(second)], False) == 2
    asyncio.run(store.cleanup())
    remaining = store.list_gallery()
    assert len(remaining) == 1
    assert remaining[0]["id"] == _id(third)
    _close(store)


def test_detail_reports_metadata_and_size(tmp_path):
    store = ImageStore(tmp_path, {})
    path = _save(
        store,
        provider_id="p9",
        model="m9",
        size="512x512",
        prompt="a fox",
    )
    image_id = _id(path)

    item = store.detail(image_id)
    assert item["prompt"] == "a fox"
    assert item["provider"] == "p9"
    assert item["model"] == "m9"
    assert item["size"] == "512x512"
    assert item["kind"] == "image"
    assert item["favorite"] == 0
    assert item["created_at"] > 0
    assert item["bytes"] == path.stat().st_size == len(PNG_1PX)
    assert item["filename"] == path.name
    assert item["exists"] is True

    path.unlink()
    gone = store.detail(image_id)
    assert gone["exists"] is False
    assert gone["bytes"] == len(PNG_1PX)
    assert store.detail("missing") is None
    _close(store)
