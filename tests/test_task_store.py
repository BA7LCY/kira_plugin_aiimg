"""TaskStore persistence tests (SQLite)."""

import time

from core.task_store import TaskRecord, TaskStore


def _store(tmp_path) -> TaskStore:
    return TaskStore(tmp_path)


def test_create_and_get(tmp_path):
    store = _store(tmp_path)
    record = store.create(kind="generate_image", sid="s1", user_id="u1", prompt="猫")
    assert record.task_id
    assert record.status == "queued"
    fetched = store.get(record.task_id)
    assert fetched is not None
    assert fetched.kind == "generate_image"
    assert fetched.user_id == "u1"
    assert fetched.prompt == "猫"
    store.close()


def test_upsert_and_roundtrip(tmp_path):
    store = _store(tmp_path)
    record = store.create(kind="generate_video", user_id="u2")
    record.provider = "agnes"
    record.chain = ["a", "b"]
    record.prompt_layers = [{"name": "user_prompt", "text": "x"}]
    record.extra = {"aspect_ratio": "16:9"}
    store.upsert(record)
    fetched = store.get(record.task_id)
    assert isinstance(fetched, TaskRecord)
    assert fetched.provider == "agnes"
    assert fetched.chain == ["a", "b"]
    assert fetched.extra == {"aspect_ratio": "16:9"}
    store.close()


def test_status_transitions(tmp_path):
    store = _store(tmp_path)
    record = store.create(kind="generate_image", user_id="u1")
    store.set_status(record.task_id, "running", provider="p1")
    running = store.get(record.task_id)
    assert running.status == "running"
    assert running.provider == "p1"
    assert running.started_at > 0
    store.set_result(record.task_id, '{"ok": true}')
    done = store.get(record.task_id)
    assert done.status == "succeeded"
    assert done.result == '{"ok": true}'
    assert done.finished_at > 0
    store.close()


def test_error_transition(tmp_path):
    store = _store(tmp_path)
    record = store.create(kind="generate_image", user_id="u1")
    store.set_error(record.task_id, "boom")
    got = store.get(record.task_id)
    assert got.status == "failed"
    assert got.error == "boom"
    store.close()


def test_cancel(tmp_path):
    store = _store(tmp_path)
    record = store.create(kind="generate_image", user_id="u1")
    assert store.cancel(record.task_id) is True
    assert store.get(record.task_id).status == "cancelled"
    assert store.cancel(record.task_id) is False
    store.close()


def test_counts(tmp_path):
    store = _store(tmp_path)
    for i in range(3):
        store.create(kind="generate_image", user_id=f"u{i}")
    store.create(kind="generate_video", user_id="u0")
    assert store.count_active() == 4
    assert store.count_active_by_kind("generate_video") == 1
    assert store.count_active_by_user("u0") == 2
    assert store.count_active_by_user("u0", "generate_video") == 1
    store.close()


def test_list_recent_order_and_filter(tmp_path):
    store = _store(tmp_path)
    first = store.create(kind="generate_image", user_id="u1")
    second = store.create(kind="generate_video", user_id="u2")
    recent = store.list_recent(limit=10)
    assert [r.task_id for r in recent] == [second.task_id, first.task_id]
    videos = store.list_recent(kind="generate_video")
    assert [r.task_id for r in videos] == [second.task_id]
    store.close()


def test_prune_old(tmp_path):
    store = _store(tmp_path)
    old = store.create(kind="generate_image", user_id="u1")
    store.set_error(old.task_id, "old failure")
    store._db.execute(
        "UPDATE tasks SET finished_at = ? WHERE task_id = ?",
        (int(time.time()) - 100000, old.task_id),
    )
    store._db.commit()
    fresh = store.create(kind="generate_image", user_id="u2")
    store.set_result(fresh.task_id, "{}")
    removed = store.prune_old(retention_hours=24)
    assert removed == 1
    assert store.get(old.task_id) is None
    assert store.get(fresh.task_id) is not None
    store.close()
