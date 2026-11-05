"""Owner-lease and state-machine tests for TaskStore (SQLite)."""

import asyncio
import sqlite3
import time

import pytest

from core.task_store import TaskStore, new_owner_id

# Schema as shipped before the lease columns existed (task_store < 0.3.4).
OLD_SCHEMA = """
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    sid TEXT NOT NULL DEFAULT '',
    session_type TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL DEFAULT '',
    prompt TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    result TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    created_at INTEGER NOT NULL,
    started_at INTEGER NOT NULL DEFAULT 0,
    finished_at INTEGER NOT NULL DEFAULT 0,
    chain TEXT NOT NULL DEFAULT '[]',
    prompt_layers TEXT NOT NULL DEFAULT '[]',
    final_prompt TEXT NOT NULL DEFAULT '',
    size TEXT NOT NULL DEFAULT '',
    extra TEXT NOT NULL DEFAULT '{}'
)
"""

OWNER_A = "aiimg-1001-aaaaaaaa"
OWNER_B = "aiimg-1001-bbbbbbbb"


@pytest.fixture
def store(tmp_path):
    instance = TaskStore(tmp_path)
    yield instance
    instance.close()


def _force_lease(store: TaskStore, task_id: str, deadline: float) -> None:
    """Simulate the clock moving past (or before) a lease without sleeping."""
    store._db.execute(
        "UPDATE tasks SET lease_until = ? WHERE task_id = ?",
        (float(deadline), task_id),
    )
    store._db.commit()


def test_new_owner_id_shape():
    first = new_owner_id()
    second = new_owner_id()
    assert first.startswith("aiimg-")
    prefix, pid, suffix = first.split("-")
    assert prefix == "aiimg"
    assert pid.isdigit()
    assert len(suffix) == 8
    int(suffix, 16)
    assert first != second


async def test_migration_from_old_schema(tmp_path):
    """An old database file keeps working and gains the lease columns."""
    db_path = tmp_path / "tasks.sqlite3"
    old_db = sqlite3.connect(str(db_path))
    old_db.executescript(OLD_SCHEMA)
    old_db.execute(
        "INSERT INTO tasks (task_id, kind, status, created_at, prompt) "
        "VALUES (?, ?, ?, ?, ?)",
        ("old-1", "generate_image", "queued", int(time.time()), "旧任务"),
    )
    old_db.commit()
    old_db.close()

    store = TaskStore(tmp_path)
    try:
        record = store.get("old-1")
        assert record is not None
        assert record.prompt == "旧任务"
        assert record.status == "queued"
        assert record.owner == ""
        assert record.lease_until == 0.0
        assert record.attempts == 0

        columns = {
            str(row["name"])
            for row in store._db.execute("PRAGMA table_info(tasks)").fetchall()
        }
        assert {"owner", "lease_until", "attempts"} <= columns

        assert await store.claim("old-1", OWNER_A, 60.0) is True
        claimed = store.get("old-1")
        assert claimed.status == "running"
        assert claimed.owner == OWNER_A
        assert claimed.attempts == 1
    finally:
        store.close()


async def test_claim_is_exclusive(store):
    record = store.create(kind="generate_image", user_id="u1")

    assert await store.claim(record.task_id, OWNER_A, 60.0) is True
    claimed = store.get(record.task_id)
    assert claimed.status == "running"
    assert claimed.owner == OWNER_A
    assert claimed.attempts == 1
    assert claimed.started_at > 0
    assert claimed.lease_until > time.time()

    assert await store.claim(record.task_id, OWNER_B, 60.0) is False
    still = store.get(record.task_id)
    assert still.owner == OWNER_A
    assert still.attempts == 1


async def test_claim_takes_over_expired_lease(store):
    record = store.create(kind="generate_image", user_id="u1")
    assert await store.claim(record.task_id, OWNER_A, 60.0) is True
    assert await store.claim(record.task_id, OWNER_B, 60.0) is False

    _force_lease(store, record.task_id, time.time() - 1.0)
    assert await store.claim(record.task_id, OWNER_B, 60.0) is True
    taken = store.get(record.task_id)
    assert taken.owner == OWNER_B
    assert taken.attempts == 2


async def test_claim_takes_over_ownerless_running(store):
    record = store.create(kind="generate_image", user_id="u1")
    store.set_status(record.task_id, "running")
    _force_lease(store, record.task_id, time.time() + 600.0)
    store._db.execute(
        "UPDATE tasks SET owner = '' WHERE task_id = ?", (record.task_id,)
    )
    store._db.commit()

    assert await store.claim(record.task_id, OWNER_A, 60.0) is True
    assert store.get(record.task_id).owner == OWNER_A


async def test_claim_after_short_ttl(store):
    record = store.create(kind="generate_image", user_id="u1")
    assert await store.claim(record.task_id, OWNER_A, 0.01) is True
    await asyncio.sleep(0.05)
    assert await store.claim(record.task_id, OWNER_B, 60.0) is True
    assert store.get(record.task_id).owner == OWNER_B


async def test_claim_rejects_finished_task(store):
    record = store.create(kind="generate_image", user_id="u1")
    assert store.cancel(record.task_id) is True
    assert await store.claim(record.task_id, OWNER_A, 60.0) is False
    assert store.get(record.task_id).status == "cancelled"


async def test_concurrent_claims_have_one_winner(store):
    record = store.create(kind="generate_image", user_id="u1")
    results = await asyncio.gather(
        *(store.claim(record.task_id, f"aiimg-1-{i:08x}", 60.0) for i in range(8))
    )
    assert sum(1 for won in results if won) == 1
    assert store.get(record.task_id).attempts == 1


async def test_two_stores_share_one_file(tmp_path):
    first = TaskStore(tmp_path)
    second = TaskStore(tmp_path)
    try:
        record = first.create(kind="generate_image", user_id="u1")
        assert await first.claim(record.task_id, OWNER_A, 60.0) is True
        assert await second.claim(record.task_id, OWNER_B, 60.0) is False
        assert await second.stale_running() == []
        assert second.get(record.task_id).owner == OWNER_A
    finally:
        first.close()
        second.close()


async def test_heartbeat(store):
    record = store.create(kind="generate_image", user_id="u1")
    assert await store.claim(record.task_id, OWNER_A, 30.0) is True
    before = store.get(record.task_id).lease_until

    assert await store.heartbeat(record.task_id, OWNER_A, 120.0) is True
    after = store.get(record.task_id).lease_until
    assert after > before

    assert await store.heartbeat(record.task_id, OWNER_A, 120.0) is True
    assert await store.heartbeat(record.task_id, OWNER_B, 120.0) is False
    assert store.get(record.task_id).owner == OWNER_A

    store.set_result(record.task_id, "{}")
    assert await store.heartbeat(record.task_id, OWNER_A, 120.0) is False


async def test_release(store):
    record = store.create(kind="generate_image", user_id="u1")
    assert await store.claim(record.task_id, OWNER_A, 60.0) is True

    await store.release(record.task_id, OWNER_B)  # non-holder: no-op
    assert store.get(record.task_id).owner == OWNER_A

    await store.release(record.task_id, OWNER_A)
    freed = store.get(record.task_id)
    assert freed.owner == ""
    assert freed.lease_until == 0.0
    assert await store.claim(record.task_id, OWNER_B, 60.0) is True


async def test_stale_running(store):
    now = time.time()
    queued = store.create(kind="generate_image", user_id="u1")

    older = store.create(kind="generate_image", user_id="u2")
    store.set_status(older.task_id, "running")
    newer = store.create(kind="generate_image", user_id="u3")
    store.set_status(newer.task_id, "running")

    live = store.create(kind="generate_image", user_id="u4")
    store.set_status(live.task_id, "running")

    done = store.create(kind="generate_image", user_id="u5")
    store.set_result(done.task_id, "{}")

    _force_lease(store, older.task_id, now - 30.0)
    _force_lease(store, newer.task_id, now - 5.0)
    _force_lease(store, live.task_id, now + 300.0)
    # Oldest created_at first, even though "newer" was inserted later.
    store._db.execute(
        "UPDATE tasks SET created_at = ? WHERE task_id = ?",
        (int(now) - 100, newer.task_id),
    )
    store._db.commit()

    stale = await store.stale_running()
    assert [item.task_id for item in stale] == [newer.task_id, older.task_id]
    assert {item.task_id for item in stale}.isdisjoint(
        {queued.task_id, live.task_id, done.task_id}
    )

    assert [item.task_id for item in await store.stale_running(now=now - 10.0)] == [
        older.task_id
    ]


async def test_bump_attempts(store):
    record = store.create(kind="generate_image", user_id="u1")
    assert await store.bump_attempts(record.task_id) == 1
    assert await store.bump_attempts(record.task_id) == 2
    assert store.get(record.task_id).attempts == 2
    assert await store.bump_attempts("missing-task") == 0


def test_state_machine_allows_normal_flow(store):
    record = store.create(kind="generate_image", user_id="u1")
    store.set_status(record.task_id, "running")
    store.set_status(record.task_id, "running")  # idempotent no-op
    store.set_status(record.task_id, "failed", finished_at=123)
    finished = store.get(record.task_id)
    assert finished.status == "failed"
    assert finished.finished_at == 123


def test_state_machine_rejects_restart(store):
    record = store.create(kind="generate_image", user_id="u1")
    store.set_status(record.task_id, "running")
    store.set_result(record.task_id, "{}")

    with pytest.raises(ValueError) as excinfo:
        store.set_status(record.task_id, "running")
    assert "任务已结束" in str(excinfo.value)

    with pytest.raises(ValueError):
        store.set_status(record.task_id, "queued")
    with pytest.raises(ValueError):
        store.set_status(record.task_id, "failed")
    assert store.get(record.task_id).status == "succeeded"


def test_state_machine_rejects_unknown_status(store):
    record = store.create(kind="generate_image", user_id="u1")
    with pytest.raises(ValueError):
        store.set_status(record.task_id, "exploded")
    assert store.get(record.task_id).status == "queued"
