"""TaskQueue concurrency, limits and finish-callback tests."""

import asyncio

import pytest

from core.queue import QueueCancelled, QueueFullError, TaskQueue, UserLimitError
from core.task_store import TaskStore


def _queue(tmp_path, **kwargs):
    store = TaskStore(tmp_path)
    q = TaskQueue(store, **kwargs)
    return store, q


async def _idle(seconds=0.15):
    await asyncio.sleep(seconds)


async def _wait_status(store, task_id, statuses, timeout=15.0):
    """Poll instead of sleeping a fixed amount: sqlite commits can be slow."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if store.get(task_id).status in statuses:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"task {task_id} stuck at {store.get(task_id).status!r}")


def test_success_and_finish_callback(tmp_path):
    store, q = _queue(tmp_path, max_running=1, max_queued=8)
    finished = []

    async def cb(record):
        finished.append(record.task_id)

    q.on_finished(cb)

    async def scenario():
        await q.start()
        task_id = store.create(kind="generate_image", user_id="u1", prompt="猫").task_id

        async def runner(rec):
            await asyncio.sleep(0.01)
            return {"ok": True, "media": ["/x.png"]}

        await q.submit(store.get(task_id), runner)
        await _wait_status(store, task_id, {"succeeded"})
        await q.stop()
        return task_id

    task_id = asyncio.run(scenario())
    got = store.get(task_id)
    assert got.status == "succeeded"
    assert '"ok": true' in got.result
    assert finished == [task_id]
    store.close()


def test_runner_cancelled_maps_to_cancelled_status(tmp_path):
    store, q = _queue(tmp_path, max_running=1)
    finished = []

    async def cb(record):
        finished.append(record.task_id)

    q.on_finished(cb)

    async def scenario():
        await q.start()
        task_id = store.create(kind="generate_image", user_id="u1").task_id

        async def runner(rec):
            raise QueueCancelled("stop")

        await q.submit(store.get(task_id), runner)
        await _wait_status(store, task_id, {"cancelled"})
        await q.stop()
        return task_id

    task_id = asyncio.run(scenario())
    assert store.get(task_id).status == "cancelled"
    assert finished == [task_id]
    store.close()


def test_runner_failure_sets_error(tmp_path):
    store, q = _queue(tmp_path, max_running=1)

    async def scenario():
        await q.start()
        task_id = store.create(kind="generate_image", user_id="u1").task_id

        async def runner(rec):
            raise RuntimeError("boom")

        await q.submit(store.get(task_id), runner)
        await _wait_status(store, task_id, {"failed"})
        await q.stop()
        return task_id

    task_id = asyncio.run(scenario())
    got = store.get(task_id)
    assert got.status == "failed"
    assert got.error == "boom"
    store.close()


def test_user_image_limit(tmp_path):
    store, q = _queue(tmp_path, max_running=1, max_user_image_tasks=1)

    async def runner(rec):
        await _idle(0.05)
        return {}

    async def scenario():
        await q.start()
        first = store.create(kind="generate_image", user_id="u1").task_id
        await q.submit(store.get(first), runner)
        second = store.create(kind="generate_image", user_id="u1").task_id
        with pytest.raises(UserLimitError):
            await q.submit(store.get(second), runner)
        await _idle()
        await q.stop()

    asyncio.run(scenario())
    store.close()


def test_image_and_video_limits_independent(tmp_path):
    store, q = _queue(tmp_path, max_running=1, max_user_image_tasks=1, max_user_video_tasks=1)

    async def runner(rec):
        await _idle(0.05)
        return {}

    async def scenario():
        await q.start()
        img = store.create(kind="generate_image", user_id="u1").task_id
        await q.submit(store.get(img), runner)
        vid = store.create(kind="generate_video", user_id="u1").task_id
        await q.submit(store.get(vid), runner)
        img2 = store.create(kind="generate_image", user_id="u1").task_id
        with pytest.raises(UserLimitError):
            await q.submit(store.get(img2), runner)
        await _idle()
        await q.stop()

    asyncio.run(scenario())
    store.close()


def test_queue_full(tmp_path):
    store, q = _queue(tmp_path, max_running=1, max_queued=2)

    async def runner(rec):
        await _idle(0.05)
        return {}

    async def scenario():
        await q.start()
        a = store.create(kind="generate_image", user_id="u1").task_id
        await q.submit(store.get(a), runner)
        b = store.create(kind="generate_image", user_id="u2").task_id
        await q.submit(store.get(b), runner)
        c = store.create(kind="generate_image", user_id="u3").task_id
        with pytest.raises(QueueFullError):
            await q.submit(store.get(c), runner)
        await _idle()
        await q.stop()

    asyncio.run(scenario())
    store.close()


def test_snapshot_shape(tmp_path):
    store, q = _queue(tmp_path, max_running=2)

    async def scenario():
        await q.start()
        task_id = store.create(kind="generate_image", user_id="u1").task_id

        async def runner(rec):
            await _idle(0.05)
            return {}

        await q.submit(store.get(task_id), runner)
        snap = q.snapshot()
        assert snap["max_running"] == 2
        assert "waiting" in snap and "running" in snap
        assert len(snap["recent"]) == 1
        await _idle()
        await q.stop()

    asyncio.run(scenario())
    store.close()


def test_restart_reschedules_active(tmp_path):
    store, q = _queue(tmp_path, max_running=1)
    task_id = store.create(kind="generate_image", user_id="u1").task_id
    store.set_status(task_id, "running", provider="p1")

    async def scenario():
        await q.start()
        await _wait_status(store, task_id, {"failed"})
        await q.stop()
        return task_id

    asyncio.run(scenario())
    got = store.get(task_id)
    # No runner registered -> task failed with the missing-runner error.
    assert got.status == "failed"
    assert "未注册" in got.error
    store.close()
