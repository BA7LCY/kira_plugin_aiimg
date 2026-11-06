"""TaskQueue concurrency, limits and finish-callback tests."""

import asyncio
import logging

import pytest

from core.queue import (
    NoRunnerError,
    QueueCancelled,
    QueueFullError,
    TaskQueue,
    UserLimitError,
)
from core.task_store import TaskRecord, TaskStore


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


def test_cancel_interrupts_running_runner_and_keeps_worker(tmp_path):
    store, q = _queue(tmp_path, max_running=1, max_queued=8)

    async def scenario():
        await q.start()
        started = asyncio.Event()
        interrupted = asyncio.Event()
        task_id = store.create(kind="generate_image", user_id="u1").task_id

        async def runner(rec):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                interrupted.set()
                raise
            return {"ok": True, "media": ["/x.png"]}

        await q.submit(store.get(task_id), runner)
        await asyncio.wait_for(started.wait(), 5)
        assert await q.cancel(task_id) is True
        await _wait_status(store, task_id, {"cancelled"})
        assert interrupted.is_set()

        second_id = store.create(kind="generate_image", user_id="u2").task_id

        async def second(rec):
            return {"ok": True, "media": ["/y.png"]}

        await q.submit(store.get(second_id), second)
        await _wait_status(store, second_id, {"succeeded"})
        await q.stop()
        return task_id, second_id

    task_id, second_id = asyncio.run(scenario())
    assert store.get(task_id).status == "cancelled"
    assert store.get(second_id).status == "succeeded"
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
        with pytest.raises(UserLimitError) as info:
            await q.submit(store.get(second), runner)
        # 给用户看的那句话只有事实，不含任何数字。
        assert info.value.code == "too_many_running"
        assert str(info.value) == "同时进行的任务太多"
        assert not any(ch.isdigit() for ch in str(info.value))
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
        with pytest.raises(UserLimitError) as info:
            await q.submit(store.get(img2), runner)
        assert info.value.code == "too_many_running"
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
        with pytest.raises(QueueFullError) as info:
            await q.submit(store.get(c), runner)
        assert info.value.code == "queue_full"
        assert str(info.value) == "队列已满"
        assert not any(ch.isdigit() for ch in str(info.value))
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


def test_exception_codes_are_stable():
    assert QueueFullError.code == "queue_full"
    assert UserLimitError.code == "too_many_running"
    assert QueueCancelled.code == "cancelled"
    assert NoRunnerError.code == "no_runner"

    # 不传话时也说一句普通中文，且不带任何数字。
    for exc_type, expected in (
        (QueueFullError, "队列已满"),
        (UserLimitError, "同时进行的任务太多"),
        (QueueCancelled, "任务已取消"),
        (NoRunnerError, "没有可用的执行器"),
    ):
        text = str(exc_type())
        assert text == expected
        assert not any(ch.isdigit() for ch in text)


def test_steps_are_logged_in_chinese(tmp_path, caplog):
    store, q = _queue(tmp_path, max_running=1)

    async def runner(rec):
        await asyncio.sleep(0.01)
        return {"ok": True}

    async def scenario():
        await q.start()
        task_id = store.create(kind="generate_image", user_id="u1", prompt="猫").task_id
        await q.submit(store.get(task_id), runner)
        await _wait_status(store, task_id, {"succeeded"})
        await q.stop()

    with caplog.at_level(logging.INFO):
        asyncio.run(scenario())

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "队列启动" in text
    assert "入队任务" in text
    assert "开始执行任务" in text
    assert "结束：状态 succeeded" in text
    assert "队列已停止" in text
    store.close()


def test_restart_reschedules_active(tmp_path, caplog):
    store, q = _queue(tmp_path, max_running=1)
    task_id = store.create(kind="generate_image", user_id="u1").task_id
    store.set_status(task_id, "running", provider="p1")

    async def scenario():
        await q.start()
        await _wait_status(store, task_id, {"failed"})
        await q.stop()
        return task_id

    with caplog.at_level(logging.INFO):
        asyncio.run(scenario())
    got = store.get(task_id)
    # No runner registered -> task failed with the missing-runner error.
    assert got.status == "failed"
    assert got.error == "没有可用的执行器"
    assert NoRunnerError.code == "no_runner"
    assert any(
        "没有可用的执行器" in record.getMessage() for record in caplog.records
    )
    store.close()


def test_lease_keeps_two_instances_from_running_the_same_task(tmp_path):
    store, first = _queue(tmp_path, max_running=1, owner="instance-a", lease_seconds=60)
    second = TaskQueue(store, max_running=1, owner="instance-b", lease_seconds=60)
    runs = []
    finished = []

    async def runner(rec):
        runs.append(rec.task_id)
        await asyncio.sleep(0.4)
        return {"media": ["a.png"]}

    async def cb(record):
        finished.append(record.task_id)

    first.on_finished(cb)

    async def scenario():
        await first.start()
        task_id = store.create(kind="generate_image", user_id="u1", prompt="猫").task_id
        await first.submit(store.get(task_id), runner)
        await asyncio.sleep(0.1)
        await second.start()
        await asyncio.sleep(0.5)
        await second.stop()
        await first.stop()
        return task_id

    task_id = asyncio.run(scenario())

    assert runs == [task_id]
    assert finished == [task_id]
    assert store.get(task_id).status == "succeeded"
    store.close()


def test_resume_runner_takes_over_an_interrupted_task(tmp_path):
    store, q = _queue(tmp_path, max_running=1, owner="instance-a", lease_seconds=60)
    task_id = store.create(kind="generate_image", user_id="u1", prompt="猫").task_id
    store.set_status(task_id, "running", provider="p1")

    resumed = []

    async def resume(rec):
        resumed.append((rec.task_id, rec.attempts))
        return {"media": ["b.png"]}

    q.on_resume(resume)

    async def scenario():
        await q.start()
        await _wait_status(store, task_id, {"succeeded"})
        await q.stop()

    asyncio.run(scenario())

    assert [task_id] == [item[0] for item in resumed]
    assert resumed[0][1] >= 1
    assert store.get(task_id).status == "succeeded"
    assert store.get(task_id).owner == ""
    store.close()
