"""Concurrency-controlled async task queue.

Global and per-user limits come from config and are enforced at submit time;
the worker pool runs up to `max_running` tasks concurrently. Batch jobs run
as a single queue slot; the caller may parallelize sub-images internally with
its own limit. Status transitions are persisted through the TaskStore so the
queue survives plugin restarts (active tasks re-schedule on start).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .task_store import TaskRecord, TaskStore

logger = logging.getLogger("kira_plugin_aiimg.queue")

IMAGE_KINDS = ("generate_image", "edit_image", "generate_selfie", "generate_image_batch")


class QueueFullError(RuntimeError):
    pass


class UserLimitError(RuntimeError):
    pass


class QueueCancelled(Exception):
    pass


@dataclass
class QueueStats:
    running: int
    waiting: int
    total_capacity: int


class TaskQueue:
    def __init__(
        self,
        store: TaskStore,
        *,
        max_running: int = 2,
        max_queued: int = 16,
        max_user_image_tasks: int = 1,
        max_user_video_tasks: int = 1,
        retention_hours: int = 24,
    ):
        self._store = store
        self._max_running = max(1, int(max_running))
        self._max_queued = max(0, int(max_queued))
        self._max_user_image = max(1, int(max_user_image_tasks))
        self._max_user_video = max(1, int(max_user_video_tasks))
        self._retention_hours = int(retention_hours)
        self._queue: asyncio.Queue[tuple[TaskRecord, Callable[[TaskRecord], Awaitable[dict]]]] = (
            asyncio.Queue()
        )
        self._workers: list[asyncio.Task] = []
        self._retire = 0
        self._on_finished: Callable[[TaskRecord], Awaitable[None]] | None = None
        self._closed = False

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Re-schedule queued/running tasks that survived a restart, then boot."""
        for record in self._store.list_active():
            if record.status == "running":
                # A previous run may have been interrupted; reset to queued so
                # the worker picks it up again.
                record.status = "queued"
                record.started_at = 0
                self._store.upsert(record)
            self._queue.put_nowait((record, self._missing_runner))

        for _ in range(self._max_running):
            self._workers.append(asyncio.create_task(self._worker_loop()))

    async def stop(self) -> None:
        self._closed = True
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    def on_finished(self, callback: Callable[[TaskRecord], Awaitable[None]]) -> None:
        self._on_finished = callback

    def apply_limits(
        self,
        *,
        max_running: int | None = None,
        max_queued: int | None = None,
        max_user_image_tasks: int | None = None,
        max_user_video_tasks: int | None = None,
        retention_hours: int | None = None,
    ) -> None:
        """Apply new limits without interrupting running tasks.

        Extra workers retire after the task they hold, so a shrinking
        ``max_running`` never cancels work in flight.
        """
        if max_queued is not None:
            self._max_queued = max(0, int(max_queued))
        if max_user_image_tasks is not None:
            self._max_user_image = max(1, int(max_user_image_tasks))
        if max_user_video_tasks is not None:
            self._max_user_video = max(1, int(max_user_video_tasks))
        if retention_hours is not None:
            self._retention_hours = int(retention_hours)
        if max_running is None:
            return
        target = max(1, int(max_running))
        self._workers = [worker for worker in self._workers if not worker.done()]
        live = len(self._workers)
        if target > live:
            for _ in range(target - live):
                self._workers.append(asyncio.create_task(self._worker_loop()))
        elif target < live:
            self._retire += live - target
        self._max_running = target

    # -- submission ---------------------------------------------------------

    async def submit(
        self,
        record: TaskRecord,
        runner: Callable[[TaskRecord], Awaitable[dict]],
        *,
        is_batch: bool = False,
    ) -> str:
        if self._closed:
            raise RuntimeError("队列已关闭")
        active = self._store.count_active()
        if self._max_queued and active > self._max_queued:
            raise QueueFullError(
                f"任务队列已满（{self._max_queued} 个），请稍后再试"
            )
        if record.user_id:
            if record.kind == "generate_video":
                limit = self._max_user_video
                current = self._store.count_active_by_user(
                    record.user_id, "generate_video", exclude_task_id=record.task_id
                )
                label = "视频任务"
            else:
                limit = self._max_user_image
                current = self._store.count_active_by_user(
                    record.user_id, "", exclude_task_id=record.task_id
                ) - self._store.count_active_by_user(
                    record.user_id, "generate_video", exclude_task_id=record.task_id
                )
                label = "图片任务"
            if current >= limit:
                raise UserLimitError(
                    f"同一账号最多同时 {limit} 个{label}，请等待完成"
                )
        self._queue.put_nowait((record, runner))
        logger.info(
            "accepted %s task %s user=%s provider=%s chain=%s waiting=%d running=%d",
            record.kind,
            record.task_id,
            record.user_id or "-",
            record.provider or "-",
            ",".join(record.chain or []) or "-",
            self._queue.qsize(),
            max(0, self._store.count_active() - self._queue.qsize()),
        )
        return record.task_id

    async def cancel(self, task_id: str) -> bool:
        if not self._store.cancel(task_id):
            logger.info("cancel %s: not cancellable", task_id)
            return False
        logger.info("cancelled task %s", task_id)
        return True

    # -- stats --------------------------------------------------------------

    def stats(self) -> QueueStats:
        return QueueStats(
            running=self._store.count_active_by_kind("") - self._queue.qsize(),
            waiting=self._queue.qsize(),
            total_capacity=self._max_queued,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "max_running": self._max_running,
            "max_queued": self._max_queued,
            "waiting": self._queue.qsize(),
            "running": max(0, self._store.count_active() - self._queue.qsize()),
            "recent": [
                record.to_dict() for record in self._store.list_recent(limit=50)
            ],
        }

    # -- workers ------------------------------------------------------------

    async def _worker_loop(self) -> None:
        while not self._closed:
            if self._retire > 0:
                self._retire -= 1
                break
            record, runner = await self._queue.get()
            try:
                await self._run_one(record, runner)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("queue worker failure for %s", record.task_id)
            finally:
                self._queue.task_done()

    async def _run_one(
        self,
        record: TaskRecord,
        runner: Callable[[TaskRecord], Awaitable[dict]],
    ) -> None:
        fresh = self._store.get(record.task_id) or record
        if fresh.status == "cancelled":
            await self._finish(fresh)
            return
        fresh.status = "running"
        fresh.started_at = int(time.time())
        self._store.upsert(fresh)
        started = time.monotonic()
        logger.info(
            "start %s task %s provider=%s size=%s waiting=%d",
            fresh.kind,
            fresh.task_id,
            fresh.provider or "-",
            fresh.size or "-",
            self._queue.qsize(),
        )
        try:
            result = await runner(fresh)
        except QueueCancelled:
            self._store.cancel(record.task_id)
            fresh = self._store.get(record.task_id) or fresh
            await self._finish(fresh)
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._store.set_error(record.task_id, str(exc), provider=fresh.provider)
            logger.warning(
                "task %s failed after %dms: %s",
                record.task_id,
                int((time.monotonic() - started) * 1000),
                exc,
            )
            fresh = self._store.get(record.task_id) or record
            await self._finish(fresh)
            return

        # A cancel may have raced with completion; respect it.
        fresh = self._store.get(record.task_id) or record
        if fresh.status == "cancelled":
            await self._finish(fresh)
            return
        self._store.set_result(
            record.task_id,
            json.dumps(result, ensure_ascii=False),
            provider=fresh.provider,
        )
        logger.info(
            "done task %s in %dms provider=%s",
            record.task_id,
            int((time.monotonic() - started) * 1000),
            fresh.provider or "-",
        )
        fresh = self._store.get(record.task_id) or record
        await self._finish(fresh)

    async def _finish(self, record: TaskRecord) -> None:
        if self._on_finished is not None:
            try:
                await self._on_finished(record)
            except Exception:
                logger.exception("on_finished callback failed for %s", record.task_id)

    @staticmethod
    async def _missing_runner(record: TaskRecord) -> dict:
        raise RuntimeError(f"未注册的任务执行器: {record.kind}")
