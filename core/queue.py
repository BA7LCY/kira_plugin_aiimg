"""Concurrency-controlled async task queue.

Global and per-user limits come from config and are enforced at submit time;
the worker pool runs up to `max_running` tasks concurrently. Batch jobs run
as a single queue slot; the caller may parallelize sub-images internally with
its own limit. Status transitions are persisted through the TaskStore so the
queue survives plugin restarts (active tasks re-schedule on start).

拒绝入队时抛出的异常只带一句普通话和一个稳定的 code；排队数、上限这类内部数字
只写进日志，不进模型看到的那句话。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from .logbook import elapsed_ms
from .task_store import TaskRecord, TaskStore

logger = logging.getLogger("kira_plugin_aiimg.queue")

IMAGE_KINDS = ("generate_image", "edit_image", "generate_selfie", "generate_image_batch")


class QueueFullError(RuntimeError):
    """队列已经满了，这次请求没排上。

    ``code`` 让调用方按稳定值决定怎么说，消息本身只说事实，不带数字。
    """

    code = "queue_full"

    def __init__(self, message: str = "队列已满") -> None:
        super().__init__(message)


class UserLimitError(RuntimeError):
    """同一账号手上的任务还没结束，先不加新的。"""

    code = "too_many_running"

    def __init__(self, message: str = "同时进行的任务太多") -> None:
        super().__init__(message)


class QueueCancelled(Exception):
    """执行器发现任务已被取消，用它把自己停下来。"""

    code = "cancelled"

    def __init__(self, message: str = "任务已取消") -> None:
        super().__init__(message)


class NoRunnerError(RuntimeError):
    """任务类型没有注册执行器，只能按失败收尾。"""

    code = "no_runner"

    def __init__(self, message: str = "没有可用的执行器") -> None:
        super().__init__(message)


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
        owner: str = "",
        lease_seconds: float = 0.0,
    ):
        self._store = store
        self._max_running = max(1, int(max_running))
        self._max_queued = max(0, int(max_queued))
        self._max_user_image = max(1, int(max_user_image_tasks))
        self._max_user_video = max(1, int(max_user_video_tasks))
        self._retention_hours = int(retention_hours)
        self._owner = str(owner or "")
        self._lease_seconds = max(0.0, float(lease_seconds or 0.0))
        self._queue: asyncio.Queue[tuple[TaskRecord, Callable[[TaskRecord], Awaitable[dict]]]] = (
            asyncio.Queue()
        )
        self._workers: list[asyncio.Task] = []
        self._running: dict[str, asyncio.Task] = {}
        self._retire = 0
        self._on_finished: Callable[[TaskRecord], Awaitable[None]] | None = None
        self._resume: Callable[[TaskRecord], Awaitable[dict]] | None = None
        self._closed = False

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Re-schedule queued/running tasks that survived a restart, then boot."""
        runner = self._resume or self._missing_runner
        restored = self._store.list_active()
        if restored:
            logger.info("启动恢复：%d 个未完成任务等待接手", len(restored))
            # 只为前若干条逐条说明，任务再多也只占一行日志。
            for record in restored[:20]:
                logger.info(
                    "恢复任务 %s：类型 %s，原状态 %s，此前已尝试 %d 次",
                    record.task_id,
                    record.kind,
                    record.status,
                    record.attempts,
                )
        for record in restored:
            # Ownership is decided by the lease: a task another instance is
            # still running gets dropped when this worker cannot claim it.
            self._queue.put_nowait((record, runner))

        for _ in range(self._max_running):
            self._workers.append(asyncio.create_task(self._worker_loop()))
        logger.info(
            "队列启动：%d 个工作线程，排队上限 %d 个",
            self._max_running,
            self._max_queued,
        )

    async def stop(self) -> None:
        self._closed = True
        for worker in self._workers:
            worker.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("队列已停止")

    def on_finished(self, callback: Callable[[TaskRecord], Awaitable[None]]) -> None:
        self._on_finished = callback

    def on_resume(self, callback: Callable[[TaskRecord], Awaitable[dict]]) -> None:
        """Runner used for tasks restored from disk after a restart."""
        self._resume = callback

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
        if max_running is not None:
            target = max(1, int(max_running))
            self._workers = [worker for worker in self._workers if not worker.done()]
            live = len(self._workers)
            if target > live:
                for _ in range(target - live):
                    self._workers.append(asyncio.create_task(self._worker_loop()))
            elif target < live:
                self._retire += live - target
            self._max_running = target
        logger.info(
            "队列参数调整：工作线程 %d 个，排队上限 %d 个，"
            "单用户图片 %d 个，单用户视频 %d 个，记录保留 %d 小时",
            self._max_running,
            self._max_queued,
            self._max_user_image,
            self._max_user_video,
            self._retention_hours,
        )

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
            # 人数只留在日志里；给用户的那句话不报数字。
            logger.info(
                "拒绝入队 %s：队列已满，当前未结束 %d 个，上限 %d 个",
                record.task_id,
                active,
                self._max_queued,
            )
            raise QueueFullError("队列已满")
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
                logger.info(
                    "拒绝入队 %s：用户 %s 已有 %d 个%s在跑，上限 %d 个",
                    record.task_id,
                    record.user_id,
                    current,
                    label,
                    limit,
                )
                raise UserLimitError("同时进行的任务太多")
        self._queue.put_nowait((record, runner))
        logger.info(
            "入队任务 %s：类型 %s，用户 %s，提供方 %s，链路 %s，排队 %d，运行 %d",
            record.task_id,
            record.kind,
            record.user_id or "-",
            record.provider or "-",
            ",".join(record.chain or []) or "-",
            self._queue.qsize(),
            max(0, self._store.count_active() - self._queue.qsize()),
        )
        return record.task_id

    async def cancel(self, task_id: str) -> bool:
        if not self._store.cancel(task_id):
            logger.info("取消任务 %s 未生效：任务已结束或不存在", task_id)
            return False
        running = self._running.get(task_id)
        if running is not None and not running.done():
            logger.info("取消任务 %s：正在执行，通知它停下", task_id)
            running.cancel()
            # Give the worker a chance to unwind before the caller polls again.
            await asyncio.sleep(0)
        logger.info("取消任务 %s 已生效", task_id)
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
                if self._closed:
                    raise
                # 用户取消的是这一条任务，工作线程本身还要继续接活。
                logger.info("任务 %s 已停下，工作线程继续待命", record.task_id)
            except Exception:
                logger.exception("工作线程处理任务 %s 时出错", record.task_id)
            finally:
                self._queue.task_done()

    async def _run_one(
        self,
        record: TaskRecord,
        runner: Callable[[TaskRecord], Awaitable[dict]],
    ) -> None:
        fresh = self._store.get(record.task_id) or record
        if fresh.status == "cancelled":
            logger.info("跳过任务 %s：排队期间已被取消", record.task_id)
            await self._finish(fresh)
            return
        if self._owner and not await self._claim(record.task_id):
            holder = self._store.get(record.task_id)
            logger.info(
                "跳过任务 %s：另一个实例正在运行它（持有者 %s）",
                record.task_id,
                (holder.owner if holder else "") or "-",
            )
            return
        fresh = self._store.get(record.task_id) or fresh
        if fresh.status == "cancelled":
            logger.info("跳过任务 %s：开始执行前已被取消", record.task_id)
            await self._finish(fresh)
            return
        heartbeat = asyncio.create_task(self._heartbeat(record.task_id)) if self._owner else None
        current = asyncio.current_task()
        if current is not None:
            self._running[record.task_id] = current
        try:
            await self._execute(fresh, runner)
        finally:
            self._running.pop(record.task_id, None)
            if heartbeat is not None:
                heartbeat.cancel()
                await asyncio.gather(heartbeat, return_exceptions=True)
            if self._owner:
                try:
                    # Shielded: a shutdown right as the task finishes must not
                    # leave the row looking like someone still owns it.
                    await asyncio.shield(
                        self._store.release(record.task_id, self._owner)
                    )
                except Exception:
                    logger.exception("任务 %s 的租约释放失败", record.task_id)

    async def _claim(self, task_id: str) -> bool:
        claimed = bool(await self._store.claim(task_id, self._owner, self._lease_seconds))
        if claimed:
            logger.debug("任务 %s 抢占租约成功，持有者 %s", task_id, self._owner)
        return claimed

    async def _heartbeat(self, task_id: str) -> None:
        interval = max(5.0, self._lease_seconds / 3) if self._lease_seconds else 15.0
        renewals = 0
        while True:
            await asyncio.sleep(interval)
            try:
                held = await self._store.heartbeat(task_id, self._owner, self._lease_seconds)
            except Exception:
                logger.exception("任务 %s 的续租失败", task_id)
                return
            if not held:
                logger.warning("任务 %s 的租约被别的实例接管了，停止续租", task_id)
                return
            renewals += 1
            # 第一次续租说一句，之后降到 debug：长任务每十几秒一行会淹掉别的日志。
            if renewals == 1:
                logger.info(
                    "任务 %s 续租成功：租约延长 %d 秒",
                    task_id,
                    int(self._lease_seconds),
                )
            else:
                logger.debug("任务 %s 第 %d 次续租成功", task_id, renewals)

    async def _execute(
        self,
        fresh: TaskRecord,
        runner: Callable[[TaskRecord], Awaitable[dict]],
    ) -> None:
        fresh.status = "running"
        fresh.started_at = int(time.time())
        self._store.upsert(fresh)
        task_id = fresh.task_id
        started = time.monotonic()
        logger.info(
            "开始执行任务 %s：类型 %s，提供方 %s，尺寸 %s，排队 %d",
            task_id,
            fresh.kind,
            fresh.provider or "-",
            fresh.size or "-",
            self._queue.qsize(),
        )
        try:
            result = await runner(fresh)
        except QueueCancelled:
            logger.info("任务 %s 被取消，停止执行", task_id)
            self._store.cancel(task_id)
            fresh = self._store.get(task_id) or fresh
            await self._finish(fresh)
            return
        except asyncio.CancelledError:
            logger.info("任务 %s 被中断，下次启动会重新接手", task_id)
            raise
        except Exception as exc:
            self._store.set_error(task_id, str(exc), provider=fresh.provider)
            logger.warning(
                "任务 %s 执行失败，用时 %d 毫秒：%s",
                task_id,
                elapsed_ms(started),
                exc,
            )
            fresh = self._store.get(task_id) or fresh
            await self._finish(fresh)
            return

        # A cancel may have raced with completion; respect it.
        fresh = self._store.get(task_id) or fresh
        if fresh.status == "cancelled":
            logger.info("任务 %s 已出结果但先前被取消，按取消收尾", task_id)
            await self._finish(fresh)
            return
        self._store.set_result(
            task_id,
            json.dumps(result, ensure_ascii=False),
            provider=fresh.provider,
        )
        fresh = self._store.get(task_id) or fresh
        logger.info(
            "任务 %s 结束：状态 %s，用时 %d 毫秒，提供方 %s",
            task_id,
            fresh.status,
            elapsed_ms(started),
            fresh.provider or "-",
        )
        await self._finish(fresh)

    async def _finish(self, record: TaskRecord) -> None:
        if self._on_finished is not None:
            try:
                await self._on_finished(record)
            except Exception:
                logger.exception("任务 %s 的完成回调出错", record.task_id)

    @staticmethod
    async def _missing_runner(record: TaskRecord) -> dict:
        logger.warning("任务 %s 没有可用的执行器：类型 %s", record.task_id, record.kind)
        raise NoRunnerError("没有可用的执行器")
