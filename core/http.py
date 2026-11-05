"""Provider 共用的 HTTP 辅助函数（进程内共享连接版）。

设计要点：
1. 整个进程只保留一个 aiohttp.ClientSession，第一次请求时懒创建；
2. 创建会话时记录所属事件循环，循环变化时先关旧的再建新的，避免跨循环复用报错；
3. 连接池统一限制为 32 条连接、单主机 8 条，DNS 缓存 300 秒；
4. 关闭入口有两个：异步 close_http()（推荐）和同步兜底 close_http_sync()，两者都幂等。
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from typing import Any
from urllib.parse import urljoin

import aiohttp

from .image_format import validate_image_bytes
from .logbook import clip, get_logger, mask
from .net import URLFetchPolicy, ensure_peer_allowed, ensure_url_allowed

logger = get_logger("http")

# 连接池上限：全局 32 条、单个主机 8 条，既复用连接又不把对端打爆。
_CONNECTION_LIMIT = 32
_CONNECTION_LIMIT_PER_HOST = 8
# DNS 缓存 300 秒：出图服务通常是同一个域名反复请求。
_DNS_CACHE_SECONDS = 300
# 单个请求最多跟随的重定向跳数，防止重定向环。
_MAX_REDIRECTS = 5
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
# 关闭旧会话最多等这么久，跨事件循环时不会把调用方卡死。
_CLOSE_TIMEOUT_SECONDS = 5.0
# 30x 里只有这几种会把请求方法降级成 GET。
_GET_REDIRECT_STATUSES = frozenset({301, 302, 303})

# 进程内共享会话及其所属事件循环，两者必须同时更新。
_shared_session: aiohttp.ClientSession | None = None
_shared_loop: asyncio.AbstractEventLoop | None = None
# 懒创建会话用的锁，随事件循环切换而重建。
_create_lock: asyncio.Lock | None = None
_create_lock_loop: asyncio.AbstractEventLoop | None = None


def _safe_url(url: str) -> str:
    """日志里不暴露查询串：查询串里经常带着密钥。"""
    text = str(url or "")
    head, sep, _query = text.partition("?")
    return f"{head}?…" if sep else head


def backoff_delay(attempt: int, *, base: float = 0.5, cap: float = 8.0) -> float:
    """带抖动的指数退避，attempt 从 0 开始。"""
    delay = min(cap, base * (2 ** max(0, int(attempt))))
    delay += delay * 0.15 * (2 * random.random() - 1)
    return max(0.05, min(cap, round(delay, 3)))


def retry_after_seconds(value: str | None) -> float | None:
    """解析 Retry-After 头；HTTP 日期格式按设计忽略。"""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        return None
    return max(0.0, min(300.0, seconds))


def _connector_kwargs() -> dict[str, Any]:
    """连接器参数；Python 3.14 起已自行处理这类连接清理，传了只会告警。"""
    kwargs: dict[str, Any] = {
        "limit": _CONNECTION_LIMIT,
        "limit_per_host": _CONNECTION_LIMIT_PER_HOST,
        "ttl_dns_cache": _DNS_CACHE_SECONDS,
    }
    if sys.version_info < (3, 14):
        kwargs["enable_cleanup_closed"] = True
    return kwargs


def _new_session() -> aiohttp.ClientSession:
    """新建共享会话；单次请求的超时由调用方逐次传入，会话本身不设总上限。"""
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(**_connector_kwargs()),
        timeout=aiohttp.ClientTimeout(total=None),
    )


async def _close_quietly(session: aiohttp.ClientSession) -> None:
    """关闭会话；清理路径上任何异常（含取消）都只记日志，绝不抛给调用方。"""
    try:
        await asyncio.wait_for(session.close(), timeout=_CLOSE_TIMEOUT_SECONDS)
    except (Exception, asyncio.CancelledError) as exc:  # 清理阶段不向调用方抛异常
        logger.warning("共享会话：关闭时出现异常，已忽略（%s: %s）", type(exc).__name__, exc)


async def _ensure_lock() -> asyncio.Lock:
    """取创建锁；锁绑定事件循环，循环变了就重建，避免跨循环加锁报错。"""
    global _create_lock, _create_lock_loop
    loop = asyncio.get_running_loop()
    if _create_lock is None or _create_lock_loop is not loop:
        _create_lock = asyncio.Lock()
        _create_lock_loop = loop
    return _create_lock


def peek_session() -> aiohttp.ClientSession | None:
    """只读查看当前共享会话，不会触发创建，可能返回已关闭的会话。"""
    return _shared_session


async def current_session() -> aiohttp.ClientSession:
    """返回进程内共享会话；首次调用时创建，事件循环变化时重建。"""
    global _shared_session, _shared_loop
    loop = asyncio.get_running_loop()
    lock = await _ensure_lock()
    async with lock:
        session = _shared_session
        if session is not None and session.closed:
            logger.info("共享会话：发现旧会话已被关闭，准备重新创建")
            _shared_session = None
            _shared_loop = None
            session = None
        if session is not None and _shared_loop is not loop:
            logger.info("共享会话：检测到事件循环已变化，先关闭旧会话再重建")
            _shared_session = None
            _shared_loop = None
            await _close_quietly(session)
            session = None
        if session is None:
            session = _new_session()
            _shared_session = session
            _shared_loop = loop
            logger.info(
                "共享会话：已创建（连接上限 %d，单主机 %d，DNS 缓存 %d 秒）",
                _CONNECTION_LIMIT,
                _CONNECTION_LIMIT_PER_HOST,
                _DNS_CACHE_SECONDS,
            )
        return session


async def close_http() -> None:
    """幂等关闭共享会话；重复调用只会打日志，不会报错。"""
    global _shared_session, _shared_loop
    session = _shared_session
    _shared_session = None
    _shared_loop = None
    if session is None:
        logger.info("共享会话：当前没有会话，无需关闭")
        return
    if session.closed:
        logger.info("共享会话：会话已经是关闭状态，跳过")
        return
    logger.info("共享会话：开始关闭")
    await _close_quietly(session)
    logger.info("共享会话：关闭完成")


def close_http_sync() -> None:
    """同步兜底关闭：在事件循环外也能用，异常一律捕获，绝不外抛。"""
    global _shared_session, _shared_loop
    session = _shared_session
    loop = _shared_loop
    _shared_session = None
    _shared_loop = None
    if session is None:
        logger.info("共享会话：当前没有会话，无需同步关闭")
        return
    if session.closed:
        logger.info("共享会话：会话已经是关闭状态，跳过同步关闭")
        return
    logger.info("共享会话：开始同步关闭")
    try:
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None:
            # 当前线程已有事件循环在跑：不能阻塞等待，交给它异步关闭
            running.create_task(_close_quietly(session))
            logger.info("共享会话：当前事件循环正在运行，已安排异步关闭")
            return
        if loop is not None and not loop.is_closed():
            if loop.is_running():
                # 会话所属循环跑在别的线程：投递关闭任务并等待结果
                future = asyncio.run_coroutine_threadsafe(_close_quietly(session), loop)
                try:
                    future.result(timeout=_CLOSE_TIMEOUT_SECONDS)
                except (Exception, asyncio.CancelledError) as exc:  # 等待超时或取消都只记日志
                    logger.warning(
                        "共享会话：跨线程同步关闭未正常完成，已忽略（%s: %s）",
                        type(exc).__name__,
                        exc,
                    )
                else:
                    logger.info("共享会话：同步关闭完成")
                return
            loop.run_until_complete(_close_quietly(session))
            logger.info("共享会话：同步关闭完成")
            return
        # 旧循环已经销毁：在临时循环里尽力关闭
        temp_loop = asyncio.new_event_loop()
        try:
            temp_loop.run_until_complete(_close_quietly(session))
        finally:
            temp_loop.close()
        logger.info("共享会话：在临时事件循环里同步关闭完成")
    except (Exception, asyncio.CancelledError) as exc:  # 兜底路径，任何异常都不外抛
        logger.warning(
            "共享会话：同步关闭失败，已忽略（%s: %s）", type(exc).__name__, exc
        )


class ProviderHTTP:
    """Aiohttp 封装：所有请求复用进程内共享会话，策略仍可注入以便测试。"""

    def __init__(
        self,
        timeout: int = 600,
        policy: URLFetchPolicy | None = None,
        *,
        max_bytes: int | None = None,
        dns_timeout: float | None = None,
    ):
        self.timeout = timeout
        self.policy = policy or URLFetchPolicy()
        self.max_bytes = max_bytes
        self.dns_timeout = dns_timeout

    def _request_timeout(self) -> aiohttp.ClientTimeout:
        """单次请求超时；语义与改造前完全一致。"""
        return aiohttp.ClientTimeout(
            total=max(1, int(self.timeout)),
            sock_connect=(self.dns_timeout if self.dns_timeout is not None else None),
        )

    async def request_json(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        data: Any | None = None,
        enforce_policy: bool = True,
    ) -> dict[str, Any]:
        """发请求并解析 JSON；enforce_policy 为真时逐跳校验目标地址。"""
        timeout = self._request_timeout()
        started = time.monotonic()
        verb = method.upper()
        logger.info(
            "请求 %s %s json=%s 认证=%s",
            verb,
            _safe_url(url),
            bool(json_body),
            mask(((headers or {}).get("Authorization") or "").replace("Bearer ", "")) or "-",
        )
        if enforce_policy:
            logger.info("地址校验：请求前先过 net 策略（%s）", _safe_url(url))
            await ensure_url_allowed(url, self.policy)
        session = await current_session()
        current_url = str(url or "")
        current_method = method
        current_json = json_body
        current_data = data
        for hop in range(_MAX_REDIRECTS + 1):
            async with session.request(
                current_method,
                current_url,
                headers=headers,
                json=current_json,
                data=current_data,
                timeout=timeout,
                # 启用策略校验时手动跟跳，才能每一跳都校验；否则保持原有的自动跟随
                allow_redirects=not enforce_policy,
            ) as response:
                if enforce_policy and response.status in _REDIRECT_STATUSES:
                    location = (response.headers.get("Location") or "").strip()
                    if location:
                        current_url = urljoin(current_url, location)
                        logger.info(
                            "地址校验：第 %d 跳重定向目标 %s",
                            hop + 1,
                            _safe_url(current_url),
                        )
                        # 关键：重定向后的地址必须重新校验，不能绕过内网限制
                        await ensure_url_allowed(current_url, self.policy)
                        if (
                            response.status in _GET_REDIRECT_STATUSES
                            and current_method.upper() != "HEAD"
                        ):
                            current_method = "GET"
                            current_json = None
                            current_data = None
                        continue
                body = await response.read()
                cost = int((time.monotonic() - started) * 1000)
                if response.status < 200 or response.status >= 300:
                    detail = body[:500].decode("utf-8", errors="replace")
                    logger.warning(
                        "响应 %s %s HTTP %s，耗时 %d 毫秒：%s",
                        verb,
                        _safe_url(current_url),
                        response.status,
                        cost,
                        clip(detail, 240),
                    )
                    raise RuntimeError(f"接口返回 HTTP {response.status}: {detail}")
                logger.info(
                    "响应 %s %s HTTP %s，耗时 %d 毫秒，正文 %d 字节",
                    verb,
                    _safe_url(current_url),
                    response.status,
                    cost,
                    len(body),
                )
                if not body:
                    return {}
                try:
                    result = json.loads(body)
                except Exception as exc:
                    raise RuntimeError("接口返回的内容不是有效 JSON") from exc
                if not isinstance(result, dict):
                    raise RuntimeError("接口返回的 JSON 格式不正确")
                return result
        raise RuntimeError(f"接口重定向次数过多（超过 {_MAX_REDIRECTS} 次）")

    async def download_image(
        self, url: str, *, follow_redirects: bool = False
    ) -> bytes:
        """下载图片；follow_redirects 为真时手动跟跳并逐跳校验地址。"""
        timeout = self._request_timeout()
        started = time.monotonic()
        logger.info("下载图片：%s", _safe_url(url))
        current_url = str(url or "").strip()
        # 首跳先校验：非法地址不建会话、不发请求
        logger.info("地址校验：第 1 跳 %s", _safe_url(current_url))
        await ensure_url_allowed(current_url, self.policy)
        session = await current_session()
        data = b""
        for hop in range(_MAX_REDIRECTS + 1):
            if hop:
                # 后续每一跳都单独校验，重定向不能把请求带到内网
                logger.info("地址校验：第 %d 跳 %s", hop + 1, _safe_url(current_url))
                await ensure_url_allowed(current_url, self.policy)
            async with session.get(
                current_url, allow_redirects=False, timeout=timeout
            ) as response:
                # 连上之后再核对一次对端地址，挡住解析结果被换掉的情况
                await ensure_peer_allowed(response, current_url, self.policy)
                if follow_redirects and response.status in _REDIRECT_STATUSES:
                    location = (response.headers.get("Location") or "").strip()
                    if not location:
                        logger.warning(
                            "下载失败：重定向缺少地址（HTTP %s）", response.status
                        )
                        raise RuntimeError(f"图片下载失败: HTTP {response.status}")
                    current_url = urljoin(current_url, location)
                    logger.info(
                        "下载重定向：第 %d 跳 -> %s", hop + 1, _safe_url(current_url)
                    )
                    continue
                if response.status != 200:
                    logger.warning(
                        "下载失败：HTTP %s，地址 %s",
                        response.status,
                        _safe_url(current_url),
                    )
                    raise RuntimeError(f"图片下载失败: HTTP {response.status}")
                data = await response.read()
                break
        else:
            raise RuntimeError(f"图片重定向次数过多（超过 {_MAX_REDIRECTS} 次）")
        logger.info(
            "下载完成：%d 字节，耗时 %d 毫秒",
            len(data),
            int((time.monotonic() - started) * 1000),
        )
        cap = int(self.max_bytes) if self.max_bytes else 24 * 1024 * 1024
        if len(data) > cap:
            raise ValueError(f"图片超过 {cap // (1024 * 1024)} MB")
        validate_image_bytes(data, cap)
        return data
