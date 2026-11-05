"""带重试的投递：一张画好的图不该被一次不稳定的请求弄丢。

这里刻意不依赖框架：调用方把消息链和一个小工厂函数传进来，`core/` 才能在没有
机器人运行时的环境里被导入和测试。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable


class DeliveryResult:
    """一条消息链投递完的结果。"""

    def __init__(self, ok: bool, attempts: int, error: str = "") -> None:
        self.ok = ok
        self.attempts = attempts
        self.error = error

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        return f"DeliveryResult(ok={self.ok}, attempts={self.attempts}, error={self.error!r})"


def element_kind(element: Any) -> str:
    name = type(element).__name__
    return "媒体" if name in {"Image", "Video", "File", "Record"} else "文本"


# 适配器最终拿到的是本地文件，还是一段字节，日志里分开说，出问题时少猜一步。
_PAYLOAD_ATTRS = ("file", "image", "path", "url", "src")


def _is_bytes_payload(value: Any) -> bool:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return True
    if isinstance(value, str):
        return value.startswith("base64://") or value.startswith("data:")
    return False


def payload_route(element: Any) -> str:
    """这个元素最终走文件、字节还是文本。"""
    for name in _PAYLOAD_ATTRS:
        try:
            value = getattr(element, name, None)
        except Exception:
            continue
        if value is None or value == "":
            continue
        return "字节" if _is_bytes_payload(value) else "文件"
    try:
        data = getattr(element, "data", None)
    except Exception:
        data = None
    if data is not None and data != "":
        return "字节" if _is_bytes_payload(data) else "文件"
    return element_kind(element)


def route_summary(elements: list[Any]) -> str:
    """一行说清这批元素各有几个走文件、字节和文本。"""
    counts: dict[str, int] = {}
    for element in elements:
        route = payload_route(element)
        counts[route] = counts.get(route, 0) + 1
    return "、".join(f"{name} {count} 个" for name, count in counts.items()) or "空"


async def _send_once(
    sender: Callable[[str, Any], Any],
    sid: str,
    chain: Any,
    timeout: float = 0.0,
) -> tuple[bool, str, bool]:
    """发一次，返回（是否成功, 失败原因, 是否抛了异常）。

    适配器有的返回失败对象、有的直接抛错，这里统一成一种形状，调用方只管重试。
    给了超时就用它兜住整个发送，避免一次卡住的请求把任务永远挂在那里。
    """
    try:
        if timeout > 0:
            result = await asyncio.wait_for(sender(sid, chain), timeout=timeout)
        else:
            result = await sender(sid, chain)
    except asyncio.TimeoutError:
        return False, f"发送超时（{int(timeout)} 秒）", True
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", True
    if getattr(result, "ok", True):
        return True, "", False
    return False, str(getattr(result, "err", "") or "适配器返回失败"), False


async def deliver_chain(
    sender: Callable[[str, Any], Any],
    sid: str,
    chain: Any,
    *,
    make_chain: Callable[[list[Any]], Any],
    logger,
    attempts: int = 3,
    base_delay: float = 1.0,
    split_media: bool = True,
    timeout_seconds: float = 0.0,
) -> DeliveryResult:
    """先整条重试，实在不行再逐段发送。

    适配器遇到一个格式不对的元素会拒掉整条消息，所以最后一步是把每个元素单独
    发出去：一张坏图不该连累同一批的另外三张。
    """
    total = max(1, int(attempts))
    delay = max(0.0, float(base_delay))
    last_error = ""
    elements = list(chain)
    route = route_summary(elements)
    logger.info(
        "[aiimg] 投递开始：%d 个元素（%s），最多 %d 轮",
        len(elements),
        route,
        total,
    )
    started = time.monotonic()
    for attempt in range(1, total + 1):
        ok, error, raised = await _send_once(sender, sid, chain, timeout_seconds)
        if ok:
            logger.info(
                "[aiimg] 投递成功：第 %d/%d 轮，走%s，用时 %d 毫秒",
                attempt,
                total,
                route,
                int((time.monotonic() - started) * 1000),
            )
            return DeliveryResult(True, attempt)
        last_error = error
        if raised:
            logger.warning("[aiimg] 投递失败（第 %d/%d 轮）：%s", attempt, total, error)
        else:
            logger.warning("[aiimg] 投递被拒绝（第 %d/%d 轮）：%s", attempt, total, error)
        if attempt < total:
            await asyncio.sleep(min(delay * (2 ** (attempt - 1)), 8.0))
    if not split_media or len(elements) <= 1:
        logger.warning(
            "[aiimg] 投递最终失败：%d 轮都没成功，%s",
            total,
            last_error or "原因未知",
        )
        return DeliveryResult(False, total, last_error)

    logger.info(
        "[aiimg] 整条投递失败，改为逐段重发（%d 个元素：%s）",
        len(elements),
        route,
    )
    sent_any = False
    sent_ok = 0
    sent_fail = 0
    for element in elements:
        piece = make_chain([element])
        element_route = payload_route(element)
        element_error = ""
        for attempt in range(1, total + 1):
            ok, error, _raised = await _send_once(sender, sid, piece, timeout_seconds)
            if ok:
                sent_any = True
                sent_ok += 1
                break
            element_error = error
            if attempt < total:
                await asyncio.sleep(min(delay * (2 ** (attempt - 1)), 8.0))
        else:
            sent_fail += 1
            last_error = element_error or last_error
            logger.warning(
                "[aiimg] 单个%s元素（走%s）投递失败：%s",
                element_kind(element),
                element_route,
                element_error or "原因未知",
            )
    # 逐段发送的量可能不小，只汇总一行，不为每段都写日志。
    logger.info(
        "[aiimg] 逐段投递结束：成功 %d 段，失败 %d 段，用时 %d 毫秒",
        sent_ok,
        sent_fail,
        int((time.monotonic() - started) * 1000),
    )
    return DeliveryResult(sent_any, total, last_error)


async def deliver_text(
    sender: Callable[[str, Any], Any],
    sid: str,
    text: str,
    *,
    make_text: Callable[[str], Any],
    make_chain: Callable[[list[Any]], Any],
    logger,
) -> DeliveryResult:
    """一句短提示，只重试不拆段。"""
    logger.info("[aiimg] 投递文字提示：%d 字", len(str(text or "")))
    return await deliver_chain(
        sender, sid, make_chain([make_text(text)]), make_chain=make_chain,
        logger=logger, attempts=2, split_media=False,
    )
