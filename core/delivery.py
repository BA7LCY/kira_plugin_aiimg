"""Delivery with retries: a finished image must not be lost to one bad request.

The module stays framework-free on purpose: callers hand in the chain object
and a tiny factory so `core/` can be imported without the bot runtime.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable


class DeliveryResult:
    """Outcome of one chain delivery attempt sequence."""

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
) -> DeliveryResult:
    """Send a chain, retrying transient failures and then salvaging element by element.

    Adapters reject a whole chain when one element is malformed, so the last
    resort is to send each element as its own message: one broken image can no
    longer swallow a batch of four.
    """
    total = max(1, int(attempts))
    delay = max(0.0, float(base_delay))
    last_error = ""
    for attempt in range(1, total + 1):
        try:
            result = await sender(sid, chain)
        except Exception as exc:  # adapter raised instead of returning a failure
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("[aiimg] 投递失败（第 %d/%d 次）：%s", attempt, total, last_error)
        else:
            if getattr(result, "ok", True):
                return DeliveryResult(True, attempt)
            last_error = str(getattr(result, "err", "") or "适配器返回失败")
            logger.warning("[aiimg] 投递被拒绝（第 %d/%d 次）：%s", attempt, total, last_error)
        if attempt < total:
            await asyncio.sleep(min(delay * (2 ** (attempt - 1)), 8.0))
    elements = list(chain)
    if not split_media or len(elements) <= 1:
        return DeliveryResult(False, total, last_error)

    logger.info("[aiimg] 整条投递失败，改为逐段重发（%d 个元素）", len(elements))
    sent_any = False
    for element in elements:
        piece = make_chain([element])
        for attempt in range(1, total + 1):
            try:
                result = await sender(sid, piece)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                result = None
            if result is None or getattr(result, "ok", True):
                sent_any = True
                break
            last_error = str(getattr(result, "err", "") or "适配器返回失败")
            if attempt < total:
                await asyncio.sleep(min(delay * (2 ** (attempt - 1)), 8.0))
        else:
            logger.warning("[aiimg] 单个%s元素投递失败：%s", element_kind(element), last_error)
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
    """Short text notice with one retry; no element splitting involved."""
    return await deliver_chain(
        sender, sid, make_chain([make_text(text)]), make_chain=make_chain,
        logger=logger, attempts=2, split_media=False,
    )
