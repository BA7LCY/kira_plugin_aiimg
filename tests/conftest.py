"""Make the plugin package importable when tests run from the tests dir."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def _close_shared_http_sync():
    """同步收尾：同步用例里用 asyncio.run 建出来的会话，循环已经销毁。"""
    yield
    from core.http import close_http_sync

    close_http_sync()


@pytest.fixture(autouse=True)
async def _close_shared_http(_close_shared_http_sync):
    """异步收尾：在当前事件循环里关掉共享连接池。"""
    yield
    from core.http import close_http

    await close_http()
