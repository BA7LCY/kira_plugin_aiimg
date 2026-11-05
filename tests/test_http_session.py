"""共享 HTTP 会话的行为测试：全部打本地 aiohttp 服务，不访问外网。"""

from __future__ import annotations

import asyncio
import io
import socket
import threading

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from PIL import Image as PILImage

from core.http import (
    ProviderHTTP,
    close_http,
    close_http_sync,
    current_session,
    peek_session,
)
from core.net import URLFetchPolicy
from core.task_store import TaskStore

PRIVATE_IP = "10.1.2.3"
# 服务端收到的请求记录，用 AppKey 避免 aiohttp 的告警
CALLS_KEY = web.AppKey("calls", list)


def _png() -> bytes:
    """生成一张最小可用的 PNG，供下载路径做真实校验。"""
    out = io.BytesIO()
    PILImage.new("RGB", (16, 16), (12, 200, 90)).save(out, format="PNG")
    return out.getvalue()


def _fake_dns(monkeypatch, address: str) -> None:
    """把所有域名解析固定成指定地址，避免任何真实 DNS 查询。"""

    def resolve(host, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)


def _policy_for(server: TestServer) -> URLFetchPolicy:
    """把本地测试服务登记成受信任来源，其它地址仍按默认策略拒绝。"""
    return URLFetchPolicy(trusted_origins=frozenset({f"http://127.0.0.1:{server.port}"}))


def _policy_for_both(*servers: TestServer) -> URLFetchPolicy:
    """把多个本地服务都登记成受信任来源，供跨 origin 跳转的用例使用。"""
    return URLFetchPolicy(
        trusted_origins=frozenset(
            f"http://127.0.0.1:{server.port}" for server in servers
        )
    )


@pytest.fixture(autouse=True)
def clean_shared_session():
    """每个用例前后都清掉共享会话，避免用例之间互相污染。"""
    close_http_sync()
    yield
    close_http_sync()


@pytest.fixture()
async def demo_server():
    """本地测试服务：覆盖成功、500、超时、重定向、图片下载几条路径。"""
    app = web.Application()
    app[CALLS_KEY] = []

    async def ok(request: web.Request) -> web.Response:
        app[CALLS_KEY].append({"path": request.path, "method": request.method})
        return web.json_response({"ok": True, "path": request.path})

    async def echo(request: web.Request) -> web.Response:
        app[CALLS_KEY].append({"path": request.path, "method": request.method})
        return web.json_response(
            {
                "authorization": request.headers.get("Authorization", ""),
                "trace": request.headers.get("X-Trace", ""),
            }
        )

    async def fail(request: web.Request) -> web.Response:
        app[CALLS_KEY].append({"path": request.path, "method": request.method})
        return web.Response(status=500, text="接口炸了")

    async def slow(request: web.Request) -> web.Response:
        await asyncio.sleep(1.2)
        return web.json_response({"late": True})

    async def redirect_internal(request: web.Request) -> web.Response:
        app[CALLS_KEY].append({"path": request.path, "method": request.method})
        raise web.HTTPFound("http://blocked.internal/secret.png")

    async def redirect_image(request: web.Request) -> web.Response:
        app[CALLS_KEY].append({"path": request.path, "method": request.method})
        raise web.HTTPFound("/image")

    async def redirect_ok(request: web.Request) -> web.Response:
        app[CALLS_KEY].append({"path": request.path, "method": request.method})
        raise web.HTTPFound("/ok")

    async def image(request: web.Request) -> web.Response:
        app[CALLS_KEY].append({"path": request.path, "method": request.method})
        return web.Response(body=_png(), content_type="image/png")

    app.router.add_get("/ok", ok)
    app.router.add_get("/echo", echo)
    app.router.add_get("/fail", fail)
    app.router.add_get("/slow", slow)
    app.router.add_get("/redirect-internal", redirect_internal)
    app.router.add_get("/redirect-image", redirect_image)
    app.router.add_get("/redirect-ok", redirect_ok)
    app.router.add_get("/image", image)

    server = TestServer(app)
    await server.start_server()
    try:
        yield server
    finally:
        await server.close()


def _base(server: TestServer) -> str:
    return str(server.make_url("/"))


@pytest.mark.asyncio
async def test_两次请求复用同一个会话(demo_server, monkeypatch):
    """两次请求必须落在同一个 ClientSession 与同一个连接器上。"""
    created: list[aiohttp.ClientSession] = []
    real_session = aiohttp.ClientSession

    def recording_session(*args, **kwargs):
        session = real_session(*args, **kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(aiohttp, "ClientSession", recording_session)

    client = ProviderHTTP(policy=_policy_for(demo_server))
    base = _base(demo_server)
    first = await client.request_json("GET", base + "ok")
    session_one = peek_session()
    second = await client.request_json("GET", base + "ok")
    session_two = peek_session()

    assert first == {"ok": True, "path": "/ok"}
    assert second == {"ok": True, "path": "/ok"}
    assert session_one is not None and session_two is session_one
    assert session_one.connector is session_two.connector
    assert session_one.connector is not None
    assert len(created) == 1, "共享会话只应创建一次"
    assert not session_one.closed


@pytest.mark.asyncio
async def test_请求头与返回值结构保持原样(demo_server):
    """headers 原样透传，返回值仍是 dict。"""
    client = ProviderHTTP(policy=_policy_for(demo_server))
    result = await client.request_json(
        "GET",
        _base(demo_server) + "echo",
        headers={"Authorization": "Bearer sk-secret-123456", "X-Trace": "trace-1"},
    )
    assert result == {"authorization": "Bearer sk-secret-123456", "trace": "trace-1"}


@pytest.mark.asyncio
async def test_关闭后再次请求会自动重建会话(demo_server):
    """close_http 之后共享会话应为空，下一次请求自动重建。"""
    client = ProviderHTTP(policy=_policy_for(demo_server))
    base = _base(demo_server)
    await client.request_json("GET", base + "ok")
    first = peek_session()
    assert first is not None

    await close_http()
    assert first.closed
    assert peek_session() is None

    await client.request_json("GET", base + "ok")
    second = peek_session()
    assert second is not None
    assert second is not first
    assert not second.closed


@pytest.mark.asyncio
async def test_关闭会话是幂等的(demo_server):
    """连续关闭多次不应报错，重复关闭只打日志。"""
    await current_session()
    await close_http()
    await close_http()
    await close_http()
    assert peek_session() is None
    # 在事件循环里调用同步兜底也不能炸，只会安排异步关闭
    close_http_sync()
    assert peek_session() is None


def test_同步兜底能在事件循环外关闭会话():
    """没有事件循环时 close_http_sync 也能关闭会话，并且幂等。"""

    async def make_session() -> aiohttp.ClientSession:
        return await current_session()

    session = asyncio.run(make_session())
    assert session is not None and not session.closed
    assert peek_session() is session

    close_http_sync()
    assert session.closed
    assert peek_session() is None
    close_http_sync()


def test_事件循环变化会重建共享会话():
    """换事件循环后必须新建会话，不能复用旧循环上的会话。"""

    async def first() -> aiohttp.ClientSession:
        return await current_session()

    async def second() -> aiohttp.ClientSession:
        return await current_session()

    session_one = asyncio.run(first())
    session_two = asyncio.run(second())
    assert session_two is not session_one
    assert session_two is peek_session()
    assert not session_two.closed


@pytest.mark.asyncio
async def test_连接池参数符合要求():
    """连接池上限、单主机上限、DNS 缓存时间必须与约定一致。"""
    session = await current_session()
    connector = session.connector
    assert connector is not None
    assert connector.limit == 32
    assert connector.limit_per_host == 8
    assert connector._cached_hosts._ttl == 300
    # 单次请求各自带超时，会话本身不设总上限
    assert session.timeout.total is None


@pytest.mark.asyncio
async def test_请求前先拦截内网地址(monkeypatch):
    """地址校验发生在建会话/发请求之前，非法地址不会留下会话。"""
    _fake_dns(monkeypatch, PRIVATE_IP)
    client = ProviderHTTP()
    with pytest.raises(ValueError, match="内网"):
        await client.request_json("GET", "http://files.internal/a.json")
    assert peek_session() is None


@pytest.mark.asyncio
async def test_下载前先拦截内网地址(monkeypatch):
    """下载路径同样先校验再建会话，非法地址不会留下会话。"""
    _fake_dns(monkeypatch, PRIVATE_IP)
    client = ProviderHTTP()
    with pytest.raises(ValueError, match="内网"):
        await client.download_image("http://files.internal/a.png")
    assert peek_session() is None


@pytest.mark.asyncio
async def test_下载重定向到内网会被拦截(demo_server, monkeypatch):
    """第一跳合法，重定向到内网域名时必须在第二跳之前拦下。"""
    _fake_dns(monkeypatch, PRIVATE_IP)
    client = ProviderHTTP(policy=_policy_for(demo_server))
    with pytest.raises(ValueError, match="内网"):
        await client.download_image(
            _base(demo_server) + "redirect-internal", follow_redirects=True
        )
    # 说明校验发生在收到第一跳的重定向之后、发起第二跳之前
    assert {"path": "/redirect-internal", "method": "GET"} in demo_server.app[CALLS_KEY]


@pytest.mark.asyncio
async def test_接口重定向到内网会被拦截(demo_server, monkeypatch):
    """接口请求开启策略校验时，重定向同样逐跳校验。"""
    _fake_dns(monkeypatch, PRIVATE_IP)
    client = ProviderHTTP(policy=_policy_for(demo_server))
    with pytest.raises(ValueError, match="内网"):
        await client.request_json("GET", _base(demo_server) + "redirect-internal")
    assert {"path": "/redirect-internal", "method": "GET"} in demo_server.app[CALLS_KEY]


@pytest.mark.asyncio
async def test_合法重定向仍能正常请求(demo_server):
    """逐跳校验不会误伤合法重定向，最终仍拿到目标地址的结果。"""
    client = ProviderHTTP(policy=_policy_for(demo_server))
    result = await client.request_json("GET", _base(demo_server) + "redirect-ok")
    assert result == {"ok": True, "path": "/ok"}
    hits = demo_server.app[CALLS_KEY]
    assert {"path": "/redirect-ok", "method": "GET"} in hits
    assert {"path": "/ok", "method": "GET"} in hits


@pytest.mark.asyncio
async def test_下载默认不跟随重定向保持原异常(demo_server):
    """默认参数下 302 仍然直接报错，与改造前的行为一致。"""
    client = ProviderHTTP(policy=_policy_for(demo_server))
    with pytest.raises(RuntimeError, match="HTTP 302"):
        await client.download_image(_base(demo_server) + "redirect-image")


@pytest.mark.asyncio
async def test_跟随重定向后能正常下载图片(demo_server):
    """显式开启跟随后，合法重定向可以下载到图片并走图片校验。"""
    client = ProviderHTTP(policy=_policy_for(demo_server))
    data = await client.download_image(
        _base(demo_server) + "redirect-image", follow_redirects=True
    )
    assert data == _png()
    assert peek_session() is not None


@pytest.mark.asyncio
async def test_非200响应仍抛RuntimeError(demo_server):
    """非 2xx 的异常类型与提示保持原样。"""
    client = ProviderHTTP(policy=_policy_for(demo_server))
    with pytest.raises(RuntimeError) as excinfo:
        await client.request_json("GET", _base(demo_server) + "fail")
    message = str(excinfo.value)
    assert message.startswith("接口返回 HTTP 500: ")
    assert "接口炸了" in message


@pytest.mark.asyncio
async def test_超时异常类型不变且会话仍可用(demo_server):
    """超时仍抛 TimeoutError，超时不会把共享会话弄坏。"""
    client = ProviderHTTP(timeout=1, policy=_policy_for(demo_server))
    base = _base(demo_server)
    with pytest.raises((asyncio.TimeoutError, TimeoutError)):
        await client.request_json("GET", base + "slow")
    session = peek_session()
    assert session is not None and not session.closed
    assert await client.request_json("GET", base + "ok") == {"ok": True, "path": "/ok"}


@pytest.mark.asyncio
async def test_下载图片复用共享会话(demo_server):
    """图片下载路径也落在同一个共享会话上。"""
    client = ProviderHTTP(policy=_policy_for(demo_server))
    base = _base(demo_server)
    data = await client.download_image(base + "image")
    session = peek_session()
    assert data == _png()
    assert session is not None
    assert await client.request_json("GET", base + "ok") == {"ok": True, "path": "/ok"}
    assert peek_session() is session


# -- 跨 origin 重定向的凭据处理 -------------------------------------------
#
# 用两个不同端口的本地服务模拟「一个主机把请求跳到另一个主机」。
# 两个 origin 都登记成受信任来源，地址校验因此不会拦住第二跳，
# 这样才能单独观察 Authorization 等凭据头有没有被带过去。

def _credential_report(request: web.Request) -> dict:
    """把服务端实际收到的凭据头原样回显，便于断言。"""
    return {
        "authorization": request.headers.get("Authorization", ""),
        "proxy_authorization": request.headers.get("Proxy-Authorization", ""),
        "cookie": request.headers.get("Cookie", ""),
        "trace": request.headers.get("X-Trace", ""),
        "path": request.path,
    }


@pytest.fixture()
async def redirect_pair():
    """source 服务按 ?to= 跳转，target 服务回显收到的请求头。"""
    seen: list[dict] = []

    async def inspect(request: web.Request) -> web.Response:
        report = _credential_report(request)
        seen.append(report)
        return web.json_response(report)

    async def hop(request: web.Request) -> web.Response:
        raise web.HTTPFound(request.query.get("to") or "/inspect")

    async def hop_image(request: web.Request) -> web.Response:
        raise web.HTTPFound(request.query.get("to") or "/image")

    async def image(request: web.Request) -> web.Response:
        seen.append(_credential_report(request))
        return web.Response(body=_png(), content_type="image/png")

    source_app = web.Application()
    source_app.router.add_get("/hop", hop)
    source_app.router.add_get("/hop-image", hop_image)
    source_app.router.add_get("/inspect", inspect)
    source_app.router.add_get("/image", image)

    target_app = web.Application()
    target_app.router.add_get("/inspect", inspect)
    target_app.router.add_get("/image", image)

    source = TestServer(source_app)
    await source.start_server()
    target = TestServer(target_app)
    await target.start_server()
    try:
        yield source, target, seen
    finally:
        await target.close()
        await source.close()


def _credential_headers() -> dict[str, str]:
    return {
        "Authorization": "Bearer sk-secret-123456",
        "Proxy-Authorization": "Basic cHJveHk=",
        "Cookie": "sid=abc",
        "X-Trace": "trace-1",
    }


@pytest.mark.asyncio
async def test_同源重定向保留凭据头(redirect_pair):
    """跳转目标与原始地址同 origin 时，凭据头必须照旧带上。"""
    source, _target, _seen = redirect_pair
    client = ProviderHTTP(policy=_policy_for(source))
    result = await client.request_json(
        "GET", _base(source) + "hop?to=/inspect", headers=_credential_headers()
    )
    assert result["path"] == "/inspect"
    assert result["authorization"] == "Bearer sk-secret-123456"
    assert result["proxy_authorization"] == "Basic cHJveHk="
    assert result["cookie"] == "sid=abc"
    assert result["trace"] == "trace-1"


@pytest.mark.asyncio
async def test_跨域重定向剥掉凭据头(redirect_pair):
    """跳到另一个 origin 时，三个凭据头都必须被剥掉，普通头照常保留。"""
    source, target, _seen = redirect_pair
    client = ProviderHTTP(policy=_policy_for_both(source, target))
    to = str(target.make_url("/inspect"))
    result = await client.request_json(
        "GET", f"{_base(source)}hop?to={to}", headers=_credential_headers()
    )
    assert result["path"] == "/inspect"
    assert result["authorization"] == ""
    assert result["proxy_authorization"] == ""
    assert result["cookie"] == ""
    assert result["trace"] == "trace-1"


@pytest.mark.asyncio
async def test_图片下载跨域重定向也剥掉凭据头(redirect_pair):
    """下载路径同样逐跳重算请求头，跨域时不能把凭据带过去。"""
    source, target, seen = redirect_pair
    client = ProviderHTTP(policy=_policy_for_both(source, target))
    to = str(target.make_url("/image"))
    data = await client.download_image(
        f"{_base(source)}hop-image?to={to}",
        follow_redirects=True,
        headers=_credential_headers(),
    )
    assert data == _png()
    last = seen[-1]
    assert last["path"] == "/image"
    assert last["authorization"] == ""
    assert last["proxy_authorization"] == ""
    assert last["cookie"] == ""
    assert last["trace"] == "trace-1"


# -- 任务库跨线程回归 -----------------------------------------------------
#
# 后台清理由 main.py 通过 asyncio.to_thread 调度，会在另一个线程里用同一个
# sqlite 连接，所以连接要允许跨线程、写操作要串行化。测试改动范围只放开
# 本文件与 test_net_policy.py，任务库的这两条回归用例因此放在这里。


@pytest.mark.asyncio
async def test_任务库能在后台线程里执行清理(tmp_path):
    """to_thread 里调用 prune_old 不能因为线程检查而失败。"""
    store = TaskStore(tmp_path)
    try:
        record = store.create(kind="image", user_id="u1")
        store.set_result(record.task_id, "ok")
        # 直接把完成时间改到很久以前，避免用例依赖真实时间流逝
        store._db.execute(
            "UPDATE tasks SET finished_at = 1 WHERE task_id = ?", (record.task_id,)
        )
        removed = await asyncio.to_thread(store.prune_old, 1)
        assert removed == 1
        assert store.get(record.task_id) is None
    finally:
        store.close()


def test_任务库支持多线程并发写(tmp_path):
    """多个线程同时经同一个连接写库时，不能丢记录也不能抛异常。"""
    store = TaskStore(tmp_path)
    created: list[str] = []
    failures: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            record = store.create(kind="image", user_id=f"u{index}")
            created.append(record.task_id)
            store.set_status(record.task_id, "running")
            store.set_result(record.task_id, "ok")
        except BaseException as exc:  # noqa: BLE001 - 收集到主线程统一断言
            failures.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
        assert not failures, failures
        assert len(created) == 8
        assert len(set(created)) == 8
        assert store.count_active() == 0
        for task_id in created:
            record = store.get(task_id)
            assert record is not None
            assert record.status == "succeeded"
    finally:
        store.close()
