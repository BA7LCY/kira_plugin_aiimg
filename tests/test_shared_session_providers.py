"""服务商与视频下载必须真正复用 core.http 的共享会话。

这些用例不联网：要么用本地 aiohttp 服务，要么用假会话只检查调用参数，
重点是三件事——共享同一个 ClientSession、close() 不关共享会话、
每个请求仍带自己的超时。
"""

import asyncio
import base64
import io

import aiohttp
import aiohttp.web
import pytest
from PIL import Image

from core import http as core_http
from core.net import URLFetchPolicy
from core.providers.base import ProviderSettings
from core.providers.openai_chat import OpenAIChatImageBackend
from core.providers.openai_compat import OpenAICompatBackend
from core.providers.openai_full_url import OpenAIFullURLBackend
from core import video_file


def _png() -> bytes:
    buffer = io.BytesIO()
    # 64x64 纯色图足够大，不会被占位图检测误判。
    Image.new("RGB", (64, 64), (30, 60, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


def _settings(**overrides) -> ProviderSettings:
    values = dict(
        id="shared-session",
        template_key="openai_images",
        base_url="https://example.test/v1",
        api_keys=["sk-test"],
        model="z-image-turbo",
        max_retries=0,
        timeout=42,
    )
    values.update(overrides)
    return ProviderSettings(**values)


class _FakeResponse:
    """只实现服务商代码用到的那几个属性。"""

    def __init__(self, body: bytes = b"{}", status: int = 200, headers=None):
        self.status = status
        self._body = body
        self.headers = headers or {"Content-Type": "application/json"}

    async def read(self) -> bytes:
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeStreamResponse(_FakeResponse):
    class _Content:
        def __init__(self, data: bytes):
            self._data = data

        async def iter_chunked(self, size: int):
            for index in range(0, len(self._data), size):
                yield self._data[index : index + size]

    def __init__(self, body: bytes, headers):
        super().__init__(body=body, headers=headers)
        self.content = self._Content(body)


class _FakeSession:
    """记录每次请求的关键字参数，用来确认超时确实逐请求传入。"""

    def __init__(self, response):
        self.response = response
        self.calls: list[tuple[str, str, dict]] = []

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return self.response

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return self.response


def _video_bytes() -> bytes:
    def box(kind: bytes, payload: bytes) -> bytes:
        # 前 4 字节是整盒长度（含 8 字节头），少了它容器解析会判为不完整。
        return (len(payload) + 8).to_bytes(4, "big") + kind + payload

    handler = box(b"hdlr", b"\0" * 8 + b"vide" + b"\0" * 12)
    return (
        box(b"ftyp", b"isom\0\0\0\0isom")
        + box(b"moov", box(b"trak", box(b"mdia", handler)))
        + box(b"mdat", b"sample payload")
    )


@pytest.fixture(autouse=True)
def clean_shared_session():
    """每个用例前后都把共享会话清干净，避免事件循环切换留下旧连接。"""
    core_http.close_http_sync()
    yield
    core_http.close_http_sync()


def test_all_backends_share_one_session_and_close_does_not_touch_it():
    """三个服务商拿到同一个会话；close() 只清自己的引用，不关共享会话。"""

    async def scenario():
        backends = [
            OpenAICompatBackend(_settings()),
            OpenAIChatImageBackend(_settings()),
            OpenAIFullURLBackend(_settings()),
        ]
        sessions = [await backend._session_get() for backend in backends]
        shared = await core_http.current_session()
        assert all(session is shared for session in sessions)

        for backend in backends:
            await backend.close()
        # 关键点：服务商 close() 之后共享会话仍可用，等插件统一关闭
        assert not shared.closed
        assert core_http.peek_session() is shared
        await core_http.close_http()

    asyncio.run(scenario())


def test_two_backends_really_reuse_one_connection_pool(monkeypatch):
    """真实请求两次：只允许创建一个 ClientSession（连接池共享）。"""
    created: list[aiohttp.ClientSession] = []
    real_session = aiohttp.ClientSession

    def recording_session(*args, **kwargs):
        session = real_session(*args, **kwargs)
        created.append(session)
        return session

    monkeypatch.setattr(aiohttp, "ClientSession", recording_session)

    async def handler(request):
        return aiohttp.web.json_response({"data": [{"b64_json": base64.b64encode(_png()).decode()}]})

    async def scenario():
        app = aiohttp.web.Application()
        app.router.add_post("/v1/images/generations", handler)
        runner = aiohttp.web.AppRunner(app)
        await runner.setup()
        site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            root = f"http://127.0.0.1:{port}/v1"
            first = OpenAICompatBackend(_settings(base_url=root))
            second = OpenAICompatBackend(_settings(base_url=root))
            for backend in (first, second):
                result = (await backend.generate("一只猫"))[0]
                assert result.data == _png()
                await backend.close()
            assert len(created) == 1, "两次请求应当共用一个 ClientSession"
            assert not created[0].closed
        finally:
            await runner.cleanup()
            await core_http.close_http()

    asyncio.run(scenario())


def test_provider_requests_keep_their_own_timeout(monkeypatch):
    """请求上要带各自的 ClientTimeout，不能依赖共享会话的默认值。"""

    async def scenario():
        # 图片接口 JSON 请求
        compat = OpenAICompatBackend(_settings(timeout=42))
        compat_session = _FakeSession(_FakeResponse(b"{}"))

        async def compat_session_get():
            return compat_session

        monkeypatch.setattr(compat, "_session_get", compat_session_get)
        await compat._post_json("https://example.test/v1/images/generations", "k", {})
        timeout = compat_session.calls[0][2]["timeout"]
        assert isinstance(timeout, aiohttp.ClientTimeout)
        assert timeout.total == 42

        # 对话接口请求
        chat = OpenAIChatImageBackend(_settings(timeout=57))
        chat_session = _FakeSession(_FakeResponse(b'{"choices": []}'))

        async def chat_session_get():
            return chat_session

        monkeypatch.setattr(chat, "_session_get", chat_session_get)
        await chat._post_chat(key="k", model="m", messages=[], extra_body={}, stream=False)
        timeout = chat_session.calls[0][2]["timeout"]
        assert timeout.total == 57

        # 完整地址接口请求
        full = OpenAIFullURLBackend(_settings(timeout=68))
        full_session = _FakeSession(_FakeResponse(b"{}"))

        async def full_session_get():
            return full_session

        monkeypatch.setattr(full, "_session_get", full_session_get)
        await full._post_json("https://example.test/v1/generate", {"prompt": "x"})
        timeout = full_session.calls[0][2]["timeout"]
        assert timeout.total == 68

    asyncio.run(scenario())


def test_video_download_uses_shared_session_with_per_request_timeout(monkeypatch, tmp_path):
    """视频分段下载不再自建会话，且每次 GET 都带原来的总超时与读超时。"""
    payload = _video_bytes()
    session = _FakeSession(
        _FakeStreamResponse(
            payload,
            {"Content-Type": "video/mp4", "Content-Length": str(len(payload))},
        )
    )

    async def fake_current_session():
        return session

    monkeypatch.setattr(video_file, "current_session", fake_current_session)

    def fail_client_session(*args, **kwargs):
        raise AssertionError("视频下载不应再新建 ClientSession")

    monkeypatch.setattr(aiohttp, "ClientSession", fail_client_session)

    policy = URLFetchPolicy(trusted_origins=frozenset({"http://127.0.0.1:9"}))
    path = asyncio.run(
        video_file.download_video(
            "http://127.0.0.1:9/clip.mp4",
            tmp_path,
            policy,
            timeout=33,
            parallel=False,
        )
    )

    assert path.read_bytes() == payload
    assert session.calls, "下载应当走共享会话发起请求"
    for _method, _url, kwargs in session.calls:
        timeout = kwargs["timeout"]
        assert timeout.total == 33
        assert timeout.sock_read == 30
    # 没有新建共享会话，说明视频模块没有再自建连接池
    assert core_http.peek_session() is None
