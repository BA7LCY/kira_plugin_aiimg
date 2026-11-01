"""AgnesVideoService tests against a local aiohttp mock server."""

import asyncio
import base64
import pathlib
import tempfile

import aiohttp
import aiohttp.web
import pytest

from core.providers.agnes_video import AgnesVideoError, AgnesVideoService
from core.providers.base import ProviderSettings

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _settings(base_url: str, **kw) -> ProviderSettings:
    values = dict(
        id="agn",
        template_key="agnes_video",
        base_url=base_url,
        api_keys=["ak-test"],
        model="agnes-video-2.5-flash",
    )
    values.update(kw)
    return ProviderSettings(**values)


class MockAgnes:
    def __init__(self, statuses=None):
        self.creates = []
        self.polls = []
        self.statuses = statuses or ["processing", "completed"]
        self.completed_payload = {
            "status": "completed",
            "data": {"url": "https://cdn.example.com/video.mp4"},
        }
        self.create_status = 200
        self.create_body = {"video_id": "v1"}
        self.create_retry_then_ok = 0
        self.status_endpoint_fails = False
        self.app = aiohttp.web.Application()
        self.app.router.add_post("/v1/videos", self.handle_create)
        self.app.router.add_get("/agnesapi", self.handle_status)
        self.app.router.add_get("/v1/videos/{task_id}", self.handle_status)

    async def __aenter__(self):
        self.runner = aiohttp.web.AppRunner(self.app)
        await self.runner.setup()
        self.site = aiohttp.web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        self.port = self.site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        return self

    async def __aexit__(self, *exc):
        await self.runner.cleanup()

    async def handle_create(self, request: aiohttp.web.Request):
        payload = await request.json()
        self.creates.append(payload)
        if self.create_retry_then_ok:
            self.create_retry_then_ok -= 1
            return aiohttp.web.json_response({"error": "slow down"}, status=429)
        return aiohttp.web.json_response(self.create_body, status=self.create_status)

    async def handle_status(self, request: aiohttp.web.Request):
        self.polls.append(str(request.rel_url))
        if self.status_endpoint_fails and request.path == "/agnesapi":
            return aiohttp.web.json_response({"error": "not found"}, status=404)
        status = self.statuses.pop(0) if self.statuses else "completed"
        if status == "completed":
            return aiohttp.web.json_response(self.completed_payload)
        if status == "failed":
            return aiohttp.web.json_response({"status": "failed", "error": "model exploded"})
        return aiohttp.web.json_response({"status": status})


async def _run(coro):
    return await asyncio.wait_for(coro, timeout=10)


@pytest.fixture
def mock():
    return MockAgnes()


def _service(mock, tmp_path, **kw):
    service = AgnesVideoService(_settings(mock.base_url, **kw), tmp_path)
    service.poll_interval = 0.05
    service.interval = 0
    return service


def _png(tmp_path) -> pathlib.Path:
    path = pathlib.Path(tempfile.mkdtemp(dir=tmp_path)) / "a.png"
    path.write_bytes(PNG_1PX)
    return path


def test_endpoints_are_derived_from_base(mock, tmp_path):
    async def scenario():
        async with mock:
            service = _service(mock, tmp_path)
            assert service.api_url == f"{mock.base_url}/videos"
            assert service.status_url == f"http://127.0.0.1:{mock.port}/agnesapi"

        async with mock:
            service = AgnesVideoService(_settings(f"{mock.base_url}/videos"), tmp_path)
            assert service.api_url == f"{mock.base_url}/videos"

    asyncio.run(scenario())


def test_generate_video_success(mock, tmp_path):
    async def scenario():
        async with mock:
            service = _service(mock, tmp_path)
            result = await _run(
                service.generate_video("一只猫在跑步", seconds="5", extra={"aspect_ratio": "16:9"})
            )
            assert result.url == "https://cdn.example.com/video.mp4"
            assert result.model == "agnes-video-2.5-flash"
            assert result.size == "720P"
            assert result.duration == "5"
            payload = mock.creates[0]
            assert payload["seconds"] == "5"
            assert payload["size"] == "720P"
            assert payload["aspect_ratio"] == "16:9"
            assert payload["n"] == 1
            assert payload["mode"] == "text"
            await service.close()

    asyncio.run(scenario())


def test_invalid_seconds(mock, tmp_path):
    async def scenario():
        async with mock:
            service = _service(mock, tmp_path)
            with pytest.raises(AgnesVideoError) as exc:
                await _run(service.generate_video("x", seconds="99"))
            assert exc.value.stop_provider_chain is True
            assert "4 至 12" in str(exc.value)
            await service.close()

    asyncio.run(scenario())


def test_invalid_aspect_ratio(mock, tmp_path):
    async def scenario():
        async with mock:
            service = _service(mock, tmp_path)
            with pytest.raises(AgnesVideoError) as exc:
                await _run(
                    service.generate_video("x", seconds="5", extra={"aspect_ratio": "5:1"})
                )
            assert "画幅" in str(exc.value)
            await service.close()

    asyncio.run(scenario())


def test_non_720p_size_rejected(mock, tmp_path):
    async def scenario():
        async with mock:
            service = _service(mock, tmp_path, size="1080P")
            with pytest.raises(AgnesVideoError) as exc:
                await _run(service.generate_video("x", seconds="5"))
            assert "720P" in str(exc.value)
            await service.close()

    asyncio.run(scenario())


def test_no_api_key(mock, tmp_path):
    async def scenario():
        async with mock:
            service = _service(mock, tmp_path, api_keys=[])
            with pytest.raises(AgnesVideoError) as exc:
                await _run(service.generate_video("x"))
            assert "API Key" in str(exc.value)
            assert exc.value.stop_provider_chain is True
            await service.close()

    asyncio.run(scenario())


def test_failed_status(mock, tmp_path):
    async def scenario():
        async with mock:
            service = _service(mock, tmp_path)
            mock.statuses = ["failed"]
            with pytest.raises(AgnesVideoError) as exc:
                await _run(service.generate_video("x"))
            assert "model exploded" in str(exc.value)
            await service.close()

    asyncio.run(scenario())


def test_completed_without_url(mock, tmp_path):
    async def scenario():
        async with mock:
            service = _service(mock, tmp_path)
            mock.completed_payload = {"status": "completed", "data": {"state": "done"}}
            with pytest.raises(AgnesVideoError) as exc:
                await _run(service.generate_video("x"))
            assert "未返回视频地址" in str(exc.value)
            assert exc.value.stop_provider_chain is True
            await service.close()

    asyncio.run(scenario())


def test_url_returned_even_when_status_failed(mock, tmp_path):
    async def scenario():
        async with mock:
            service = _service(mock, tmp_path)
            mock.completed_payload = {
                "status": "failed",
                "data": {"url": "https://cdn.example.com/still.mp4"},
            }
            result = await _run(service.generate_video("x"))
            assert result.url == "https://cdn.example.com/still.mp4"
            await service.close()

    asyncio.run(scenario())


def test_text_mode_rejects_images(mock, tmp_path):
    async def scenario():
        async with mock:
            service = _service(mock, tmp_path)
            with pytest.raises(AgnesVideoError) as exc:
                await _run(service.generate_video("x", mode="text", images=[_png(tmp_path)]))
            assert "纯文本" in str(exc.value)
            await service.close()

    asyncio.run(scenario())


def test_auto_mode_uses_keyframe_for_image(mock, tmp_path):
    async def scenario():
        async with mock:
            service = _service(mock, tmp_path)
            mock.statuses = ["completed"]
            result = await _run(service.generate_video("x", seconds="5", images=[_png(tmp_path)]))
            assert result.url == "https://cdn.example.com/video.mp4"
            payload = mock.creates[0]
            assert payload["mode"] == "keyframe"
            assert payload["first_frame"].startswith("data:image/png;base64,")
            assert "images" not in payload
            await service.close()

    asyncio.run(scenario())


def test_reference_mode_keeps_images_list(mock, tmp_path):
    async def scenario():
        async with mock:
            service = _service(mock, tmp_path)
            mock.statuses = ["completed"]
            await _run(
                service.generate_video("x", seconds="5", mode="reference", images=[_png(tmp_path)])
            )
            payload = mock.creates[0]
            assert payload["mode"] == "reference"
            assert payload["images"][0].startswith("data:image/png;base64,")
            await service.close()

    asyncio.run(scenario())


def test_create_retries_on_429(mock, tmp_path):
    async def scenario():
        async with mock:
            service = _service(mock, tmp_path)
            service.timeout = 120
            mock.create_retry_then_ok = 1
            mock.statuses = ["completed"]
            original_sleep = asyncio.sleep

            async def fast_sleep(seconds, *args, **kwargs):
                await original_sleep(0)

            asyncio.sleep = fast_sleep
            try:
                result = await _run(service.generate_video("x", seconds="5"))
            finally:
                asyncio.sleep = original_sleep
            assert result.url == "https://cdn.example.com/video.mp4"
            assert len(mock.creates) == 2
            await service.close()

    asyncio.run(scenario())


def test_create_500_stops_chain(mock, tmp_path):
    async def scenario():
        async with mock:
            service = _service(mock, tmp_path)
            mock.create_status = 500
            with pytest.raises(AgnesVideoError) as exc:
                await _run(service.generate_video("x", seconds="5"))
            assert exc.value.stop_provider_chain is True
            await service.close()

    asyncio.run(scenario())


def test_poll_falls_back_to_task_endpoint(mock, tmp_path):
    async def scenario():
        async with mock:
            service = _service(mock, tmp_path)
            mock.status_endpoint_fails = True
            mock.statuses = ["processing", "completed"]
            result = await _run(service.generate_video("x", seconds="5"))
            assert result.url == "https://cdn.example.com/video.mp4"
            assert any("/v1/videos/v1" in item for item in mock.polls)
            await service.close()

    asyncio.run(scenario())
