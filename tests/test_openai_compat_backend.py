"""OpenAICompatBackend tests against a local aiohttp mock server."""

import asyncio
import base64
import io

import aiohttp
import aiohttp.web
import pytest

from core.net import URLFetchPolicy, origin
from core.output_spec import OutputIntent
from core.providers.base import ProviderSettings
from core.providers.openai_compat import OpenAICompatBackend

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
B64_1PX = base64.b64encode(PNG_1PX).decode()


def _settings(base_url: str, **kw) -> ProviderSettings:
    values = dict(
        id="mock",
        template_key="openai_images",
        base_url=base_url,
        api_keys=["sk-test"],
        model="z-image-turbo",
        max_retries=1,
    )
    values.update(kw)
    return ProviderSettings(**values)


class MockServer:
    def __init__(self):
        self.requests = []
        self.responses = {}
        self.generations_404 = False
        self.require_response_format = False
        self.include_internal_url = False
        self.async_task = False
        self.async_mode_payloads = []
        self.task_polls = 0
        self.task_poll_states = ["running", "success"]
        self.app = aiohttp.web.Application()
        self.app.router.add_post("/v1/images/generations", self.handle_generations)
        self.app.router.add_post("/v1/images/edits", self.handle_edits)
        self.app.router.add_get("/api/image-tasks", self.handle_task_status)
        self.app.router.add_get("/files/out.png", self.handle_file)

    async def __aenter__(self):
        self.runner = aiohttp.web.AppRunner(self.app)
        await self.runner.setup()
        self.site = aiohttp.web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        self.port = self.site._server.sockets[0].getsockname()[1]
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self.root_url = f"http://127.0.0.1:{self.port}"
        return self

    async def __aexit__(self, *exc):
        await self.runner.cleanup()

    async def handle_generations(self, request: aiohttp.web.Request):
        payload = await request.json()
        self.requests.append(("generations", payload))
        if self.generations_404:
            return aiohttp.web.Response(status=404, text="endpoint missing")
        if self.require_response_format and payload.get("response_format") != "b64_json":
            return aiohttp.web.json_response(
                {"error": {"code": "bad_request", "message": "bad_request"}}, status=400
            )
        if self.async_task:
            self.async_mode_payloads.append(bool(payload.get("async")))
            task_id = "api_gen_test_1"
            return aiohttp.web.json_response(
                {
                    "object": "image_task",
                    "id": task_id,
                    "task_id": task_id,
                    "status": "queued",
                    "poll_url": f"{self.root_url}/api/image-tasks?ids={task_id}",
                },
                status=202,
            )
        item = {"b64_json": B64_1PX}
        if self.include_internal_url:
            item["url"] = "http://127.0.0.1:9/never-downloaded.png"
        return aiohttp.web.json_response({"data": [item]})

    async def handle_task_status(self, request: aiohttp.web.Request):
        index = min(self.task_polls, len(self.task_poll_states) - 1)
        state = self.task_poll_states[index]
        self.task_polls += 1
        item = {"id": "api_gen_test_1", "status": state, "terminal": state == "success"}
        if state == "success":
            item["results"] = [{"url": f"{self.root_url}/files/out.png", "b64_json": ""}]
        return aiohttp.web.json_response({"items": [item]})

    async def handle_file(self, request: aiohttp.web.Request):
        return aiohttp.web.Response(body=PNG_1PX, content_type="image/png")

    async def handle_edits(self, request: aiohttp.web.Request):
        body = await request.read()
        ctype = request.headers.get("Content-Type", "")
        self.requests.append(("edits", {"body": body, "content_type": ctype}))
        return aiohttp.web.json_response({"data": [{"b64_json": B64_1PX}]})


@pytest.fixture
def server():
    return MockServer()


async def _run(coro):
    return await asyncio.wait_for(coro, timeout=10)


def test_generate_b64_json(server):
    async def scenario():
        async with server:
            backend = OpenAICompatBackend(_settings(server.base_url))
            result = (await _run(backend.generate("一只猫")))[0]
            assert result.data == PNG_1PX
            assert result.mime == "image/png"
            assert result.model == "z-image-turbo"
            assert result.size == "1024x1024"
            method, payload = server.requests[0]
            assert method == "generations"
            assert payload["model"] == "z-image-turbo"
            assert payload["prompt"] == "一只猫"
            assert payload["size"] == "1024x1024"
            await backend.close()

    asyncio.run(scenario())


def test_generate_prefers_inline_bytes_over_internal_url(server):
    async def scenario():
        async with server:
            backend = OpenAICompatBackend(_settings(server.base_url))
            server.include_internal_url = True
            result = (await _run(backend.generate("一只猫")))[0]
            assert result.data == PNG_1PX
            await backend.close()

    asyncio.run(scenario())


def test_generate_retries_with_inline_format_on_bad_request(server):
    async def scenario():
        async with server:
            backend = OpenAICompatBackend(_settings(server.base_url))
            server.require_response_format = True
            result = (await _run(backend.generate("一只猫")))[0]
            assert result.data == PNG_1PX
            assert len(server.requests) == 2
            _, retry_payload = server.requests[1]
            assert retry_payload["response_format"] == "b64_json"
            await backend.close()

    asyncio.run(scenario())


def test_generate_size_hint(server):
    async def scenario():
        async with server:
            backend = OpenAICompatBackend(_settings(server.base_url))
            await _run(backend.generate("狗", size_hint="512x768"))
            _, payload = server.requests[0]
            assert payload["size"] == "512x768"
            await backend.close()

    asyncio.run(scenario())


def test_generate_404_cooldown(server):
    async def scenario():
        async with server:
            backend = OpenAICompatBackend(_settings(server.base_url))
            server.generations_404 = True
            with pytest.raises(RuntimeError) as e1:
                await _run(backend.generate("x"))
            assert "404" in str(e1.value)
            # Second call hits the cooldown.
            with pytest.raises(RuntimeError) as e2:
                await _run(backend.generate("y"))
            assert "temporarily disabled" in str(e2.value)
            await backend.close()

    asyncio.run(scenario())


def test_edit_single_image_multipart(server):
    async def scenario():
        async with server:
            backend = OpenAICompatBackend(_settings(server.base_url))
            png_path = _write_temp(server)
            result = (await _run(backend.edit("改一下", [png_path])))[0]
            assert result.data == PNG_1PX
            method, entry = server.requests[0]
            assert method == "edits"
            assert "multipart/form-data" in entry["content_type"]
            body = entry["body"]
            assert b"z-image-turbo" in body
            assert b"model" in body
            assert "改一下".encode("utf-8") in body
            assert PNG_1PX in body
            await backend.close()

    asyncio.run(scenario())


def _write_temp(server) -> "pathlib.Path":
    import pathlib
    import tempfile

    path = pathlib.Path(tempfile.mkdtemp()) / "input.png"
    path.write_bytes(PNG_1PX)
    return path


def test_edit_not_supported(server):
    async def scenario():
        async with server:
            backend = OpenAICompatBackend(_settings(server.base_url, supports_edit=False))
            png_path = _write_temp(server)
            with pytest.raises(RuntimeError) as exc:
                await _run(backend.edit("x", [png_path]))
            assert "does not support edit" in str(exc.value)
            await backend.close()

    asyncio.run(scenario())


def test_generate_no_model(server):
    async def scenario():
        async with server:
            backend = OpenAICompatBackend(_settings(server.base_url, model=""))
            with pytest.raises(RuntimeError) as exc:
                await _run(backend.generate("x"))
            assert "no model" in str(exc.value)
            await backend.close()

    asyncio.run(scenario())


def test_no_api_key_raises(server):
    async def scenario():
        async with server:
            backend = OpenAICompatBackend(_settings(server.base_url, api_keys=[]))
            with pytest.raises(RuntimeError) as exc:
                await _run(backend.generate("x"))
            assert "no API key" in str(exc.value)
            await backend.close()

    asyncio.run(scenario())


def test_url_download_uses_trusted_origins(server):
    async def scenario():
        async with server:
            backend = OpenAICompatBackend(
                _settings(server.base_url),
                policy=URLFetchPolicy(trusted_origins={origin(server.base_url)}),
            )
            result = (await _run(backend.generate("url")))[0]
            assert result.data == PNG_1PX
            await backend.close()

    asyncio.run(scenario())


def test_generate_polls_async_task_to_completion(server):
    async def scenario():
        async with server:
            backend = OpenAICompatBackend(
                _settings(server.base_url, async_mode="on"),
                policy=URLFetchPolicy(trusted_origins={origin(server.base_url)}),
            )
            backend.extra_body = {"poll_interval": 1}
            server.async_task = True
            result = (await _run(backend.generate("异步出图")))[0]
            assert result.data == PNG_1PX
            assert server.async_mode_payloads == [True]
            assert server.task_polls == 2
            await backend.close()

    asyncio.run(scenario())


def test_generate_auto_mode_detects_task_response(server):
    async def scenario():
        async with server:
            backend = OpenAICompatBackend(
                _settings(server.base_url),
                policy=URLFetchPolicy(trusted_origins={origin(server.base_url)}),
            )
            backend.extra_body = {"poll_interval": 1}
            server.async_task = True
            result = (await _run(backend.generate("自动探测")))[0]
            assert result.data == PNG_1PX
            assert server.async_mode_payloads == [False]
            await backend.close()

    asyncio.run(scenario())


def _plain_backend(**kw) -> OpenAICompatBackend:
    return OpenAICompatBackend(_settings("http://127.0.0.1:9/v1", **kw))


def test_ratio_and_resolution_become_size_without_allowed_sizes():
    backend = _plain_backend()
    resolved = backend.resolve_output_intent(OutputIntent(aspect_ratio="16:9", resolution="4K"))
    assert resolved == {"size": "3840x2160"}


def test_ratio_alone_uses_1k_tier_without_allowed_sizes():
    backend = _plain_backend()
    resolved = backend.resolve_output_intent(OutputIntent(aspect_ratio="16:9"))
    assert resolved == {"size": "1280x720"}


def test_resolution_hint_is_converted_instead_of_sent_raw():
    backend = _plain_backend()
    final_size, _, _ = backend._resolve_size(None, "4k", size_hint="4K")
    assert final_size == "4096x4096"


def test_allowed_sizes_still_win_over_ratio_table():
    backend = _plain_backend(allowed_sizes=["1920x1080"])
    resolved = backend.resolve_output_intent(OutputIntent(aspect_ratio="16:9", resolution="4K"))
    assert resolved == {"size": "1920x1080"}
