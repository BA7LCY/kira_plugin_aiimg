"""Video downloads must finish and contain a complete video container."""
import struct

from aiohttp import web
import pytest

from core import video_file
from core.net import URLFetchPolicy


def box(kind, payload):
    return struct.pack(">I4s", len(payload) + 8, kind) + payload


def video_bytes(handler=b"vide"):
    handler_box = box(b"hdlr", b"\0" * 8 + handler + b"\0" * 12)
    return (box(b"ftyp", b"isom\0\0\0\0isom")
            + box(b"moov", box(b"trak", box(b"mdia", handler_box)))
            + box(b"mdat", b"sample payload"))


@pytest.mark.parametrize("payload", [b"", b"<html>error</html>", b'{"ok":true}',
                                      video_bytes()[:-1], video_bytes(b"soun"),
                                      box(b"ftyp", b"isom")])
def test_invalid_video_rejected(tmp_path, payload):
    path = tmp_path / "invalid.mp4"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        video_file.validate_video_file(path)


def test_complete_video_container_accepted(tmp_path):
    path = tmp_path / "valid.mp4"
    path.write_bytes(video_bytes())
    video_file.validate_video_file(path)


@pytest.mark.parametrize("case", ["valid", "html", "partial", "redirect", "private", "loop", "oversize"])
async def test_download_validates_response_and_redirects(tmp_path, monkeypatch, case):
    payload = video_bytes()

    async def handler(request):
        if request.path == "/redirect":
            raise web.HTTPFound("/valid")
        if request.path == "/private":
            raise web.HTTPFound("http://127.0.0.1:9/private")
        if request.path == "/loop":
            raise web.HTTPFound("/loop")
        if request.path == "/html":
            return web.Response(text="<html>not a video</html>", content_type="video/mp4")
        if request.path == "/partial":
            return web.Response(body=payload[:30], status=206)
        return web.Response(body=payload, content_type="video/mp4")

    app = web.Application()
    app.router.add_get("/{name}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    root = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    policy = URLFetchPolicy(trusted_origins=frozenset({root}))
    if case == "oversize":
        monkeypatch.setattr(video_file, "MAX_VIDEO_BYTES", 10)
    try:
        if case in {"valid", "redirect"}:
            path = await video_file.download_video(root + "/" + case, tmp_path, policy)
            assert path.read_bytes() == payload
        else:
            with pytest.raises(ValueError):
                await video_file.download_video(root + "/" + case, tmp_path, policy)
            assert not list(tmp_path.iterdir())
        assert not list(tmp_path.glob("*.part"))
    finally:
        await runner.cleanup()
