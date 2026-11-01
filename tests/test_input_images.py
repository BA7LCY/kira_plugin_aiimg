"""Input images come from the message, never from a fetch we make ourselves."""

import asyncio

from test_plugin_lifecycle import _Ctx, _config, plugin_module  # noqa: F401


class _Gallery:
    def __init__(self, paths=None):
        self.paths = list(paths or [])

    def recent_images(self, sid):
        return list(self.paths)


class _Quoted:
    """Stand-in for a Reply element that carries the quoted message's chain."""

    def __init__(self, chain):
        self.message_id = "m1"
        self.chain = chain


def _plugin(plugin_module, tmp_path, recent=None):
    plugin = plugin_module.AIImagePlugin(_Ctx(tmp_path / "data"), _config())
    plugin._gallery = _Gallery(recent)
    plugin._ready = True
    return plugin


def _image(plugin_module, path, *, to_path=None):
    element = plugin_module.Image(image=str(path))
    element.to_path = to_path or (lambda: _ready(path))
    return element


async def _ready(path):
    return str(path)


def _message(plugin_module, chain):
    return plugin_module.KiraIMMessage(chain=chain)


def _event(plugin_module, messages):
    event = plugin_module.KiraMessageBatchEvent(messages=messages)
    event.sid = "s1"
    return event


def test_current_message_image_is_used_when_image_is_blank(plugin_module, tmp_path):
    photo = tmp_path / "photo.png"
    photo.write_bytes(b"\x89PNG")
    plugin = _plugin(plugin_module, tmp_path)
    event = _event(plugin_module, [_message(plugin_module, [_image(plugin_module, photo)])])

    assert asyncio.run(plugin._resolve_input_image(event, "")) == photo


def test_quoted_message_image_is_used(plugin_module, tmp_path):
    photo = tmp_path / "quoted.png"
    photo.write_bytes(b"\x89PNG")
    plugin = _plugin(plugin_module, tmp_path)
    event = _event(
        plugin_module,
        [_message(plugin_module, [_Quoted([_image(plugin_module, photo)])])],
    )

    assert asyncio.run(plugin._resolve_input_image(event, "")) == photo


def test_url_arguments_reuse_the_attachment_and_never_fetch(plugin_module, tmp_path):
    photo = tmp_path / "photo.png"
    photo.write_bytes(b"\x89PNG")
    plugin = _plugin(plugin_module, tmp_path)
    event = _event(plugin_module, [_message(plugin_module, [_image(plugin_module, photo)])])

    resolved = asyncio.run(
        plugin._resolve_input_image(event, "https://multimedia.nt.qq.com/x.png")
    )

    assert resolved == photo


def test_url_without_an_attachment_resolves_to_nothing(plugin_module, tmp_path):
    plugin = _plugin(plugin_module, tmp_path)
    event = _event(plugin_module, [])

    assert asyncio.run(plugin._resolve_input_image(event, "https://example.com/a.png")) is None


def test_latest_still_uses_the_gallery(plugin_module, tmp_path):
    photo = tmp_path / "stored.png"
    photo.write_bytes(b"\x89PNG")
    plugin = _plugin(plugin_module, tmp_path, recent=[str(photo)])
    event = _event(plugin_module, [])

    assert asyncio.run(plugin._resolve_input_image(event, "latest")) == photo
    # A blank argument prefers attachments, then falls back to the gallery.
    assert asyncio.run(plugin._resolve_input_image(event, "")) == photo


def test_broken_attachment_falls_back_to_the_gallery(plugin_module, tmp_path):
    stored = tmp_path / "stored.png"
    stored.write_bytes(b"\x89PNG")
    plugin = _plugin(plugin_module, tmp_path, recent=[str(stored)])

    async def boom():
        raise RuntimeError("download failed")

    element = _image(plugin_module, "https://example.com/x.png", to_path=boom)
    event = _event(plugin_module, [_message(plugin_module, [element])])

    assert asyncio.run(plugin._resolve_input_image(event, "")) == stored


def test_attached_paths_takes_every_picture_and_caps_them(plugin_module, tmp_path):
    paths = []
    for index in range(6):
        path = tmp_path / f"p{index}.png"
        path.write_bytes(b"\x89PNG")
        paths.append(path)
    plugin = _plugin(plugin_module, tmp_path)
    event = _event(
        plugin_module,
        [_message(plugin_module, [_image(plugin_module, path) for path in paths])],
    )

    resolved = asyncio.run(plugin._attached_paths(event))

    assert resolved == paths[:4]


def test_quoted_picture_comes_before_the_new_one(plugin_module, tmp_path):
    quoted = tmp_path / "quoted.png"
    fresh = tmp_path / "fresh.png"
    quoted.write_bytes(b"\x89PNG")
    fresh.write_bytes(b"\x89PNG")
    plugin = _plugin(plugin_module, tmp_path)
    event = _event(
        plugin_module,
        [
            _message(
                plugin_module,
                [
                    _Quoted([_image(plugin_module, quoted)]),
                    _image(plugin_module, fresh),
                ],
            )
        ],
    )

    resolved = asyncio.run(plugin._attached_paths(event))

    assert resolved == [quoted, fresh]


def test_media_hint_counts_the_attachments(plugin_module, tmp_path):
    plugin = _plugin(plugin_module, tmp_path)
    plugin._features = lambda key, default=None: True if key == "edit_enabled" else default
    event = _event(
        plugin_module,
        [_message(plugin_module, [_image(plugin_module, "a.png")])],
    )

    hint = plugin._media_hint(event)

    assert "1 张可用图片" in hint
    assert "image 留空" in hint


def test_media_hint_warns_when_the_quote_has_no_media(plugin_module, tmp_path):
    plugin = _plugin(plugin_module, tmp_path)
    plugin._features = lambda key, default=None: True if key == "edit_enabled" else default
    event = _event(plugin_module, [_message(plugin_module, [_Quoted(None)])])

    hint = plugin._media_hint(event)

    assert "没有提供其中的图片内容" in hint
    assert "重新发送" in hint


def test_media_hint_stays_quiet_without_media_or_quotes(plugin_module, tmp_path):
    plugin = _plugin(plugin_module, tmp_path)
    plugin._features = lambda key, default=None: True if key == "edit_enabled" else default
    event = _event(plugin_module, [_message(plugin_module, [])])

    assert plugin._media_hint(event) == ""
