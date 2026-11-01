"""Web API tests for gallery search, bulk actions, details and task retry."""

import base64

from test_plugin_lifecycle import _Ctx, _config, plugin_module  # noqa: F401

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _id(path) -> str:
    return path.stem.split("_", 1)[1]


async def _start(plugin_module, tmp_path):
    plugin = plugin_module.AIImagePlugin(_Ctx(tmp_path / "data"), _config())
    await plugin.initialize()
    return plugin


def _capture_submit(plugin) -> list:
    calls: list = []

    async def fake_submit(record, runner, is_batch=False):
        calls.append((record, runner, is_batch))
        return "captured"

    plugin._queue.submit = fake_submit
    return calls


async def test_gallery_api_search_filter_and_bulk(plugin_module, tmp_path):
    plugin = await _start(plugin_module, tmp_path)
    try:
        gallery = plugin._gallery
        first = await gallery.save_image(
            PNG_1PX,
            provider_id="p1",
            model="m1",
            size="1x1",
            sid="s",
            task_id="t1",
            prompt="a red fox",
        )
        second = await gallery.save_image(
            PNG_1PX,
            provider_id="p2",
            model="m2",
            size="1x1",
            sid="s",
            task_id="t2",
            prompt="a blue fox",
        )
        video = await gallery.save_image(
            PNG_1PX,
            provider_id="p3",
            model="m3",
            size="1x1",
            sid="s",
            task_id="t3",
            kind="video",
            prompt="clip",
        )
        first_id, second_id, video_id = _id(first), _id(second), _id(video)

        data = await plugin.api_gallery()
        assert data["total"] == 3
        assert len(data["items"]) == 3
        assert "path" not in data["items"][0]
        assert "." in data["items"][0]["filename"]

        seek = await plugin.api_gallery(q="fox")
        assert seek["total"] == 2
        assert {item["id"] for item in seek["items"]} == {first_id, second_id}

        video_only = await plugin.api_gallery(kind="video")
        assert video_only["total"] == 1
        assert video_only["items"][0]["id"] == video_id

        unknown_kind = await plugin.api_gallery(kind="audio")
        assert unknown_kind["total"] == 3

        assert (await plugin.api_gallery(favorite=1))["total"] == 0
        bulk = await plugin.api_gallery_bulk(
            {"action": "favorite", "ids": [first_id]}
        )
        assert bulk == {"ok": True, "action": "favorite", "updated": 1}
        favorite = await plugin.api_gallery(favorite=1)
        assert favorite["total"] == 1
        assert favorite["items"][0]["id"] == first_id
        assert (await plugin.api_gallery(favorite=0))["total"] == 2

        unfav = await plugin.api_gallery_bulk(
            {"action": "unfavorite", "ids": [first_id]}
        )
        assert unfav["updated"] == 1
        assert (await plugin.api_gallery(favorite=1))["total"] == 0

        assert (await plugin.api_gallery_bulk({"action": "delete", "ids": "x"})) == {
            "ok": False,
            "error": "ids 必须是数组",
        }
        assert (await plugin.api_gallery_bulk(
            {"action": "zap", "ids": [first_id]}
        )) == {"ok": False, "error": "未知操作"}

        removed = await plugin.api_gallery_bulk(
            {"action": "delete", "ids": [second_id]}
        )
        assert removed["deleted"] == 1
        assert (await plugin.api_gallery())["total"] == 2
    finally:
        await plugin.terminate()


async def test_gallery_api_filters_the_selfie_kind(plugin_module, tmp_path):
    plugin = await _start(plugin_module, tmp_path)
    try:
        gallery = plugin._gallery
        image = await gallery.save_image(
            PNG_1PX, provider_id="p1", model="m", size="1x1", sid="s", task_id="t1",
        )
        selfie = await gallery.save_image(
            PNG_1PX, provider_id="p1", model="m", size="1x1", sid="s", task_id="t2",
            kind="selfie",
        )
        video = await gallery.save_image(
            PNG_1PX, provider_id="p1", model="m", size="1x1", sid="s", task_id="t3",
            kind="video",
        )

        selfies = await plugin.api_gallery(kind="selfie")
        assert selfies["total"] == 1
        assert [item["id"] for item in selfies["items"]] == [_id(selfie)]
        assert (await plugin.api_gallery(kind="image"))["items"][0]["id"] == _id(image)
        assert (await plugin.api_gallery(kind="video"))["items"][0]["id"] == _id(video)
    finally:
        await plugin.terminate()


async def test_gallery_item_returns_metadata(plugin_module, tmp_path):
    plugin = await _start(plugin_module, tmp_path)
    try:
        path = await plugin._gallery.save_image(
            PNG_1PX,
            provider_id="p9",
            model="m9",
            size="512x512",
            sid="s",
            task_id="t9",
            prompt="a fox",
        )
        image_id = _id(path)

        data = await plugin.api_gallery_item(image_id)
        assert data["ok"] is True
        item = data["item"]
        for key in (
            "prompt",
            "provider",
            "model",
            "size",
            "kind",
            "created_at",
            "favorite",
            "bytes",
            "filename",
            "exists",
        ):
            assert key in item
        assert "path" not in item
        assert item["prompt"] == "a fox"
        assert item["provider"] == "p9"
        assert item["model"] == "m9"
        assert item["size"] == "512x512"
        assert item["bytes"] == path.stat().st_size
        assert item["exists"] is True

        missing = await plugin.api_gallery_item("missing")
        assert missing == {"ok": False, "error": "图片不存在"}
    finally:
        await plugin.terminate()


async def test_tasks_api_filters_and_pages(plugin_module, tmp_path):
    plugin = await _start(plugin_module, tmp_path)
    try:
        store = plugin._store
        failed = store.create(kind="generate_image", prompt="a", sid="s")
        store.set_error(failed.task_id, "boom")
        running = store.create(kind="generate_image", prompt="b", sid="s")
        store.set_status(running.task_id, "running")
        queued = store.create(kind="generate_image", prompt="c", sid="s")
        done = store.create(kind="generate_video", prompt="d", sid="s")
        store.set_result(done.task_id, "ok")

        everything = await plugin.api_tasks()
        assert everything["total"] == 4
        assert len(everything["tasks"]) == 4

        assert (await plugin.api_tasks(state="all"))["total"] == 4

        active = await plugin.api_tasks(state="running")
        assert active["total"] == 2
        assert {task["task_id"] for task in active["tasks"]} == {
            running.task_id,
            queued.task_id,
        }

        failed_only = await plugin.api_tasks(state="failed")
        assert failed_only["total"] == 1
        assert failed_only["tasks"][0]["task_id"] == failed.task_id

        done_only = await plugin.api_tasks(state="done")
        assert done_only["total"] == 1
        assert done_only["tasks"][0]["task_id"] == done.task_id

        cancelled = store.create(kind="generate_image", prompt="e", sid="s")
        assert store.cancel(cancelled.task_id) is True
        stopped = await plugin.api_tasks(state="cancelled")
        assert stopped["total"] == 1
        assert stopped["tasks"][0]["task_id"] == cancelled.task_id

        page = await plugin.api_tasks(limit=2, offset=1)
        assert page["total"] == 5
        assert len(page["tasks"]) == 2
    finally:
        await plugin.terminate()


async def test_retry_requeues_failed_task_with_same_parameters(plugin_module, tmp_path):
    plugin = await _start(plugin_module, tmp_path)
    try:
        record = plugin._store.create(
            kind="edit_image",
            sid="sid-1",
            session_type="group",
            user_id="user-1",
            prompt="make it red",
            provider="p2",
            chain=["p2", "p3"],
            size="1024x1024",
            extra={
                "adapter": "qq",
                "session_type": "group",
                "session_id": "sid-1",
                "input_image": "/tmp/a.png",
                "prompt_layers": [{"text": "old layer"}],
                "final_prompt": "old final",
            },
        )
        plugin._store.set_error(record.task_id, "boom")
        calls = _capture_submit(plugin)

        result = await plugin.api_retry_task({}, record.task_id)
        assert result["ok"] is True
        assert result["source"] == record.task_id
        assert result["task_id"] != record.task_id
        assert len(calls) == 1

        retry, runner, is_batch = calls[0]
        assert is_batch is False
        assert retry.task_id == result["task_id"]
        assert retry.kind == "edit_image"
        assert retry.prompt == "make it red"
        assert retry.provider == "p2"
        assert retry.chain == ["p2", "p3"]
        assert retry.size == "1024x1024"
        assert retry.sid == "sid-1"
        assert retry.session_type == "group"
        assert retry.user_id == "user-1"
        assert retry.extra["input_image"] == "/tmp/a.png"
        assert retry.extra["adapter"] == "qq"
        assert "prompt_layers" not in retry.extra
        assert "final_prompt" not in retry.extra

        original = plugin._store.get(record.task_id)
        assert original.status == "failed"
        assert original.error == "boom"

        batch = plugin._store.create(
            kind="generate_image_batch",
            prompt="two shots",
            extra={"prompts": ["one", "two"]},
        )
        plugin._store.set_error(batch.task_id, "boom")
        await plugin.api_retry_task({}, batch.task_id)
        retry, _, is_batch = calls[-1]
        assert is_batch is True
        assert retry.extra["prompts"] == ["one", "two"]
    finally:
        await plugin.terminate()


async def test_retry_rejects_non_failed_tasks(plugin_module, tmp_path):
    plugin = await _start(plugin_module, tmp_path)
    try:
        calls = _capture_submit(plugin)

        running = plugin._store.create(kind="generate_image", prompt="x")
        plugin._store.set_status(running.task_id, "running")
        rejected = await plugin.api_retry_task({}, running.task_id)
        assert rejected == {"ok": False, "error": "只有失败的任务可以重试"}

        done = plugin._store.create(kind="generate_image", prompt="y")
        plugin._store.set_result(done.task_id, "ok")
        assert (await plugin.api_retry_task({}, done.task_id))["ok"] is False

        assert (await plugin.api_retry_task({}, "missing")) == {
            "ok": False,
            "error": "任务不存在",
        }
        assert calls == []
    finally:
        await plugin.terminate()
