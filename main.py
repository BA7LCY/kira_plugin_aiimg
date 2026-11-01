"""Kira AI image plugin.

Multi-provider image generation, editing, selfie, batch and Agnes video, with
a durable task queue, a prompt-audit trail (every layer is visible and
editable), a gallery and a WebUI. No prompt is hard-coded: each one can be
edited or cleared in the settings page.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import weakref
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

import aiohttp

from core.plugin import BasePlugin, PluginPage, PageMenu, Priority, logger, on, register
from core.chat import (
    Group,
    KiraIMMessage,
    KiraMessageBatchEvent,
    KiraMessageEvent,
    MessageChain,
    Session,
    User,
)
from core.chat.message_elements import Image, Text, Video
from core.prompt_manager import Prompt

from .core import refs as refs_mod
from .core import settings as settings_mod
from .core.chains import ChainItem, chain_for
from .core.image_store import ImageStore
from .core.image_format import validate_image_bytes
from .core.logbook import clip, elapsed_ms, mask
from .core.net import URLFetchPolicy, ensure_url_allowed, trusted_origins_from_config
from .core.output_spec import (
    OutputIntent,
    apply_conflict_rule,
    extract_output_intent_from_prompt,
    format_output_intent,
    merge_output_intents,
    parse_output_intent,
)
from .core.prompt import reference_note, render_template
from .core.provider_registry import ProviderRegistry
from .core.provider_stats import ProviderStats
from .core.providers.base import StopProviderChain
from .core.queue import QueueFullError, TaskQueue, UserLimitError
from .core.task_store import TaskRecord, TaskStore
from .core.video_file import download_video

# Tool names registered by this plugin; removed from the LLM tool set during
# the terminal-notification turn to avoid recursion.
TOOL_NAMES = (
    "generate_image",
    "edit_image",
    "generate_selfie",
    "generate_image_batch",
    "generate_video",
    "get_task_status",
)

IMAGE_KINDS = ("generate_image", "edit_image", "generate_selfie", "generate_image_batch")

_VIDEO_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
_VIDEO_SECONDS = (4, 12)

_NOTIFY_MARKER = "_aiimg_task"
_ERROR_MARKER = "_aiimg_task_error"
# How long a submitted task may still be waiting for its acknowledgement.
_ACK_WATCH_TTL = 300.0

# Uploads under refs/ are the only source of selfie reference photos, but the
# provider request itself caps how many images can be attached at once.
MAX_REFERENCE_IMAGES = 8

PLUGIN_ID = "kira_plugin_aiimg"

_VERSION_CACHE: str | None = None


def _plugin_version() -> str:
    """Read the manifest version once for the WebUI footer."""
    global _VERSION_CACHE
    if _VERSION_CACHE is None:
        try:
            data = json.loads(
                (Path(__file__).resolve().parent / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            _VERSION_CACHE = str(data.get("version") or "")
        except Exception:
            _VERSION_CACHE = ""
    return _VERSION_CACHE

# Tags registered by the framework's own ``kira-ai`` plugin.
_BUILTIN_MEDIA_TAGS = frozenset({"img", "selfie", "video"})


def _is_builtin_media_tag(tag: Any) -> bool:
    """True only for tag instances declared by the built-in plugin.

    Other plugins may reuse the same tag names (NAI Studio registers ``img``),
    so the declaring module is the discriminator, not the tag name.
    """
    if getattr(tag, "name", "") not in _BUILTIN_MEDIA_TAGS:
        return False
    return "builtin_plugins" in (type(tag).__module__ or "")


def _facts_line(facts: dict[str, str]) -> str:
    """One factual line for notifications whose template is left empty.

    Without it the model receives a bare image and no instruction, which makes
    an empty ``<msg />`` reply the likely outcome.
    """
    bits = [f"任务 {facts.get('task_id') or '-'}"]
    for key, label in (
        ("kind", "类型"),
        ("status", "状态"),
        ("size", "尺寸"),
        ("provider", "服务商"),
        ("error_kind", "失败类别"),
    ):
        value = str(facts.get(key) or "").strip()
        if value:
            bits.append(f"{label} {value}")
    return "；".join(bits) + "。"


_NOTICE_HINT = (
    "后台任务已经结束，请用当前人设自然地跟用户说一句结果，"
    "不要念出任务号或字段名。"
)


def _notification_extra(event: Any) -> dict:
    """Find the aiimg markers on either event shape.

    The workflow hands plugins a ``KiraMessageBatchEvent`` (``messages`` list)
    while older call sites expose a single ``message``; reading only the latter
    silently skipped every notification.
    """
    candidates: list[Any] = list(getattr(event, "messages", None) or [])
    single = getattr(event, "message", None)
    if single is not None:
        candidates.append(single)
    extras: list[Any] = [getattr(item, "extra", None) for item in candidates]
    extras.append(getattr(event, "extra", None))
    for extra in extras:
        if isinstance(extra, dict) and (
            extra.get(_NOTIFY_MARKER) or extra.get(_ERROR_MARKER)
        ):
            return extra
    return {}


def _event_elements_kw(adapter: Any) -> dict:
    """Cross-generation keyword for supported elements on a synthetic event."""
    params = set(inspect.signature(KiraMessageEvent.__init__).parameters)
    elements = getattr(adapter, "supported_elements", None)
    if elements is None:
        elements = list(
            getattr(
                getattr(adapter, "capabilities", None),
                "supported_elements",
                ["text", "image"],
            )
        )
    if "supported_elements" in params:
        return {"supported_elements": elements}
    return {"message_types": elements}


def _split_user_key(user_id: str) -> tuple[str, str]:
    """'adapter:user' -> (adapter, user); otherwise ('', user)."""
    text = str(user_id or "").strip()
    if ":" in text:
        adapter, _, uid = text.partition(":")
        return adapter.strip(), uid.strip()
    return "", text



def _looks_like_image(data: bytes) -> bool:
    """Require a verified, decodable image rather than a magic prefix."""
    try:
        validate_image_bytes(data)
    except ValueError:
        return False
    return True


class AIImagePlugin(BasePlugin):
    def __init__(self, ctx, cfg):
        super().__init__(ctx, cfg)
        self._data_dir: Path | None = None
        self._registry: ProviderRegistry | None = None
        self._stats = ProviderStats(None)
        self._limiters: dict[str, tuple[int, asyncio.Semaphore]] = {}
        self._store: TaskStore | None = None
        self._gallery: ImageStore | None = None
        self._queue: TaskQueue | None = None
        self._config_lock = asyncio.Lock()
        self._ready = False
        self._builtin_tags_logged = False
        # Turns that submitted a task and still owe the user an acknowledgement.
        self._ack_watch: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        self._data_dir = self.ctx.get_plugin_data_dir()
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._stats = ProviderStats(self._data_dir)
        self._configure_stats()
        self._store = TaskStore(self._data_dir)
        self._gallery = ImageStore(self._data_dir, {"gallery": self._cfg("section_gallery")})
        self._registry = ProviderRegistry(self._provider_config(), data_dir=self._data_dir)
        self._build_queue()
        self._queue.on_finished(self._on_task_finished)
        await self._queue.start()
        self._prune_old_tasks()
        self._ready = True
        for error in self._registry.validate():
            logger.warning("[aiimg] provider config: %s", error)
        gallery = self._gallery.stats() or {}
        logger.info(
            "[aiimg] 数据目录 %s，画廊 %d 张（收藏 %d，占用 %.1f MB）",
            self._data_dir,
            int(gallery.get("count") or 0),
            int(gallery.get("favorites") or 0),
            int(gallery.get("bytes") or 0) / (1024 * 1024),
        )
        for provider in self._registry.image_provider_ids():
            settings = self._registry.get_settings(provider)
            logger.info(
                "[aiimg] 服务商 %s 类型=%s 模型=%s 并发=%s 异步=%s",
                provider,
                settings.template_key,
                settings.model or "-",
                settings.max_concurrent or "不限",
                settings.async_mode,
            )
        for provider in self._registry.video_provider_ids():
            settings = self._registry.get_settings(provider)
            logger.info(
                "[aiimg] 视频服务商 %s 模型=%s 尺寸=%s 比例=%s 并发=%s",
                provider,
                settings.model or "-",
                settings.size or "-",
                settings.aspect_ratio or "-",
                settings.max_concurrent or "不限",
            )
        queue_cfg = self._cfg("section_queue", default={}) or {}
        logger.info(
            "[aiimg] 队列：并行 %s，容量 %s，批量 %s×%s，每用户图片 %s / 视频 %s",
            queue_cfg.get("max_running", "-"),
            queue_cfg.get("max_queued", "-"),
            queue_cfg.get("max_batch_items", "-"),
            queue_cfg.get("batch_concurrency", "-"),
            queue_cfg.get("max_user_image_tasks", "-"),
            queue_cfg.get("max_user_video_tasks", "-"),
        )
        logger.info("[aiimg] initialized with %d image / %d video providers",
                    len(self._registry.image_provider_ids()),
                    len(self._registry.video_provider_ids()))

    async def terminate(self):
        self._ready = False
        if self._queue is not None:
            await self._queue.stop()
        if self._gallery is not None:
            await self._gallery.close()
        if self._registry is not None:
            await self._registry.close()
        if self._store is not None:
            self._store.close()

    # ------------------------------------------------------------------
    # config helpers
    # ------------------------------------------------------------------

    def _cfg(self, *path: str, default: Any = None) -> Any:
        node = self.plugin_cfg or {}
        for key in path:
            if not isinstance(node, dict):
                return default
            node = node.get(key)
        return node if node is not None else default

    def _provider_config(self) -> dict:
        providers = self._cfg("providers", default=[])
        chains = self._cfg("chains", default={})
        return {
            "providers": providers if isinstance(providers, list) else [],
            "chains": chains if isinstance(chains, dict) else {},
        }

    def _features(self, key: str, default: str = "") -> str:
        value = self._cfg("section_features", key)
        if value is None:
            # Older configs predate newer fields; fall back to the declared
            # default so a fresh template still applies without a save.
            value = settings_mod.field_default("section_features", key, default)
        return str(value or default).strip()

    async def _download_video(self, url: str) -> Path:
        policy = URLFetchPolicy(
            trusted_origins=trusted_origins_from_config(self._provider_config())
        )
        return await download_video(url, self._data_dir / "gallery", policy)

    def _consecutive_failures(self, provider_id: str) -> int:
        for entry in self._stats.snapshot():
            if entry["id"] == provider_id:
                return int(entry["consecutive_failures"])
        return 0

    def _denied(self, reason: str, action: str) -> str:
        """Tool result for a request the plugin refused to start."""
        logger.info("[aiimg] 拒绝%s：%s", action, reason)
        template = self._features("denied_notification_prompt", "")
        if template:
            return render_template(template, reason=reason, action=action).strip()
        return f"请求没有执行：现在不能做{action}，原因是{reason}。"

    def _feature_on(self, key: str, default: bool = True) -> bool:
        value = self._cfg("section_features", key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in ("1", "true", "yes", "on")

    def _compat(self, key: str, default: Any = None) -> Any:
        value = self._cfg("section_compat", key)
        return default if value is None else value

    @staticmethod
    def _lines(value: Any) -> list[str]:
        """Read a line-based setting that may be stored as text or a list."""
        if isinstance(value, str):
            return [line.strip() for line in value.splitlines() if line.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if str(item).strip()]
        return []

    def _preset_map(self, kind: str) -> dict[str, str]:
        """Parse "name: prompt" preset lines into a lookup table."""
        key = "draw_presets" if kind == "generate" else "edit_presets"
        raw = self._lines(self._cfg("section_features", key, default=[]))
        presets: dict[str, str] = {}
        for line in raw:
            text = str(line or "").strip()
            if not text:
                continue
            name, sep, value = text.partition(":")
            if not sep or not name.strip() or not value.strip():
                continue
            presets[name.strip().lower()] = value.strip()
        return presets

    def _apply_preset(self, prompt: str, preset: str, kind: str) -> tuple[str, str]:
        """Prepend a named preset to the caller's prompt; unknown names are ignored."""
        name = str(preset or "").strip()
        if not name:
            return prompt, ""
        text = self._preset_map(kind).get(name.lower())
        if not text:
            return prompt, ""
        return f"{text} {str(prompt or '').strip()}".strip(), name

    async def _persona_reference(self) -> Path | None:
        """Current persona reference photo, or None when there is no usable one."""
        manager = getattr(self.ctx, "persona_mgr", None)
        if manager is None:
            return None
        try:
            reference = await manager.get_reference_image()
        except Exception as exc:
            logger.warning("[aiimg] failed to read persona reference: %s", exc)
            return None
        if not reference:
            return None
        path = Path(str(reference))
        return path if path.exists() else None

    async def _selfie_reference_plan(
        self, roles: list[str] | None = None
    ) -> list[tuple[Path, str]]:
        """Final selfie references as (path, role) in send order.

        Persona mode sends only the persona photo unless the merge switch (or an
        explicit role filter) also asks for uploads; without persona mode uploads
        are used alone.
        """
        uploads = self._ref_paths(roles)
        if not self._feature_on("selfie_use_persona", True):
            return uploads
        persona = await self._persona_reference()
        if persona is None:
            # Never drop a photo: with the merge switch on, uploads stay usable.
            if self._feature_on("selfie_include_uploads", False) or roles is not None:
                return uploads
            return []
        if roles is not None or self._feature_on("selfie_include_uploads", False):
            return [(persona, "persona")] + uploads
        return [(persona, "persona")]

    async def _selfie_references(self) -> list[Path]:
        return [path for path, _role in await self._selfie_reference_plan()]

    def _can_replace_builtin_tag(self, name: str) -> bool:
        """Only take over a built-in tag when this plugin can serve it."""
        if self._registry is None:
            return False
        if name == "video":
            return self._feature_on("video_enabled", default=False) and bool(
                self._registry.video_provider_ids()
            )
        if not self._feature_on("draw_enabled"):
            return False
        if not self._registry.image_provider_ids():
            return False
        if name == "selfie":
            if not self._feature_on("selfie_enabled"):
                return False
            use_persona = self._feature_on("selfie_use_persona", True)
            if not use_persona:
                return bool(self._ref_files())
            return True
        return True

    def _default_output_for(self, feature: str) -> str:
        table = {
            "generate_image": ("draw_default_output", "1024x1024"),
            "edit_image": ("edit_default_output", "1024x1024"),
            "generate_selfie": ("selfie_default_output", "768x1024"),
            "generate_image_batch": ("draw_default_output", "1024x1024"),
        }
        key, fallback = table.get(feature, ("", ""))
        return self._features(key, fallback) if key else fallback

    def _build_queue(self) -> None:
        section = self._cfg("section_queue")
        self._queue = TaskQueue(
            self._store,
            max_running=int(section.get("max_running", 2) or 2),
            max_queued=int(section.get("max_queued", 16) or 16),
            max_user_image_tasks=int(section.get("max_user_image_tasks", 1) or 1),
            max_user_video_tasks=int(section.get("max_user_video_tasks", 1) or 1),
            retention_hours=int(section.get("task_retention_hours", 24) or 24),
        )

    def _configure_stats(self) -> None:
        section = self._cfg("section_queue") or {}
        self._stats.configure(
            threshold=int(section.get("provider_failure_threshold", 3) or 3),
            cooldown=float(section.get("provider_cooldown_seconds", 120) or 120),
        )

    def _provider_limiter(self, provider_id: str) -> asyncio.Semaphore | None:
        if self._registry is None:
            return None
        try:
            provider = self._registry.get_settings(provider_id)
        except KeyError:
            return None
        limit = max(0, int(getattr(provider, "max_concurrent", 0) or 0))
        if limit <= 0:
            return None
        cached = self._limiters.get(provider_id)
        if cached is not None and cached[0] == limit:
            return cached[1]
        semaphore = asyncio.Semaphore(limit)
        self._limiters[provider_id] = (limit, semaphore)
        return semaphore

    @asynccontextmanager
    async def _provider_slot(self, provider_id: str):
        semaphore = self._provider_limiter(provider_id)
        if semaphore is None:
            yield
            return
        async with semaphore:
            yield

    def _skip_cooling(self, provider_id: str, upcoming: list[str]) -> bool:
        """Skip a cooling provider unless everything left is cooling too."""
        if self._stats.cooling_seconds(provider_id) <= 0:
            return False
        return any(
            other != provider_id and self._stats.cooling_seconds(other) <= 0
            for other in upcoming
        )

    async def _reload_registry(self) -> None:
        old = self._registry
        self._registry = ProviderRegistry(self._provider_config(), data_dir=self._data_dir)
        if old is not None:
            await old.close()

    async def _apply_config(self) -> None:
        """Reload providers and limits after the WebUI saved a new config."""
        await self._reload_registry()
        self._limiters.clear()
        self._configure_stats()
        if self._queue is not None:
            section = self._cfg("section_queue") or {}
            self._queue.apply_limits(
                max_running=int(section.get("max_running", 2) or 2),
                max_queued=int(section.get("max_queued", 16) or 16),
                max_user_image_tasks=int(section.get("max_user_image_tasks", 1) or 1),
                max_user_video_tasks=int(section.get("max_user_video_tasks", 1) or 1),
                retention_hours=int(section.get("task_retention_hours", 24) or 24),
            )
        if self._gallery is not None:
            self._gallery.update_config({"gallery": self._cfg("section_gallery") or {}})
        self._prune_old_tasks()

    def _prune_old_tasks(self) -> None:
        try:
            pruned = self._store.prune_old(
                int(self._cfg("section_queue", "task_retention_hours", default=24) or 24)
            )
            if pruned:
                logger.info("[aiimg] pruned %d old task records", pruned)
        except Exception as exc:
            logger.warning("[aiimg] prune failed: %s", exc)

    # ------------------------------------------------------------------
    # access control
    # ------------------------------------------------------------------

    def _matches_entry(self, entry: str, keys: list[str]) -> bool:
        entry_adapter, entry_id = _split_user_key(entry)
        if not entry_id:
            return False
        for key in keys:
            if not key:
                continue
            if entry_adapter:
                if key == entry:
                    return True
            elif key.split(":", 1)[-1] == entry_id:
                return True
        return False

    def _can_use(self, user_id: str, adapter_name: str = "", group_id: str = "") -> bool:
        if not user_id:
            return False
        whitelist = self._lines(self._cfg("section_access", "whitelist", default=[]))
        if not whitelist:
            return True
        keys = [key for key in (user_id, group_id) if key]
        return any(self._matches_entry(entry, keys) for entry in whitelist)

    # ------------------------------------------------------------------
    # chains
    # ------------------------------------------------------------------

    def _first_image_provider(self) -> str:
        ids = self._registry.image_provider_ids() if self._registry else []
        return ids[0] if ids else ""

    def _first_video_provider(self) -> str:
        ids = self._registry.video_provider_ids() if self._registry else []
        return ids[0] if ids else ""

    def _image_chain(self, feature: str) -> list[ChainItem]:
        if not self._registry:
            return []
        available = set(self._registry.image_provider_ids())
        chains = self._provider_config()["chains"] or {}
        items = chain_for(chains, feature, available=available, default_provider=self._first_image_provider())
        return self._append_fallback(items, sorted(available))

    @staticmethod
    def _append_fallback(items: list[ChainItem], available: list[str]) -> list[ChainItem]:
        """Append providers that are configured but missing from the chain."""
        listed = {item.provider_id for item in items}
        merged = list(items)
        for provider_id in available:
            if provider_id and provider_id not in listed:
                merged.append(ChainItem(provider_id=provider_id, output=""))
                listed.add(provider_id)
        return merged

    def _edit_chain(self) -> list[ChainItem]:
        chain = self._image_chain("edit_image")
        if not self._registry:
            return []
        return [
            item
            for item in chain
            if item.provider_id in self._registry.image_provider_ids()
            and self._registry.get_backend(item.provider_id).supports_edit
        ]

    def _video_chain(self) -> list[ChainItem]:
        if not self._registry:
            return []
        available = set(self._registry.video_provider_ids())
        chains = self._provider_config()["chains"] or {}
        items = chain_for(chains, "generate_video", available=available, default_provider=self._first_video_provider())
        return self._append_fallback(items, sorted(available))

    def _selfie_chain(self) -> list[ChainItem]:
        chain = self._image_chain("generate_selfie")
        if chain:
            return chain
        if self._feature_on("selfie_follow_edit_chain", True):
            return self._edit_chain()
        return []

    # ------------------------------------------------------------------
    # intent
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_intent_text(text: str) -> OutputIntent:
        intent = OutputIntent()
        for token in str(text or "").split():
            try:
                intent = merge_output_intents(intent, parse_output_intent(token))
            except ValueError:
                continue
        return intent

    def _chain_intent(self, record: TaskRecord, chain_output: str) -> OutputIntent:
        pieces: list[OutputIntent] = []
        explicit = record.extra.get("explicit") or {}
        # Highest precedence first: prompt tokens, tool fields, chain output.
        # Provider defaults come next (applied by the backend), and the
        # feature-level default is only a last resort when a provider has no
        # default size of its own.
        pieces.append(extract_output_intent_from_prompt(record.prompt))
        for key in ("size", "aspect_ratio", "resolution"):
            value = explicit.get(key)
            if value:
                pieces.append(self._parse_intent_text(str(value)))
        if chain_output:
            pieces.append(self._parse_intent_text(chain_output))
        intent = merge_output_intents(*pieces)
        prefer_resolution = bool(
            self._cfg("section_features", "prefer_resolution", default=True)
        )
        return apply_conflict_rule(intent, prefer_resolution=prefer_resolution)

    @staticmethod
    def _intent_is_empty(intent: OutputIntent) -> bool:
        return not (intent.exact_size or intent.aspect_ratio or intent.resolution)

    # ------------------------------------------------------------------
    # task creation helpers
    # ------------------------------------------------------------------

    def _task_extra(self, event: KiraMessageBatchEvent, **values: Any) -> dict:
        session = getattr(event, "session", None)
        extra = {
            "adapter": self._adapter_name(event),
            "session_type": str(getattr(session, "session_type", "") or ""),
            "session_id": str(getattr(session, "session_id", "") or ""),
        }
        extra.update(values)
        return extra

    async def _queue_image_task(
        self,
        event: KiraMessageBatchEvent,
        kind: str,
        *,
        prompt: str,
        provider: str = "",
        size: str = "",
        aspect_ratio: str = "",
        resolution: str = "",
        preset: str = "",
        preset_text: str = "",
        caller_prompt: str = "",
    ) -> str:
        if not self._ready or not self._gallery:
            return "插件未就绪"
        if not self._can_use(self._user_id(event), self._adapter_name(event), self._group_id(event)):
            return self._denied("不在允许使用生图的白名单里", "生图")
        if not self._feature_on("draw_enabled"):
            return self._denied("文生图功能已在插件设置里关闭", "文生图")
        prompt = str(prompt or "").strip()
        if not prompt:
            return "提示词不能为空"
        chain = self._image_chain(kind)
        if not chain:
            return self._denied("没有配置可用的生图服务商", "生图")
        if provider:
            chain = [item for item in chain if item.provider_id == provider]
            if not chain:
                return f"服务商 {provider} 不可用或不在默认链路中"
        extra = self._task_extra(
            event,
            explicit=self._explicit(size, aspect_ratio, resolution),
            default_output=self._default_output_for(kind),
            chain_items=[self._chain_item_dict(item) for item in chain],
        )
        if preset:
            extra["preset"] = preset
            extra["preset_text"] = preset_text
            extra["caller_prompt"] = caller_prompt or prompt
        record = self._store.create(
            kind=kind,
            sid=event.sid,
            user_id=self._user_id(event),
            prompt=prompt,
            provider=chain[0].provider_id,
            chain=[item.provider_id for item in chain],
            size=format_output_intent(self._chain_intent_base(record_hint=extra)),
            extra=extra,
        )
        try:
            await self._submit(record, self._run_generate)
        except (QueueFullError, UserLimitError) as exc:
            return str(exc)
        return self._ack_text(record, "已加入生图队列", event)

    async def _submit(
        self,
        record: TaskRecord,
        runner: Callable[[TaskRecord], Any],
        *,
        is_batch: bool = False,
    ) -> str:
        if not self._queue:
            raise RuntimeError("插件未就绪")
        logger.info(
            "[aiimg] 提交 %s 任务 %s 用户=%s 会话=%s 链路=%s 尺寸=%s 提示词=%s",
            record.kind,
            record.task_id,
            record.user_id or "-",
            record.sid or "-",
            ",".join(record.chain or []) or "-",
            record.size or "-",
            clip(record.prompt, 120) or "-",
        )
        try:
            return await self._queue.submit(record, runner, is_batch=is_batch)
        except QueueFullError as exc:
            self._store.set_error(record.task_id, str(exc))
            logger.warning("[aiimg] 队列已满，拒绝 %s：%s", record.task_id, exc)
            raise
        except UserLimitError as exc:
            self._store.set_error(record.task_id, str(exc))
            logger.warning("[aiimg] 超出用户并发，拒绝 %s：%s", record.task_id, exc)
            raise

    def _task_runner(
        self, kind: str
    ) -> tuple[Callable[[TaskRecord], Any], bool]:
        """Pick the runner for a task kind; (runner, is_batch) for the queue."""
        runners = {
            "generate_image": (self._run_generate, False),
            "edit_image": (self._run_edit_image, False),
            "generate_selfie": (self._run_selfie, False),
            "generate_image_batch": (self._run_batch, True),
            "generate_video": (self._run_video, False),
        }
        return runners.get(kind, (self._run_generate, False))

    def _retry_record(self, record: TaskRecord) -> TaskRecord:
        """Clone a failed task so it keeps its prompt, chain and parameters."""
        extra = dict(record.extra or {})
        # Prompt layers and the final prompt are rebuilt by the runner.
        for key in ("prompt_layers", "final_prompt"):
            extra.pop(key, None)
        return self._store.create(
            kind=record.kind,
            sid=record.sid,
            session_type=str(extra.get("session_type") or record.session_type),
            user_id=record.user_id,
            prompt=record.prompt,
            provider=record.provider,
            chain=list(record.chain or []),
            size=record.size,
            extra=extra,
        )

    def _ack_text(self, record: TaskRecord, message: str, event: Any = None) -> str:
        payload: dict[str, Any] = {
            "status": "accepted",
            "task_id": record.task_id,
            "kind": record.kind,
            "state": record.status,
            "summary": message,
        }
        instruction = self._features("ack_notification_prompt", "")
        if instruction:
            payload["instruction"] = render_template(
                instruction,
                task_id=record.task_id,
                kind=record.kind,
                state=record.status,
            ).strip()
        if event is not None:
            self._remember_ack(event, record.task_id)
        return json.dumps(payload, ensure_ascii=False)

    def _remember_ack(self, event: Any, task_id: str) -> None:
        """Track the turn that still owes the user an acknowledgement."""
        now = time.monotonic()
        for key, entry in list(self._ack_watch.items()):
            if now - entry["at"] > _ACK_WATCH_TTL:
                self._ack_watch.pop(key, None)
        entry = self._ack_watch.get(id(event))
        if entry is not None and entry["ref"]() is event:
            entry["task_ids"].add(task_id)
            entry["at"] = now
            return
        try:
            ref: Any = weakref.ref(event)
        except TypeError:
            ref = lambda: event
        self._ack_watch[id(event)] = {
            "ref": ref,
            "task_ids": {task_id},
            "delivered": False,
            "at": now,
        }

    def _mark_ack_delivered(self, task_id: str) -> None:
        """Record that the media went out before the turn spoke."""
        for entry in self._ack_watch.values():
            if task_id in entry["task_ids"]:
                entry["delivered"] = True

    @on.after_xml_parse()
    async def _skip_ack_after_delivery(self, event, actions, *_):
        entry = self._ack_watch.pop(id(event), None)
        if not entry or not entry["delivered"]:
            return
        if entry["ref"]() is not event:
            return
        actions.clear()
        logger.info(
            "[aiimg] 任务 %s 的媒体先到，跳过这一轮接单话术",
            ",".join(sorted(entry["task_ids"])),
        )

    # ------------------------------------------------------------------
    # tools
    # ------------------------------------------------------------------

    @register.tool(
        name="generate_image",
        description="根据文字描述生成图片。任务进入队列，完成后会自动把图片发送给发起者。可指定服务商、尺寸（如 1024x1024）、比例（如 16:9）或分辨率（1K/2K/4K）。",
        params={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "图片内容描述，可包含尺寸比例要求"},
                "provider": {"type": "string", "description": "服务商 id，留空用默认链路"},
                "size": {"type": "string", "description": "精确尺寸，如 1024x1024"},
                "aspect_ratio": {"type": "string", "description": "比例，如 16:9 / 3:4"},
                "resolution": {"type": "string", "description": "分辨率档位，1K/2K/4K"},
                "preset": {"type": "string", "description": "预设提示词的名字，留空表示不用"},
            },
            "required": ["prompt"],
        },
    )
    async def generate_image(
        self,
        event: KiraMessageBatchEvent,
        prompt: str,
        provider: str = "",
        size: str = "",
        aspect_ratio: str = "",
        resolution: str = "",
        preset: str = "",
    ) -> str:
        caller_prompt = str(prompt or "")
        prompt, preset_name = self._apply_preset(caller_prompt, preset, "generate")
        preset_text = self._preset_map("generate").get(preset_name.lower(), "") if preset_name else ""
        return await self._queue_image_task(
            event, "generate_image", prompt=prompt, provider=provider,
            size=size, aspect_ratio=aspect_ratio, resolution=resolution,
            preset=preset_name, preset_text=preset_text, caller_prompt=caller_prompt,
        )

    @register.tool(
        name="edit_image",
        description="基于现有图片进行修改（图生图）。image 传 'latest' 表示用本会话最近生成的图片，或传图片文件路径。可用 reference_roles 按角色挑参考图（subject 主体、clothing 服装、style 画风、pose 姿势、background 背景），选中的上传参考图会和原图一起发过去。",
        params={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "修改要求"},
                "image": {"type": "string", "description": "'latest' 或图片文件路径"},
                "provider": {"type": "string", "description": "服务商 id，留空用默认链路"},
                "size": {"type": "string", "description": "精确尺寸，如 1024x1024"},
                "aspect_ratio": {"type": "string", "description": "比例，如 16:9"},
                "resolution": {"type": "string", "description": "分辨率档位，1K/2K/4K"},
                "preset": {"type": "string", "description": "预设提示词的名字，留空表示不用"},
                "reference_roles": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["subject", "clothing", "style", "pose", "background"],
                    },
                    "description": "按角色挑参考图：只带上这些角色对应的上传图，可多选；不填则不带上传图",
                },
            },
            "required": ["prompt", "image"],
        },
    )
    async def edit_image(
        self,
        event: KiraMessageBatchEvent,
        prompt: str,
        image: str,
        provider: str = "",
        size: str = "",
        aspect_ratio: str = "",
        resolution: str = "",
        preset: str = "",
        reference_roles: list[str] | None = None,
    ) -> str:
        if not self._ready or not self._gallery:
            return "插件未就绪"
        if not self._can_use(self._user_id(event), self._adapter_name(event), self._group_id(event)):
            return self._denied("不在允许使用生图的白名单里", "生图")
        if not self._feature_on("edit_enabled"):
            return self._denied("改图功能已在插件设置里关闭", "改图")
        try:
            wanted_roles = self._wanted_reference_roles(reference_roles)
        except ValueError as exc:
            return str(exc)
        input_path = await self._resolve_input_image(event, image)
        if not input_path:
            return "没有可用的输入图片：传 'latest' 使用最近生成的图片，或提供图片文件路径"
        chain = self._edit_chain()
        if not chain:
            return "未配置支持改图的服务商"
        extra_refs = self._ref_paths(wanted_roles) if wanted_roles is not None else []
        logger.info(
            "[aiimg] 改图请求：输入图 %s，角色 %s，附带参考图 %d 张",
            Path(input_path).name,
            ",".join(wanted_roles) if wanted_roles is not None else "未指定",
            len(extra_refs),
        )
        if len(extra_refs) + 1 > MAX_REFERENCE_IMAGES:
            return self._too_many_references(len(extra_refs) + 1)
        caller_prompt = str(prompt or "")
        prompt, preset_name = self._apply_preset(caller_prompt, preset, "edit")
        extra = self._task_extra(
            event,
            explicit=self._explicit(size, aspect_ratio, resolution),
            default_output=self._default_output_for("edit_image"),
            input_image=str(input_path),
            chain_items=[self._chain_item_dict(item) for item in chain],
        )
        if extra_refs:
            extra["reference_images"] = [
                str(path.resolve()) for path, _role in extra_refs
            ]
            extra["reference_roles"] = [role for _path, role in extra_refs]
        if preset_name:
            extra["preset"] = preset_name
            extra["preset_text"] = self._preset_map("edit").get(preset_name.lower(), "")
            extra["caller_prompt"] = caller_prompt
        record = self._store.create(
            kind="edit_image",
            sid=event.sid,
            user_id=self._user_id(event),
            prompt=prompt,
            provider=chain[0].provider_id,
            chain=[item.provider_id for item in chain],
            size=format_output_intent(self._chain_intent_base(record_hint=extra)),
            extra=extra,
        )
        try:
            await self._submit(record, self._run_edit_image)
        except (QueueFullError, UserLimitError) as exc:
            return str(exc)
        return self._ack_text(record, "已加入改图队列", event)

    @register.tool(
        name="generate_selfie",
        description="使用配置好的单人参考照片生成自拍/人像图。需要先在插件设置里配置参考图路径。可用 reference_roles 按角色挑参考图（subject 主体、clothing 服装、style 画风、pose 姿势、background 背景），选中的上传参考图会和人物参考一起发过去。",
        params={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "自拍场景描述，可留空"},
                "provider": {"type": "string", "description": "服务商 id，留空用默认链路"},
                "size": {"type": "string", "description": "精确尺寸"},
                "aspect_ratio": {"type": "string", "description": "比例，如 3:4"},
                "resolution": {"type": "string", "description": "分辨率档位，1K/2K/4K"},
                "reference_roles": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["subject", "clothing", "style", "pose", "background"],
                    },
                    "description": "按角色挑参考图：只带上这些角色对应的上传图，可多选；不填则按插件设置",
                },
            },
            "required": [],
        },
    )
    async def generate_selfie(
        self,
        event: KiraMessageBatchEvent,
        prompt: str = "",
        provider: str = "",
        size: str = "",
        aspect_ratio: str = "",
        resolution: str = "",
        reference_roles: list[str] | None = None,
    ) -> str:
        if not self._ready or not self._gallery:
            return "插件未就绪"
        if not self._can_use(self._user_id(event), self._adapter_name(event), self._group_id(event)):
            return self._denied("不在允许使用生图的白名单里", "生图")
        if not self._feature_on("selfie_enabled"):
            return self._denied("自拍功能已在插件设置里关闭", "自拍")
        try:
            wanted_roles = self._wanted_reference_roles(reference_roles)
        except ValueError as exc:
            return str(exc)
        plan = await self._selfie_reference_plan(wanted_roles)
        if not plan:
            use_persona = self._feature_on("selfie_use_persona", True)
            if not use_persona:
                return "还没有可用的自拍参考图：去网页设置里上传参考图片"
            return "还没有可用的自拍参考图：去框架的人设配置里设置参考人像，或在网页设置里改成本地上传"
        if len(plan) > MAX_REFERENCE_IMAGES:
            return self._too_many_references(len(plan))
        chain = self._selfie_chain()
        if not chain:
            return self._denied("没有配置可用的生图服务商", "生图")
        extra = self._task_extra(
            event,
            explicit=self._explicit(size, aspect_ratio, resolution),
            default_output=self._default_output_for("generate_selfie"),
            reference_images=[str(path.resolve()) for path, _role in plan],
            reference_roles=[role for _path, role in plan],
            chain_items=[self._chain_item_dict(item) for item in chain],
        )
        record = self._store.create(
            kind="generate_selfie",
            sid=event.sid,
            user_id=self._user_id(event),
            prompt=prompt,
            provider=chain[0].provider_id,
            chain=[item.provider_id for item in chain],
            size=format_output_intent(self._chain_intent_base(record_hint=extra)),
            extra=extra,
        )
        try:
            await self._submit(record, self._run_selfie)
        except (QueueFullError, UserLimitError) as exc:
            return str(exc)
        return self._ack_text(record, "已加入自拍队列", event)

    @register.tool(
        name="generate_image_batch",
        description="一次生成多张图片，每张对应一条提示词。任务进入队列，完成后把全部图片一起发送。",
        params={
            "type": "object",
            "properties": {
                "prompts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "多条提示词",
                },
                "provider": {"type": "string", "description": "服务商 id"},
                "size": {"type": "string", "description": "精确尺寸"},
                "aspect_ratio": {"type": "string", "description": "比例"},
                "resolution": {"type": "string", "description": "分辨率档位"},
            },
            "required": ["prompts"],
        },
    )
    async def generate_image_batch(
        self,
        event: KiraMessageBatchEvent,
        prompts: list[str],
        provider: str = "",
        size: str = "",
        aspect_ratio: str = "",
        resolution: str = "",
    ) -> str:
        if not self._ready or not self._gallery:
            return "插件未就绪"
        if not self._can_use(self._user_id(event), self._adapter_name(event), self._group_id(event)):
            return self._denied("不在允许使用生图的白名单里", "生图")
        if not self._feature_on("batch_enabled"):
            return self._denied("批量生图功能已在插件设置里关闭", "批量生图")
        if not prompts or not any(str(p).strip() for p in prompts):
            return "至少需要一条提示词"
        prompts = [str(p).strip() for p in prompts if str(p).strip()]
        limit = int(self._cfg("section_queue", "batch_max_count", default=8) or 8)
        if len(prompts) > limit:
            prompts = prompts[:limit]
        chain = self._image_chain("generate_image")
        if not chain:
            return self._denied("没有配置可用的生图服务商", "生图")
        extra = self._task_extra(
            event,
            explicit=self._explicit(size, aspect_ratio, resolution),
            default_output=self._default_output_for("generate_image"),
            prompts=prompts,
            chain_items=[self._chain_item_dict(item) for item in chain],
        )
        record = self._store.create(
            kind="generate_image_batch",
            sid=event.sid,
            user_id=self._user_id(event),
            prompt=prompts[0],
            provider=chain[0].provider_id,
            chain=[item.provider_id for item in chain],
            size=format_output_intent(self._chain_intent_base(record_hint=extra)),
            extra=extra,
        )
        try:
            await self._submit(record, self._run_batch, is_batch=True)
        except (QueueFullError, UserLimitError) as exc:
            return str(exc)
        return self._ack_text(record, f"已加入批量队列（{len(prompts)} 张）", event)

    @register.tool(
        name="generate_video",
        description="用 Agnes 生成 AI 视频。image 传 'latest' 使用最近生成的图片作为关键帧，或传图片路径。任务生成较慢，完成后自动发送。",
        params={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "视频内容描述"},
                "image": {"type": "string", "description": "'latest' 或图片文件路径，作关键帧"},
                "provider": {"type": "string", "description": "视频服务商 id"},
                "seconds": {"type": "string", "description": "时长（秒），4 到 12"},
                "aspect_ratio": {"type": "string", "description": "比例，16:9/9:16/1:1/3:4/4:3/21:9"},
                "mode": {"type": "string", "description": "生成模式 auto/text/keyframe/reference"},
            },
            "required": ["prompt"],
        },
    )
    async def generate_video(
        self,
        event: KiraMessageBatchEvent,
        prompt: str,
        image: str = "",
        provider: str = "",
        seconds: str = "",
        aspect_ratio: str = "",
        mode: str = "",
    ) -> str:
        if not self._ready or not self._gallery:
            return "插件未就绪"
        if not self._can_use(self._user_id(event), self._adapter_name(event), self._group_id(event)):
            return self._denied("不在允许使用生图的白名单里", "生图")
        if not self._feature_on("video_enabled", default=False):
            return self._denied("视频生成功能已在插件设置里关闭（默认关闭）", "视频生成")
        chain = self._video_chain()
        if not chain:
            return self._denied("没有配置可用的视频服务商", "视频生成")
        seconds_text = str(seconds or "").strip()
        if seconds_text:
            if not seconds_text.isdigit():
                return "seconds 必须是数字（4-12）"
            seconds_int = int(seconds_text)
            if not (_VIDEO_SECONDS[0] <= seconds_int <= _VIDEO_SECONDS[1]):
                return f"seconds 必须在 {_VIDEO_SECONDS[0]}-{_VIDEO_SECONDS[1]} 之间"
            seconds_text = str(seconds_int)
        ratio = str(aspect_ratio or "").strip()
        if ratio and ratio not in _VIDEO_RATIOS:
            return f"不支持的视频比例: {ratio}"
        mode_text = str(mode or "").strip().lower()
        if mode_text and mode_text not in ("auto", "text", "keyframe", "reference"):
            return "mode 只能是 auto/text/keyframe/reference"
        input_path = ""
        if image:
            input_path = str(await self._resolve_input_image(event, image) or "")
        extra = self._task_extra(
            event,
            seconds=seconds_text,
            aspect_ratio=ratio,
            mode=mode_text,
            input_image=input_path,
            chain_items=[self._chain_item_dict(item) for item in chain],
        )
        record = self._store.create(
            kind="generate_video",
            sid=event.sid,
            user_id=self._user_id(event),
            prompt=prompt,
            provider=chain[0].provider_id,
            chain=[item.provider_id for item in chain],
            size=ratio or "",
            extra=extra,
        )
        try:
            await self._submit(record, self._run_video)
        except (QueueFullError, UserLimitError) as exc:
            return str(exc)
        return self._ack_text(record, "已加入视频队列", event)

    @register.tool(
        name="get_task_status",
        description="查询任务状态。不传 task_id 时返回队列概览和该用户最近任务。",
        params={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "任务编号，可省略"}
            },
            "required": [],
        },
    )
    async def get_task_status(self, event: KiraMessageBatchEvent, task_id: str = "") -> str:
        if not self._store:
            return "插件未就绪"
        user_id = self._user_id(event)
        if task_id:
            record = self._store.get(task_id.strip())
            if record is None:
                return f"未找到任务 {task_id}"
            if record.user_id and record.user_id != user_id:
                return "无权查看该任务"
            return self._describe_task(record)
        recent = [
            record
            for record in self._store.list_recent(limit=30)
            if not record.user_id or record.user_id == user_id
        ]
        snapshot = self._queue.snapshot() if self._queue else {}
        lines = [
            f"队列: 运行中 {snapshot.get('running', 0)} / 等待 {snapshot.get('waiting', 0)}",
        ]
        if recent:
            lines.append("最近任务:")
            for record in recent[:8]:
                lines.append(
                    f"- {record.task_id} [{record.kind}] {record.status}"
                    + (f" ({record.error})" if record.error else "")
                )
        else:
            lines.append("暂无任务")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # shared tool plumbing
    # ------------------------------------------------------------------

    def _user_id(self, event: KiraMessageBatchEvent) -> str:
        sender = getattr(getattr(event, "messages", [None])[-1] if event.messages else None, "sender", None)
        user_id = getattr(sender, "user_id", "") or ""
        adapter_name = self._adapter_name(event)
        if not user_id:
            return ""
        if ":" not in user_id:
            return f"{adapter_name}:{user_id}" if adapter_name else user_id
        return user_id

    def _group_id(self, event: KiraMessageBatchEvent) -> str:
        message = event.messages[-1] if getattr(event, "messages", None) else None
        group = getattr(message, "group", None)
        group_id = getattr(group, "group_id", "") or ""
        adapter_name = self._adapter_name(event)
        if not group_id:
            return ""
        if ":" not in group_id:
            return f"{adapter_name}:{group_id}" if adapter_name else group_id
        return group_id

    @staticmethod
    def _adapter_name(event: KiraMessageBatchEvent) -> str:
        adapter = getattr(event, "adapter", None) or ""
        return str(getattr(adapter, "name", "") or adapter or "")

    @staticmethod
    def _explicit(size: str, aspect_ratio: str, resolution: str) -> dict:
        out: dict[str, str] = {}
        for key, value in (("size", size), ("aspect_ratio", aspect_ratio), ("resolution", resolution)):
            text = str(value or "").strip()
            if text and text.lower() not in {"auto", "default", "默认"}:
                out[key] = text
        return out

    @staticmethod
    def _chain_item_dict(item: ChainItem) -> dict:
        return {"provider": item.provider_id, "output": item.output}

    def _chain_intent_base(self, *, record_hint: dict) -> OutputIntent:
        explicit = record_hint.get("explicit") or {}
        pieces: list[OutputIntent] = [OutputIntent()]
        for key in ("size", "aspect_ratio", "resolution"):
            if explicit.get(key):
                pieces.append(self._parse_intent_text(str(explicit[key])))
        chain_items = record_hint.get("chain_items") or []
        first_output = (chain_items[0] or {}).get("output") if chain_items else ""
        if first_output:
            pieces.append(self._parse_intent_text(str(first_output)))
        return merge_output_intents(*pieces)

    async def _resolve_input_image(self, event: KiraMessageBatchEvent, image: str) -> Path | None:
        text = str(image or "").strip()
        lowered = text.lower()
        if lowered in ("latest", "last", "最近"):
            recent = self._gallery.recent_images(event.sid)
            if not recent:
                return None
            path = Path(recent[0])
            return path if path.exists() else None
        if text.startswith("http://") or text.startswith("https://"):
            try:
                from .core.http import ProviderHTTP
                from .core.net import URLFetchPolicy
                policy = URLFetchPolicy()
                data = await ProviderHTTP(timeout=60, policy=policy).download_image(text)
                from .core.image_format import guess_image_mime_and_ext
                _mime, ext = guess_image_mime_and_ext(data)
                tmp = self._data_dir / "input"
                tmp.mkdir(parents=True, exist_ok=True)
                target = tmp / f"input_{int(time.time() * 1000)}.{ext}"
                target.write_bytes(data)
                return target
            except Exception as exc:
                logger.warning("[aiimg] failed to fetch input image: %s", exc)
                return None
        path = Path(text)
        return path if path.exists() else None

    def _describe_task(self, record: TaskRecord) -> str:
        result = ""
        if record.result:
            try:
                payload = json.loads(record.result or "{}")
                media = payload.get("media") or []
                result = f"，产出 {len(media)} 张" if media else ""
                if payload.get("url"):
                    result = "，视频已生成"
            except Exception:
                pass
        lines = [
            f"任务 {record.task_id}",
            f"类型: {record.kind}",
            f"状态: {record.status}",
            f"服务商: {record.provider or '未指定'}",
        ]
        if record.prompt:
            lines.append(f"提示词: {record.prompt}")
        if record.final_prompt:
            lines.append(f"最终提示词: {record.final_prompt}")
        if record.error:
            lines.append(f"错误: {record.error}")
        if result:
            lines.append(f"结果: {result}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # runners (executed by the queue workers)
    # ------------------------------------------------------------------

    async def _run_generate(self, record: TaskRecord) -> dict:
        return await self._run_image_chain(record, kind="generate")

    @staticmethod
    def _reference_prompt(prompt: str, roles: list[str]) -> tuple[str, str]:
        """Final prompt with the reference note appended once two images are sent."""
        note = reference_note(roles)
        text = str(prompt or "").strip()
        if not note:
            return text, ""
        return (f"{text}\n\n{note}" if text else note), note

    async def _run_edit_image(self, record: TaskRecord) -> dict:
        input_path = record.extra.get("input_image") or ""
        if not input_path or not Path(input_path).exists():
            raise RuntimeError("输入图片不存在")
        pairs: list[tuple[Path, str]] = [(Path(input_path), "base")]
        references = record.extra.get("reference_images") or []
        stored_roles = record.extra.get("reference_roles") or []
        for index, item in enumerate(references):
            path = Path(str(item))
            if not path.exists():
                continue
            role = str(stored_roles[index]) if index < len(stored_roles) else "subject"
            pairs.append((path, role))
        roles = [role for _path, role in pairs]
        final_prompt, note = self._reference_prompt(record.prompt, roles)
        if note:
            record.extra["final_prompt"] = final_prompt
            record.extra["prompt_layers"] = [
                {"name": "user_prompt", "source": "caller", "text": str(record.prompt or "")},
                {"name": "reference_note", "source": "plugin", "text": note},
            ]
            self._store.upsert(record)
        return await self._run_image_chain(
            record, kind="edit", images=[path for path, _role in pairs]
        )

    async def _run_selfie(self, record: TaskRecord) -> dict:
        references = record.extra.get("reference_images") or []
        stored_roles = record.extra.get("reference_roles") or []
        pairs: list[tuple[Path, str]] = []
        for index, item in enumerate(references):
            path = Path(str(item))
            if not path.exists():
                continue
            role = str(stored_roles[index]) if index < len(stored_roles) else "subject"
            pairs.append((path, role))
        if not pairs:
            raise RuntimeError("自拍参考图不存在，请重新配置")
        from .core.prompt import apply_prefix

        prefix = self._features("selfie_prompt_prefix", "")
        final_prompt, layer = apply_prefix(record.prompt, prefix, "selfie_prefix")
        final_prompt, note = self._reference_prompt(
            final_prompt, [role for _path, role in pairs]
        )
        records = [{"name": "user_prompt", "source": "caller", "text": record.prompt}]
        if layer:
            records.append(layer)
        if note:
            records.append({"name": "reference_note", "source": "plugin", "text": note})
        record.extra["prompt_layers"] = records
        record.extra["final_prompt"] = final_prompt
        self._store.upsert(record)
        return await self._run_image_chain(
            record, kind="edit", images=[path for path, _role in pairs]
        )

    async def _run_batch(self, record: TaskRecord) -> dict:
        prompts = record.extra.get("prompts") or [record.prompt]
        concurrency = int(self._cfg("section_queue", "batch_concurrency", default=2) or 2)
        semaphore = asyncio.Semaphore(max(1, concurrency))
        media: list[str] = []
        provider = ""
        model = ""
        size = ""
        errors: list[str] = []

        async def one(index: int, prompt_text: str) -> None:
            nonlocal provider, model, size
            async with semaphore:
                sub = TaskRecord(
                    task_id=f"{record.task_id}-{index}",
                    kind=record.kind,
                    sid=record.sid,
                    user_id=record.user_id,
                    prompt=prompt_text,
                    provider=record.provider,
                    chain=record.chain,
                    size=record.size,
                    extra=dict(record.extra),
                )
                try:
                    result = await self._run_image_chain(sub, kind="generate")
                    media.extend(result.get("media") or [])
                    provider = result.get("provider") or provider
                    model = result.get("model") or model
                    size = result.get("size") or size
                except Exception as exc:
                    errors.append(f"{index + 1}: {exc}")

        await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
        if not media:
            raise RuntimeError("批量生成全部失败: " + " | ".join(errors))
        return {
            "media": media,
            "provider": provider,
            "model": model,
            "size": size,
            "caption": f"批量任务共 {len(media)} 张",
            "count": len(media),
            "partial_errors": errors,
        }

    async def _render_video_frame(self, record: TaskRecord, source: Path) -> Path | None:
        """Build the opening still frame when a frame template is configured."""
        template = self._features("selfie_video_frame_template", "")
        if not template or not self._registry:
            return None
        chain = self._edit_chain()
        if not chain:
            return None
        prompt = render_template(template, prompt=record.prompt).strip() or record.prompt
        intent = self._parse_intent_text(self._features("selfie_default_output", "3:4 4K"))
        for item in chain:
            try:
                backend = self._registry.get_backend(item.provider_id)
                if not backend.supports_edit:
                    continue
                results = await backend.edit(prompt, [source], intent=intent, size_hint="")
            except Exception as exc:
                logger.warning("[aiimg] video frame via %s failed: %s", item.provider_id, exc)
                continue
            if not results:
                continue
            path = await self._gallery.save_image(
                results[0].data,
                provider_id=item.provider_id,
                model=results[0].model or "",
                size=results[0].size or "",
                sid=record.sid,
                task_id=record.task_id,
                kind="frame",
                prompt=prompt,
            )
            return Path(path)
        return None

    async def _run_video(self, record: TaskRecord) -> dict:
        chain_items = record.extra.get("chain_items") or [
            {"provider": p, "output": ""} for p in record.chain
        ]
        source_path = str(record.extra.get("input_image") or "")
        if source_path and Path(source_path).exists():
            logger.info("[aiimg] 视频任务 %s 先把参考图转成开场帧：%s", record.task_id, Path(source_path).name)
            frame = await self._render_video_frame(record, Path(source_path))
            if frame is not None:
                record.extra["input_image"] = str(frame)
                self._store.upsert(record)
                logger.info("[aiimg] 视频任务 %s 开场帧就绪：%s", record.task_id, Path(frame).name)
            else:
                logger.info("[aiimg] 视频任务 %s 没有开场帧，直接用参考图", record.task_id)
        last_error = ""
        upcoming = [item["provider"] for item in chain_items]
        for index, item in enumerate(chain_items):
            provider_id = item["provider"]
            if self._skip_cooling(provider_id, upcoming[index + 1 :]):
                cooling = int(self._stats.cooling_seconds(provider_id) + 0.5)
                last_error = f"{provider_id} 冷却中，{cooling} 秒后重试"
                logger.warning(
                    "[aiimg] 任务 %s 跳过服务商 %s：冷却中，还需 %d 秒",
                    record.task_id,
                    provider_id,
                    cooling,
                )
                continue
            try:
                backend = self._registry.get_video_backend(provider_id)
            except KeyError as exc:
                last_error = str(exc)
                continue
            started = time.monotonic()
            try:
                input_path = record.extra.get("input_image") or ""
                images = [Path(input_path)] if input_path and Path(input_path).exists() else []
                async with self._provider_slot(provider_id):
                    result = await backend.generate_video(
                        record.prompt,
                        seconds=record.extra.get("seconds") or "",
                        mode=record.extra.get("mode") or "",
                        images=images or None,
                        extra={"aspect_ratio": record.extra.get("aspect_ratio") or ""},
                    )
                video_path = await self._download_video(result.url)
                try:
                    await self._gallery.register_video(
                        video_path, provider_id=provider_id, model=result.model or "",
                        size=result.size or "", sid=record.sid, task_id=record.task_id,
                        prompt=record.prompt,
                    )
                except BaseException:
                    video_path.unlink(missing_ok=True)
                    raise
                record.provider = provider_id
                self._store.upsert(record)
                cost = (time.monotonic() - started) * 1000
                await self._stats.record_success(provider_id, cost)
                logger.info(
                    "[aiimg] 视频任务 %s 服务商 %s 成功，%dms 模型 %s 尺寸 %s 时长 %s",
                    record.task_id,
                    provider_id,
                    int(cost),
                    result.model or "-",
                    result.size or "-",
                    result.duration or "-",
                )
                return {
                    "media": [str(video_path)],
                    "url": "",
                    "provider": provider_id,
                    "model": result.model,
                    "size": result.size,
                    "duration": result.duration,
                }
            except StopProviderChain as exc:
                last_error = str(exc) or "provider requested chain stop"
                await self._stats.record_failure(
                    provider_id, (time.monotonic() - started) * 1000, last_error
                )
                break
            except Exception as exc:
                last_error = str(exc)
                await self._stats.record_failure(
                    provider_id, (time.monotonic() - started) * 1000, last_error
                )
                logger.warning(
                    "[aiimg] 视频任务 %s 服务商 %s 失败（%dms）：%s",
                    record.task_id,
                    provider_id,
                    int((time.monotonic() - started) * 1000),
                    exc,
                )
                if getattr(exc, "stop_provider_chain", False):
                    break
                continue
        raise RuntimeError(f"视频生成失败: {last_error or '无可用服务商'}")

    async def _run_image_chain(
        self,
        record: TaskRecord,
        *,
        kind: str,
        images: list[Path] | None = None,
    ) -> dict:
        chain_items = record.extra.get("chain_items") or [
            {"provider": p, "output": ""} for p in record.chain
        ]
        last_error = ""
        upcoming = [item["provider"] for item in chain_items]
        for index, item in enumerate(chain_items):
            provider_id = item["provider"]
            if self._skip_cooling(provider_id, upcoming[index + 1 :]):
                cooling = int(self._stats.cooling_seconds(provider_id) + 0.5)
                last_error = f"{provider_id} 冷却中，{cooling} 秒后重试"
                logger.warning(
                    "[aiimg] 任务 %s 跳过服务商 %s：冷却中，还需 %d 秒",
                    record.task_id,
                    provider_id,
                    cooling,
                )
                continue
            try:
                backend = self._registry.get_backend(provider_id)
            except KeyError as exc:
                last_error = str(exc)
                continue
            started = time.monotonic()
            logger.info(
                "[aiimg] 任务 %s 第 %d/%d 次尝试：服务商 %s%s",
                record.task_id,
                index + 1,
                len(chain_items),
                provider_id,
                f"（{kind}）" if kind else "",
            )
            try:
                intent = self._chain_intent(record, item.get("output") or "")
                if self._intent_is_empty(intent):
                    settings = None
                    try:
                        settings = self._registry.get_settings(provider_id)
                    except KeyError:
                        settings = None
                    provider_default = (settings.default_size if settings else "") or ""
                    if not provider_default.strip():
                        fallback = (record.extra.get("default_output") or "").strip()
                        if fallback:
                            intent = self._parse_intent_text(fallback)
                final_prompt = record.extra.get("final_prompt") or record.prompt
                async with self._provider_slot(provider_id):
                    if kind == "edit":
                        results = await backend.edit(
                            final_prompt,
                            list(images or []),
                            intent=intent,
                            size_hint=record.size,
                        )
                    else:
                        results = await backend.generate(
                            final_prompt, intent=intent, size_hint=record.size
                        )
                # HTTP 200 only means the request was accepted; a step counts
                # as a success only when real image bytes come back.
                usable = []
                for item in results or []:
                    raw = getattr(item, "data", b"") or b""
                    if not await asyncio.to_thread(_looks_like_image, raw):
                        logger.warning(
                            "[aiimg] 任务 %s 服务商 %s 返回的数据不是图片，丢弃（%dB）",
                            record.task_id,
                            provider_id,
                            len(raw),
                        )
                        continue
                    usable.append(item)
                if not usable:
                    raise RuntimeError("服务商没有返回可用的图片")
                collected = usable
                media: list[str] = []
                model = ""
                size = ""
                gallery_kind = "selfie" if record.kind == "generate_selfie" else "image"
                for result in collected:
                    path = await self._gallery.save_image(
                        result.data,
                        provider_id=provider_id,
                        model=result.model or "",
                        size=result.size or "",
                        sid=record.sid,
                        task_id=record.task_id,
                        kind=gallery_kind,
                        prompt=final_prompt,
                    )
                    media.append(str(path))
                    model = result.model or model
                    size = result.size or size
                    logger.info(
                        "[aiimg] 任务 %s 第 %d 张已存画廊：%s（%dB）",
                        record.task_id,
                        len(media),
                        Path(path).name,
                        len(result.data),
                    )
                record.provider = provider_id
                record.extra["prompt_layers"] = record.extra.get("prompt_layers") or [
                    {"name": "user_prompt", "source": "caller", "text": final_prompt}
                ]
                record.extra["final_prompt"] = final_prompt
                self._store.set_audit(
                    record.task_id,
                    prompt_layers=record.extra.get("prompt_layers"),
                    final_prompt=record.extra.get("final_prompt", ""),
                )
                cost = (time.monotonic() - started) * 1000
                await self._stats.record_success(provider_id, cost)
                logger.info(
                    "[aiimg] 任务 %s 服务商 %s 成功，%dms 已保存 %d 张",
                    record.task_id, provider_id, int(cost), len(media),
                )
                return {
                    "media": media,
                    "provider": provider_id,
                    "model": model,
                    "size": size,
                    "caption": "",
                }
            except StopProviderChain as exc:
                last_error = str(exc) or "provider requested chain stop"
                await self._stats.record_failure(
                    provider_id, (time.monotonic() - started) * 1000, last_error
                )
                break
            except Exception as exc:
                last_error = str(exc)
                await self._stats.record_failure(
                    provider_id, (time.monotonic() - started) * 1000, last_error
                )
                logger.warning(
                    "[aiimg] 任务 %s 服务商 %s 失败（%dms）：%s",
                    record.task_id,
                    provider_id,
                    int((time.monotonic() - started) * 1000),
                    exc,
                )
                cooling = self._stats.cooling_seconds(provider_id)
                if cooling:
                    logger.warning(
                        "[aiimg] 服务商 %s 进入冷却 %d 秒（连续失败 %d 次）",
                        provider_id,
                        int(cooling + 0.5),
                        self._consecutive_failures(provider_id),
                    )
                continue
        raise RuntimeError(f"所有服务商均失败: {last_error or '无可用服务商'}")

    # ------------------------------------------------------------------
    # terminal notification
    # ------------------------------------------------------------------

    @staticmethod
    def _error_kind(error: str) -> str:
        """Coarse failure category; the raw text stays in logs and the record."""
        text = str(error or "").lower()
        if "超时" in text or "timeout" in text or "timed out" in text:
            return "超时"
        if "未返回图片" in text or "可用的图片" in text or "no image" in text or "empty image" in text:
            return "没出图"
        if "不存在" in text or "not found" in text or "missing" in text:
            return "输入缺失"
        if "均失败" in text or "http" in text or "状态码" in text:
            return "服务商报错"
        return "内部错误"

    async def _notify_failure(self, record: TaskRecord) -> None:
        """Let the model explain a failure in persona; ask once, then give up quietly."""
        error = str(record.error or "").strip()
        facts = {
            "task_id": record.task_id,
            "kind": record.kind,
            "status": "failed",
            "provider": record.provider or "",
            "size": record.size or "",
            "error_kind": self._error_kind(error),
        }
        template = self._features("error_notification_prompt", "")
        # A failure has no media to carry, and the route stage drops events
        # whose chain is empty, so the turn needs one short line to ride along.
        delivered = await self._publish_notification(
            record,
            [Text(_facts_line(facts))],
            template,
            facts,
            marker=_ERROR_MARKER,
            template_key="error_notification_prompt",
        )
        logger.info(
            "[aiimg] 任务 %s 失败通知%s",
            record.task_id,
            "已交给模型" if delivered else "未发布（模型侧不可用）",
        )
        if not delivered:
            await self._safe_send(
                record.sid,
                MessageChain([Text(f"任务 {record.task_id} 没成功，这次没出图。")]),
            )

    async def _on_task_finished(self, record: TaskRecord) -> None:
        if not self._ready:
            return
        logger.info(
            "[aiimg] 任务 %s 结束：状态=%s 服务商=%s",
            record.task_id,
            record.status,
            record.provider or "-",
        )
        if record.status == "cancelled":
            await self._safe_send(record.sid, MessageChain([Text(f"任务 {record.task_id} 已取消")]))
            return
        if record.status == "failed":
            await self._notify_failure(record)
            return
        if record.status != "succeeded":
            return
        try:
            payload = json.loads(record.result or "{}")
        except Exception:
            payload = {}
        elements: list[Any] = []
        facts: dict[str, str] = {
            "task_id": record.task_id,
            "kind": record.kind,
            "status": record.status,
            "provider": record.provider or "",
            "prompt": record.prompt or "",
            "final_prompt": record.final_prompt or record.prompt or "",
            "size": record.size or payload.get("size") or "",
            "model": payload.get("model") or "",
            "error": "",
        }
        media = payload.get("media") or []
        video_url = payload.get("url") or ""
        if video_url:
            elements.append(Video(file=video_url))
            facts["kind"] = "video"
        for path in media:
            if str(path).lower().endswith((".mp4", ".webm", ".mov")):
                elements.append(Video(file=str(path)))
                facts["kind"] = "video"
            else:
                elements.append(Image(image=str(path)))
        if payload.get("caption"):
            facts["caption"] = str(payload["caption"])
        if payload.get("count"):
            facts["count"] = str(payload["count"])
        if not elements:
            return
        template = self._features("terminal_notification_prompt", "")
        # Deliver the media ourselves before asking the model for a sentence.
        # The synthetic event below reaches the model only; whatever it replies
        # is a separate message, so an empty reply can no longer swallow the
        # image.
        await self._safe_send(record.sid, MessageChain(elements))
        self._mark_ack_delivered(record.task_id)
        logger.info(
            "[aiimg] 任务 %s 媒体已发送：%d 个元素（%s）",
            record.task_id,
            len(elements),
            "视频" if facts["kind"] == "video" else "图片",
        )
        published = await self._publish_notification(record, elements, template, facts)
        logger.info(
            "[aiimg] 任务 %s 完成通知%s",
            record.task_id,
            "已交给模型" if published else "未发布（模型侧不可用）",
        )

    async def _safe_send(self, sid: str, chain: MessageChain) -> None:
        try:
            await self.ctx.send_message_chain(sid, chain)
        except Exception as exc:
            logger.warning("[aiimg] failed to deliver to %s: %s", sid, exc)

    async def _publish_notification(
        self,
        record: TaskRecord,
        elements: list[Any],
        template: str,
        facts: dict[str, str],
        *,
        marker: str = _NOTIFY_MARKER,
        template_key: str = "terminal_notification_prompt",
    ) -> bool:
        adapter_name = record.extra.get("adapter") or ""
        session_type = record.extra.get("session_type") or ""
        session_id = record.extra.get("session_id") or ""
        if not adapter_name or not session_id:
            return False
        try:
            adapter = self.ctx.adapter_mgr.get_adapter(adapter_name)
        except Exception as exc:
            logger.warning("[aiimg] adapter %s unavailable: %s", adapter_name, exc)
            return False
        group = Group(group_id=session_id) if session_type == "gm" else None
        event = KiraMessageEvent(
            adapter=adapter.info,
            message=KiraIMMessage(
                timestamp=int(time.time()),
                sender=User(user_id="system_proactive", nickname="系统"),
                group=group,
                message_id=f"aiimg_{record.task_id}",
                self_id=str((adapter.config or {}).get("self_id", "") or ""),
                chain=MessageChain(elements),
                is_notice=False,
                is_mentioned=True,
            ),
            timestamp=int(time.time()),
            **_event_elements_kw(adapter),
        )
        event.session = Session(
            adapter_name=adapter_name,
            session_type=session_type or "dm",
            session_id=session_id,
        )
        event.message.extra = {
            marker: record.task_id,
            "_aiimg_facts": facts,
            "_aiimg_template_key": template_key,
        }
        # Ask the pipeline to answer this message instead of discarding it.
        trigger = getattr(event, "trigger", None)
        if callable(trigger):
            try:
                trigger(force=True)
            except TypeError:
                trigger()
        try:
            await self.ctx.event_bus.publish(event)
        except Exception as exc:
            logger.warning("[aiimg] notification publish failed: %s", exc)
            return False
        return True

    @on.llm_request()
    async def _inject_task_notification(self, event, req: LLMRequest, *_):
        extra = _notification_extra(event)
        if not extra:
            return
        facts = extra.get("_aiimg_facts") or {}
        template_key = extra.get("_aiimg_template_key") or "terminal_notification_prompt"
        template = self._features(template_key, "")
        rendered = render_template(template, **facts).strip() if template else ""
        if not rendered:
            rendered = f"{_facts_line(facts)}\n{_NOTICE_HINT}"
        if rendered:
            req.user_prompt.append(
                Prompt(
                    rendered,
                    name="aiimg_task_notification",
                    source="aiimg",
                    persist=False,
                )
            )
        for tool_name in TOOL_NAMES:
            try:
                req.tool_set.remove(tool_name)
            except Exception:
                pass

    @on.llm_request()
    async def _inject_presets(self, event, req: LLMRequest, *_):
        """Tell the model which named presets it may pass to the image tools."""
        draw = self._preset_map("generate")
        edit = self._preset_map("edit")
        if not draw and not edit:
            return
        lines: list[str] = []
        intro = str(self._features("presets_prompt", "") or "").strip()
        if intro:
            lines.append(intro)
        for name, text in draw.items():
            lines.append(f"- {name}（文生图）：{text}")
        for name, text in edit.items():
            lines.append(f"- {name}（改图）：{text}")
        content = "\n".join(lines)
        for prompt in getattr(req, "system_prompt", []) or []:
            if getattr(prompt, "name", "") == "aiimg_presets":
                prompt.content = content
                return
        try:
            req.system_prompt.append(
                Prompt(content, name="aiimg_presets", source="aiimg", persist=False)
            )
        except Exception as exc:
            logger.debug("[aiimg] failed to inject presets: %s", exc)

    @on.llm_request(priority=Priority.LOW)
    async def _hide_builtin_media_tags(self, event, req, tag_set, *_):
        """Take over image/video generation from the framework's built-in tags.

        Runs after the built-in plugin registered its tags and after plugins that
        swap in their own ``img`` tag, so only untouched built-in instances are
        removed.
        """
        if not self._compat("replace_builtin_media", True):
            return
        get_all = getattr(tag_set, "get_all", None)
        unregister = getattr(tag_set, "unregister", None)
        if get_all is None or unregister is None:
            return
        tags = list(get_all() or [])
        get_all_root = getattr(tag_set, "get_all_root", None)
        if get_all_root is not None:
            tags.extend(get_all_root() or [])
        grouped: dict[str, list[Any]] = {}
        for tag in tags:
            name = getattr(tag, "name", "")
            if name in _BUILTIN_MEDIA_TAGS:
                grouped.setdefault(name, []).append(tag)
        removable = [
            name
            for name, items in grouped.items()
            if self._can_replace_builtin_tag(name)
            and all(_is_builtin_media_tag(tag) for tag in items)
        ]
        if not removable:
            return
        try:
            unregister(*removable)
        except Exception as exc:
            logger.debug("[aiimg] failed to hide builtin media tags: %s", exc)
            return
        if not self._builtin_tags_logged:
            self._builtin_tags_logged = True
            logger.info(
                "[aiimg] builtin media tags disabled: %s",
                ", ".join(sorted(removable)),
            )

    # ------------------------------------------------------------------
    # WebUI
    # ------------------------------------------------------------------

    @register.page(
        "/aiimg",
        auth=True,
        menu=PageMenu(label={"zh": "AI 生图"}, icon="Picture", order=100),
    )
    def page_dashboard(self):
        return PluginPage.from_folder("./web")

    def _status_payload(self) -> dict:
        registry = self._registry
        queue = self._queue.snapshot() if self._queue else {}
        queue.pop("recent", None)
        return {
            "ready": self._ready,
            "version": _plugin_version(),
            "providers": registry.list_settings() if registry else [],
            "image_providers": registry.image_provider_ids() if registry else [],
            "video_providers": registry.video_provider_ids() if registry else [],
            "provider_errors": registry.validate() if registry else [],
            "queue": queue,
            "gallery": self._gallery.stats() if self._gallery else {},
        }

    @register.api(method="GET", path="/aiimg/status", auth=True, summary="插件状态")
    async def api_status(self):
        return self._status_payload()

    @register.api(method="GET", path="/aiimg/metrics", auth=True, summary="服务商健康与延迟")
    async def api_metrics(self, payload: dict | None = None):
        return {
            "providers": self._stats.snapshot(),
            "concurrency": {
                provider_id: limit for provider_id, (limit, _sem) in self._limiters.items()
            },
            "generated_at": int(time.time()),
        }

    @register.api(method="GET", path="/aiimg/config", auth=True, summary="读取全部配置")
    async def api_config(self):
        manager = getattr(self.ctx, "plugin_mgr", None)
        try:
            cfg = (
                manager.get_plugin_config(PLUGIN_ID)
                if manager is not None
                else (self.plugin_cfg or {})
            )
        except Exception:
            cfg = self.plugin_cfg or {}
        return {
            "revision": settings_mod.revision(cfg),
            "config": settings_mod.with_defaults(cfg),
            "sections": settings_mod.sections(),
            "templates": settings_mod.template_catalog(),
            "fields": settings_mod.field_meta(),
            "provider_defaults": {
                template["key"]: settings_mod.provider_defaults(template["key"])
                for template in settings_mod.template_catalog()
            },
            "required": {
                template["key"]: list(
                    settings_mod.REQUIRED_FIELDS.get(template["key"], ())
                )
                for template in settings_mod.template_catalog()
            },
            "features": [
                {"key": key, **settings_mod.FEATURE_LABELS.get(key, {})}
                for key in settings_mod.IMPORT_FEATURES
            ],
            "notes": settings_mod.warnings(cfg),
            "status": self._status_payload(),
        }

    @staticmethod
    def _merge_config(current: dict, incoming: dict) -> dict:
        """Merge one page's changes into the stored config.

        Section dictionaries merge field by field so saving the provider page
        cannot wipe the queue page, and the rest replaces as a whole.
        """
        merged = dict(current or {})
        for key, value in (incoming or {}).items():
            existing = merged.get(key)
            if key.startswith("section_") and isinstance(value, dict) and isinstance(existing, dict):
                section = dict(existing)
                section.update(value)
                merged[key] = section
            else:
                merged[key] = value
        return merged

    @register.api(method="PUT", path="/aiimg/config", auth=True, summary="保存全部配置")
    async def api_save_config(self, payload: dict):
        from fastapi import HTTPException

        manager = getattr(self.ctx, "plugin_mgr", None)
        if manager is None:
            raise HTTPException(503, "插件管理器没有连接，暂时不能保存配置")
        data = payload if isinstance(payload, dict) else {}
        async with self._config_lock:
            try:
                current = manager.get_plugin_config(PLUGIN_ID)
            except Exception:
                current = self.plugin_cfg or {}
            if str(data.get("revision") or "") != settings_mod.revision(current):
                logger.warning(
                    "[aiimg] 保存配置被拒：revision 不匹配（页面 %s / 当前 %s）",
                    str(data.get("revision") or "")[:12] or "-",
                    settings_mod.revision(current)[:12],
                )
                raise HTTPException(409, "配置已经在别处改过了，刷新页面再试")
            active = self._store.count_active() if self._store else 0
            if active and not data.get("allow_reload"):
                logger.warning("[aiimg] 保存配置被拒：还有 %d 个任务在跑", active)
                raise HTTPException(
                    409, f"还有 {active} 个任务在跑，确认后会立刻应用新配置"
                )
            try:
                submitted = data.get("config")
                incoming = submitted if isinstance(submitted, dict) else {}
                merged = self._merge_config(current, incoming)
                config = settings_mod.normalize(merged)
            except ValueError as exc:
                logger.warning("[aiimg] 配置校验失败：%s", exc)
                raise HTTPException(422, str(exc)) from exc
            logger.info(
                "[aiimg] 保存配置：页=%s 服务商 %d 家 链路 %d 条 变更字段 %s",
                ",".join(sorted(key for key in incoming if str(key).startswith("section_"))) or "-",
                len(config.get("providers") or []),
                len(config.get("chains") or {}),
                ",".join(sorted(
                    field
                    for key, value in incoming.items()
                    if isinstance(value, dict) and str(key).startswith("section_")
                    for field in value
                ))[:300] or "-",
            )
            updated = await manager.update_plugin_config(PLUGIN_ID, config)
            # The framework re-initializes a plugin after a config save, so this
            # instance may already be retired; report from the live one.
            self.plugin_cfg = updated if isinstance(updated, dict) else config
            live = None
            getter = getattr(manager, "get_plugin_inst", None)
            if callable(getter):
                try:
                    live = getter(PLUGIN_ID)
                except Exception:
                    live = None
            if live is not None and live is not self:
                config_now = getattr(live, "plugin_cfg", None) or self.plugin_cfg
                status = (
                    live._status_payload()
                    if hasattr(live, "_status_payload")
                    else self._status_payload()
                )
            else:
                await self._apply_config()
                config_now = self.plugin_cfg
                status = self._status_payload()
            return {
                "revision": settings_mod.revision(config_now),
                "config": settings_mod.with_defaults(config_now),
                "status": status,
                "notes": settings_mod.warnings(config_now),
            }

    @register.api(method="POST", path="/aiimg/models", auth=True, summary="按接口地址拉取模型列表")
    async def api_fetch_models(self, payload: dict):
        from fastapi import HTTPException

        data = payload if isinstance(payload, dict) else {}
        base_url = str(data.get("base_url") or "").strip()
        api_key = str(data.get("api_key") or "").strip()
        server_url = str(data.get("server_url") or "").strip()
        root = base_url or server_url
        if not root:
            raise HTTPException(400, "先填好接口地址再获取模型")
        url = root.rstrip("/") + "/models"
        policy = URLFetchPolicy(
            trusted_origins=trusted_origins_from_config(self._provider_config())
        )
        try:
            await ensure_url_allowed(url, policy)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as resp:
                    text = await resp.text()
                    if resp.status != 200:
                        raise HTTPException(502, f"接口返回 {resp.status}，检查地址和密钥")
                    body = json.loads(text)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(502, f"获取失败：{exc}") from exc
        models: list[str] = []
        items = body
        if isinstance(body, dict):
            items = body.get("data") or body.get("models") or []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    name = str(item.get("id") or item.get("name") or "").strip()
                else:
                    name = str(item or "").strip()
                if name and name not in models:
                    models.append(name)
        return {"models": models}

    @register.api(method="POST", path="/aiimg/tasks/cancel", auth=True, summary="取消任务")
    async def api_cancel_task(self, payload: dict):
        if not self._queue:
            return {"ok": False, "error": "插件未就绪"}
        data = payload if isinstance(payload, dict) else {}
        ok = await self._queue.cancel(str(data.get("task_id") or ""))
        return {"ok": ok}

    @register.api(method="GET", path="/aiimg/tasks", auth=True, summary="任务列表")
    async def api_tasks(self, state: str = "", limit: int = 50, offset: int = 0):
        if not self._store:
            return {"tasks": [], "total": 0}
        records, total = self._store.search_tasks(
            state=state,
            limit=min(max(1, limit), 200),
            offset=max(0, offset),
        )
        return {"tasks": [record.to_dict() for record in records], "total": total}

    @register.api(method="POST", path="/aiimg/tasks/{task_id}/retry", auth=True, summary="重试失败任务")
    async def api_retry_task(self, payload: dict, task_id: str):
        if not self._store or not self._queue:
            return {"ok": False, "error": "插件未就绪"}
        data = payload if isinstance(payload, dict) else {}
        target = str(task_id or data.get("task_id") or "")
        record = self._store.get(target)
        if record is None:
            return {"ok": False, "error": "任务不存在"}
        if record.status != "failed":
            return {"ok": False, "error": "只有失败的任务可以重试"}
        retry = self._retry_record(record)
        runner, is_batch = self._task_runner(record.kind)
        try:
            await self._submit(retry, runner, is_batch=is_batch)
        except (QueueFullError, UserLimitError) as exc:
            return {"ok": False, "error": str(exc), "task_id": retry.task_id}
        return {"ok": True, "task_id": retry.task_id, "source": record.task_id}

    @register.api(method="GET", path="/aiimg/gallery", auth=True, summary="画廊")
    async def api_gallery(
        self,
        offset: int = 0,
        limit: int = 0,
        q: str = "",
        kind: str = "",
        favorite: int = -1,
        sort: str = "new",
    ):
        if not self._gallery:
            return {"items": [], "total": 0}
        wanted_kind = str(kind or "").strip().lower()
        if wanted_kind not in ("image", "selfie", "video"):
            wanted_kind = ""
        try:
            fav_flag = int(favorite)
        except (TypeError, ValueError):
            fav_flag = -1
        fav = bool(fav_flag) if fav_flag in (0, 1) else None
        images, total = self._gallery.search_gallery(
            offset=max(0, offset),
            limit=max(0, limit),
            query=q,
            kind=wanted_kind,
            favorite=fav,
            sort="old" if str(sort or "").lower() == "old" else "new",
        )
        out = []
        for im in images:
            item = {key: value for key, value in im.items() if key != "path"}
            item["filename"] = Path(im.get("path") or "").name
            out.append(item)
        return {"items": out, "total": total}

    @register.api(method="POST", path="/aiimg/gallery/bulk", auth=True, summary="批量删除/收藏")
    async def api_gallery_bulk(self, payload: dict):
        if not self._gallery:
            return {"ok": False, "error": "插件未就绪"}
        data = payload if isinstance(payload, dict) else {}
        action = str(data.get("action") or "").strip().lower()
        ids = data.get("ids")
        if not isinstance(ids, (list, tuple)):
            return {"ok": False, "error": "ids 必须是数组"}
        if action == "delete":
            result = self._gallery.delete_many(list(ids))
            return {"ok": True, "action": action, **result}
        if action in ("favorite", "unfavorite"):
            changed = self._gallery.favorite_many(
                list(ids), action == "favorite"
            )
            return {"ok": True, "action": action, "updated": changed}
        return {"ok": False, "error": "未知操作"}

    @register.api(method="GET", path="/aiimg/gallery/item/{image_id}", auth=True, summary="图片详情")
    async def api_gallery_item(self, image_id: str):
        record = self._gallery.detail(image_id) if self._gallery else None
        if record is None:
            return {"ok": False, "error": "图片不存在"}
        item = {key: value for key, value in record.items() if key != "path"}
        return {"ok": True, "item": item}

    def _gallery_file(self, image_id: str) -> Path | None:
        if not self._gallery:
            return None
        record = self._gallery.get(image_id)
        if not record:
            return None
        path = Path(record.get("path") or "")
        return path if path.is_file() else None

    @register.api(method="GET", path="/aiimg/gallery/file", auth=True, summary="画廊原图")
    async def api_gallery_file(self, image_id: str):
        from fastapi.responses import FileResponse

        path = self._gallery_file(image_id)
        if path is None:
            return {"ok": False, "error": "图片不存在或已清理"}
        suffix = path.suffix.lower()
        media = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
        }.get(suffix, "application/octet-stream")
        return FileResponse(
            path,
            media_type=media,
            headers={"Cache-Control": "private, max-age=31536000, immutable"},
        )

    @register.api(method="GET", path="/aiimg/gallery/thumb", auth=True, summary="画廊缩略图")
    async def api_gallery_thumb(self, image_id: str):
        from fastapi.responses import FileResponse, Response

        record = self._gallery.get(image_id) if self._gallery else None
        if record and str(record.get("kind") or "image") == "video":
            return Response(status_code=404)
        path = self._gallery_file(image_id)
        if path is None:
            return {"ok": False, "error": "图片不存在或已清理"}
        thumb = self._gallery.thumb_path(image_id)
        if not thumb.is_file():
            from .core.image_format import thumbnail_webp

            data = await asyncio.to_thread(path.read_bytes)
            data = await asyncio.to_thread(thumbnail_webp, data)
            thumb.write_bytes(data)
        return FileResponse(
            thumb,
            media_type="image/webp",
            headers={"Cache-Control": "private, max-age=31536000, immutable"},
        )

    @register.api(method="GET", path="/aiimg/gallery/preview", auth=True, summary="画廊预览图")
    async def api_gallery_preview(self, image_id: str):
        from fastapi.responses import FileResponse, Response

        record = self._gallery.get(image_id) if self._gallery else None
        if record and str(record.get("kind") or "image") == "video":
            return Response(status_code=404)
        path = self._gallery_file(image_id)
        if path is None:
            return {"ok": False, "error": "图片不存在或已清理"}
        preview = self._gallery.preview_path(image_id)
        if not preview.is_file():
            from .core.image_format import preview_jpeg
            from .core.image_store import ImageStore

            data = await asyncio.to_thread(path.read_bytes)
            data = await asyncio.to_thread(preview_jpeg, data)
            await asyncio.to_thread(ImageStore._atomic_write, preview, data)
        return FileResponse(
            preview,
            media_type="image/jpeg",
            headers={"Cache-Control": "private, max-age=31536000, immutable"},
        )

    @register.api(method="GET", path="/aiimg/gallery/image", auth=True, summary="画廊图片（data URL）")
    async def api_gallery_image(self, image_id: str):
        if not self._gallery:
            return {"ok": False, "error": "插件未就绪"}
        im = self._gallery.get(image_id)
        if not im:
            return {"ok": False, "error": "图片不存在"}
        path = Path(im.get("path") or "")
        if not path.exists():
            return {"ok": False, "error": "文件缺失"}
        from .core.image_format import guess_image_mime_and_ext, image_data_url
        data = path.read_bytes()
        _mime, _ext = guess_image_mime_and_ext(data)
        return {
            "ok": True,
            "data_url": image_data_url(data),
            "filename": path.name,
            "prompt": str(im.get("prompt") or ""),
        }

    @register.api(method="POST", path="/aiimg/gallery/favorite", auth=True, summary="收藏/取消收藏")
    async def api_gallery_favorite(self, payload: dict):
        if not self._gallery:
            return {"ok": False}
        data = payload if isinstance(payload, dict) else {}
        image_id = str(data.get("image_id") or "")
        return {"ok": self._gallery.set_favorite(image_id, bool(data.get("favorite")))}

    @register.api(method="POST", path="/aiimg/gallery/delete", auth=True, summary="删除图片")
    async def api_gallery_delete(self, payload: dict):
        if not self._gallery:
            return {"ok": False}
        data = payload if isinstance(payload, dict) else {}
        return {"ok": self._gallery.delete(str(data.get("image_id") or ""))}

    # ------------------------------------------------------------------
    # selfie reference files
    # ------------------------------------------------------------------

    def _refs_dir(self) -> Path:
        path = self._data_dir / "refs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _ref_entries(self) -> list[dict]:
        """Ordered index entries; unindexed files append by mtime as subject."""
        if self._data_dir is None:
            return []
        return refs_mod.list_entries(self._data_dir)

    def _ref_files(self) -> list[str]:
        """Reference photos live on disk only, so uploads never touch the config.

        Keeping the list out of the config means an upload cannot change the
        config revision and cannot collide with a settings save.
        """
        return [entry["file"] for entry in self._ref_entries()]

    def _ref_paths(self, roles: list[str] | None = None) -> list[tuple[Path, str]]:
        """Existing uploads in index order, optionally filtered by role."""
        wanted = set(roles) if roles is not None else None
        found: list[tuple[Path, str]] = []
        for entry in self._ref_entries():
            if wanted is not None and entry["role"] not in wanted:
                continue
            path = self._data_dir / entry["file"]
            if path.exists():
                found.append((path, entry["role"]))
        return found

    @staticmethod
    def _wanted_reference_roles(raw: Any) -> list[str] | None:
        """Validated role filter; None means the caller did not ask for one."""
        if raw is None:
            return None
        items = [raw] if isinstance(raw, str) else list(raw or [])
        wanted: list[str] = []
        for item in items:
            role = str(item or "").strip().lower()
            if not role:
                continue
            if role not in refs_mod.ROLES:
                raise ValueError(
                    "reference_roles 只能是 subject、clothing、style、pose、background"
                )
            if role not in wanted:
                wanted.append(role)
        return wanted

    @staticmethod
    def _too_many_references(count: int) -> str:
        return (
            f"参考图最多 {MAX_REFERENCE_IMAGES} 张，现在有 {count} 张："
            "请减少上传的参考图，或用 reference_roles 只挑需要的角色。"
        )

    def _ref_items(self) -> list[dict]:
        items = []
        for entry in self._ref_entries():
            file = self._data_dir / entry["file"]
            exists = file.exists()
            items.append(
                {
                    "name": entry["file"],
                    "role": entry["role"],
                    "size": file.stat().st_size if exists else 0,
                    "exists": exists,
                }
            )
        return items

    @register.api(method="GET", path="/aiimg/files", auth=True, summary="自拍参考图列表")
    async def api_files(self):
        return {"files": self._ref_items(), "mode": "persona" if self._feature_on("selfie_use_persona", True) else "files"}

    @register.api(method="POST", path="/aiimg/files/upload", auth=True, summary="上传自拍参考图")
    async def api_files_upload(self, payload: dict):
        from .core.image_format import (
            decode_base64_image,
            guess_image_mime_and_ext,
            validate_image_bytes,
        )

        body = payload if isinstance(payload, dict) else {}
        name = str(body.get("name") or "")
        data_url = str(body.get("data_url") or "")
        data = decode_base64_image(data_url or "")
        if not data:
            return {"ok": False, "error": "没有读到图片内容"}
        if len(data) > 10 * 1024 * 1024:
            return {"ok": False, "error": "单张参考图不要超过 10MB"}
        try:
            validate_image_bytes(data)
        except Exception:
            return {"ok": False, "error": "只支持 jpg、png、webp"}
        _mime, ext = guess_image_mime_and_ext(data)
        ext = (ext or "").lstrip(".").lower()
        if ext not in {"jpg", "jpeg", "png", "webp"}:
            return {"ok": False, "error": "只支持 jpg、png、webp"}
        stem = Path(str(name or "reference")).stem[:40] or "reference"
        safe = "".join(ch for ch in stem if ch.isalnum() or ch in "-_") or "reference"
        filename = f"refs/{safe[:24]}-{int(__import__('time').time())}.{ext}"
        target = self._data_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        refs_mod.append_entry(self._data_dir, filename)
        logger.info(
            "[aiimg] 上传参考图 %s（%dB，原始名 %s），现共 %d 张",
            filename,
            len(data),
            clip(name, 60) or "-",
            len(self._ref_items()),
        )
        return {"ok": True, "files": self._ref_items()}

    @register.api(method="POST", path="/aiimg/files/role", auth=True, summary="设置自拍参考图角色")
    async def api_files_role(self, payload: dict):
        from fastapi import HTTPException

        body = payload if isinstance(payload, dict) else {}
        name = str(body.get("name") or "").strip()
        role = str(body.get("role") or "").strip().lower()
        if not name:
            return {"ok": False, "error": "没有指定文件"}
        if self._data_dir is None:
            return {"ok": False, "error": "插件数据目录还没有准备好"}
        try:
            refs_mod.set_role(self._data_dir, name, role)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except KeyError:
            return {"ok": False, "error": "文件不存在"}
        logger.info("[aiimg] 参考图 %s 角色改为 %s", name, role)
        return {"ok": True, "files": self._ref_items()}

    @register.api(method="POST", path="/aiimg/files/order", auth=True, summary="调整自拍参考图顺序")
    async def api_files_order(self, payload: dict):
        from fastapi import HTTPException

        body = payload if isinstance(payload, dict) else {}
        names = body.get("names")
        if not isinstance(names, list) or not names:
            raise HTTPException(400, "排序需要提供全部参考图文件名")
        if self._data_dir is None:
            return {"ok": False, "error": "插件数据目录还没有准备好"}
        try:
            refs_mod.reorder(self._data_dir, list(names))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        logger.info("[aiimg] 参考图顺序改为：%s", " → ".join(Path(str(n)).name for n in names))
        return {"ok": True, "files": self._ref_items()}

    @register.api(method="POST", path="/aiimg/files/delete", auth=True, summary="删除自拍参考图")
    async def api_files_delete(self, payload: dict):
        body = payload if isinstance(payload, dict) else {}
        target = str(body.get("name") or "").strip()
        if not target:
            return {"ok": False, "error": "没有指定文件"}
        path = (self._data_dir / target).resolve()
        root = self._refs_dir().resolve()
        if root not in path.parents or path.name in {refs_mod.INDEX_NAME, refs_mod.INDEX_NAME + ".tmp"}:
            return {"ok": False, "error": "文件不存在"}
        if path.exists():
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("[aiimg] failed to delete reference file: %s", exc)
        refs_mod.remove_entry(self._data_dir, target)
        logger.info("[aiimg] 删除参考图 %s，剩 %d 张", target, len(self._ref_items()))
        return {"ok": True, "files": self._ref_items()}

    @register.api(method="GET", path="/aiimg/files/raw", auth=True, summary="查看自拍参考图")
    async def api_files_raw(self, name: str = ""):
        target = str(name or "").strip()
        path = (self._data_dir / target).resolve()
        root = self._refs_dir().resolve()
        if (
            not target
            or root not in path.parents
            or not path.exists()
            or path.name == refs_mod.INDEX_NAME
        ):
            return {"ok": False, "error": "文件不存在"}
        from .core.image_format import image_data_url

        return {"ok": True, "data_url": image_data_url(path.read_bytes()), "name": path.name}

