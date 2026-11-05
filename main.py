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
from core.provider import LLMRequest

from .core import planner
from .core import refs as refs_mod
from .core import settings as settings_mod
from .core.chains import ChainItem, chain_for
from .core.debounce import Debouncer
from .core.delivery import deliver_chain
from .core.http import close_http, current_session
from .core.image_store import ImageStore
from .core.image_format import validate_image_bytes
from .core.inspiration import InspirationStore
from .core.memory import ImageMemory
from .core.logbook import clip, elapsed_ms, mask
from .core.net import URLFetchPolicy, ensure_url_allowed, origin, trusted_origins_from_config
from .core.output_spec import (
    OutputIntent,
    apply_conflict_rule,
    extract_output_intent_from_prompt,
    format_output_intent,
    merge_output_intents,
    parse_output_intent,
)
from .core.prompt import reference_note, render_template
from .core import refbridge
from .core.provider_registry import ProviderRegistry
from .core.provider_stats import ProviderStats
from .core.providers.base import StopProviderChain
from .core.queue import QueueFullError, TaskQueue, UserLimitError
from .core.task_store import TaskRecord, TaskStore, new_owner_id
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
    "resend_image",
    "find_image",
    "save_inspiration",
    "pick_inspiration",
)

# Read-only lookups: never gated by the whitelist, but still kept inside the
# caller's own session unless isolation is turned off.
READ_ONLY_TOOLS = ("get_task_status", "resend_image", "find_image", "pick_inspiration")

IMAGE_KINDS = ("generate_image", "edit_image", "generate_selfie", "generate_image_batch")

_VIDEO_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
_VIDEO_SECONDS = (4, 12)

_NOTIFY_MARKER = "_aiimg_task"
_ERROR_MARKER = "_aiimg_task_error"
# How long a submitted task may still be waiting for its acknowledgement.
_ACK_WATCH_TTL = 300.0

# Pictures taken straight from the conversation; the rest of the budget stays
# free for role-tagged reference uploads. The hard cap lives in refbridge.
_ATTACHMENT_LIMIT = 4

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


def _facts_text(facts: dict[str, Any]) -> str:
    """Machine-readable facts for the turns whose prompt settings are empty.

    A canned sentence here would be the plugin talking in its own voice; a JSON
    line keeps every visible word under the persona's control while still giving
    the model the fields it needs to phrase the outcome.
    """
    cleaned = {
        str(key): value
        for key, value in (facts or {}).items()
        if str(value if value is not None else "").strip()
    }
    return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))


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
        self._memory: ImageMemory | None = None
        self._inspiration: InspirationStore | None = None
        self._queue: TaskQueue | None = None
        # Identifies this process when several instances share one data dir.
        self._owner = new_owner_id()
        self._config_lock = asyncio.Lock()
        self._ready = False
        self._builtin_tags_logged = False
        # Collapses an identical request repeated within a few seconds.
        self._debounce = Debouncer(0.0)
        # Turns that submitted a task and still owe the user an acknowledgement.
        self._ack_watch: dict[int, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def initialize(self):
        started = time.perf_counter()
        self._data_dir = self.ctx.get_plugin_data_dir()
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._debounce.configure(
            float(self._cfg("section_queue", "debounce_seconds", default=3) or 0)
        )
        self._stats = ProviderStats(self._data_dir)
        self._configure_stats()
        self._store = TaskStore(self._data_dir)
        self._gallery = ImageStore(self._data_dir, {
            "gallery": self._cfg("section_gallery"),
        })
        self._memory = ImageMemory(self._data_dir / "image_memory.sqlite3")
        self._inspiration = InspirationStore(self._data_dir / "inspiration.json", limit=200)
        logger.info(
            "[aiimg] 记忆与灵感：记忆库 %s，灵感库 %d 条（上限 200）",
            self._memory.db_path.name,
            len(self._inspiration),
        )
        self._registry = ProviderRegistry(
            self._provider_config(),
            data_dir=self._data_dir,
            policy=self._fetch_policy(),
            image_max_bytes=self._advanced_max_bytes(),
        )
        self._build_queue()
        self._queue.on_finished(self._on_task_finished)
        restored = self._store.list_active()
        if restored:
            logger.info("[aiimg] 发现 %d 个没跑完的任务，接着处理", len(restored))
        await self._queue.start()
        self._ready = True
        for error in self._registry.validate():
            logger.warning("[aiimg] 服务商配置有误：%s", error)
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
        logger.info("[aiimg] 已载入 %d 个生图服务商 / %d 个视频服务商",
                    len(self._registry.image_provider_ids()),
                    len(self._registry.video_provider_ids()))
        logger.info(
            "[aiimg] 初始化完成，用时 %.0f 毫秒", (time.perf_counter() - started) * 1000
        )
        # Old task rows never affect the first request, so sweeping them runs
        # once the plugin has already reported ready.
        self._housekeeping = asyncio.create_task(self._deferred_housekeeping())

    async def _deferred_housekeeping(self) -> None:
        await asyncio.sleep(0)
        started = time.perf_counter()
        await asyncio.to_thread(self._prune_old_tasks)
        logger.info(
            "[aiimg] 后台清理用时 %.0f 毫秒", (time.perf_counter() - started) * 1000
        )

    async def terminate(self):
        self._ready = False
        housekeeping = getattr(self, "_housekeeping", None)
        if housekeeping is not None and not housekeeping.done():
            housekeeping.cancel()
        if self._queue is not None:
            await self._queue.stop()
        if self._gallery is not None:
            await self._gallery.close()
        if self._memory is not None:
            self._memory.close()
        if self._registry is not None:
            await self._registry.close()
        if self._store is not None:
            self._store.close()
        await close_http()

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

    def _fetch_policy(self) -> URLFetchPolicy:
        """Build the fetch policy from provider + advanced section config."""
        advanced = self._cfg("section_advanced") or {}
        if not isinstance(advanced, dict):
            advanced = {}
        trusted_result = set()
        for line in str(advanced.get("trusted_result_origins") or "").splitlines():
            value = origin(str(line).strip())
            if value:
                trusted_result.add(value)
        return URLFetchPolicy(
            trusted_origins=trusted_origins_from_config(self._provider_config()),
            trusted_result_origins=frozenset(trusted_result),
            allow_private_result_origins=bool(advanced.get("allow_private_result_origins")),
        )

    def _advanced_max_bytes(self) -> int | None:
        """Byte cap from the advanced section; None keeps the built-in default."""
        advanced = self._cfg("section_advanced") or {}
        if not isinstance(advanced, dict):
            advanced = {}
        value = advanced.get("image_max_mb")
        try:
            return max(1, int(value)) * 1024 * 1024 if value else None
        except (TypeError, ValueError):
            return None

    async def _download_video(self, url: str, provider_id: str = "") -> Path:
        advanced = self._cfg("section_advanced") or {}
        if not isinstance(advanced, dict):
            advanced = {}
        max_bytes = int(advanced.get("video_max_mb") or 256) * 1024 * 1024
        timeout = int(advanced.get("download_timeout_seconds") or 120)
        parallel = advanced.get("video_parallel_download")
        return await download_video(
            url,
            self._data_dir / "gallery",
            self._fetch_policy(),
            max_bytes=max_bytes,
            timeout=timeout,
            headers=self._video_headers(url, provider_id),
            parallel=True if parallel is None else bool(parallel),
        )

    def _video_headers(self, url: str, provider_id: str) -> dict[str, str]:
        """Credentials for the provider origin, withheld from CDN redirects."""
        if not provider_id:
            return {}
        try:
            settings = self._registry.get_settings(provider_id)
        except Exception:
            return {}
        if settings is None:
            return {}
        host = origin(str(url or ""))
        provider_origins = {
            origin(str(value))
            for value in (
                settings.base_url,
                settings.server_url,
                settings.full_generate_url,
                settings.full_edit_url,
            )
            if value
        }
        if not host or host not in provider_origins:
            return {}
        key = settings.api_keys[0] if settings.api_keys else ""
        return {"Authorization": f"Bearer {key}"} if key else {}

    def _consecutive_failures(self, provider_id: str) -> int:
        for entry in self._stats.snapshot():
            if entry["id"] == provider_id:
                return int(entry["consecutive_failures"])
        return 0

    def _denied(self, action: str, reason: str) -> str:
        """Refuse a request with facts only.

        The model sees why nothing started and can say it in persona; the plugin
        never ships a canned sentence of its own.
        """
        logger.info("[aiimg] 拒绝 %s：%s", action, reason)
        return json.dumps(
            {"status": "denied", "action": action, "reason": reason},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _queue_refusal(self, action: str, exc: Exception) -> str:
        """Queue rejections become facts too; the counters stay in the log."""
        reason = "队列已满" if isinstance(exc, QueueFullError) else "同时进行的太多"
        return self._denied(action, reason)

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
            logger.warning("[aiimg] 读取人设参考图失败：%s", exc)
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
        section = self._cfg("section_queue") or {}
        advanced = self._cfg("section_advanced") or {}
        self._queue = TaskQueue(
            self._store,
            max_running=int(section.get("max_running", 2) or 2),
            max_queued=int(section.get("max_queued", 16) or 16),
            max_user_image_tasks=int(section.get("max_user_image_tasks", 1) or 1),
            max_user_video_tasks=int(section.get("max_user_video_tasks", 1) or 1),
            retention_hours=int(section.get("task_retention_hours", 24) or 24),
            owner=self._owner,
            lease_seconds=float(advanced.get("task_lease_seconds", 120) or 120),
        )
        self._queue.on_resume(self._resume_task)

    async def _resume_task(self, record: TaskRecord) -> dict:
        """Re-run a task that was interrupted when the process stopped."""
        limit = int(self._cfg("section_advanced", "task_max_attempts", default=2) or 2)
        if record.attempts > limit:
            raise RuntimeError("任务在进程中断前没跑完，重试次数已经用尽")
        runner, _ = self._task_runner(record.kind)
        logger.info(
            "[aiimg] 接手中断的任务 %s（第 %d 次尝试）", record.task_id, record.attempts
        )
        return await runner(record)

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
        self._registry = ProviderRegistry(
            self._provider_config(),
            data_dir=self._data_dir,
            policy=self._fetch_policy(),
            image_max_bytes=self._advanced_max_bytes(),
        )
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
            self._gallery.update_config({
                "gallery": self._cfg("section_gallery") or {},
            })
        self._prune_old_tasks()

    def _prune_old_tasks(self) -> None:
        try:
            pruned = self._store.prune_old(
                int(self._cfg("section_queue", "task_retention_hours", default=24) or 24)
            )
            if pruned:
                logger.info("[aiimg] 清理了 %d 条过期任务记录", pruned)
        except Exception as exc:
            logger.warning("[aiimg] 清理任务记录失败：%s", exc)

    # ------------------------------------------------------------------
    # access control
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_entry(entry: str, keys: list[str]) -> bool:
        """Match one whitelist line against the adapter-qualified ids of a sender.

        Accepted shapes: a bare id ("123456"), an adapter-qualified id
        ("qq:123456"), or a whole session id ("qq:dm:123456"). In the session
        form the middle segment is a session type, so only the adapter and the
        trailing id take part in matching.
        """
        parts = [part.strip() for part in str(entry or "").split(":") if part.strip()]
        if not parts:
            return False
        entry_id = parts[-1]
        entry_adapter = parts[0] if len(parts) > 1 else ""
        for key in keys:
            if not key:
                continue
            key_adapter, _, key_id = str(key).partition(":")
            if not key_id:
                key_adapter, key_id = "", str(key)
            if entry_adapter and entry_adapter != key_adapter:
                continue
            if entry_id == key_id:
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

    def _video_allowed(self, event: Any) -> bool:
        """Second gate for video; an empty list means everyone on the main list."""
        entries = self._lines(self._cfg("section_features", "video_whitelist", default=[]))
        if not entries:
            return True
        keys = [key for key in (self._user_id(event), self._group_id(event)) if key]
        return any(self._matches_entry(entry, keys) for entry in entries)

    def _scoped(self) -> bool:
        """Whether lookups stay inside the caller's own session."""
        value = self._cfg("section_access", "session_scope", default=True)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() not in ("0", "false", "no", "off")

    def _tool_available(self, name: str, event: Any) -> bool:
        """Whether this turn should even see the tool.

        An unavailable tool is removed from the turn, so the model never offers
        a capability that would only end in a refusal.
        """
        if name == "generate_video":
            return (
                self._feature_on("video_enabled", default=False)
                and self._video_allowed(event)
                and bool(self._video_chain())
            )
        if name == "generate_selfie":
            return self._feature_on("selfie_enabled") and bool(self._selfie_chain())
        if name == "generate_image_batch":
            return self._feature_on("batch_enabled") and bool(
                self._image_chain("generate_image")
            )
        if name == "edit_image":
            return self._feature_on("edit_enabled") and bool(self._image_chain("edit_image"))
        if name == "generate_image":
            return self._feature_on("draw_enabled") and bool(
                self._image_chain("generate_image")
            )
        return True

    @on.llm_request()
    async def _hide_unavailable_tools(self, event, req: LLMRequest, *_):
        """Trim the turn's tool list down to what this sender can actually use."""
        if not self._ready:
            return
        allowed = self._can_use(
            self._user_id(event), self._adapter_name(event), self._group_id(event)
        )
        removed: list[str] = []
        for tool_name in TOOL_NAMES:
            if tool_name in READ_ONLY_TOOLS:
                continue
            if allowed and self._tool_available(tool_name, event):
                continue
            try:
                req.tool_set.remove(tool_name)
                removed.append(tool_name)
            except Exception:
                continue
        if removed:
            logger.info(
                "[aiimg] 本轮不提供工具：%s（白名单=%s）",
                "、".join(removed),
                "通过" if allowed else "未通过",
            )

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
        plan_output = record.extra.get("plan_output")
        if plan_output:
            pieces.append(self._parse_intent_text(str(plan_output)))
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

    def _duplicate_task(
        self, event: Any, kind: str, prompt: str, extra: dict, *, input_path: str = ""
    ) -> str:
        """Return the task id of an identical request sent moments ago, or ""."""
        if not self._debounce.enabled:
            return ""
        key = self._debounce.make_key(
            self._user_id(event),
            self._group_id(event),
            kind,
            str(prompt or "").strip(),
            input_path,
            json.dumps(extra.get("explicit") or {}, sort_keys=True),
            extra.get("default_output") or "",
            (extra.get("chain_items") or [{}])[0].get("output")
            if extra.get("chain_items")
            else "",
        )
        existing = self._debounce.duplicate_of(key)
        if existing:
            logger.info("[aiimg] 重复请求合并：%s 复用任务 %s", kind, existing)
        return existing

    def _remember_task(self, event: Any, kind: str, prompt: str, extra: dict, task_id: str,
                       *, input_path: str = "") -> None:
        if not self._debounce.enabled:
            return
        key = self._debounce.make_key(
            self._user_id(event),
            self._group_id(event),
            kind,
            str(prompt or "").strip(),
            input_path,
            json.dumps(extra.get("explicit") or {}, sort_keys=True),
            extra.get("default_output") or "",
            (extra.get("chain_items") or [{}])[0].get("output")
            if extra.get("chain_items")
            else "",
        )
        self._debounce.remember(key, task_id)

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
            return self._denied(kind, "白名单")
        if not self._feature_on("draw_enabled"):
            return self._denied(kind, "功能关闭")
        prompt = str(prompt or "").strip()
        if not prompt:
            return "提示词不能为空"
        chain = self._image_chain(kind)
        if not chain:
            return self._denied(kind, "没有服务商")
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
        duplicate = self._duplicate_task(event, kind, prompt, extra)
        if duplicate:
            return self._duplicate_ack(duplicate, "已加入生图队列", event)
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
            return self._queue_refusal(kind, exc)
        self._remember_task(event, kind, prompt, extra, record.task_id)
        return self._ack_text(record, "已加入生图队列", event)

    def _duplicate_ack(self, task_id: str, message: str, event: Any) -> str:
        """Acknowledge a collapsed duplicate without queueing a second task."""
        record = self._store.get(task_id) if self._store else None
        logger.info("[aiimg] 重复请求复用任务 %s：%s", task_id, message)
        if record is not None:
            payload = json.loads(self._ack_text(record, message, event))
            payload["state"] = "reused"
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return json.dumps(
            {
                "status": "accepted",
                "task_id": task_id,
                "state": "reused",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

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
        """Facts-only acknowledgement; ``message`` is kept for the log only."""
        logger.info("[aiimg] 接单回执 %s：%s", record.task_id, message)
        payload: dict[str, Any] = {
            "status": "accepted",
            "task_id": record.task_id,
            "kind": record.kind,
            "state": record.status,
        }
        if record.size:
            payload["size"] = record.size
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
        description="根据文字描述生成图片。任务进入队列，完成后会自动把图片发送给发起者。可指定服务商、尺寸（如 1024x1024）、比例（如 16:9）或分辨率（1K/2K/4K）。用户只是随口描述画面、没说要图时不要调用。",
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
        description="基于现有图片进行修改（图生图）。image 留空时用当前消息或所回复消息里的图片，可以一次带多张（最多 4 张，其余作为参考图）；传 'latest' 用本会话最近生成的图片；也可以传 find_image 给出的图片编号或图片文件路径。可用 reference_roles 按角色挑参考图（subject 主体、clothing 服装、style 画风、pose 姿势、background 背景），选中的上传参考图会和原图一起发过去。",
        params={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "修改要求"},
                "image": {
                    "type": "string",
                    "description": "留空用消息里的图片（多张时第一张是原图）；'latest' 用最近生成的图；也可以给 find_image 返回的编号或本地图片路径",
                },
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
            "required": ["prompt"],
        },
    )
    async def edit_image(
        self,
        event: KiraMessageBatchEvent,
        prompt: str,
        image: str = "",
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
            return self._denied("edit_image", "白名单")
        if not self._feature_on("edit_enabled"):
            return self._denied("edit_image", "功能关闭")
        try:
            wanted_roles = self._wanted_reference_roles(reference_roles)
        except ValueError as exc:
            return str(exc)
        attached = await self._attached_paths(event)
        if str(image or "").strip():
            resolved = await self._resolve_input_image(event, image)
            if not resolved:
                return self._denied("edit_image", "没有可用的图片")
            input_path = resolved
            message_extras: list[Path] = [path for path in attached if path != resolved]
        else:
            if not attached:
                latest = await self._gallery_latest(event)
                if latest:
                    attached = [latest]
            if not attached:
                return "没有可用的输入图片：直接发一张图（或引用一张图）再来，或传 'latest' 用最近生成的图"
            input_path = attached[0]
            message_extras = attached[1:]
        chain = self._edit_chain()
        if not chain:
            return "未配置支持改图的服务商"
        candidates: list[tuple[Path, str]] = [(Path(input_path), "base")]
        candidates.extend(
            (path, role) for path, role in self._ref_paths(wanted_roles)
        )
        candidates.extend((path, "subject") for path in message_extras)
        try:
            extra_refs, notes = await self._plan_references(event, candidates)
        except ValueError as exc:
            return str(exc)
        for note in notes:
            logger.info("[aiimg] 改图参考图：%s", note)
        if not extra_refs or extra_refs[0][1] != "base":
            return "输入图片来源不明：请把要改的那张图重新发一次"
        input_path, extra_refs = extra_refs[0][0], extra_refs[1:]
        logger.info(
            "[aiimg] 改图请求：输入图 %s，角色 %s，附带参考图 %d 张",
            Path(input_path).name,
            ",".join(wanted_roles) if wanted_roles is not None else "未指定",
            len(extra_refs),
        )
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
            return self._queue_refusal("edit_image", exc)
        return self._ack_text(record, "已加入改图队列", event)

    @register.tool(
        name="generate_selfie",
        description="使用当前人设的参考照片（或自己上传的参考图）生成自拍/人像图。参考图在哪配由插件设置决定，调用方不需要提供。可用 reference_roles 按角色挑参考图（subject 主体、clothing 服装、style 画风、pose 姿势、background 背景），选中的上传参考图会和人物参考一起发过去。",
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
            return self._denied("generate_selfie", "白名单")
        if not self._feature_on("selfie_enabled"):
            return self._denied("generate_selfie", "功能关闭")
        try:
            wanted_roles = self._wanted_reference_roles(reference_roles)
        except ValueError as exc:
            return str(exc)
        plan = await self._selfie_reference_plan(wanted_roles)
        if not plan:
            use_persona = self._feature_on("selfie_use_persona", True)
            if not use_persona:
                return self._denied("generate_selfie", "还没有上传参考图")
            return self._denied("generate_selfie", "人设还没有参考人像")
        persona = await self._persona_reference()
        try:
            kept, notes = refbridge.plan_references(
                plan,
                gallery_dir=getattr(self._gallery, "gallery_dir", None),
                refs_dir=self._refs_dir() if self._data_dir is not None else None,
                persona=persona,
            )
        except ValueError as exc:
            return str(exc)
        for note in notes:
            logger.info("[aiimg] 自拍参考图：%s", note)
        plan = [(item.path, item.role) for item in kept]
        if not plan:
            return self._denied("generate_selfie", "参考图来源不明")
        chain = self._selfie_chain()
        if not chain:
            return self._denied("generate_selfie", "没有服务商")
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
            return self._queue_refusal("generate_selfie", exc)
        return self._ack_text(record, "已加入自拍队列", event)

    @register.tool(
        name="generate_image_batch",
        description=(
            "一次生成多张图片。可以直接给 prompts（每张一条提示词），"
            "也可以只给 brief 一句话，由插件调用模型拆成多条风格不同、画幅不同的提示词。"
            "任务进入队列，完成后把全部图片一起发送。"
        ),
        params={
            "type": "object",
            "properties": {
                "prompts": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {
                                    "prompt": {"type": "string"},
                                    "output": {"type": "string"},
                                },
                                "required": ["prompt"],
                            },
                        ]
                    },
                    "description": "多条提示词；每项可以是字符串，也可以是 {prompt, output} 对象（output 写画幅，如 3:4）。给了它就不用 brief",
                },
                "brief": {
                    "type": "string",
                    "description": "一句话需求，插件会拆成多条提示词；prompts 为空时使用",
                },
                "count": {
                    "type": "integer",
                    "description": "打算生成几张，1 到批量上限；0 用设置里的默认值",
                },
                "provider": {"type": "string", "description": "服务商 id"},
                "size": {"type": "string", "description": "精确尺寸"},
                "aspect_ratio": {"type": "string", "description": "比例"},
                "resolution": {"type": "string", "description": "分辨率档位"},
            },
            "required": [],
        },
    )
    async def generate_image_batch(
        self,
        event: KiraMessageBatchEvent,
        prompts: list[str] | None = None,
        brief: str = "",
        count: int = 0,
        provider: str = "",
        size: str = "",
        aspect_ratio: str = "",
        resolution: str = "",
    ) -> str:
        if not self._ready or not self._gallery:
            return "插件未就绪"
        if not self._can_use(self._user_id(event), self._adapter_name(event), self._group_id(event)):
            return self._denied("generate_image_batch", "白名单")
        if not self._feature_on("batch_enabled"):
            return self._denied("generate_image_batch", "功能关闭")
        prompts = self._batch_prompts(prompts)
        if not prompts:
            idea = str(brief or "").strip()
            if not idea:
                return self._denied("generate_image_batch", "没有提示词")
            planned, problem = await self._plan_batch(idea, count)
            if problem:
                return problem
            prompts = planned
        limit = int(self._cfg("section_queue", "batch_max_count", default=8) or 8)
        if len(prompts) > limit:
            prompts = prompts[:limit]
        chain = self._image_chain("generate_image")
        if not chain:
            return self._denied("generate_image_batch", "没有服务商")
        extra = self._task_extra(
            event,
            explicit=self._explicit(size, aspect_ratio, resolution),
            default_output=self._default_output_for("generate_image"),
            prompts=prompts,
            chain_items=[self._chain_item_dict(item) for item in chain],
        )
        first_prompt = prompts[0]["prompt"] if isinstance(prompts[0], dict) else prompts[0]
        record = self._store.create(
            kind="generate_image_batch",
            sid=event.sid,
            user_id=self._user_id(event),
            prompt=first_prompt,
            provider=chain[0].provider_id,
            chain=[item.provider_id for item in chain],
            size=format_output_intent(self._chain_intent_base(record_hint=extra)),
            extra=extra,
        )
        try:
            await self._submit(record, self._run_batch, is_batch=True)
        except (QueueFullError, UserLimitError) as exc:
            return self._queue_refusal("generate_image_batch", exc)
        return self._ack_text(record, f"已加入批量队列（{len(prompts)} 张）", event)

    @staticmethod
    def _batch_prompts(items: Any) -> list[Any]:
        """Normalize batch entries, keeping per-image output specs intact."""
        cleaned: list[Any] = []
        for item in items or []:
            if isinstance(item, dict):
                text = str(item.get("prompt") or "").strip()
                if not text:
                    continue
                output = str(item.get("output") or item.get("aspect_ratio") or "").strip()
                cleaned.append({"prompt": text, "output": output} if output else {"prompt": text})
                continue
            text = str(item or "").strip()
            if text:
                cleaned.append(text)
        return cleaned

    async def _plan_batch(self, brief: str, count: int) -> tuple[list[dict[str, str]], str]:
        """Ask the fast model to expand one line into a varied batch plan."""
        limit = int(self._cfg("section_queue", "batch_max_count", default=8) or 8)
        wanted = int(count or 0) or int(self._cfg("section_queue", "planner_count", default=4) or 4)
        wanted = max(1, min(wanted, limit))
        template = str(self._cfg("section_queue", "planner_prompt", default="") or "").strip()
        template = template or planner.DEFAULT_PLANNER_TEMPLATE
        ratios = planner.ratio_choices(self._cfg("section_queue", "planner_ratios", default=""))
        client = None
        getter = getattr(self.ctx, "get_default_fast_llm_client", None)
        if callable(getter):
            try:
                client = getter()
            except Exception as exc:
                logger.warning("[aiimg] 取不到规划用的模型: %s", exc)
                client = None
        if client is None:
            return [], "没有可用的模型来拆分提示词，请直接给 prompts"
        request_text = planner.build_request(template, brief=brief, count=wanted, ratios=ratios)
        try:
            response = await client.chat(
                LLMRequest(messages=[{"role": "user", "content": request_text}])
            )
        except Exception as exc:
            logger.warning("[aiimg] 批量规划失败: %s", exc)
            return [], "拆分提示词失败，请稍后再试，或者直接给 prompts"
        text = str(getattr(response, "text_response", "") or "")
        try:
            items = planner.plan(text, count=wanted, ratios=ratios)
        except ValueError as exc:
            logger.warning("[aiimg] 批量规划没有可用结果: %s", exc)
            return [], str(exc)
        logger.info("[aiimg] 批量规划 %s", planner.describe(items))
        return [{"prompt": item.prompt, "output": item.aspect_ratio} for item in items], ""

    @register.tool(
        name="generate_video",
        description="生成 AI 视频。image 留空时用当前消息或所回复消息里的图片作关键帧，传 'latest' 用最近生成的图片，也可以传 find_image 给出的编号或图片路径。任务生成较慢，完成后自动发送。",
        params={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "视频内容描述"},
                "image": {
                    "type": "string",
                    "description": "留空用消息里的图片作关键帧；'latest' 用最近生成的图；也可以给 find_image 返回的编号或本地图片路径",
                },
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
            return self._denied("generate_video", "白名单")
        if not self._video_allowed(event):
            return self._denied("generate_video", "白名单")
        if not self._feature_on("video_enabled", default=False):
            return self._denied("generate_video", "功能关闭")
        chain = self._video_chain()
        if not chain:
            return self._denied("generate_video", "没有服务商")
        seconds_text = str(seconds or "").strip()
        if seconds_text:
            if not seconds_text.isdigit():
                return self._denied("generate_video", "时长不是数字")
            seconds_int = int(seconds_text)
            if not (_VIDEO_SECONDS[0] <= seconds_int <= _VIDEO_SECONDS[1]):
                return self._denied("generate_video", "时长超出范围")
            seconds_text = str(seconds_int)
        ratio = str(aspect_ratio or "").strip()
        if ratio and ratio not in _VIDEO_RATIOS:
            return self._denied("generate_video", "不支持的比例")
        mode_text = str(mode or "").strip().lower()
        if mode_text and mode_text not in ("auto", "text", "keyframe", "reference"):
            return "mode 只能是 auto/text/keyframe/reference"
        input_path = ""
        if image:
            input_path = str(await self._resolve_input_image(event, image) or "")
        elif mode_text != "text":
            # A photo sent with the request is the obvious keyframe.
            attached = await self._first_attached_image(event)
            input_path = str(attached or "")
        if input_path:
            # A keyframe is an input picture like any other: refuse a path that
            # did not come from this message, the gallery or the reference pool.
            try:
                kept, notes = await self._plan_references(
                    event, [(Path(input_path), "base")]
                )
            except ValueError as exc:
                return str(exc)
            for note in notes:
                logger.info("[aiimg] 视频关键帧：%s", note)
            input_path = str(kept[0][0]) if kept else ""
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
            return self._queue_refusal("generate_video", exc)
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
        manager = self._can_use(user_id, self._adapter_name(event), self._group_id(event))
        if task_id:
            record = self._store.get(task_id.strip())
            if record is None:
                return json.dumps(
                    {"status": "missing", "task_id": task_id.strip()},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            if not self._visible(event, record):
                return json.dumps(
                    {"status": "denied", "action": "get_task_status", "reason": "不在这个会话"},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            return self._describe_task(record)
        recent = [
            record
            for record in self._store.list_recent(limit=30)
            if self._visible(event, record)
        ]
        payload: dict[str, Any] = {"status": "ok", "recent": []}
        # Queue depth is an operator number; only the people who can spend on
        # generation get to see it.
        if manager and self._queue:
            snapshot = self._queue.snapshot() or {}
            payload["queue"] = {
                "running": snapshot.get("running", 0),
                "waiting": snapshot.get("waiting", 0),
            }
        for record in recent[:8]:
            item: dict[str, Any] = {
                "task_id": record.task_id,
                "kind": record.kind,
                "status": record.status,
            }
            if record.size:
                item["size"] = record.size
            if record.error:
                item["error_kind"] = self._error_kind(record.error)
            payload["recent"].append(item)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @register.tool(
        name="resend_image",
        description="重新发送已经生成过的图片或视频，不重新生成。用户说“再发一次”“刚才那张再给我看看”或图片没收到时用它。不传参数时取这个会话最近一次成功的结果，也可以给 find_image 返回的编号。",
        params={
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "任务编号，留空用最近一次成功的任务"},
                "image_ref": {"type": "string", "description": "图片编号（find_image 返回的 img- 开头编号），留空则按任务或最近一次"},
            },
            "required": [],
        },
    )
    async def resend_image(
        self, event: KiraMessageBatchEvent, task_id: str = "", image_ref: str = ""
    ) -> str:
        if not self._ready or not self._store:
            return "插件未就绪"
        if image_ref.strip():
            return await self._resend_by_ref(event, image_ref.strip())
        record = self._latest_media_task(event, task_id)
        if record is None:
            return self._denied("resend_image", "还没有可重发的图")
        elements = self._media_elements(record)
        if not elements:
            return self._denied("resend_image", "那条结果不在了")
        await self._safe_send(event.sid, MessageChain(elements))
        self._mark_ack_delivered(record.task_id)
        logger.info(
            "[aiimg] 重发任务 %s 的 %d 个元素到 %s",
            record.task_id,
            len(elements),
            event.sid,
        )
        return json.dumps(
            {
                "status": "resent",
                "task_id": record.task_id,
                "kind": record.kind,
                "count": len(elements),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    async def _resend_by_ref(self, event: Any, ref: str) -> str:
        """Send one remembered file back by its short ref."""
        path = ""
        if self._memory is not None:
            try:
                path = self._memory.resolve(
                    ref, sid=getattr(event, "sid", ""), scoped=self._scoped()
                ) or ""
            except Exception as exc:
                logger.warning("[aiimg] 按编号查图失败 %s：%s", ref, exc)
        if not path:
            return json.dumps(
                {"status": "missing", "image_ref": ref},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        target = Path(path)
        is_video = target.suffix.lower() in (".mp4", ".webm", ".mov")
        element = Video(file=str(target)) if is_video else Image(image=str(target))
        sent = await self._safe_send(getattr(event, "sid", ""), MessageChain([element]))
        logger.info(
            "[aiimg] 按编号 %s 重发%s：%s",
            ref,
            "视频" if is_video else "图片",
            "成功" if sent else "失败",
        )
        return json.dumps(
            {
                "status": "resent" if sent else "failed",
                "image_ref": ref,
                "kind": "video" if is_video else "image",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @register.tool(
        name="find_image",
        description="翻这个会话画过的东西：按描述或类型列出编号和内容，只回清单不发图。想重发其中某张时，把编号交给 resend_image。",
        params={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "按提示词里出现过的词找，留空表示不筛"},
                "kind": {
                    "type": "string",
                    "description": "只看某类：image 图片、selfie 自拍、video 视频，留空表示全部",
                    "enum": ["", "image", "selfie", "video"],
                },
                "limit": {"type": "integer", "description": "最多几条，默认 8，最多 20"},
            },
            "required": [],
        },
    )
    async def find_image(
        self, event: KiraMessageBatchEvent, query: str = "", kind: str = "", limit: int = 8
    ) -> str:
        if self._memory is None:
            return "插件未就绪"
        try:
            size = max(1, min(int(limit or 8), 20))
        except (TypeError, ValueError):
            size = 8
        wanted = str(kind or "").strip()
        if wanted not in ("image", "selfie", "video"):
            wanted = ""
        try:
            rows = self._memory.find(
                query=str(query or ""),
                kind=wanted,
                sid=getattr(event, "sid", ""),
                scoped=self._scoped(),
                limit=size,
            )
        except Exception as exc:
            logger.warning("[aiimg] 翻图片记忆失败：%s", exc)
            return self._denied("find_image", "内部错误")
        images = [
            {
                "image_ref": row.get("ref", ""),
                "kind": row.get("kind", ""),
                "prompt": clip(row.get("prompt", ""), 80),
            }
            for row in rows
        ]
        logger.info(
            "[aiimg] 翻图片记忆：条件=%s/%s，命中 %d 条（会话 %s）",
            clip(query, 40) or "-",
            wanted or "全部",
            len(images),
            getattr(event, "sid", "") or "-",
        )
        return json.dumps(
            {"status": "ok", "count": len(images), "images": images},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @register.tool(
        name="save_inspiration",
        description="把一条以后还能复用的画面描述存进灵感库。用户说“记下来”“存个点子”“这个想法留着”时用它。",
        params={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要存的画面描述"},
                "tags": {"type": "string", "description": "顺手打几个标签，用空格或逗号隔开，可留空"},
            },
            "required": ["text"],
        },
    )
    async def save_inspiration(
        self, event: KiraMessageBatchEvent, text: str, tags: str = ""
    ) -> str:
        if self._inspiration is None:
            return "插件未就绪"
        body = str(text or "").strip()
        if not body:
            return self._denied("save_inspiration", "内容为空")
        if not self._can_use(
            self._user_id(event), self._adapter_name(event), self._group_id(event)
        ):
            return self._denied("save_inspiration", "白名单")
        raw_tags = str(tags or "")
        for mark in ("，", ",", "、", ";"):
            raw_tags = raw_tags.replace(mark, " ")
        labels = raw_tags.split()
        try:
            entry = self._inspiration.add(
                body,
                tags=labels,
                sid=getattr(event, "sid", ""),
                user_id=self._user_id(event),
            )
        except ValueError as exc:
            return self._denied("save_inspiration", str(exc))
        except OSError as exc:
            logger.error("[aiimg] 灵感库写入失败：%s", exc)
            return self._denied("save_inspiration", "写入失败")
        logger.info(
            "[aiimg] 灵感库新增 %s，标签 %s，当前 %d 条",
            entry.get("id", "-"),
            "、".join(labels) or "-",
            len(self._inspiration),
        )
        return json.dumps(
            {"status": "saved", "id": entry.get("id", ""), "tags": labels},
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @register.tool(
        name="pick_inspiration",
        description="从灵感库里随机翻几条以前存下的画面描述，用来给接下来要画的东西找方向。",
        params={
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "要几条，默认 3，最多 8"},
            },
            "required": [],
        },
    )
    async def pick_inspiration(self, event: KiraMessageBatchEvent, count: int = 3) -> str:
        if self._inspiration is None:
            return "插件未就绪"
        try:
            size = max(1, min(int(count or 3), 8))
        except (TypeError, ValueError):
            size = 3
        try:
            rows = self._inspiration.random(
                count=size, sid=getattr(event, "sid", ""), scoped=self._scoped()
            )
        except Exception as exc:
            logger.warning("[aiimg] 翻灵感库失败：%s", exc)
            return self._denied("pick_inspiration", "内部错误")
        logger.info("[aiimg] 灵感库随机取 %d 条", len(rows))
        return json.dumps(
            {
                "status": "ok",
                "count": len(rows),
                "ideas": [
                    {"id": row.get("id", ""), "text": row.get("text", ""), "tags": row.get("tags", [])}
                    for row in rows
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _remember_media(
        self, paths: list[str], *, record: TaskRecord, kind: str, prompt: str = ""
    ) -> list[str]:
        """Give every saved file a short ref the model can quote later."""
        if self._memory is None or not paths:
            return []
        try:
            refs = self._memory.remember(
                list(paths),
                sid=record.sid,
                kind=kind,
                task_id=record.task_id,
                prompt=prompt,
            )
        except Exception as exc:
            logger.warning("[aiimg] 任务 %s 登记图片记忆失败：%s", record.task_id, exc)
            return []
        logger.info(
            "[aiimg] 任务 %s 已登记 %d 张图片记忆：%s",
            record.task_id,
            len(refs),
            "、".join(refs),
        )
        return refs

    def _visible(self, event: Any, record: TaskRecord) -> bool:
        """Whether this session may look at that record.

        With isolation on, a group chat only sees what that group made and a
        private chat only sees its own work.
        """
        if not self._scoped():
            return True
        return record.sid == getattr(event, "sid", "")

    def _latest_media_task(self, event: Any, task_id: str = "") -> TaskRecord | None:
        """Find a succeeded task that still has deliverable media."""
        if task_id:
            record = self._store.get(task_id.strip()) if self._store else None
            if record is None or not self._visible(event, record):
                return None
            return record if self._media_elements(record) else None
        for record in self._store.list_recent(limit=40):
            if record.status != "succeeded":
                continue
            if not self._visible(event, record):
                continue
            if self._media_elements(record):
                return record
        return None

    def _media_elements(self, record: TaskRecord) -> list[Any]:
        """Rebuild the media chain of a finished task from its stored result."""
        try:
            payload = json.loads(record.result or "{}")
        except Exception:
            payload = {}
        elements: list[Any] = []
        video_url = str(payload.get("url") or "").strip()
        if video_url:
            elements.append(Video(file=video_url))
        for path in payload.get("media") or []:
            text = str(path or "").strip()
            if not text:
                continue
            if text.lower().endswith((".mp4", ".webm", ".mov")):
                elements.append(Video(file=text))
            else:
                elements.append(Image(image=text))
        return elements

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

    @staticmethod
    def _message_image_sources(event: Any) -> list[tuple[Any, str]]:
        """(element, source) pairs: quoted pictures first, then the new ones.

        The framework hands plugins a batch event, so the newest message is the
        last entry. Quoted messages carry their own chain, which is where an
        "use the photo I replied to" request lives.
        """
        messages = list(getattr(event, "messages", None) or [])
        single = getattr(event, "message", None)
        if single is not None:
            messages.append(single)
        for message in reversed(messages):
            chain = list(getattr(message, "chain", None) or [])
            quoted: list[tuple[Any, str]] = []
            for element in chain:
                inner = getattr(element, "chain", None)
                if inner is None or not getattr(element, "message_id", None):
                    continue
                quoted.extend(
                    (item, refbridge.SOURCE_QUOTED)
                    for item in inner
                    if isinstance(item, Image)
                )
            own = [
                (element, refbridge.SOURCE_MESSAGE)
                for element in chain
                if isinstance(element, Image)
            ]
            found = quoted + own
            if found:
                return found
        return []

    @staticmethod
    def _message_images(event: Any) -> list[Any]:
        """Images attached to this turn: quoted pictures first, own ones second."""
        return [element for element, _source in AIImagePlugin._message_image_sources(event)]

    async def _element_path(self, element: Any) -> Path | None:
        """Materialise an attached image through the framework's own helper."""
        to_path = getattr(element, "to_path", None)
        if to_path is None:
            return None
        try:
            path = await to_path()
        except Exception as exc:
            logger.warning("[aiimg] 读取消息里的图片失败：%s", exc)
            return None
        if not path:
            return None
        resolved = Path(str(path))
        return resolved if resolved.exists() else None

    async def _first_attached_image(self, event: Any) -> Path | None:
        images = self._message_images(event)
        if not images:
            return None
        return await self._element_path(images[0])

    async def _attached_paths(self, event: Any, limit: int = _ATTACHMENT_LIMIT) -> list[Path]:
        """Materialise every attached picture, quoted message first."""
        return [
            path
            for path, _source in await self._attached_sources(event, limit=limit)
        ]

    async def _attached_sources(
        self, event: Any, limit: int = _ATTACHMENT_LIMIT
    ) -> list[tuple[Path, str]]:
        """Attached pictures with where each one came from."""
        found: list[tuple[Path, str]] = []
        seen: list[Path] = []
        for element, source in self._message_image_sources(event)[:limit]:
            path = await self._element_path(element)
            if path is None or path in seen:
                continue
            seen.append(path)
            found.append((path, source))
        return found

    async def _gallery_latest(self, event: Any) -> Path | None:
        """Newest image this session may use; isolation keeps the lookup local."""
        if not self._gallery:
            return None
        sid = event.sid if self._scoped() else ""
        for path in self._gallery.latest_image_paths(sid, limit=5):
            candidate = Path(path)
            if candidate.exists():
                return candidate
        for path in self._gallery.recent_images(sid):
            candidate = Path(path)
            if candidate.exists():
                return candidate
        return None

    async def _plan_references(
        self, event: Any, pairs: list[tuple[Path, str]]
    ) -> tuple[list[tuple[Path, str]], list[str]]:
        """Validate provenance, drop duplicates and enforce the byte budget."""
        attachments = await self._attached_paths(event)
        quoted = [
            path
            for path, source in await self._attached_sources(event)
            if source == refbridge.SOURCE_QUOTED
        ]
        gallery_dir = getattr(self._gallery, "gallery_dir", None)
        refs_dir = self._refs_dir() if self._data_dir is not None else None
        planned, notes = refbridge.plan_references(
            pairs,
            attachments=attachments,
            quoted=quoted,
            gallery_dir=gallery_dir,
            refs_dir=refs_dir,
            persona=await self._persona_reference(),
        )
        return [(item.path, item.role) for item in planned], notes

    async def _resolve_input_image(self, event: KiraMessageBatchEvent, image: str) -> Path | None:
        text = str(image or "").strip()
        lowered = text.lower()
        if lowered in ("latest", "last", "最近"):
            return await self._gallery_latest(event)
        if not text or lowered in ("auto", "message", "current", "附件", "当前消息"):
            attached = await self._first_attached_image(event)
            if attached is not None:
                return attached
            return await self._gallery_latest(event)
        if text.startswith("http://") or text.startswith("https://"):
            # Never fetch a URL ourselves: the framework already materialises
            # attachments, so a URL argument just means "the picture I sent".
            return await self._first_attached_image(event)
        if lowered.startswith("img-"):
            if self._memory is None:
                return None
            try:
                found = self._memory.resolve(
                    text, sid=getattr(event, "sid", ""), scoped=self._scoped()
                )
            except Exception as exc:
                logger.warning("[aiimg] 按编号取输入图失败 %s：%s", text, exc)
                return None
            if not found:
                logger.info("[aiimg] 输入图编号 %s 不在这个会话或已失效", text)
                return None
            logger.info("[aiimg] 输入图来自编号 %s：%s", text, Path(found).name)
            return Path(found)
        path = Path(text)
        return path if path.exists() else None

    def _media_hint(self, event: Any) -> str:
        """Tell the model what this turn actually carries.

        Without it the model guesses, and a quoted message whose media the
        platform never delivered looks the same as one that has a photo.
        """
        if not self._ready or not self._feature_on("edit_enabled"):
            return ""
        images = self._message_images(event)
        if images:
            return (
                f"本轮消息里有 {len(images)} 张可用图片，改图或图生视频时 image 留空即可直接使用。"
            )
        messages = list(getattr(event, "messages", None) or [])
        for message in reversed(messages):
            for element in getattr(message, "chain", None) or []:
                if not getattr(element, "message_id", None):
                    continue
                quoted = getattr(element, "chain", None) or []
                if any(isinstance(item, Image) for item in quoted):
                    continue
                return (
                    "用户引用了历史消息，但平台没有提供其中的图片内容；"
                    "需要改图时请把那张图片重新发一次。"
                )
        return ""

    def _describe_task(self, record: TaskRecord) -> str:
        payload: dict[str, Any] = {
            "status": "ok",
            "task_id": record.task_id,
            "kind": record.kind,
            "state": record.status,
        }
        if record.provider:
            payload["provider"] = record.provider
        if record.size:
            payload["size"] = record.size
        if record.prompt:
            payload["prompt"] = record.prompt
        if record.final_prompt and record.final_prompt != record.prompt:
            payload["final_prompt"] = record.final_prompt
        if record.error:
            payload["error_kind"] = self._error_kind(record.error)
        if record.result:
            try:
                result = json.loads(record.result or "{}")
            except Exception:
                result = {}
            media = result.get("media") or []
            if media:
                payload["media_count"] = len(media)
            if result.get("url"):
                payload["kind"] = "video"
                payload["media_count"] = 1
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

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
        items = record.extra.get("prompts") or [record.prompt]
        concurrency = int(self._cfg("section_queue", "batch_concurrency", default=2) or 2)
        semaphore = asyncio.Semaphore(max(1, concurrency))
        media: list[str] = []
        provider = ""
        model = ""
        size = ""
        errors: list[str] = []

        async def one(index: int, item: Any) -> None:
            nonlocal provider, model, size
            if isinstance(item, dict):
                prompt_text = str(item.get("prompt") or "").strip()
                plan_output = str(item.get("output") or "").strip()
            else:
                prompt_text = str(item or "").strip()
                plan_output = ""
            if not prompt_text:
                return
            async with semaphore:
                extra = dict(record.extra)
                if plan_output:
                    extra["plan_output"] = plan_output
                sub = TaskRecord(
                    task_id=f"{record.task_id}-{index}",
                    kind=record.kind,
                    sid=record.sid,
                    user_id=record.user_id,
                    prompt=prompt_text,
                    provider=record.provider,
                    chain=record.chain,
                    size=record.size,
                    extra=extra,
                )
                try:
                    result = await self._run_image_chain(sub, kind="generate")
                    media.extend(result.get("media") or [])
                    provider = result.get("provider") or provider
                    model = result.get("model") or model
                    size = result.get("size") or size
                except Exception as exc:
                    errors.append(f"{index + 1}: {exc}")

        await asyncio.gather(*(one(i, p) for i, p in enumerate(items)))
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
                logger.warning("[aiimg] 用 %s 取视频开场帧失败：%s", item.provider_id, exc)
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
                dedupe=True,
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
        logger.info(
            "[aiimg] 视频任务 %s 开始生成：提示词=%s，服务商顺序=%s，关键帧=%s，时长=%s，比例=%s",
            record.task_id,
            clip(record.prompt, 60) or "-",
            " → ".join(upcoming) or "-",
            Path(source_path).name if source_path else "无",
            record.extra.get("seconds") or "-",
            record.extra.get("aspect_ratio") or "-",
        )
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
                video_path = await self._download_video(result.url, provider_id)
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
                refs = self._remember_media(
                    [str(video_path)],
                    record=record,
                    kind="video",
                    prompt=record.prompt or "",
                )
                return {
                    "media": [str(video_path)],
                    "url": "",
                    "refs": refs,
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
        logger.info(
            "[aiimg] 任务 %s 开始出图：类型=%s，提示词=%s，尺寸=%s，服务商顺序=%s，输入图 %d 张",
            record.task_id,
            kind or "-",
            clip(record.prompt, 60) or "-",
            record.size or "-",
            " → ".join(upcoming) or "-",
            len(images or []),
        )
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
                        dedupe=True,
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
                refs = self._remember_media(
                    media,
                    record=record,
                    kind=gallery_kind,
                    prompt=final_prompt,
                )
                return {
                    "media": media,
                    "refs": refs,
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

    async def _publish_outcome(self, record: TaskRecord, facts: dict[str, Any]) -> bool:
        """Hand one finished task to the model with facts only.

        Failures and cancellations carry no media, and a turn whose chain is
        empty gets dropped, so the JSON line doubles as the ride-along element.
        """
        return await self._publish_notification(
            record,
            [Text(_facts_text(facts))],
            self._features("result_notification_prompt", ""),
            facts,
            marker=_ERROR_MARKER if facts.get("status") != "succeeded" else "",
            template_key="result_notification_prompt",
        )

    async def _notify_failure(self, record: TaskRecord) -> None:
        """Ask the model once to phrase a failure in persona; stay silent after."""
        facts = {
            "task_id": record.task_id,
            "kind": record.kind,
            "status": record.status or "failed",
            "provider": record.provider or "",
            "size": record.size or "",
            "error_kind": self._error_kind(str(record.error or "")),
        }
        delivered = await self._publish_outcome(record, facts)
        logger.info(
            "[aiimg] 任务 %s 失败通知%s",
            record.task_id,
            "已交给模型" if delivered else "未发布（模型侧不可用）",
        )
        if not delivered:
            await self._safe_send(
                record.sid,
                MessageChain([Text(_facts_text({"status": "failed", "task_id": record.task_id}))]),
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
            facts = {"task_id": record.task_id, "kind": record.kind, "status": "cancelled"}
            delivered = await self._publish_outcome(record, facts)
            logger.info(
                "[aiimg] 任务 %s 取消通知%s",
                record.task_id,
                "已交给模型" if delivered else "未发布（模型侧不可用）",
            )
            if not delivered:
                await self._safe_send(
                    record.sid,
                    MessageChain([Text(_facts_text(facts))]),
                )
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
        refs = [str(item) for item in (payload.get("refs") or []) if item]
        if refs:
            facts["image_ref"] = refs[0]
            if len(refs) > 1:
                facts["image_refs"] = "、".join(refs)
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
            await self._notify_failure(record)
            return
        template = self._features("result_notification_prompt", "")
        # Deliver the media ourselves before asking the model for a sentence.
        # The synthetic event below reaches the model only; whatever it replies
        # is a separate message, so an empty reply can no longer swallow the
        # image.
        delivered = await self._safe_send(record.sid, MessageChain(elements))
        if delivered:
            self._mark_ack_delivered(record.task_id)
            logger.info(
                "[aiimg] 任务 %s 媒体已发送：%d 个元素（%s）",
                record.task_id,
                len(elements),
                "视频" if facts["kind"] == "video" else "图片",
            )
        else:
            logger.error(
                "[aiimg] 任务 %s 媒体投递彻底失败，%d 个元素都没发出去",
                record.task_id,
                len(elements),
            )
            facts["delivery"] = "failed"
        published = await self._publish_notification(
            record,
            elements,
            template,
            facts,
            template_key="result_notification_prompt",
        )
        logger.info(
            "[aiimg] 任务 %s 完成通知%s",
            record.task_id,
            "已交给模型" if published else "未发布（模型侧不可用）",
        )

    async def _safe_send(self, sid: str, chain: MessageChain) -> bool:
        """Deliver a chain with retries; report whether anything reached the chat.

        A finished image is expensive to lose, so transient adapter failures are
        retried, and a chain the adapter keeps rejecting is retried element by
        element: one malformed image no longer swallows the rest of a batch.
        """
        advanced = self._cfg("section_advanced") or {}
        if not isinstance(advanced, dict):
            advanced = {}
        attempts = int(advanced.get("delivery_attempts") or 3)
        return await deliver_chain(
            self.ctx.send_message_chain,
            sid,
            chain,
            make_chain=MessageChain,
            logger=logger,
            attempts=attempts,
        )

    async def _publish_notification(
        self,
        record: TaskRecord,
        elements: list[Any],
        template: str,
        facts: dict[str, Any],
        *,
        marker: str = _NOTIFY_MARKER,
        template_key: str = "result_notification_prompt",
    ) -> bool:
        adapter_name = record.extra.get("adapter") or ""
        session_type = record.extra.get("session_type") or ""
        session_id = record.extra.get("session_id") or ""
        if not adapter_name or not session_id:
            return False
        try:
            adapter = self.ctx.adapter_mgr.get_adapter(adapter_name)
        except Exception as exc:
            logger.warning("[aiimg] 适配器 %s 不可用：%s", adapter_name, exc)
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
            logger.warning("[aiimg] 通知事件发布失败：%s", exc)
            return False
        return True

    @on.llm_request()
    async def _inject_task_notification(self, event, req: LLMRequest, *_):
        extra = _notification_extra(event)
        if not extra:
            return
        facts = extra.get("_aiimg_facts") or {}
        template_key = extra.get("_aiimg_template_key") or "result_notification_prompt"
        template = self._features(template_key, "")
        rendered = render_template(template, **facts).strip() if template else ""
        if not rendered:
            rendered = _facts_text(facts)
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
    async def _inject_media_hint(self, event, req: LLMRequest, *_):
        """Say whether this turn carries a picture the image tools can use."""
        hint = self._media_hint(event)
        if not hint:
            return
        try:
            req.user_prompt.append(
                Prompt(hint, name="aiimg_media_hint", source="aiimg", persist=False)
            )
        except Exception as exc:
            logger.debug("[aiimg] 注入图片提示失败：%s", exc)

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
            logger.debug("[aiimg] 注入预设说明失败：%s", exc)

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
            logger.debug("[aiimg] 摘除框架自带媒体标签失败：%s", exc)
            return
        if not self._builtin_tags_logged:
            self._builtin_tags_logged = True
            logger.info(
                "[aiimg] 已接管框架自带的媒体标签：%s",
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
    async def api_config(self, lite: bool = False):
        manager = getattr(self.ctx, "plugin_mgr", None)
        try:
            cfg = (
                manager.get_plugin_config(PLUGIN_ID)
                if manager is not None
                else (self.plugin_cfg or {})
            )
        except Exception:
            cfg = self.plugin_cfg or {}
        try:
            merged = settings_mod.with_defaults(cfg)
        except Exception:
            # A malformed stored value must not blank the whole settings page.
            logger.exception("[aiimg] 配置读进来后补默认值失败，先用原始内容")
            merged = dict(cfg or {})
        templates = settings_mod.template_catalog()
        logger.info(
            "[aiimg] 下发配置：服务商 %d 家，密钥只给掩码",
            len(merged.get("providers") or []),
        )
        payload = {
            "revision": settings_mod.revision(cfg),
            "config": settings_mod.mask_providers(merged),
            "sections": settings_mod.sections(),
            "templates": templates,
            "fields": settings_mod.field_meta(),
            "provider_defaults": {
                template["key"]: settings_mod.provider_defaults(template["key"])
                for template in templates
            },
            "required": {
                template["key"]: list(
                    settings_mod.REQUIRED_FIELDS.get(template["key"], ())
                )
                for template in templates
            },
            "features": [
                {"key": key, **settings_mod.FEATURE_LABELS.get(key, {})}
                for key in settings_mod.IMPORT_FEATURES
            ],
            "notes": [],
            "status": {},
        }
        if lite:
            # The page paints from this and asks again for the live numbers.
            return payload
        try:
            payload["notes"] = settings_mod.warnings(cfg)
        except Exception:
            logger.exception("[aiimg] 配置提示生成失败")
        try:
            payload["status"] = self._status_payload()
        except Exception:
            logger.exception("[aiimg] 状态汇总失败")
        return payload

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
                incoming = settings_mod.restore_keys(incoming, current)
                logger.info("[aiimg] 保存配置：页面上的掩码密钥已换回本机保存的密钥")
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
                "config": settings_mod.mask_providers(
                    settings_mod.with_defaults(config_now)
                ),
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
        if api_key:
            # 设置页只会看到掩码，拉模型时把它换回真正的密钥
            api_key = settings_mod.unmask_key(api_key, self.plugin_cfg)
        root = base_url or server_url
        if not root:
            raise HTTPException(400, "先填好接口地址再获取模型")
        url = root.rstrip("/") + "/models"
        policy = self._fetch_policy()
        try:
            await ensure_url_allowed(url, policy)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            session = await current_session()
            async with session.get(url, headers=headers, timeout=timeout) as resp:
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
        params = self._gallery.encoding_params()
        if not thumb.is_file():
            from .core.image_format import thumbnail_webp

            data = await asyncio.to_thread(path.read_bytes)
            data = await asyncio.to_thread(
                thumbnail_webp,
                data,
                params["thumb_size"],
                params["thumb_quality"],
                params["thumb_format"],
            )
            thumb.write_bytes(data)
        return FileResponse(
            thumb,
            media_type="image/jpeg" if params["thumb_format"].lower() == "jpeg" else "image/webp",
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
        params = self._gallery.encoding_params()
        if not preview.is_file():
            from .core.image_format import preview_jpeg
            from .core.image_store import ImageStore

            data = await asyncio.to_thread(path.read_bytes)
            data = await asyncio.to_thread(
                preview_jpeg,
                data,
                params["preview_size"],
                params["preview_quality"],
                params["preview_format"],
            )
            await asyncio.to_thread(ImageStore._atomic_write, preview, data)
        fmt = str(params["preview_format"] or "jpeg").lower()
        return FileResponse(
            preview,
            media_type="image/webp" if fmt == "webp" else "image/jpeg",
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
        # 先按字符串长度挡一道，超大内容不进入解码。
        if len(data_url) > 14 * 1024 * 1024:
            logger.warning(
                "[aiimg] 上传参考图被挡：内容 %.1f MB，超过 10MB 上限",
                len(data_url) / (1024 * 1024),
            )
            return {"ok": False, "error": "单张参考图不要超过 10MB"}
        try:
            data = decode_base64_image(data_url)
        except ValueError as exc:
            return {"ok": False, "error": str(exc) or "只支持 jpg、png、webp"}
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
                logger.warning("[aiimg] 删除参考图失败：%s", exc)
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

