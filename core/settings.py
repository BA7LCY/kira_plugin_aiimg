"""Config schema helpers shared by the plugin and its WebUI.

The framework reads ``schema.json`` to render its own settings page. The plugin
WebUI renders a richer editor from the same file plus the provider catalog
below, so there is exactly one description of every option.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from .chains import FEATURE_ALIASES, FEATURES
from .provider_registry import REQUIRED_FIELDS, normalize_template_key

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schema.json"

IMPORT_FEATURES: tuple[str, ...] = tuple(
    feature for feature in FEATURES if feature != "generate_image_batch"
)

FEATURE_LABELS: dict[str, dict[str, str]] = {
    "generate_image": {
        "label": "文生图",
        "hint": "只给文字时的出图顺序。留空时用第一个可用的生图服务商。",
    },
    "edit_image": {
        "label": "改图",
        "hint": "带参考图改图时的顺序。服务商不支持改图会自动跳到下一家。",
    },
    "generate_selfie": {
        "label": "自拍",
        "hint": "用参考照片生成自拍时的顺序。",
    },
    "generate_image_batch": {
        "label": "批量生图",
        "hint": "一次画多张时的顺序。",
    },
    "generate_video": {
        "label": "视频",
        "hint": "文生视频、图生视频的顺序。",
    },
}

TEMPLATE_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "key": "openai_images",
        "label": "OpenAI Images",
        "kind": "image",
        "summary": "标准生图接口：单独的生图、改图地址，最常见的对接方式。",
        "fields": (
            "base_url",
            "api_keys",
            "model",
            "default_size",
            "supports_edit",
            "generate_request_mode",
            "edit_request_mode",
            "async_mode",
            "poll_interval",
            "extra_body",
            "proxy_url",
            "timeout",
            "max_retries",
            "max_concurrent",
        ),
    },
    {
        "key": "openai_chat",
        "label": "OpenAI Chat",
        "kind": "image",
        "summary": "对话接口出图：把提示词发给对话模型，再从回复里取出图片；Gemini 系和各类网关常用。",
        "fields": (
            "base_url",
            "api_keys",
            "model",
            "default_size",
            "prompt_template",
            "generate_request_mode",
            "edit_request_mode",
            "extra_body",
            "proxy_url",
            "timeout",
            "max_retries",
            "max_concurrent",
        ),
    },
    {
        "key": "openai_full_url_images",
        "label": "完整地址",
        "kind": "image",
        "summary": "生图和改图各填一个完整地址，接口路径不标准时用这个。",
        "fields": (
            "full_generate_url",
            "full_edit_url",
            "api_keys",
            "model",
            "edit_model",
            "default_size",
            "generate_request_mode",
            "edit_request_mode",
            "async_mode",
            "poll_interval",
            "extra_body",
            "proxy_url",
            "timeout",
            "max_retries",
            "max_concurrent",
        ),
    },
    {
        "key": "agnes_video",
        "label": "Agnes 视频",
        "kind": "video",
        "summary": "提交视频任务后按固定间隔查询结果，支持文生视频和图生视频。",
        "fields": (
            "label",
            "base_url",
            "api_keys",
            "model",
            "mode",
            "seconds",
            "size",
            "aspect_ratio",
            "request_interval_seconds",
            "timeout_seconds",
            "request_timeout_seconds",
            "poll_interval_seconds",
            "max_retries",
            "max_concurrent",
            "extra_body",
        ),
        "field_overrides": {
            "base_url": {"label": "API 地址"},
            "api_keys": {"label": "API Key"},
            "model": {"label": "模型 ID"},
            "max_retries": {
                "label": "查询临时错误重试次数",
                "hint": "只重试查询，不重新创建结果不明的任务，默认 3。",
                "minimum": 0,
                "maximum": 5,
            },
            "extra_body": {
                "label": "额外请求体（高级）",
                "hint": "可配置 seed、first_frame、last_frame、images、audios。Flash 最多 5 张图片和 3 段音频，不支持 videos。",
            },
        },
    },
)

FIELD_META: dict[str, dict[str, Any]] = {
    "id": {
        "label": "服务商 ID",
        "type": "text",
        "hint": "起个好认的短名字，链路里用它引用。只能用字母、数字、下划线、短横线或中文。",
        "example": "main",
    },
    "label": {
        "label": "显示名称",
        "type": "text",
        "hint": "只用来在页面和日志里显示，可以留空。",
        "example": "主力出图",
    },
    "api_key": {
        "label": "API Key",
        "type": "textarea",
        "hint": "一行一个。填多个时会轮流使用，某个密钥失败会换下一个。",
        "example": "sk-xxxx",
    },
    "mode": {
        "label": "生成模式",
        "type": "select",
        "options": ["auto", "text", "keyframe", "reference"],
        "hint": "auto：没有图片时用文生视频，有图片时用首帧；text：只用文字；keyframe：把图片当首帧；reference：把图片当风格与内容参考。",
    },
    "seconds": {
        "label": "视频时长(秒)",
        "type": "text",
        "hint": "按服务商支持的档位填，Agnes 常见为 5、10、12、15。",
        "example": "12",
    },
    "size": {
        "label": "分辨率",
        "type": "text",
        "hint": "常见为 720P、1080P，按服务商支持的档位填。",
        "example": "720P",
    },
    "aspect_ratio": {
        "label": "画幅",
        "type": "text",
        "hint": "常见为 16:9、9:16、1:1，按服务商支持的档位填。",
        "example": "16:9",
    },
    "request_interval_seconds": {
        "label": "请求最短间隔(秒)",
        "type": "integer",
        "hint": "创建与查询共用额度，按密钥持久化限速，跨插件重载保持，默认 61。",
        "minimum": 1,
        "maximum": 600,
    },
    "timeout_seconds": {
        "label": "任务总超时(秒)",
        "type": "integer",
        "hint": "一个视频任务最多等多久，默认 1800。",
        "minimum": 120,
        "maximum": 7200,
    },
    "request_timeout_seconds": {
        "label": "单次请求超时(秒)",
        "type": "integer",
        "hint": "创建或查询单次请求最多等多久，默认 120。",
        "minimum": 10,
        "maximum": 300,
    },
    "poll_interval_seconds": {
        "label": "轮询间隔(秒)",
        "type": "integer",
        "hint": "提交后每隔多少秒查一次结果，默认 61。",
        "minimum": 1,
        "maximum": 600,
    },
    "base_url": {
        "label": "接口地址",
        "type": "text",
        "hint": "填到 /v1 这一层就行，后面的生图、改图路径会自动接上。",
        "example": "https://api.example.com/v1",
        "required": True,
    },
    "api_keys": {
        "label": "密钥",
        "type": "textarea",
        "hint": "一行一个。填多个时会轮流使用，某个密钥失败会换下一个。",
        "example": "sk-xxxx\nsk-yyyy",
    },
    "model": {
        "label": "模型名",
        "type": "text",
        "hint": "按服务商文档填写，也可以点右边的「获取模型」。",
        "required": True,
    },
    "edit_model": {
        "label": "改图模型",
        "type": "text",
        "hint": "改图要用另一个模型时才填，留空沿用上面的模型名。",
    },
    "default_size": {
        "label": "默认画幅",
        "type": "text",
        "hint": "可写 1024x1024、4K 或 3:4 4K。留空用设置里的默认输出。",
        "example": "1024x1024",
    },
    "supports_edit": {
        "label": "支持改图",
        "type": "switch",
        "hint": "服务商不能改图就关掉，改图请求会自动交给链路上的下一家。",
    },
    "extra_body": {
        "label": "额外参数",
        "type": "json",
        "hint": "原样并入请求体，例如 {\"quality\": \"high\"}。不确定就留空。",
    },
    "proxy_url": {
        "label": "代理地址",
        "type": "text",
        "hint": "需要走代理时填，例如 http://127.0.0.1:7890。留空表示直连。",
    },
    "timeout": {
        "label": "超时秒数",
        "type": "integer",
        "hint": "单次请求最多等多久，默认 600。",
        "minimum": 1,
        "maximum": 3600,
    },
    "async_mode": {
        "label": "异步任务",
        "type": "select",
        "options": ["auto", "on", "off"],
        "hint": "auto 先按普通方式提交，对方返回任务号时自动改成轮询取图；on 直接以异步方式提交；off 只等同步结果。",
    },
    "poll_interval": {
        "label": "查询间隔秒数",
        "type": "integer",
        "hint": "查异步任务结果多久问一次，默认 5 秒。",
        "minimum": 1,
        "maximum": 600,
    },
    "max_retries": {
        "label": "失败重试次数",
        "type": "integer",
        "hint": "网络抖动或服务商临时出错时重试几次，默认 2。",
        "minimum": 0,
        "maximum": 10,
    },
    "max_concurrent": {
        "label": "并发上限",
        "type": "integer",
        "hint": "这家同时最多跑几个任务，超出的在队列里等；0 表示不限。给会限流的服务商调小一点。",
        "minimum": 0,
        "maximum": 32,
    },
    "full_generate_url": {
        "label": "生图完整地址",
        "type": "text",
        "hint": "直接填完整的生图接口地址，不再自动拼路径。",
        "example": "https://api.example.com/v1/images/generations",
        "required": True,
    },
    "full_edit_url": {
        "label": "改图完整地址",
        "type": "text",
        "hint": "直接填完整的改图接口地址，留空表示这家不改图。",
        "example": "https://api.example.com/v1/images/edits",
    },
    "generate_request_mode": {
        "label": "文生图请求模式",
        "type": "select",
        "options": ["auto", "stream", "non_stream"],
        "hint": "auto 由当前后端自己决定；stream 强制优先走流式；non_stream 强制直接走非流式。不支持双路径的后端会自动忽略。",
        "row": "request_modes",
    },
    "edit_request_mode": {
        "label": "改图请求模式",
        "type": "select",
        "options": ["auto", "stream", "non_stream"],
        "hint": "auto 由当前后端自己决定；stream 强制优先走流式；non_stream 强制直接走非流式。不支持双路径的后端会自动忽略。",
        "row": "request_modes",
    },
    "prompt_template": {
        "label": "请求模板",
        "type": "textarea",
        "hint": "可用 {{prompt}}、{{size}}、{{resolution}}；默认在提示词后面带一句“只返回一张图”的英文，不需要可以删掉。",
        "example": "{{prompt}}",
    },
    "server_url": {
        "label": "服务地址",
        "type": "text",
        "hint": "视频服务的地址，按服务商文档填写。",
        "example": "https://api.example.com",
    },
    "interval": {
        "label": "查询间隔秒",
        "type": "integer",
        "hint": "提交后每隔多少秒查一次结果，默认 61。",
        "minimum": 1,
        "maximum": 600,
    },
}

_LIST_FIELDS = ("allowed_sizes",)
_JSON_FIELDS = ("extra_body", "ratio_default_sizes")
_INT_RANGES = {
    "timeout": (1, 3600),
    "max_retries": (0, 10),
    "max_concurrent": (0, 32),
    "interval": (1, 600),
    "poll_interval": (1, 600),
    "request_interval_seconds": (1, 600),
    "timeout_seconds": (120, 7200),
    "request_timeout_seconds": (10, 300),
    "poll_interval_seconds": (1, 600),
}
_INT_FIELDS = tuple(_INT_RANGES)
_PROVIDER_INT_FIELDS = _INT_FIELDS
_PROVIDER_STR_FIELDS = (
    "base_url",
    "model",
    "default_size",
    "proxy_url",
    "full_generate_url",
    "full_edit_url",
    "edit_model",
    "generate_request_mode",
    "edit_request_mode",
    "prompt_template",
    "server_url",
    "label",
    "mode",
    "seconds",
    "size",
    "aspect_ratio",
    "async_mode",
)
_ASYNC_MODES = ("auto", "on", "off")
_VIDEO_MODES = ("auto", "text", "keyframe", "reference")

# Chat-style image models often answer with prose or media links unless told
# what to produce. The default stays visible and editable in the provider form.
CHAT_PROMPT_TEMPLATE = (
    "{{prompt}}\n\n"
    "Return exactly one image. Do NOT return video/mp4, HTML, or explanations."
)


# Long text and upload fields live only in the console: the framework page has
# no upload widget and caps an expanded section at 2000px, which made the
# feature section scroll inside itself.
CONSOLE_ONLY_FEATURES: dict[str, dict[str, Any]] = {
    "selfie_follow_edit_chain": {
        "type": "switch",
        "name": "自拍沿用改图链路",
        "default": True,
        "note": "自拍没单独配链路时，自动借用改图那条链路兜底。关掉就必须给自拍单独配链路。",
    },
    "selfie_reference_files": {
        "type": "textarea",
        "name": "自拍参考图文件",
        "default": "",
        "note": "自己上传的参考照片，可多张；在下面点「管理文件」上传。",
    },
    "selfie_include_uploads": {
        "type": "switch",
        "name": "自拍同时使用上传图",
        "default": False,
        "note": "打开后人设参考人像会和上传的参考图一起使用：人设图固定人物身份，上传图按各自的角色补充服装、画风、姿势或背景。关掉只用人设图。",
    },
    "selfie_use_persona": {
        "type": "switch",
        "name": "使用人设自拍图",
        "default": True,
        "note": "打开用当前人设的参考人像；关掉用自己上传的图片。",
    },
    "selfie_prompt_prefix": {
        "type": "textarea",
        "name": "自拍提示词前缀",
        "default": "",
        "note": "拼在每次自拍描述前面的固定文字，用来固定人物外貌或画风。写一段描述即可，例如：A young woman with long black hair, realistic photo, soft daylight。或者“第三方视角拍摄；严禁对镜自拍，画面中不得出现手机、手机屏幕、镜子、镜面反射、相机、三脚架、其它拍摄设备或相机界面；双手严禁做举手机或拿摄像头的动作。”留空表示只用调用方给的描述。想引用调用方原文，在文字里写 {{prompt}}。",
    },
    "selfie_video_frame_template": {
        "type": "textarea",
        "name": "自拍转视频底图模板",
        "default": (
            "A single opening still frame for a video of the reference person. {{prompt}}. "
            "Keep a natural starting pose with room for the requested motion. "
            "One person, one frame, no collage or storyboard."
        ),
        "note": "把自拍参考照转成视频开场画面时用的模板。可用 {{prompt}} 代表调用方描述；留空表示跳过这一步，直接把参考照当关键帧。",
    },
    "prompts_heading": {
        "type": "info",
        "name": "插件状态提示词设置",
        "default": "",
        "note": "填了 = 人设 + 你写的指令；留空 = 人设 + 插件给的最小事实。",
    },
    "ack_notification_prompt": {
        "type": "textarea",
        "name": "接单提示词",
        "default": "",
        "note": "留空时，工具只返回任务事实（任务号、类型、状态），模型自己决定怎么回应；填了就作为指令附在工具结果里。可用 {{task_id}}（任务 ID）、{{kind}}（任务类型）、{{state}}（任务状态）。",
    },
    "terminal_notification_prompt": {
        "type": "textarea",
        "name": "任务完成通知提示词",
        "default": "",
        "note": "媒体已经发出去之后，这段话用来决定模型怎么补一句；留空时给任务事实（任务号、类型、状态、尺寸），并让模型用一句自然的话交代结果。可用 {{task_id}}（任务 ID）、{{kind}}（任务类型）、{{size}}（图片尺寸）、{{model}}（服务商返回的模型名）。",
    },
    "error_notification_prompt": {
        "type": "textarea",
        "name": "任务失败通知提示词",
        "default": "",
        "note": "这次没画成时交给模型的话术指令，只给失败类别，不给原始报错；留空时同样只给类别，由模型自己说。可用 {{error_kind}}（失败类别：超时、没出图、输入缺失、服务商报错、内部错误）。",
    },
    "denied_notification_prompt": {
        "type": "textarea",
        "name": "未执行提示词",
        "default": "",
        "note": "白名单不命中、功能关闭、没有可用服务商时返回给模型的指令；留空时只给原因，由模型自己说。可用 {{reason}}（没执行的原因）、{{action}}（被拦下的操作）。",
    },
    "draw_presets": {
        "type": "textarea",
        "name": "文生图预设提示词",
        "default": "",
    },
    "edit_presets": {
        "type": "textarea",
        "name": "改图预设提示词",
        "default": "",
    },
    "presets_prompt": {
        "type": "textarea",
        "name": "预设说明",
        "default": "画图预设：用户用这些名字描述风格时，把名字填进工具参数 preset。",
        "note": "加在系统提示词里的一句话，用来告诉模型预设怎么用；下面会自动列出预设清单。留空则只列清单、不加说明。",
    },
}


def schema() -> dict[str, Any]:
    """Load the on-disk schema; the plugin root is the single source."""
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def field_default(section: str, key: str, fallback: Any = None) -> Any:
    """Declared default for one field, including console-only features."""
    definition = (schema().get(section) or {}).get("fields", {}).get(key)
    if isinstance(definition, dict) and "default" in definition:
        return copy.deepcopy(definition.get("default"))
    if section == "section_features" and key in CONSOLE_ONLY_FEATURES:
        return copy.deepcopy(CONSOLE_ONLY_FEATURES[key].get("default"))
    return fallback


def sections() -> dict[str, dict[str, Any]]:
    """Section definitions with their field notes exposed as hints.

    The framework form turns `hint` into both placeholder and description, so
    the schema keeps field guidance under `note`; the console still shows it.
    """
    result: dict[str, dict[str, Any]] = {}
    for key, value in schema().items():
        if not isinstance(value, dict) or value.get("type") != "section":
            continue
        section = copy.deepcopy(value)
        fields = section.get("fields") or {}
        for name, field in fields.items():
            if not isinstance(field, dict) or field.get("hint") or not field.get("note"):
                continue
            if f"{name}_tip" in fields:
                continue
            field["hint"] = field["note"]
        result[key] = section
    features = result.get("section_features")
    if features is not None:
        fields = features.setdefault("fields", {})
        console: dict[str, dict[str, Any]] = {}
        for name, definition in CONSOLE_ONLY_FEATURES.items():
            item = copy.deepcopy(definition)
            item["hint"] = item.get("note") or ""
            console[name] = item
        ordered = {}
        # The console page draws its own order: the schema keeps only the two
        # toggles, and everything the framework cannot render is spliced in
        # after `prefer_resolution` so the visual order matches the saved page.
        for key, value in fields.items():
            if key in CONSOLE_ONLY_FEATURES:
                continue
            ordered[key] = value
            if key == "prefer_resolution":
                for name, item in console.items():
                    ordered[name] = item
        for name, item in console.items():
            if name not in ordered:
                ordered[name] = item
        fields.clear()
        fields.update(ordered)
        fields.clear()
        fields.update(ordered)
    return result


def with_defaults(config: dict | None) -> dict[str, Any]:
    """Fill every declared default without dropping unknown keys."""
    result = copy.deepcopy(config or {})
    definitions = schema()
    for key, definition in definitions.items():
        if not isinstance(definition, dict):
            continue
        if definition.get("type") == "info":
            continue
        default = copy.deepcopy(definition.get("default"))
        if definition.get("type") == "section":
            values = result.get(key)
            if not isinstance(values, dict):
                values = {}
            for field, item in (definition.get("fields") or {}).items():
                if isinstance(item, dict) and item.get("type") == "info":
                    continue
                values.setdefault(field, copy.deepcopy(item.get("default")))
            result[key] = values
        else:
            result.setdefault(key, default)
    features = result.get("section_features")
    if isinstance(features, dict):
        for name, definition in CONSOLE_ONLY_FEATURES.items():
            features.setdefault(name, copy.deepcopy(definition.get("default")))
    if not isinstance(result.get("providers"), list):
        result["providers"] = []
    if not isinstance(result.get("chains"), dict):
        result["chains"] = {}
    return result


def revision(config: dict | None) -> str:
    payload = json.dumps(config or {}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def template_catalog() -> list[dict[str, Any]]:
    return [copy.deepcopy(item) for item in TEMPLATE_CATALOG]


def field_meta() -> dict[str, dict[str, Any]]:
    return copy.deepcopy(FIELD_META)


def provider_defaults(template_key: str) -> dict[str, Any]:
    """Sensible starting values for a freshly added provider."""
    template = _template_entry(template_key)
    overrides = template.get("field_overrides") or {}
    values: dict[str, Any] = {"id": "", "template_key": template_key, "api_keys": []}
    if template["kind"] == "image":
        values.update(
            {
                "default_size": "",
                "timeout": 600,
                "max_retries": 2,
                "max_concurrent": 0,
                "generate_request_mode": "auto",
                "edit_request_mode": "auto",
                "async_mode": "auto",
                "poll_interval": 5,
                "prompt_template": (
                    CHAT_PROMPT_TEMPLATE if template_key == "openai_chat" else ""
                ),
                "extra_body": {},
            }
        )
        if template_key != "openai_full_url_images":
            values["supports_edit"] = True
    else:
        values.update(
            {
                "model": "agnes-video-2.5-flash",
                "mode": "auto",
                "seconds": "12",
                "size": "720P",
                "aspect_ratio": "16:9",
                "request_interval_seconds": 61,
                "timeout_seconds": 1800,
                "request_timeout_seconds": 120,
                "poll_interval_seconds": 61,
                "max_retries": 3,
                "max_concurrent": 0,
                "extra_body": {},
            }
        )
    for field in template["fields"]:
        meta = {**FIELD_META.get(field, {}), **(overrides.get(field) or {})}
        values.setdefault(field, _default_for_type(meta.get("type", "text")))
    return values


def _default_for_type(kind: str) -> Any:
    if kind == "integer":
        return 0
    if kind == "switch":
        return False
    if kind == "list":
        return []
    if kind == "json":
        return {}
    return ""


def _template_entry(template_key: str) -> dict[str, Any]:
    for item in TEMPLATE_CATALOG:
        if item["key"] == template_key:
            return item
    raise ValueError(f"未知接口类型: {template_key}")


def _section_type_check(field: str, label: str, kind: str, value: Any, item: dict) -> Any:
    if kind == "switch":
        if type(value) is not bool:
            raise ValueError(f"{label} 只能是开或关")
        return value
    if kind == "integer":
        if type(value) is not int:
            raise ValueError(f"{label} 需要填整数")
        minimum = item.get("minimum", item.get("min"))
        maximum = item.get("maximum", item.get("max"))
        if minimum is not None and value < minimum:
            raise ValueError(f"{label} 不能小于 {minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"{label} 不能大于 {maximum}")
        return value
    if kind == "float":
        if type(value) not in (int, float):
            raise ValueError(f"{label} 需要填数字")
        return float(value)
    if kind in ("string", "text", "textarea", "sensitive"):
        if isinstance(value, (list, tuple)):
            # Older configs stored these as arrays; the form now edits lines.
            value = "\n".join(str(entry) for entry in value)
        if not isinstance(value, str):
            raise ValueError(f"{label} 需要填文字")
        if len(value) > 20000:
            raise ValueError(f"{label} 太长了，最多 20000 个字符")
        return value
    if kind == "list":
        if not isinstance(value, list) or any(not isinstance(entry, str) for entry in value):
            raise ValueError(f"{label} 需要一行一项的文字列表")
        if len(value) > 500:
            raise ValueError(f"{label} 最多 500 项")
        return [entry.strip() for entry in value if entry.strip()]
    if kind == "json":
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value) if value.strip() else {}
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label}的格式不对，检查括号和引号（{exc.msg}）") from exc
        raise ValueError(f"{label}的格式不对")
    return value


def normalize_provider(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"第 {index + 1} 个服务商格式不对")
    provider_id = str(raw.get("id") or "").strip()
    if not provider_id:
        raise ValueError(f"第 {index + 1} 个服务商还没有填 ID")
    if len(provider_id) > 64 or any(ch.isspace() for ch in provider_id):
        raise ValueError(f"服务商 ID「{provider_id}」不能用空格，也不要超过 64 个字符")
    template_key = normalize_template_key(
        raw.get("__template_key") or raw.get("template_key")
    )
    if not template_key:
        raise ValueError(f"服务商「{provider_id}」还没有选接口类型")
    entry = _template_entry(template_key)

    result: dict[str, Any] = {"id": provider_id, "template_key": template_key}
    keys_raw = raw.get("api_keys")
    if isinstance(keys_raw, str):
        keys_raw = [line.strip() for line in keys_raw.splitlines()]
    if keys_raw is None:
        keys_raw = []
    if not isinstance(keys_raw, list) or any(not isinstance(key, str) for key in keys_raw):
        raise ValueError(f"服务商「{provider_id}」的密钥需要一行一个")
    result["api_keys"] = [key.strip() for key in keys_raw if key.strip()]

    for field in _PROVIDER_STR_FIELDS:
        value = raw.get(field)
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise ValueError(f"服务商「{provider_id}」的{field_label(field)}需要填文字")
        result[field] = value.strip()

    defaults = provider_defaults(template_key)
    for field in _PROVIDER_INT_FIELDS:
        if field not in defaults:
            continue
        value = raw.get(field)
        if value is None or value == "":
            value = defaults[field]
        if type(value) is not int:
            raise ValueError(f"服务商「{provider_id}」的{field_label(field)}需要填整数")
        minimum, maximum = _INT_RANGES[field]
        if not minimum <= value <= maximum:
            raise ValueError(
                f"服务商「{provider_id}」的{field_label(field)}要在 {minimum} 到 {maximum} 之间"
            )
        result[field] = value

    supports_edit = raw.get("supports_edit", entry["kind"] == "image")
    if entry["kind"] == "video":
        supports_edit = False
    elif template_key == "openai_full_url_images" and "supports_edit" not in raw:
        supports_edit = False
    if type(supports_edit) is not bool:
        raise ValueError(f"服务商「{provider_id}」的「支持改图」只能是开或关")
    result["supports_edit"] = supports_edit

    mode = str(result.get("mode") or "").strip().lower()
    if entry["kind"] == "video" and mode not in _VIDEO_MODES:
        raise ValueError(
            f"服务商「{provider_id}」的{field_label('mode')}只能是 "
            + "、".join(_VIDEO_MODES)
        )
    result["mode"] = mode

    async_mode = str(result.get("async_mode") or "auto").strip().lower()
    if entry["kind"] == "image" and async_mode not in _ASYNC_MODES:
        raise ValueError(
            f"服务商「{provider_id}」的{field_label('async_mode')}只能是 "
            + "、".join(_ASYNC_MODES)
        )
    result["async_mode"] = async_mode if entry["kind"] == "image" else "off"

    allowed = raw.get("allowed_sizes") or []
    if isinstance(allowed, str):
        allowed = [line.strip() for line in allowed.splitlines()]
    if not isinstance(allowed, list) or any(not isinstance(item, str) for item in allowed):
        raise ValueError(f"服务商「{provider_id}」的可选画幅需要一行一个")
    result["allowed_sizes"] = [item.strip() for item in allowed if item.strip()]

    for field in _JSON_FIELDS:
        value = raw.get(field) or {}
        if isinstance(value, str):
            try:
                value = json.loads(value) if value.strip() else {}
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"服务商「{provider_id}」的{field_label(field)}格式不对，检查括号和引号（{exc.msg}）"
                ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"服务商「{provider_id}」的{field_label(field)}格式不对")
        result[field] = value

    for field in REQUIRED_FIELDS.get(template_key, ()):
        value = result.get(field)
        if field == "api_keys":
            if not value:
                raise ValueError(f"服务商「{provider_id}」还缺{field_label(field)}")
        elif not str(value or "").strip():
            raise ValueError(f"服务商「{provider_id}」还缺{field_label(field)}")

    # Keep unknown keys so future fields survive an older UI save.
    for key, value in raw.items():
        if key not in result and key != "__template_key":
            result[key] = value
    return result


def field_label(field: str) -> str:
    return FIELD_META.get(field, {}).get("label", field)


def normalize_chains(raw: Any) -> dict[str, list[Any]]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("功能链路格式不对，应该按功能分组")
    chains: dict[str, list[Any]] = {}
    for feature, items in raw.items():
        key = FEATURE_ALIASES.get(str(feature or "").strip(), str(feature or "").strip())
        if key not in FEATURES and key != "default":
            continue
        if items is None:
            items = []
        if not isinstance(items, list):
            items = [items]
        chain: list[Any] = []
        for item in items:
            if isinstance(item, dict):
                provider_id = str(item.get("provider") or item.get("id") or "").strip()
                output = str(item.get("output") or "").strip()
                if not provider_id:
                    raise ValueError(f"「{FEATURE_LABELS.get(key, {}).get('label', key)}」有链条目没有选服务商")
                chain.append({"provider": provider_id, "output": output})
                continue
            text = str(item or "").strip()
            if not text:
                continue
            provider_id, _, output = text.partition(":")
            chain.append(
                {"provider": provider_id.strip(), "output": output.strip()}
            )
        chains.setdefault(key, []).extend(chain)
    return chains


def normalize(config: Any) -> dict[str, Any]:
    """Validate one full config document and return the canonical form."""
    if not isinstance(config, dict):
        raise ValueError("配置格式不对")
    result: dict[str, Any] = {}
    for key, definition in sections().items():
        values = config.get(key)
        if values is None:
            values = {}
        if not isinstance(values, dict):
            raise ValueError(f"「{definition.get('name', key)}」这一段格式不对")
        normalized: dict[str, Any] = {}
        for field, item in (definition.get("fields") or {}).items():
            if isinstance(item, dict) and item.get("type") == "info":
                continue
            value = values.get(field, copy.deepcopy(item.get("default")))
            normalized[field] = _section_type_check(
                field, item.get("name", field), item.get("type", "string"), value, item
            )
        result[key] = normalized

    # Reference photos are listed from the plugin data directory, so the field
    # exists for layout only and must never reach the stored config: an upload
    # would otherwise change the revision and break the next settings save.
    features = result.get("section_features")
    if isinstance(features, dict):
        features.pop("selfie_reference_files", None)

    providers_raw = config.get("providers") or []
    if not isinstance(providers_raw, list):
        raise ValueError("服务商配置需要是列表")
    if len(providers_raw) > 50:
        raise ValueError("服务商最多 50 个")
    providers = [normalize_provider(item, index) for index, item in enumerate(providers_raw)]
    ids = [item["id"] for item in providers]
    duplicates = {item for item in ids if ids.count(item) > 1}
    if duplicates:
        raise ValueError(f"服务商 ID 重复了：{'、'.join(sorted(duplicates))}")
    result["providers"] = providers

    result["chains"] = normalize_chains(config.get("chains") or {})
    return result


def warnings(config: dict[str, Any]) -> list[str]:
    """Non-fatal problems worth showing in the WebUI."""
    notes: list[str] = []
    providers = {
        str(item.get("id") or ""): item
        for item in (config.get("providers") or [])
        if isinstance(item, dict)
    }
    for feature, items in (config.get("chains") or {}).items():
        label = FEATURE_LABELS.get(feature, {}).get("label", feature)
        for item in items or []:
            provider_id = (
                item.get("provider") if isinstance(item, dict) else str(item or "")
            )
            if provider_id and provider_id not in providers:
                notes.append(f"「{label}」链路里的 {provider_id} 不存在，会被跳过")
    if not providers:
        notes.append("还没有配置服务商，生图功能暂时用不了")
    return notes
