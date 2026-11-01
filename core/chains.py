"""Fallback chain resolution for image and video features.

The `chains` config maps a feature name to an ordered list of providers. Each
item is either a provider id string or an object with `provider` and `output`
fields. The output override participates in output-intent resolution
(explicit request > tool fields > chain item override > feature default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Feature keys usable in the `chains` config. They match the Kira tool names
# so a user can reason about which chain a tool uses. Short aliases are
# accepted for readability.
FEATURES = (
    "generate_image",
    "edit_image",
    "generate_selfie",
    "generate_image_batch",
    "generate_video",
)

DEFAULT_CHAIN_KEY = "default"

# Accepted aliases -> canonical feature key.
FEATURE_ALIASES: dict[str, str] = {
    "draw": "generate_image",
    "generate": "generate_image",
    "edit": "edit_image",
    "selfie": "generate_selfie",
    "batch": "generate_image_batch",
    "video": "generate_video",
}


def normalize_feature(feature: str) -> str:
    key = str(feature or "").strip()
    if key in FEATURE_ALIASES:
        return FEATURE_ALIASES[key]
    return key


@dataclass(frozen=True)
class ChainItem:
    provider_id: str
    output: str = ""


def parse_chain_item(item: Any) -> ChainItem:
    """Parse one chain entry: string or {provider, output} object."""
    if isinstance(item, dict):
        provider_id = str(item.get("provider") or item.get("id") or "").strip()
        output = str(item.get("output") or "").strip()
        if not provider_id:
            raise ValueError("链条目缺少 provider")
        return ChainItem(provider_id, output)
    text = str(item or "").strip()
    if not text:
        raise ValueError("链条目为空")
    if ":" in text:
        provider_id, _, output = text.partition(":")
        provider_id = provider_id.strip()
        output = output.strip()
        if not provider_id:
            raise ValueError("链条目缺少 provider")
        return ChainItem(provider_id, output)
    return ChainItem(text, "")


def _as_items(raw: Any) -> list[Any]:
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if raw is None or raw is False:
        return []
    return [raw]


def chain_for(
    chains: dict,
    feature: str,
    *,
    available: set[str],
    default_provider: str = "",
) -> list[ChainItem]:
    """Resolve the ordered chain for a feature.

    Lookup order: feature key -> `default` key -> fall back to a single
    available provider (the first image/video provider supplied as
    `default_provider`). Unknown provider ids in a configured chain are
    dropped so a stale config degrades instead of breaking.
    """
    raw = chains or {}
    feature = normalize_feature(feature)
    configured = raw.get(feature, raw.get(DEFAULT_CHAIN_KEY))
    chain: list[ChainItem] = []
    for item in _as_items(configured):
        try:
            parsed = parse_chain_item(item)
        except ValueError:
            continue
        if parsed.provider_id in available:
            chain.append(parsed)
    if not chain and default_provider and default_provider in available:
        chain = [ChainItem(default_provider, "")]
    return chain


def explicit_chain(provider_id: str, *, available: set[str]) -> list[ChainItem]:
    """Single-provider chain for an explicit caller override (no fallback)."""
    if provider_id not in available:
        return []
    return [ChainItem(provider_id, "")]
