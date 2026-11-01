"""Size normalization and provider-size resolution tests."""

from core.output_spec import OutputIntent
from core.sizes import (
    SUPPORTED_SIZES,
    normalize_size_text,
    ratio_defaults_from_sizes,
    resolve_provider_size,
    resolve_ratio_size,
    size_to_ratio,
)


def test_normalize_size_text():
    assert normalize_size_text(" 1024×768 ") == "1024x768"


def test_supported_sizes_include_common():
    assert "1024x1024" in SUPPORTED_SIZES
    assert "2048x1152" in SUPPORTED_SIZES


def test_resolve_ratio_size_default():
    size, warning = resolve_ratio_size("1:1")
    assert size == "256x256"
    assert warning is None


def test_resolve_ratio_size_override():
    size, warning = resolve_ratio_size("1:1", overrides={"1:1": "1024x1024"})
    assert size == "1024x1024"
    assert warning is None


def test_resolve_ratio_size_unsupported():
    size, warning = resolve_ratio_size("7:1")
    assert size == ""
    assert "unsupported" in warning


def test_size_to_ratio():
    assert size_to_ratio("1024x1024") == "1:1"
    assert size_to_ratio("2048x1152") == "16:9"
    assert size_to_ratio("garbage") is None


def test_ratio_defaults_from_sizes():
    table = ratio_defaults_from_sizes(["256x256", "1024x1024", "2048x1152"])
    assert table["1:1"] == "256x256"
    assert table["16:9"] == "2048x1152"


def test_resolve_provider_size_exact_allowed():
    size, warnings = resolve_provider_size(
        OutputIntent(exact_size="1024x1024"),
        allowed_sizes=["256x256", "1024x1024"],
    )
    assert size == "1024x1024"
    assert not warnings


def test_resolve_provider_size_exact_fallback():
    size, warnings = resolve_provider_size(
        OutputIntent(exact_size="800x600"),
        allowed_sizes=["256x256", "1024x1024"],
    )
    assert size in {"256x256", "1024x1024"}
    assert warnings


def test_resolve_provider_size_hint_passthrough():
    size, _ = resolve_provider_size(OutputIntent(), size_hint="512x512")
    assert size == "512x512"


def test_resolve_provider_size_ratio_allowed():
    size, _ = resolve_provider_size(
        OutputIntent(aspect_ratio="16:9"),
        allowed_sizes=["256x256", "1024x576", "2048x1152"],
    )
    assert size == "2048x1152"


def test_resolve_provider_size_resolution():
    size, _ = resolve_provider_size(
        OutputIntent(resolution="2K"),
        allowed_sizes=["256x256", "1024x1024", "2048x1152", "4096x4096"],
    )
    assert size == "2048x1152"


def test_resolve_provider_size_default():
    size, _ = resolve_provider_size(OutputIntent(), default_size="4096x4096")
    assert size == "4096x4096"


def test_resolve_provider_size_ratio_default_table():
    from core.sizes import SUPPORTED_RATIOS

    size, _ = resolve_provider_size(
        OutputIntent(aspect_ratio="16:9"),
        allowed_ratios=SUPPORTED_RATIOS,
    )
    assert size == "1024x576"
