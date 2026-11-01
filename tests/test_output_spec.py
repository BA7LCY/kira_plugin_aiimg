"""Output-intent parsing, merging and size resolution tests."""

import pytest

from core.output_spec import (
    OutputIntent,
    apply_conflict_rule,
    aspect_ratio_from_size,
    extract_output_intent_from_prompt,
    format_output_intent,
    is_gpt_image_2_model,
    merge_output_intents,
    output_intent_dict,
    parse_output_intent,
    resolve_gpt_image_2_size,
    resolve_output_intent,
    resolution_from_size,
    select_allowed_size,
    size_from_ratio_and_resolution,
    to_request_size_args,
)


def test_parse_exact_size():
    intent = parse_output_intent("1024x1024")
    assert intent.exact_size == "1024x1024"
    assert intent.aspect_ratio is None


def test_parse_exact_size_unicode_times():
    assert parse_output_intent("1024×768").exact_size == "1024x768"


def test_parse_ratio():
    assert parse_output_intent("3:4").aspect_ratio == "3:4"


def test_parse_resolution_case_insensitive():
    assert parse_output_intent("2k").resolution == "2K"


def test_parse_auto_and_default_empty():
    assert parse_output_intent("auto").is_empty
    assert parse_output_intent("默认").is_empty


def test_parse_invalid_raises():
    with pytest.raises(ValueError):
        parse_output_intent("banana")


def test_extract_ratio_from_prompt():
    intent = extract_output_intent_from_prompt("画一张 16:9 的赛博朋克图")
    assert intent.aspect_ratio == "16:9"


def test_extract_exact_from_prompt():
    intent = extract_output_intent_from_prompt("需要 512x512 的图标")
    assert intent.exact_size == "512x512"


def test_extract_ratio_and_resolution():
    intent = extract_output_intent_from_prompt("竖版 9:16 分辨率 4K")
    assert intent.aspect_ratio == "9:16"
    assert intent.resolution == "4K"


def test_extract_none():
    assert extract_output_intent_from_prompt("普通的一句话").is_empty


def test_merge_first_wins():
    merged = merge_output_intents(
        OutputIntent(aspect_ratio="16:9"), OutputIntent(aspect_ratio="1:1")
    )
    assert merged.aspect_ratio == "16:9"


def test_merge_skips_empty_and_none():
    assert merge_output_intents(None, OutputIntent()).is_empty


def test_merge_exact_into_ratio():
    merged = merge_output_intents(
        OutputIntent(exact_size="1024x768"), OutputIntent(aspect_ratio="1:1")
    )
    assert merged.exact_size == "1024x768"
    assert merged.aspect_ratio is None


def test_resolve_prompt_wins_over_output():
    intent = resolve_output_intent(prompt="竖版 9:16", output="1:1")
    assert intent.aspect_ratio == "9:16"


def test_resolve_structured_over_output():
    intent = resolve_output_intent(prompt="", aspect_ratio="3:4", output="1:1")
    assert intent.aspect_ratio == "3:4"


def test_resolve_empty():
    assert resolve_output_intent(prompt="").is_empty


def test_output_intent_dict_and_format():
    intent = OutputIntent(aspect_ratio="3:4", resolution="2K")
    assert output_intent_dict(intent) == {
        "exact_size": None,
        "aspect_ratio": "3:4",
        "resolution": "2K",
    }
    assert format_output_intent(intent) == "3:4 2K"
    assert format_output_intent(None) == ""


def test_aspect_ratio_from_size():
    assert aspect_ratio_from_size("1024x1024") == "1:1"
    assert aspect_ratio_from_size("2048x1152") == "16:9"
    assert aspect_ratio_from_size("bad") is None


def test_to_request_size_args():
    assert to_request_size_args(OutputIntent(exact_size="1024x768")) == {"size": "1024x768"}
    assert to_request_size_args(OutputIntent(resolution="4K")) == {"resolution": "4K"}
    assert to_request_size_args(OutputIntent()) == {}


def test_resolution_from_size():
    assert resolution_from_size("1024x1024") == "1K"
    assert resolution_from_size("2048x1152") == "2K"
    assert resolution_from_size("3840x2160") == "4K"
    assert resolution_from_size("bad") is None


def test_gpt_image_2_model_detection():
    assert is_gpt_image_2_model("gpt-image-2")
    assert is_gpt_image_2_model("GPT-Image_2")
    assert not is_gpt_image_2_model("gpt-image-1")


def test_resolve_gpt_image_2_size_defaults():
    assert resolve_gpt_image_2_size(OutputIntent()) == "1024x1024"
    assert resolve_gpt_image_2_size(OutputIntent(), default_aspect_ratio="16:9") == "1280x720"


def test_resolve_gpt_image_2_size_ratio_resolution():
    assert resolve_gpt_image_2_size(OutputIntent(aspect_ratio="9:16", resolution="2K")) == "1152x2048"


def test_resolve_gpt_image_2_size_exact():
    assert resolve_gpt_image_2_size(OutputIntent(exact_size="1536x2048")) == "1536x2048"


def test_select_allowed_size_resolution():
    allowed = ["256x256", "512x512", "1024x1024", "2048x2048"]
    assert select_allowed_size(OutputIntent(resolution="2K"), allowed) == "2048x2048"


def test_select_allowed_size_default():
    allowed = ["256x256", "512x512", "1024x1024", "2048x2048"]
    assert select_allowed_size(OutputIntent(), allowed, default_size="1024x1024") == "1024x1024"


def test_select_allowed_size_exact():
    allowed = ["256x256", "512x512", "1024x1024"]
    assert select_allowed_size(OutputIntent(exact_size="512x512"), allowed) == "512x512"


def test_select_allowed_size_ratio_filter():
    allowed = ["256x256", "1024x576", "2048x1152"]
    assert select_allowed_size(OutputIntent(aspect_ratio="16:9", resolution="2K"), allowed) == "2048x1152"


def test_size_from_ratio_and_resolution():
    assert size_from_ratio_and_resolution("16:9", "4K") == "3840x2160"
    assert size_from_ratio_and_resolution("3:4", "1K") == "768x1024"
    assert size_from_ratio_and_resolution("16:9") == "1280x720"
    assert size_from_ratio_and_resolution("", "4K") is None


def test_conflict_rule_keeps_size():
    intent = OutputIntent(exact_size="1024x1024", aspect_ratio="16:9", resolution="4K")
    kept = apply_conflict_rule(intent, prefer_resolution=True)
    assert kept.exact_size == "1024x1024"
    assert kept.aspect_ratio is None
    assert kept.resolution is None


def test_conflict_rule_keeps_ratio():
    intent = OutputIntent(exact_size="1024x1024", aspect_ratio="16:9", resolution="4K")
    kept = apply_conflict_rule(intent, prefer_resolution=False)
    assert kept.exact_size is None
    assert kept.aspect_ratio == "16:9"
    assert kept.resolution == "4K"


def test_conflict_rule_leaves_single_side_alone():
    only_size = OutputIntent(exact_size="512x512")
    only_ratio = OutputIntent(aspect_ratio="3:4", resolution="2K")
    assert apply_conflict_rule(only_size, prefer_resolution=False) == only_size
    assert apply_conflict_rule(only_ratio, prefer_resolution=True) == only_ratio
