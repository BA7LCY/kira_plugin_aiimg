"""Settings helpers: defaults, revision, validation and chain normalizing."""

import pytest

from core import planner
from core import prompt as P
from core import settings as S


def _provider(template="openai_images", provider_id="main", **overrides):
    values = dict(S.provider_defaults(template))
    values["id"] = provider_id
    values["base_url"] = overrides.pop("base_url", "https://api.example.com/v1")
    values["model"] = overrides.pop("model", "gpt-image-1")
    values.update(overrides)
    return values


def test_with_defaults_fills_every_section():
    config = S.with_defaults({})

    assert config["section_queue"]["max_running"] == 2
    assert config["section_gallery"]["max_count"] == 50
    assert config["providers"] == []
    assert config["chains"] == {}
    assert set(S.sections()) <= set(config)


def test_feature_defaults_use_1024_sizes():
    features = S.with_defaults({})["section_features"]

    assert features["prefer_resolution"] is True
    assert features["draw_default_output"] == "1024x1024"
    assert features["edit_default_output"] == "1024x1024"
    assert features["selfie_default_output"] == "768x1024"


def test_revision_is_stable_and_content_sensitive():
    first = {"a": 1, "b": [1, 2]}
    second = {"b": [1, 2], "a": 1}

    assert S.revision(first) == S.revision(second)
    assert len(S.revision(first)) == 64
    assert S.revision(first) != S.revision({"a": 2, "b": [1, 2]})


def test_normalize_keeps_sections_and_merges_chain_aliases():
    config = S.with_defaults({})
    config["providers"] = [_provider(), _provider(provider_id="backup")]
    config["chains"] = {"draw": ["main"], "generate_image": ["backup:4K"]}

    normalized = S.normalize(config)

    assert normalized["section_queue"]["max_running"] == 2
    assert [item["id"] for item in normalized["providers"]] == ["main", "backup"]
    assert normalized["chains"]["generate_image"] == [
        {"provider": "main", "output": ""},
        {"provider": "backup", "output": "4K"},
    ]


def test_normalize_rejects_wrong_section_type():
    config = S.with_defaults({})
    config["section_queue"]["max_running"] = "two"

    with pytest.raises(ValueError, match="整数"):
        S.normalize(config)


def test_normalize_provider_reports_missing_field():
    config = S.with_defaults({})
    provider = _provider()
    provider["base_url"] = ""
    config["providers"] = [provider]

    with pytest.raises(ValueError, match="还缺接口地址"):
        S.normalize(config)


def test_normalize_rejects_duplicate_provider_ids():
    config = S.with_defaults({})
    config["providers"] = [_provider(), _provider()]

    with pytest.raises(ValueError, match="重复"):
        S.normalize(config)


def test_provider_defaults_match_template_kind():
    chat = S.provider_defaults("openai_chat")
    video = S.provider_defaults("agnes_video")

    assert chat["prompt_template"].startswith("{{prompt}}")
    assert "Return exactly one image" in chat["prompt_template"]
    assert chat["generate_request_mode"] == "auto"
    assert chat["max_concurrent"] == 0
    assert "supports_edit" not in video
    assert video["max_concurrent"] == 0
    assert video["request_interval_seconds"] == 61
    assert video["timeout_seconds"] == 1800
    assert video["poll_interval_seconds"] == 61
    assert video["max_retries"] == 3
    assert video["mode"] == "auto"
    assert video["size"] == "720P"
    assert video["aspect_ratio"] == "16:9"


def test_provider_concurrency_is_editable_in_every_template():
    fields = S.field_meta()

    assert fields["max_concurrent"]["type"] == "integer"
    assert fields["max_concurrent"]["maximum"] == 32
    for template in S.template_catalog():
        assert "max_concurrent" in template["fields"], template["key"]


def test_queue_exposes_circuit_breaker_fields():
    fields = S.sections()["section_queue"]["fields"]

    assert fields["provider_failure_threshold"]["default"] == 3
    assert fields["provider_cooldown_seconds"]["default"] == 120


def test_warnings_report_unknown_chain_provider():
    config = S.with_defaults({})
    config["chains"] = {"generate_image": ["ghost"]}

    notes = S.warnings(config)

    assert any("ghost" in note for note in notes)
    assert any("还没有配置服务商" in note for note in notes)


def test_textarea_fields_reject_list_values():
    config = S.with_defaults({})
    config["section_access"]["whitelist"] = ["qq:123", "456"]

    with pytest.raises(ValueError, match="需要填文字"):
        S.normalize(config)


def test_info_fields_are_not_written_to_config():
    config = S.with_defaults({})

    assert "welcome_info" not in config


def test_framework_schema_hides_console_only_fields():
    features = S.schema()["section_features"]["fields"]

    assert "selfie_reference_files" not in features
    assert "selfie_include_uploads" not in features
    assert "selfie_use_persona" in features


def test_prompt_keys_live_only_in_the_prompt_card():
    features = S.schema()["section_features"]["fields"]
    prompts = S.schema()["section_prompts"]["fields"]

    for key in (
        "selfie_prompt_prefix",
        "selfie_video_frame_template",
        "ack_notification_prompt",
        "result_notification_prompt",
        "draw_presets",
        "edit_presets",
        "presets_prompt",
    ):
        assert key not in features
        assert key in prompts


def test_console_sections_expose_console_only_fields():
    fields = S.sections()["section_features"]["fields"]

    for name, definition in S.CONSOLE_ONLY_FEATURES.items():
        assert fields[name]["name"] == definition["name"]
        assert fields[name]["hint"] == definition.get("note", "")


def test_console_only_fields_follow_the_order_of_the_saved_layout():
    keys = list(S.sections()["section_features"]["fields"])

    assert keys.index("selfie_default_output") < keys.index("prefer_resolution")
    assert keys.index("prefer_resolution") < keys.index("selfie_follow_edit_chain")
    assert keys.index("selfie_follow_edit_chain") < keys.index("selfie_reference_files")
    assert keys.index("selfie_reference_files") < keys.index("selfie_include_uploads")
    assert keys.index("selfie_include_uploads") < keys.index("selfie_use_persona")


def test_prompt_card_groups_moved_and_new_fields():
    keys = list(S.sections()["section_prompts"]["fields"])

    assert keys[:7] == [
        "selfie_prompt_prefix",
        "selfie_video_frame_template",
        "ack_notification_prompt",
        "result_notification_prompt",
        "draw_presets",
        "edit_presets",
        "presets_prompt",
    ]
    assert keys[7] == "reference_heading"
    assert keys[8:] == [
        "reference_role_persona",
        "reference_role_base",
        "reference_role_subject",
        "reference_role_object",
        "reference_role_clothing",
        "reference_role_style",
        "reference_role_pose",
        "reference_role_background",
        "reference_note_header",
        "reference_note_footer",
        "media_hint_with_images",
        "media_hint_without_images",
    ]


def test_prompt_card_defaults_match_the_prompt_module():
    fields = S.schema()["section_prompts"]["fields"]

    for role, text in P.REFERENCE_ROLE_PROMPTS.items():
        assert fields[f"reference_role_{role}"]["default"] == text
    assert fields["reference_note_header"]["default"] == P.REFERENCE_NOTE_HEADER
    assert fields["reference_note_footer"]["default"] == P.REFERENCE_NOTE_FOOTER


def test_planner_prompt_default_is_the_builtin_template():
    fields = S.schema()["section_queue"]["fields"]

    assert fields["planner_prompt"]["default"] == planner.DEFAULT_PLANNER_TEMPLATE
    assert S.field_default("section_queue", "planner_prompt") == (
        planner.DEFAULT_PLANNER_TEMPLATE
    )


def test_gallery_cleanup_defaults_are_finite():
    fields = S.schema()["section_gallery"]["fields"]

    assert fields["max_count"]["default"] == 50
    assert fields["retention_days"]["default"] == 14
    assert fields["selfie_max_count"]["default"] == 50
    assert fields["selfie_retention_days"]["default"] == 14
    assert fields["video_max_count"]["default"] == 20
    assert fields["video_retention_days"]["default"] == 14


def test_delivery_timeouts_are_declared_first_in_advanced():
    fields = S.schema()["section_advanced"]["fields"]
    keys = list(fields)

    assert keys[:2] == ["delivery_timeout_seconds", "video_send_timeout_seconds"]
    assert fields["delivery_timeout_seconds"]["default"] == 90
    assert fields["delivery_timeout_seconds"]["min"] == 10
    assert fields["delivery_timeout_seconds"]["max"] == 300
    assert fields["video_send_timeout_seconds"]["default"] == 180
    assert fields["video_send_timeout_seconds"]["min"] == 30
    assert fields["video_send_timeout_seconds"]["max"] == 600


def test_prompt_values_live_only_in_the_new_card():
    config = S.with_defaults({})
    config["section_prompts"]["draw_presets"] = "手办化:figurine"

    out = S.normalize(config)

    assert out["section_prompts"]["draw_presets"] == "手办化:figurine"
    assert "draw_presets" not in out["section_features"]


def test_old_prompt_location_is_ignored():
    config = S.with_defaults({})
    config["section_features"]["ack_notification_prompt"] = "旧位置"

    out = S.normalize(config)

    assert out["section_prompts"]["ack_notification_prompt"] == ""
    assert "ack_notification_prompt" not in out["section_features"]


def test_clearing_a_prompt_stays_cleared():
    config = S.with_defaults({})
    config["section_prompts"]["ack_notification_prompt"] = "先写一句"
    config = S.normalize(config)

    config["section_prompts"]["ack_notification_prompt"] = ""
    out = S.normalize(config)

    assert out["section_prompts"]["ack_notification_prompt"] == ""
    assert "ack_notification_prompt" not in out["section_features"]


def test_a_value_edited_in_the_new_card_beats_the_legacy_copy():
    config = S.with_defaults({})
    config["section_features"]["ack_notification_prompt"] = "旧指令"
    config["section_prompts"]["ack_notification_prompt"] = "新指令"

    out = S.normalize(config)

    assert out["section_prompts"]["ack_notification_prompt"] == "新指令"
    assert "ack_notification_prompt" not in out["section_features"]


def test_framework_schema_keeps_only_the_toggles():
    fields = S.schema()["section_features"]["fields"]

    assert "selfie_use_persona" in fields
    assert "selfie_follow_edit_chain" in fields
    assert "selfie_reference_files" not in fields


def test_layout_only_values_stay_out_of_the_saved_config():
    config = S.with_defaults({})
    assert config["section_features"]["selfie_reference_files"] == ""
    config["section_prompts"]["draw_presets"] = "手办化:figurine"

    out = S.normalize(config)

    assert out["section_prompts"]["draw_presets"] == "手办化:figurine"
    assert "draw_presets" not in out["section_features"]
    # The list is read from the data directory, so it never enters the config.
    assert "selfie_reference_files" not in out["section_features"]


def test_selfie_include_uploads_defaults_to_off():
    config = S.with_defaults({})

    assert config["section_features"]["selfie_include_uploads"] is False
    assert S.field_default("section_features", "selfie_include_uploads") is False


def test_selfie_include_uploads_preserves_a_saved_value():
    config = S.with_defaults({"section_features": {"selfie_include_uploads": True}})

    assert config["section_features"]["selfie_include_uploads"] is True


def test_sections_start_collapsed():
    for definition in S.sections().values():
        assert definition.get("collapsed") is True
