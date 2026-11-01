"""Prompt composition and audit-trail tests."""

from core.prompt import (
    REFERENCE_NOTE_FOOTER,
    REFERENCE_NOTE_HEADER,
    apply_prefix,
    compose_prompt,
    reference_note,
    render_template,
)


def test_render_template_simple():
    assert render_template("画：{{prompt}}", prompt="猫") == "画：猫"


def test_render_template_unknown_token_dropped():
    assert render_template("{{missing}} ok", prompt="x") == " ok"


def test_compose_prompt_basic():
    final, records = compose_prompt("一只猫", layers=[("selfie", "头像：{{prompt}}")])
    assert final == "头像：一只猫"
    assert records[0] == {"name": "user_prompt", "source": "caller", "text": "一只猫"}
    assert records[1] == {"name": "selfie", "source": "plugin_config", "text": "头像：一只猫"}


def test_compose_prompt_template():
    final, records = compose_prompt("猫", template="开始 {{prompt}} 结束", template_name="tpl")
    assert final == "开始 猫 结束"
    assert records[1] == {"name": "tpl", "source": "plugin_config", "text": "开始 猫 结束"}


def test_compose_prompt_layers_chain():
    final, records = compose_prompt(
        "猫",
        template="T {{prompt}}",
        template_name="template",
        layers=[("a", "前缀 {{prompt}}"), ("b", "后缀")],
    )
    assert final == "后缀"
    assert [r["name"] for r in records] == ["user_prompt", "template", "a", "b"]


def test_compose_prompt_layer_replaces_when_no_token():
    final, records = compose_prompt("猫", layers=[("raw", "只画风景")])
    assert final == "只画风景"
    assert records[1] == {"name": "raw", "source": "plugin_config", "text": "只画风景"}


def test_compose_prompt_empty_layer_skipped():
    final, records = compose_prompt("猫", layers=[("x", "   ")])
    assert final == "猫"
    assert len(records) == 1


def test_apply_prefix_prepends_text():
    final, layer = apply_prefix("坐在窗边", "A young woman with long black hair")
    assert final == "A young woman with long black hair, 坐在窗边"
    assert layer["name"] == "prefix"


def test_apply_prefix_keeps_trailing_punctuation():
    final, _ = apply_prefix("looking at the camera", "soft daylight, realistic photo.")
    assert final == "soft daylight, realistic photo. looking at the camera"


def test_apply_prefix_renders_template_token():
    final, layer = apply_prefix("坐在窗边", "自拍：{{prompt}}，写实风格", name="selfie_prefix")
    assert final == "自拍：坐在窗边，写实风格"
    assert layer == {
        "name": "selfie_prefix",
        "source": "plugin_config",
        "text": "自拍：坐在窗边，写实风格",
    }


def test_apply_prefix_blank_prefix_is_noop():
    final, layer = apply_prefix("猫", "   ")
    assert final == "猫"
    assert layer is None


def test_apply_prefix_without_user_prompt_uses_prefix():
    final, layer = apply_prefix("", "A woman reading by the window")
    assert final == "A woman reading by the window"
    assert layer["source"] == "plugin_config"


def test_reference_note_needs_at_least_two_images():
    assert reference_note([]) == ""
    assert reference_note(None) == ""
    assert reference_note(["persona"]) == ""


def test_reference_note_numbers_each_role():
    note = reference_note(["persona", "clothing", "background"])
    lines = note.splitlines()

    assert lines[0] == REFERENCE_NOTE_HEADER
    assert lines[1] == "参考图 1：固定人物身份，脸和身份保持一致"
    assert lines[2] == "参考图 2：服装参考，只取衣服的款式、颜色和材质"
    assert lines[3] == "参考图 3：背景参考，只取场景和环境"
    assert lines[-1] == REFERENCE_NOTE_FOOTER


def test_reference_note_unknown_role_falls_back_to_subject():
    note = reference_note(["persona", "face"])

    assert "参考图 2：主体参考，人物或物体的外观特征保持一致" in note


def test_reference_note_mentions_the_edited_original():
    note = reference_note(["base", "pose"])

    assert "参考图 1：待修改的原图，只改用户明确要求的部分" in note
    assert "参考图 2：姿势参考，只取动作和构图" in note
