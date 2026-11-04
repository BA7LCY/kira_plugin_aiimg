"""Batch planner: request building, reply parsing and ratio spreading."""

import pytest

from core import planner


class TestRatioNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("3:4", "3:4"),
            ("3x4", "3:4"),
            ("3X4", "3:4"),
            (" 3 ： 4 ", "3:4"),
            ("9/16", "9:16"),
            ("16×9", "16:9"),
            ("0:4", ""),
            ("", ""),
            (None, ""),
            ("square", ""),
            ("1920x1080", ""),
        ],
    )
    def test_normalize_ratio(self, raw, expected):
        assert planner.normalize_ratio(raw) == expected

    def test_ratio_choices_reads_commas_and_spaces(self):
        assert planner.ratio_choices("1:1, 3:4  16:9") == ["1:1", "3:4", "16:9"]

    def test_ratio_choices_drops_duplicates_and_keeps_order(self):
        assert planner.ratio_choices("4:3,4x3,1:1") == ["4:3", "1:1"]

    def test_ratio_choices_accepts_a_list(self):
        assert planner.ratio_choices(["9:16", "bad"]) == ["9:16"]

    def test_ratio_choices_falls_back_to_defaults(self):
        assert planner.ratio_choices("") == list(planner.DEFAULT_RATIOS)
        assert planner.ratio_choices("nonsense") == list(planner.DEFAULT_RATIOS)


class TestBuildRequest:
    def test_default_template_carries_every_placeholder(self):
        text = planner.build_request(
            "", brief="雨夜霓虹街头", count=3, ratios=["1:1", "16:9"]
        )
        assert "3" in text
        assert "1:1、16:9" in text
        assert text.rstrip().endswith("雨夜霓虹街头")
        assert "{{" not in text

    def test_custom_template_is_used_verbatim(self):
        text = planner.build_request(
            "画 {{count}} 张 {{ratios}} 关于 {{brief}}",
            brief="猫",
            count=2,
            ratios=["1:1"],
        )
        assert text == "画 2 张 1:1 关于 猫"

    def test_template_without_brief_placeholder_still_gets_the_brief(self):
        text = planner.build_request("随便画", brief="雪山日出", count=2, ratios=["1:1"])
        assert text.startswith("随便画")
        assert "雪山日出" in text

    def test_count_never_drops_below_one(self):
        text = planner.build_request("{{count}} 张", brief="x", count=0, ratios=["1:1"])
        assert text.startswith("1 张")


class TestExtractJson:
    def test_plain_object(self):
        assert planner.extract_json('{"prompts": []}') == {"prompts": []}

    def test_fenced_block(self):
        text = '好的：\n```json\n{"prompts": [{"prompt": "a"}]}\n```\n完成'
        assert planner.extract_json(text) == {"prompts": [{"prompt": "a"}]}

    def test_object_wrapped_in_prose(self):
        text = '这是结果 {"prompts": [{"prompt": "a"}]} 谢谢'
        assert planner.extract_json(text) == {"prompts": [{"prompt": "a"}]}

    def test_braces_inside_strings_do_not_end_the_block(self):
        text = '前言 {"prompts": [{"prompt": "a {b} c"}]} 后记'
        assert planner.extract_json(text) == {"prompts": [{"prompt": "a {b} c"}]}

    @pytest.mark.parametrize("text", ["", "   ", "没有 JSON", '{"prompts": ['])
    def test_unusable_replies_return_none(self, text):
        assert planner.extract_json(text) is None


class TestParsePlan:
    def test_reads_the_documented_shape(self):
        reply = (
            '{"prompts": ['
            '{"prompt": "雨夜的霓虹街头", "aspect_ratio": "16:9"},'
            '{"prompt": "雨夜的霓虹街头特写", "aspect_ratio": "3:4"}'
            "]}"
        )
        items = planner.parse_plan(reply, count=4, ratios=["1:1", "3:4", "16:9"])
        assert [item.prompt for item in items] == ["雨夜的霓虹街头", "雨夜的霓虹街头特写"]
        assert [item.aspect_ratio for item in items] == ["16:9", "3:4"]

    def test_a_bare_list_works(self):
        items = planner.parse_plan('["第一张", "第二张"]', count=4, ratios=["1:1"])
        assert [item.prompt for item in items] == ["第一张", "第二张"]
        assert all(item.aspect_ratio == "" for item in items)

    def test_alternative_keys_are_understood(self):
        reply = '{"items": [{"text": "甲", "ratio": "4:3"}, {"caption": "乙"}]}'
        items = planner.parse_plan(reply, count=4, ratios=["4:3"])
        assert [(item.prompt, item.aspect_ratio) for item in items] == [("甲", "4:3"), ("乙", "")]

    def test_a_single_object_is_one_prompt(self):
        items = planner.parse_plan('{"prompt": "只有一张"}', count=4, ratios=["1:1"])
        assert [item.prompt for item in items] == ["只有一张"]

    def test_ratio_outside_the_pool_is_dropped(self):
        reply = '{"prompts": [{"prompt": "a", "aspect_ratio": "21:9"}]}'
        items = planner.parse_plan(reply, count=4, ratios=["1:1", "16:9"])
        assert items[0].aspect_ratio == ""

    def test_duplicates_are_collapsed(self):
        reply = '{"prompts": [{"prompt": "同一张"}, {"prompt": " 同一张 "}, {"prompt": "另一张"}]}'
        items = planner.parse_plan(reply, count=4, ratios=["1:1"])
        assert [item.prompt for item in items] == ["同一张", "另一张"]

    def test_the_count_is_a_hard_cap(self):
        reply = '{"prompts": [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}]}'
        items = planner.parse_plan(reply, count=2, ratios=["1:1"])
        assert [item.prompt for item in items] == ["a", "b"]

    def test_entries_without_text_are_skipped(self):
        reply = '{"prompts": [{"aspect_ratio": "1:1"}, {"prompt": "有字"}]}'
        items = planner.parse_plan(reply, count=4, ratios=["1:1"])
        assert [item.prompt for item in items] == ["有字"]

    def test_an_empty_plan_raises(self):
        with pytest.raises(ValueError) as excinfo:
            planner.parse_plan('{"prompts": []}', count=4, ratios=["1:1"])
        assert "批量规划没有返回可用的提示词" in str(excinfo.value)


class TestSpreadRatios:
    def test_missing_frames_are_filled_with_unused_ones(self):
        items = [planner.PlannedPrompt("a"), planner.PlannedPrompt("b")]
        result = planner.spread_ratios(items, ["1:1", "3:4", "16:9"])
        assert [item.aspect_ratio for item in result] == ["1:1", "3:4"]

    def test_a_requested_frame_is_kept(self):
        items = [planner.PlannedPrompt("a", "3:4")]
        result = planner.spread_ratios(items, ["1:1", "3:4"])
        assert result[0].aspect_ratio == "3:4"

    def test_a_uniform_plan_is_forced_to_vary(self):
        items = [planner.PlannedPrompt("a", "1:1"), planner.PlannedPrompt("b", "1:1")]
        result = planner.spread_ratios(items, ["1:1", "16:9"])
        assert [item.aspect_ratio for item in result] == ["1:1", "16:9"]

    def test_unknown_frames_are_replaced(self):
        items = [planner.PlannedPrompt("a", "21:9")]
        result = planner.spread_ratios(items, ["9:16"])
        assert result[0].aspect_ratio == "9:16"

    def test_empty_input_is_returned_as_is(self):
        assert planner.spread_ratios([], ["1:1"]) == []


class TestPlan:
    def test_end_to_end(self):
        reply = '{"prompts": [{"prompt": "远山", "aspect_ratio": "1:1"}, {"prompt": "近景"}]}'
        items = planner.plan(reply, count=2, ratios=["1:1", "16:9"])
        assert [(item.prompt, item.aspect_ratio) for item in items] == [
            ("远山", "1:1"),
            ("近景", "16:9"),
        ]

    def test_describe_counts_and_frames(self):
        items = [
            planner.PlannedPrompt("a", "1:1"),
            planner.PlannedPrompt("b", "3:4"),
            planner.PlannedPrompt("c", ""),
        ]
        assert planner.describe(items) == "3 条（1:1、3:4、默认）"
        assert planner.describe([]) == "0 条"
