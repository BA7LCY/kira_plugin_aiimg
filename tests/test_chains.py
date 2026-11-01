"""Fallback chain resolution tests."""

import pytest

from core.chains import (
    ChainItem,
    DEFAULT_CHAIN_KEY,
    chain_for,
    explicit_chain,
    normalize_feature,
    parse_chain_item,
)


def test_parse_string_item():
    assert parse_chain_item("p1") == ChainItem("p1", "")


def test_parse_colon_item():
    assert parse_chain_item("p1:1024x1024") == ChainItem("p1", "1024x1024")


def test_parse_dict_item():
    assert parse_chain_item({"provider": "p1", "output": "1:1"}) == ChainItem("p1", "1:1")


def test_parse_invalid():
    with pytest.raises(ValueError):
        parse_chain_item("")
    with pytest.raises(ValueError):
        parse_chain_item({"output": "1:1"})


def test_normalize_feature_alias():
    assert normalize_feature("draw") == "generate_image"
    assert normalize_feature("generate_image") == "generate_image"
    assert normalize_feature("video") == "generate_video"


def test_chain_for_feature_key():
    chain = chain_for(
        {"generate_image": ["a", "b"]},
        "generate_image",
        available={"a", "b", "c"},
    )
    assert [c.provider_id for c in chain] == ["a", "b"]


def test_chain_for_default_key():
    chain = chain_for(
        {DEFAULT_CHAIN_KEY: ["a", "b"]},
        "generate_image",
        available={"a", "b"},
    )
    assert [c.provider_id for c in chain] == ["a", "b"]


def test_chain_for_drops_unknown():
    chain = chain_for(
        {"generate_image": ["a", "zzz", "b"]},
        "generate_image",
        available={"a", "b"},
    )
    assert [c.provider_id for c in chain] == ["a", "b"]


def test_chain_for_output_override_kept():
    chain = chain_for(
        {"generate_image": [{"provider": "a", "output": "1:1"}]},
        "generate_image",
        available={"a"},
    )
    assert chain[0].output == "1:1"


def test_chain_for_fallback_default_provider():
    chain = chain_for({}, "generate_image", available={"a", "b"}, default_provider="a")
    assert [c.provider_id for c in chain] == ["a"]


def test_chain_for_empty_when_no_match():
    assert chain_for({}, "generate_image", available={"a"}) == []


def test_explicit_chain():
    assert [c.provider_id for c in explicit_chain("a", available={"a", "b"})] == ["a"]
    assert explicit_chain("zzz", available={"a"}) == []
