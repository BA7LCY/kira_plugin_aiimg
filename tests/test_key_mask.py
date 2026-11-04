"""API keys may leave the config endpoint only in masked form."""

from __future__ import annotations

from core.settings import mask_key, mask_providers, restore_keys, unmask_key

KEY = "sk-09ba1234567890abcdef1234567890abcdef3eb7"


def test_mask_key_hides_the_middle():
    masked = mask_key(KEY)

    assert masked.startswith("sk-0")
    assert masked.endswith("3eb7")
    assert KEY not in masked
    assert "…" in masked


def test_mask_key_handles_short_and_empty_values():
    assert mask_key("") == ""
    assert mask_key("abcd") == "…"
    assert mask_key("abcdefgh") == "…"


def test_mask_providers_leaves_the_rest_of_the_config_alone():
    config = {
        "providers": [
            {"id": "p1", "api_keys": [KEY, ""], "model": "gpt-image-2"},
        ],
        "section_queue": {"max_size": 8},
    }

    masked = mask_providers(config)

    assert masked["providers"][0]["api_keys"] == [mask_key(KEY), ""]
    assert masked["providers"][0]["model"] == "gpt-image-2"
    assert masked["section_queue"] == {"max_size": 8}
    # the source config keeps the real key
    assert config["providers"][0]["api_keys"][0] == KEY


def test_restore_keys_puts_back_an_unchanged_mask():
    current = {"providers": [{"id": "p1", "api_keys": [KEY]}]}
    incoming = {"providers": [{"id": "p1", "api_keys": [mask_key(KEY)]}]}

    restored = restore_keys(incoming, current)

    assert restored["providers"][0]["api_keys"] == [KEY]


def test_restore_keys_keeps_a_newly_typed_key():
    current = {"providers": [{"id": "p1", "api_keys": [KEY]}]}
    incoming = {"providers": [{"id": "p1", "api_keys": ["sk-new-key-value"]}]}

    restored = restore_keys(incoming, current)

    assert restored["providers"][0]["api_keys"] == ["sk-new-key-value"]


def test_restore_keys_matches_a_mask_that_moved_position():
    second = "sk-second-key-0000000000000000000000"
    current = {"providers": [{"id": "p1", "api_keys": [KEY, second]}]}
    incoming = {
        "providers": [{"id": "p1", "api_keys": [mask_key(second), mask_key(KEY)]}]
    }

    restored = restore_keys(incoming, current)

    assert restored["providers"][0]["api_keys"] == [second, KEY]


def test_restore_keys_leaves_unknown_providers_untouched():
    incoming = {"providers": [{"id": "fresh", "api_keys": [mask_key(KEY)]}]}

    restored = restore_keys(incoming, {"providers": []})

    assert restored["providers"][0]["api_keys"] == [mask_key(KEY)]


def test_unmask_key_maps_one_off_requests():
    config = {"providers": [{"id": "p1", "api_keys": [KEY]}]}

    assert unmask_key(mask_key(KEY), config) == KEY
    assert unmask_key("sk-typed-by-hand", config) == "sk-typed-by-hand"
    assert unmask_key("", config) == ""
