"""Tests for the three vocabulary variants."""

import pytest

from polibias.text import VARIANTS, all_words, get_variant, no_markers, no_punct

TWEET = "Pass the #AmericanRescuePlan now, @RepCasten! See https://www.HealthCare.gov. Don't wait."


def test_all_words_only_lowercases_and_splits():
    assert all_words(TWEET) == [
        "pass", "the", "#americanrescueplan", "now,", "@repcasten!",
        "see", "https://www.healthcare.gov.", "don't", "wait.",
    ]


def test_no_punct_deletes_punctuation_but_keeps_marker_words():
    assert no_punct(TWEET) == [
        "pass", "the", "americanrescueplan", "now", "repcasten",
        "see", "httpswwwhealthcaregov", "dont", "wait",
    ]


def test_no_markers_drops_hashtags_mentions_and_urls():
    assert no_markers(TWEET) == ["pass", "the", "now", "see", "dont", "wait"]


def test_no_markers_keeps_email_addresses_and_handles_underscores():
    assert no_markers("Write to info@house.gov or @Rep_Smith #Vote_2022") == ["write", "to", "infohousegov", "or"]
    assert no_punct("@Rep_Smith") == ["repsmith"]


def test_whitespace_and_empty_text():
    for tokenize in VARIANTS.values():
        assert tokenize("") == []
        assert tokenize("  A\tb\nC  ") == ["a", "b", "c"]


def test_get_variant():
    assert get_variant("no_punct") is no_punct
    with pytest.raises(ValueError, match="Unknown vocabulary variant"):
        get_variant("stemmed")
