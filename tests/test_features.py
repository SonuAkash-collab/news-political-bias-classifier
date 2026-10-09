"""Tests for the shared binary and count vectorisers."""

import pickle

import pytest

from polibias.features import binary_vectorizer, count_vectorizer

TEXTS = [
    "Vote vote VOTE! #VotingRights",
    "Cut taxes, cut spending.",
    "Vote to cut taxes @RepBob",
]


def test_count_and_binary_values():
    counts = count_vectorizer("no_punct")
    X = counts.fit_transform(TEXTS).toarray()
    vocab = list(counts.get_feature_names_out())
    assert vocab == ["cut", "repbob", "spending", "taxes", "to", "vote", "votingrights"]
    assert X[0, vocab.index("vote")] == 3
    assert X[1, vocab.index("cut")] == 2

    binary = binary_vectorizer("no_punct")
    B = binary.fit_transform(TEXTS).toarray()
    assert (B == (X > 0)).all()


def test_variant_changes_vocabulary():
    assert "#votingrights" in binary_vectorizer("all_words").fit(TEXTS).get_feature_names_out()
    vocab = binary_vectorizer("no_markers").fit(TEXTS).get_feature_names_out()
    assert "votingrights" not in vocab and "repbob" not in vocab


def test_min_df_drops_rare_words():
    vec = count_vectorizer("no_punct", min_df=2).fit(TEXTS)
    assert list(vec.get_feature_names_out()) == ["cut", "taxes", "vote"]


def test_unseen_words_are_ignored_and_vectoriser_pickles():
    vec = binary_vectorizer("no_punct").fit(TEXTS)
    restored = pickle.loads(pickle.dumps(vec))
    assert restored.transform(["brand new words"]).nnz == 0
    assert (restored.transform(TEXTS) != vec.transform(TEXTS)).nnz == 0


def test_unknown_variant():
    with pytest.raises(ValueError):
        count_vectorizer("stemmed")
