"""Smoke tests for exp3_articles.run_exp3 on tiny fake data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polibias.experiments.exp3_articles import (
    _class_counts,
    _wilson_ci,
    run_exp3,
)


# ---------------------------------------------------------------------------
# Tiny deterministic fake data helpers
# ---------------------------------------------------------------------------

_TWEET_TEXTS_D = [
    "healthcare workers deserve support and funding",
    "voting rights must be protected for all citizens",
    "climate change is a real and urgent crisis",
    "invest in public education not private profit",
    "we need comprehensive immigration reform now",
]
_TWEET_TEXTS_R = [
    "lower taxes grow the economy and create jobs",
    "second amendment rights shall not be infringed",
    "border security is a national priority protect america",
    "small businesses are the backbone of this country",
    "we must cut government spending and reduce debt",
]
_ARTICLE_TEXTS_LEFT = [
    "The senator proposed expanding healthcare access to low income communities.",
    "New climate legislation aims to reduce carbon emissions sharply.",
    "Voting access should be improved for all eligible citizens.",
]
_ARTICLE_TEXTS_RIGHT = [
    "Tax cuts signed into law boosted economic growth last quarter.",
    "Border enforcement operations reduced illegal crossings this month.",
    "The governor signed legislation protecting gun owner rights.",
]


def _make_tweets() -> pd.DataFrame:
    texts = _TWEET_TEXTS_D * 4 + _TWEET_TEXTS_R * 4  # 20 + 20
    parties = ["D"] * (len(_TWEET_TEXTS_D) * 4) + ["R"] * (len(_TWEET_TEXTS_R) * 4)
    # 5 distinct members per party so min_df=1 doesn't drop everything
    members = [f"m_d_{i % 5}" for i in range(len(_TWEET_TEXTS_D) * 4)] + [
        f"m_r_{i % 5}" for i in range(len(_TWEET_TEXTS_R) * 4)
    ]
    return pd.DataFrame({"text": texts, "party": parties, "member_id": members})


def _make_articles() -> pd.DataFrame:
    left = pd.DataFrame(
        {
            "id": [f"L{i}" for i in range(len(_ARTICLE_TEXTS_LEFT))],
            "source": ["SourceA", "SourceB", "SourceA"],
            "source_url": [""] * len(_ARTICLE_TEXTS_LEFT),
            "date": ["2021-02-01"] * len(_ARTICLE_TEXTS_LEFT),
            "title": [f"Left article {i}" for i in range(len(_ARTICLE_TEXTS_LEFT))],
            "content": _ARTICLE_TEXTS_LEFT,
            "bias_text": ["left"] * len(_ARTICLE_TEXTS_LEFT),
            "topic": ["healthcare", "climate", "voting"],
            "split": ["test"] * len(_ARTICLE_TEXTS_LEFT),
        }
    )
    right = pd.DataFrame(
        {
            "id": [f"R{i}" for i in range(len(_ARTICLE_TEXTS_RIGHT))],
            "source": ["SourceC", "SourceC", "SourceD"],
            "source_url": [""] * len(_ARTICLE_TEXTS_RIGHT),
            "date": ["2021-02-01"] * len(_ARTICLE_TEXTS_RIGHT),
            "title": [f"Right article {i}" for i in range(len(_ARTICLE_TEXTS_RIGHT))],
            "content": _ARTICLE_TEXTS_RIGHT,
            "bias_text": ["right"] * len(_ARTICLE_TEXTS_RIGHT),
            "topic": ["economy", "immigration", "guns"],
            "split": ["test"] * len(_ARTICLE_TEXTS_RIGHT),
        }
    )
    center = pd.DataFrame(
        {
            "id": ["C0"],
            "source": ["SourceE"],
            "source_url": [""],
            "date": ["2021-02-01"],
            "title": ["Center article 0"],
            "content": ["The report covered both sides of the debate."],
            "bias_text": ["center"],
            "topic": ["general"],
            "split": ["test"],
        }
    )
    return pd.concat([left, right, center], ignore_index=True)


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


def test_wilson_ci_basic():
    lo, hi = _wilson_ci(50, 100)
    assert 0.4 < lo < 0.5
    assert 0.5 < hi < 0.6
    assert lo < hi


def test_wilson_ci_all_correct():
    lo, hi = _wilson_ci(10, 10)
    assert lo > 0.7
    assert hi == 1.0


def test_wilson_ci_zero_n():
    lo, hi = _wilson_ci(0, 0)
    assert lo == 0.0 and hi == 0.0


def test_class_counts():
    counts = _class_counts(["D", "D", "R", "D"])
    assert counts == {"D": 3, "R": 1}


# ---------------------------------------------------------------------------
# Smoke test for run_exp3
# ---------------------------------------------------------------------------


def test_run_exp3_smoke(tmp_path):
    tweets = _make_tweets()
    articles = _make_articles()

    result = run_exp3(
        tweets,
        articles,
        results_path=tmp_path / "exp3.json",
        figure_dir=tmp_path / "figures",
        seed=42,
    )

    # result dict keys
    assert "scores" in result
    assert "article_test_set" in result
    assert "tweet_training_data" in result
    assert "paper_results" in result
    assert "error_analysis" in result
    assert "figures" in result

    # correct models present
    expected_models = {
        "majority_baseline",
        "bernoulli_nb",
        "multinomial_nb",
        "linear_svc_paper_c0.06",
        "linear_svc_tuned_c0.1",
        "linear_svc_balanced_c0.1",
    }
    assert set(result["scores"].keys()) == expected_models

    # every model has required score fields
    for name, s in result["scores"].items():
        assert "accuracy" in s, f"missing accuracy for {name}"
        assert "macro_f1" in s, f"missing macro_f1 for {name}"
        assert "accuracy_ci" in s, f"missing CI for {name}"
        lo, hi = s["accuracy_ci"]
        assert 0.0 <= lo <= hi <= 1.0, f"bad CI for {name}: {s['accuracy_ci']}"

    # tweet D fraction is noted
    assert 0 < result["tweet_training_data"]["fraction_D"] < 1

    # binary subset excludes center
    assert result["article_test_set"]["n_rows"] == 6  # 3 left + 3 right

    # results JSON written
    json_path = tmp_path / "exp3.json"
    assert json_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["scores"] == result["scores"]

    # figures created
    assert (tmp_path / "figures" / "exp3_comparison.png").exists()
    assert (tmp_path / "figures" / "exp3_confusion_matrix.png").exists()
    assert (tmp_path / "figures" / "exp3_accuracy_by_source.png").exists()
    assert (tmp_path / "figures" / "exp3_accuracy_by_topic.png").exists()

    # no backslashes in figure paths
    for fig_path in result["figures"].values():
        assert "\\" not in fig_path, f"backslash in figure path: {fig_path}"


def test_run_exp3_missing_tweet_columns(tmp_path):
    tweets = _make_tweets().drop(columns=["party"])
    articles = _make_articles()
    with pytest.raises(ValueError, match="missing columns"):
        run_exp3(tweets, articles, results_path=tmp_path / "exp3.json", figure_dir=tmp_path)


def test_run_exp3_no_test_articles(tmp_path):
    tweets = _make_tweets()
    articles = _make_articles()
    articles["split"] = "train"  # no test rows
    with pytest.raises(ValueError, match="No binary articles in the test split"):
        run_exp3(tweets, articles, results_path=tmp_path / "exp3.json", figure_dir=tmp_path)
