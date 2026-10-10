"""Tests for the shared splits, metrics and plot helpers."""

import json

import numpy as np
import pandas as pd
import pytest

from polibias import config
from polibias.evaluation import (
    accuracy,
    confusion_matrix,
    grouped_split,
    macro_f1,
    majority_baseline,
    new_figure,
    random_split,
    relative_path,
    save_figure,
    scores,
    split_summary,
    write_results,
)
from polibias.experiments import model_zoo


def toy_tweets(n_members: int = 40, per_member: int = 6, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    members = np.repeat([f"M{i:03d}" for i in range(n_members)], per_member)
    party = np.repeat(rng.choice(["D", "R"], size=n_members, p=[0.64, 0.36]), per_member)
    return pd.DataFrame({"text": [f"tweet {i}" for i in range(len(members))], "member_id": members, "party": party})


def test_grouped_split_keeps_members_apart():
    frame = toy_tweets()
    train, test = grouped_split(frame)
    assert set(train["member_id"]).isdisjoint(test["member_id"])
    assert len(train) + len(test) == len(frame)
    assert test["member_id"].nunique() == 8  # 20% of 40 members


def test_random_split_shares_members_and_has_80_20_rows():
    frame = toy_tweets()
    train, test = random_split(frame)
    assert len(test) == round(0.2 * len(frame))
    assert set(train["member_id"]) & set(test["member_id"])
    assert sorted(pd.concat([train, test])["text"]) == sorted(frame["text"])


def test_splits_are_reproducible():
    frame = toy_tweets()
    for split in (grouped_split, random_split):
        a, b = split(frame)[1], split(frame)[1]
        assert list(a["text"]) == list(b["text"])


def test_grouped_split_matches_model_zoo_on_toy_data():
    frame = toy_tweets()
    _, zoo_test_idx = model_zoo._grouped_split(frame)
    _, test = grouped_split(frame)
    assert set(test["member_id"]) == set(frame.iloc[zoo_test_idx]["member_id"])
    assert sorted(test["text"]) == sorted(frame.iloc[zoo_test_idx]["text"])


@pytest.mark.skipif(not config.TWEETS_CSV.exists(), reason="processed tweets not available")
def test_grouped_split_matches_model_zoo_on_real_tweets():
    tweets = pd.read_csv(config.TWEETS_CSV, usecols=["text", "party", "member_id"])
    _, zoo_test_idx = model_zoo._grouped_split(tweets)
    _, test = grouped_split(tweets)
    zoo_test = set(tweets.iloc[zoo_test_idx]["member_id"])
    assert set(test["member_id"]) == zoo_test
    assert len(zoo_test) == 90  # as recorded by model_zoo in results/results.json


def test_split_summary_counts_shared_members():
    frame = toy_tweets()
    assert split_summary(*grouped_split(frame))["shared_members"] == 0
    assert split_summary(*random_split(frame))["shared_members"] > 0


def test_metrics():
    y_true = ["D", "D", "D", "R"]
    y_pred = ["D", "D", "R", "R"]
    assert accuracy(y_true, y_pred) == 0.75
    # F1(D) = 2*2/(2*2+1) = 0.8, F1(R) = 2*1/(2*1+1) = 2/3
    assert macro_f1(y_true, y_pred) == pytest.approx((0.8 + 2 / 3) / 2)
    assert confusion_matrix(y_true, y_pred) == [[2, 1], [0, 1]]
    assert set(scores(y_true, y_pred)) == {"accuracy", "macro_f1", "confusion_matrix"}


def test_majority_baseline_predicts_training_majority():
    baseline = majority_baseline(["D", "D", "R"], ["D", "R", "R", "D", "D"])
    assert baseline["always_predict"] == "D"
    assert baseline["accuracy"] == 0.6
    assert baseline["macro_f1"] == pytest.approx(0.75 / 2)  # F1(D) = 0.75, F1(R) = 0
    assert baseline["confusion_matrix"] == [[3, 0], [2, 0]]


def test_results_and_figures_use_forward_slashes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    fig, ax = new_figure()
    ax.bar([0, 1], [1, 2])
    stored = save_figure(fig, tmp_path / "results" / "figures" / "toy.png")
    assert stored == "results/figures/toy.png"
    assert (tmp_path / stored).stat().st_size > 0
    assert relative_path(tmp_path / "a" / "b.json") == "a/b.json"

    write_results({"figure": stored}, tmp_path / "results" / "toy.json")
    assert json.loads((tmp_path / "results" / "toy.json").read_text(encoding="utf-8")) == {"figure": stored}
