"""Smoke tests for Extension 2 (chunk voting) and Extension 3 (in-domain) on tiny fake data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polibias.experiments.ext2_chunk_vote import (
    chunk_text,
    predict_by_chunk_voting,
    run_ext2,
)
from polibias.experiments.ext3_in_domain import run_ext3
from polibias.experiments.ext_comparison import generate_comparison_figure


# ---------------------------------------------------------------------------
# Tiny fake data
# ---------------------------------------------------------------------------

_TWEET_D = [
    "healthcare workers deserve support and funding",
    "voting rights must be protected for all citizens",
    "climate change is a real and urgent crisis",
]
_TWEET_R = [
    "lower taxes grow the economy and create jobs",
    "second amendment rights shall not be infringed",
    "border security is a national priority protect america",
]

_ART_LEFT = [
    "The senator proposed expanding healthcare access. New climate legislation aims to reduce emissions.",
    "Voting access should be improved for all citizens. Public schools need federal support immediately.",
]
_ART_RIGHT = [
    "Tax cuts signed into law boosted growth. Small business owners praised the lower corporate rate.",
    "Border enforcement operations secured crossings. The governor defended second amendment gun owner rights.",
]
_ART_CENTER = [
    "The bipartisan committee met today to discuss energy policy and fiscal budget deadlines.",
]


def _make_fake_tweets() -> pd.DataFrame:
    texts = _TWEET_D * 10 + _TWEET_R * 10
    parties = ["D"] * (len(_TWEET_D) * 10) + ["R"] * (len(_TWEET_R) * 10)
    members = [f"m_d_{i % 3}" for i in range(len(_TWEET_D) * 10)] + [
        f"m_r_{i % 3}" for i in range(len(_TWEET_R) * 10)
    ]
    return pd.DataFrame({"text": texts, "party": parties, "member_id": members})


def _make_fake_articles() -> pd.DataFrame:
    records = []
    # Train
    for i, t in enumerate(_ART_LEFT * 6):
        records.append({"id": f"L_tr_{i}", "source": "S1", "content": t, "bias_text": "left", "split": "train", "topic": "pol"})
    for i, t in enumerate(_ART_RIGHT * 6):
        records.append({"id": f"R_tr_{i}", "source": "S2", "content": t, "bias_text": "right", "split": "train", "topic": "pol"})
    for i, t in enumerate(_ART_CENTER * 6):
        records.append({"id": f"C_tr_{i}", "source": "S3", "content": t, "bias_text": "center", "split": "train", "topic": "pol"})

    # Validation
    for i, t in enumerate(_ART_LEFT * 2):
        records.append({"id": f"L_val_{i}", "source": "S4", "content": t, "bias_text": "left", "split": "validation", "topic": "pol"})
    for i, t in enumerate(_ART_RIGHT * 2):
        records.append({"id": f"R_val_{i}", "source": "S5", "content": t, "bias_text": "right", "split": "validation", "topic": "pol"})

    # Test
    for i, t in enumerate(_ART_LEFT * 2):
        records.append({"id": f"L_te_{i}", "source": "S6", "content": t, "bias_text": "left", "split": "test", "topic": "pol"})
    for i, t in enumerate(_ART_RIGHT * 2):
        records.append({"id": f"R_te_{i}", "source": "S7", "content": t, "bias_text": "right", "split": "test", "topic": "pol"})
    for i, t in enumerate(_ART_CENTER * 2):
        records.append({"id": f"C_te_{i}", "source": "S8", "content": t, "bias_text": "center", "split": "test", "topic": "pol"})

    df = pd.DataFrame(records)
    df["source_url"] = ""
    df["date"] = "2021-02-01"
    df["title"] = "Test Title"
    return df


# ---------------------------------------------------------------------------
# Unit & Smoke Tests
# ---------------------------------------------------------------------------

def test_chunk_text_basic():
    text = "Sentence one is brief. Sentence two adds more words! Sentence three closes this test."
    chunks = chunk_text(text, target_words=10)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_single_long_sentence():
    text = "word " * 100
    chunks = chunk_text(text, target_words=20)
    assert len(chunks) >= 4


def test_run_ext2_smoke(tmp_path: Path):
    tweets = _make_fake_tweets()
    articles = _make_fake_articles()
    res_path = tmp_path / "ext2.json"
    fig_dir = tmp_path / "figures"

    res = run_ext2(
        tweets,
        articles,
        candidate_chunk_sizes=(10, 20),
        results_path=res_path,
        figure_dir=fig_dir,
    )

    assert res_path.exists()
    assert "test_results" in res
    assert "linear_svc" in res["test_results"]
    assert "chunk_voting" in res["test_results"]["linear_svc"]
    assert res["validation_tuning"]["chosen_chunk_size"] in (10, 20)


def test_run_ext3_smoke(tmp_path: Path):
    articles = _make_fake_articles()
    res_path = tmp_path / "ext3.json"
    model_path = tmp_path / "ext3_in_domain.joblib"
    fig_dir = tmp_path / "figures"

    res = run_ext3(
        articles,
        results_path=res_path,
        model_save_path=model_path,
        figure_dir=fig_dir,
    )

    assert res_path.exists()
    assert model_path.exists()
    # Check model size is well below 20 MB
    assert model_path.stat().st_size < 20 * 1024 * 1024
    assert "binary_task" in res
    assert "three_class_task" in res
    assert "scores" in res["binary_task"]
    assert "scores" in res["three_class_task"]


def test_generate_comparison_figure_smoke(tmp_path: Path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    fig_path = tmp_path / "figures" / "comparison.png"

    # Write dummy exp3.json and ext2.json
    (results_dir / "exp3.json").write_text(
        json.dumps({
            "scores": {
                "linear_svc_tuned_c0.1": {
                    "accuracy": 0.584,
                    "accuracy_ci": [0.553, 0.614]
                }
            }
        })
    )
    (results_dir / "ext2.json").write_text(
        json.dumps({
            "test_results": {
                "linear_svc": {
                    "chunk_voting": {
                        "chunk_size": 30,
                        "accuracy": 0.605,
                        "accuracy_ci": [0.57, 0.63]
                    }
                }
            }
        })
    )
    (results_dir / "ext3.json").write_text(
        json.dumps({
            "binary_task": {
                "best_model": "linear_svc",
                "scores": {
                    "linear_svc": {
                        "accuracy": 0.725,
                        "accuracy_ci": [0.69, 0.75]
                    }
                }
            }
        })
    )

    out = generate_comparison_figure(results_dir=results_dir, output_path=fig_path)
    assert out.exists()
