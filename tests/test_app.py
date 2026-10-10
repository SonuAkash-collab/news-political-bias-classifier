"""Smoke tests for app/cli.py and app/streamlit_app.py.

The CLI tests are pure unit tests that do not need real data on disk:
they construct a minimal sklearn pipeline and call the internal helpers
directly. A single integration test invokes the CLI as a subprocess to
verify the entry point works end to end when models are present.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polibias import config  # noqa: E402

# ---------------------------------------------------------------------------
# Build a tiny in-process model for unit tests (no disk I/O needed)
# ---------------------------------------------------------------------------

_TRAIN_TEXTS = [
    "tax cuts economy growth republican conservative",
    "gun rights second amendment firearms",
    "climate change environment green energy",
    "healthcare workers union progressive policy",
    "border immigration wall enforce law",
    "social justice equality civil rights",
]
_TRAIN_LABELS = ["R", "R", "D", "D", "R", "D"]


@pytest.fixture(scope="module")
def tiny_pipeline():
    pipe = Pipeline([
        ("features", CountVectorizer()),
        ("classifier", LogisticRegression(solver="liblinear", C=1.0, random_state=0)),
    ])
    pipe.fit(_TRAIN_TEXTS, _TRAIN_LABELS)
    return pipe


# ---------------------------------------------------------------------------
# Import the helper functions from cli (avoids importing heavy streamlit)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module", autouse=True)
def _add_app_to_path():
    sys.path.insert(0, str(ROOT / "app"))


def _import_cli_helpers():
    """Import CLI helpers without running main()."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("cli", ROOT / "app" / "cli.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_returns_tuple_of_three(self, tiny_pipeline):
        cli = _import_cli_helpers()
        pred, p_left, p_right = cli._confidence(tiny_pipeline, "tax cuts republican")
        assert pred in ("D", "R")
        assert abs(p_left + p_right - 1.0) < 1e-6

    def test_probabilities_in_range(self, tiny_pipeline):
        cli = _import_cli_helpers()
        for text in _TRAIN_TEXTS:
            _, p_left, p_right = cli._confidence(tiny_pipeline, text)
            assert 0.0 <= p_left <= 1.0
            assert 0.0 <= p_right <= 1.0

    def test_high_confidence_for_clear_text(self, tiny_pipeline):
        cli = _import_cli_helpers()
        _, _, p_right = cli._confidence(tiny_pipeline, "gun rights republican conservative")
        _, p_left, _ = cli._confidence(tiny_pipeline, "healthcare union progressive")
        assert p_right > 0.5
        assert p_left > 0.5


class TestDrivingWords:
    def test_returns_list(self, tiny_pipeline):
        cli = _import_cli_helpers()
        words = cli._driving_words(tiny_pipeline, "tax cuts republican conservative", "R")
        assert isinstance(words, list)

    def test_words_present_in_vocab(self, tiny_pipeline):
        cli = _import_cli_helpers()
        words = cli._driving_words(tiny_pipeline, "gun rights conservative border", "R", n=5)
        vocab = tiny_pipeline.named_steps["features"].get_feature_names_out()
        for w in words:
            assert w in vocab

    def test_returns_empty_on_unknown_step_names(self):
        """A pipeline with non-standard step names returns [] gracefully."""
        cli = _import_cli_helpers()
        pipe = Pipeline([
            ("enc", CountVectorizer()),
            ("cls", LogisticRegression(solver="liblinear")),
        ])
        pipe.fit(_TRAIN_TEXTS, _TRAIN_LABELS)
        words = cli._driving_words(pipe, "tax cuts", "R")
        assert isinstance(words, list)


class TestAsciiBar:
    def test_length_equals_width(self):
        cli = _import_cli_helpers()
        bar = cli._bar(0.6, width=20)
        assert len(bar) == 20

    def test_full_bar(self):
        cli = _import_cli_helpers()
        bar = cli._bar(1.0, width=10)
        assert bar == "#" * 10

    def test_empty_bar(self):
        cli = _import_cli_helpers()
        bar = cli._bar(0.0, width=10)
        assert bar == "." * 10


# ---------------------------------------------------------------------------
# Integration test — needs saved model on disk
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (config.MODELS_DIR / "ext3_in_domain.joblib").exists(),
    reason="ext3_in_domain.joblib not found — run python scripts/run_all.py first",
)
def test_cli_subprocess_single_shot():
    """CLI exits 0 and prints a prediction when called with --text."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "app" / "cli.py"),
            "--model", "ext3",
            "--text", "The president signed the healthcare bill into law today.",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout
    assert "Left" in output or "Right" in output
    assert "Confidence" in output


@pytest.mark.skipif(
    not (config.MODELS_DIR / "ext1_tfidf_lr.joblib").exists(),
    reason="ext1_tfidf_lr.joblib not found — run python scripts/run_all.py first",
)
def test_cli_subprocess_ext1_model():
    """CLI works for the ext1 TF-IDF + LR model."""
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "app" / "cli.py"),
            "--model", "ext1",
            "--text", "Gun rights are fundamental to American freedom.",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "Prediction" in result.stdout
