"""Train/test splits, metrics and plotting helpers shared by every experiment.

    from polibias.evaluation import grouped_split, random_split, scores

    train, test = grouped_split(tweets)   # no congressperson in both parts
    train, test = random_split(tweets)    # tweets shuffled individually, as in the paper

Splits. Crain and Erhan (2021) used a random 90/10 split of tweets. We use 80/20 for both of our
splits so that the random and the grouped numbers are directly comparable:

    random_split   ShuffleSplit over rows, test_size 0.2, seed 42. Tweets by the same member land
                   in both train and test, as in the paper.
    grouped_split  GroupShuffleSplit with groups = member_id, test_size 0.2, seed 42. Each member
                   is entirely in train or entirely in test. These are exactly the settings of
                   polibias.experiments.model_zoo, so all grouped results use the same test members.

The difference between the two shows how much accuracy the paper's random split gains by letting
the model recognise people it has already seen.

Plot helpers take their colours from polibias.config, the only place the palette is defined.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix as _confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from polibias import config

TEST_SIZE = 0.20
LABELS = ("D", "R")


# Splits


def _check_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"Data is missing required columns: {sorted(missing)}")


def grouped_split(
    frame: pd.DataFrame, *, test_size: float = TEST_SIZE, seed: int = config.SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows so that every member_id is entirely in train or entirely in test.

    Same settings as polibias.experiments.model_zoo (GroupShuffleSplit by member_id, test_size 0.2,
    seed 42), so the test members are the same. Returns (train, test) with a fresh index.
    """
    _check_columns(frame, {"member_id"})
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(frame, groups=frame["member_id"]))
    train = frame.iloc[train_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    if set(train["member_id"]) & set(test["member_id"]):
        raise RuntimeError("Grouped split leaked a member across train and test")
    return train, test


def random_split(
    frame: pd.DataFrame, *, test_size: float = TEST_SIZE, seed: int = config.SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows at random, ignoring who wrote them (the paper's setup, at 80/20).

    Returns (train, test) with a fresh index.
    """
    splitter = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(frame))
    return frame.iloc[train_idx].reset_index(drop=True), frame.iloc[test_idx].reset_index(drop=True)


def split_summary(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Row, member and party counts of a split, for the results file."""
    summary = {"train_rows": len(train), "test_rows": len(test)}
    if "member_id" in train.columns:
        train_members, test_members = set(train["member_id"]), set(test["member_id"])
        summary.update(
            train_members=len(train_members),
            test_members=len(test_members),
            shared_members=len(train_members & test_members),
        )
    summary["test_party_counts"] = party_counts(test["party"])
    return summary


def party_counts(labels) -> dict[str, int]:
    counts = pd.Series(labels).value_counts()
    return {label: int(counts.get(label, 0)) for label in LABELS}


# Metrics


def accuracy(y_true, y_pred) -> float:
    return float(accuracy_score(y_true, y_pred))


def macro_f1(y_true, y_pred) -> float:
    """F1 averaged over D and R with equal weight, so the smaller class counts as much."""
    return float(f1_score(y_true, y_pred, average="macro", labels=list(LABELS), zero_division=0))


def confusion_matrix(y_true, y_pred) -> list[list[int]]:
    """Rows are the true class, columns the predicted class, both in the order D, R."""
    return _confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist()


def scores(y_true, y_pred) -> dict:
    """Accuracy, macro F1 and the confusion matrix in one dict."""
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def majority_baseline(y_train, y_test) -> dict:
    """Always predict the most common training label (D for the tweets) and score it on y_test."""
    label = pd.Series(y_train).value_counts().idxmax()
    predictions = np.full(len(y_test), label, dtype=object)
    return {"always_predict": str(label), **scores(np.asarray(y_test, dtype=object), predictions)}


# Results files


def write_results(result: dict, path: Path) -> None:
    """Write one experiment's results as JSON (one file per experiment avoids merge conflicts)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def relative_path(path: Path) -> str:
    """Path relative to the repository root with forward slashes, for storing in results."""
    return Path(path).resolve().relative_to(config.ROOT).as_posix()


# Plotting

MODEL_COLORS = {
    "bernoulli_nb": config.DEEP_TEAL,
    "multinomial_nb": config.TURQUOISE,
    "linear_svc": config.GOLD,
}


def new_figure(width: float = 10, height: float = 5.5, ncols: int = 1, **kwargs):
    """A figure on the light background with recessive axes. Returns (fig, axes)."""
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), constrained_layout=True, **kwargs)
    fig.patch.set_facecolor(config.LIGHT)
    for ax in np.atleast_1d(axes):
        style_axes(ax)
    return fig, axes


def style_axes(ax) -> None:
    """Light background, horizontal grid in muted grey, no top or right frame."""
    ax.set_facecolor(config.LIGHT)
    ax.grid(axis="y", color=config.MUTED, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(config.MUTED)
    ax.tick_params(colors=config.MIDNIGHT, labelsize=10)
    ax.title.set_color(config.MIDNIGHT)
    ax.xaxis.label.set_color(config.MIDNIGHT)
    ax.yaxis.label.set_color(config.MIDNIGHT)


def bar_style(color: str) -> dict:
    """Bar keyword arguments. Gold gets a thin dark outline so it stays visible on the light background."""
    style = {"color": color}
    if color in (config.GOLD, config.GOLD_DARK):
        style.update(edgecolor=config.MIDNIGHT, linewidth=0.6)
    return style


def save_figure(fig, path: Path, dpi: int = 180) -> str:
    """Save as PNG on the light background, close the figure, return the repository-relative path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    return relative_path(path)
