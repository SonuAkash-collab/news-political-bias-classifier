"""Grouped tweet evaluation for LinearSVC and comparison models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from polibias import config
from polibias.models.svm import (
    DEFAULT_C_GRID,
    evaluate_model_zoo,
    fit_text_model_zoo,
    tune_text_complement_nb,
    tune_text_linear_svc,
    tune_text_logistic_regression,
)

FEATURE_TYPE = "count"
TEXT_VARIANT = "no_punct"
MIN_DF = 2
TEST_SIZE = 0.20
CV_SPLITS = 5
RBF_SUBSAMPLE_SIZE = 5_000
CV_CACHE_DIR = config.DATA_DIR / "interim" / "model_zoo_cv_cache"


def _grouped_split(frame: pd.DataFrame, *, seed: int = config.SEED):
    """Split by member so no person's tweets occur in both partitions."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_idx, test_idx = next(
        splitter.split(frame["text"], frame["party"], groups=frame["member_id"])
    )
    train_members = set(frame.iloc[train_idx]["member_id"])
    test_members = set(frame.iloc[test_idx]["member_id"])
    if train_members & test_members:
        raise RuntimeError("Grouped holdout leaked a member across train and test")
    return train_idx, test_idx


def _class_counts(labels) -> dict[str, int]:
    counts = pd.Series(labels).value_counts().to_dict()
    return {str(label): int(counts.get(label, 0)) for label in ("D", "R")}


def _plot_scores(scores: dict, output: Path) -> None:
    names = list(scores)
    labels = [name.replace("_", "\n") for name in names]
    accuracy = [scores[name]["accuracy"] for name in names]
    macro_f1 = [scores[name]["macro_f1"] for name in names]
    baseline_accuracy = scores[names[0]]["majority_baseline_accuracy"]
    baseline_f1 = scores[names[0]]["majority_baseline_macro_f1"]
    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    fig.patch.set_facecolor(config.LIGHT)
    ax.set_facecolor(config.LIGHT)
    ax.bar(x - width / 2, accuracy, width, label="Accuracy", color=config.OCEAN_TEAL)
    ax.bar(x + width / 2, macro_f1, width, label="Macro F1", color=config.GOLD_DARK)
    ax.axhline(
        baseline_accuracy,
        color=config.OCEAN_TEAL,
        linestyle="--",
        linewidth=1.4,
        label=f"Always D accuracy ({baseline_accuracy:.3f})",
    )
    ax.axhline(
        baseline_f1,
        color=config.GOLD_DARK,
        linestyle=":",
        linewidth=1.6,
        label=f"Always D macro F1 ({baseline_f1:.3f})",
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(x, labels)
    ax.set_title("Grouped test performance by model")
    ax.grid(axis="y", color=config.MUTED, alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncols=2, loc="upper right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def run_model_zoo(
    tweets: pd.DataFrame,
    *,
    results_path: Path = config.RESULTS_JSON,
    figure_dir: Path = config.FIGURES_DIR,
    models_dir: Path = config.MODELS_DIR,
    seed: int = config.SEED,
    c_values: tuple[float, ...] = DEFAULT_C_GRID,
    rbf_max_samples: int = RBF_SUBSAMPLE_SIZE,
) -> dict[str, Any]:
    """Tune on member-grouped training folds, fit once, and score held-out members."""
    required = {"text", "party", "member_id"}
    missing = required - set(tweets.columns)
    if missing:
        raise ValueError(f"Tweet data is missing required columns: {sorted(missing)}")
    data = tweets.loc[:, ["text", "party", "member_id"]].dropna().copy()
    data["text"] = data["text"].astype(str)
    data["party"] = data["party"].astype(str)
    data = data[data["party"].isin(["D", "R"])].reset_index(drop=True)
    if set(data["party"].unique()) != {"D", "R"}:
        raise ValueError("Expected both D and R tweets")
    if data["member_id"].nunique() < CV_SPLITS + 1:
        raise ValueError("Not enough distinct members for grouped holdout and 5-fold CV")

    train_idx, test_idx = _grouped_split(data, seed=seed)
    train = data.iloc[train_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)
    train_texts, train_y, train_groups = train["text"], train["party"], train["member_id"]
    test_texts, test_y = test["text"], test["party"]

    # All choices are made with GroupKFold on train members only. The vectorizer
    # lives inside each Pipeline, so validation-fold text cannot shape its vocab.
    searches = {}
    for weighted in (None, "balanced"):
        name = "balanced" if weighted else "unweighted"
        print(f"Tuning LinearSVC ({name}) with 5-fold member GroupKFold...")
        searches[f"linear_svc_{name}"] = tune_text_linear_svc(
            train_texts,
            train_y,
            train_groups,
            feature_type=FEATURE_TYPE,
            variant=TEXT_VARIANT,
            min_df=MIN_DF,
            c_values=c_values,
            n_splits=CV_SPLITS,
            class_weight=weighted,
            seed=seed,
            cache_dir=str(CV_CACHE_DIR),
        )
        print(f"Tuning logistic regression ({name}) with 5-fold member GroupKFold...")
        searches[f"logistic_regression_{name}"] = tune_text_logistic_regression(
            train_texts,
            train_y,
            train_groups,
            feature_type=FEATURE_TYPE,
            variant=TEXT_VARIANT,
            min_df=MIN_DF,
            c_values=c_values,
            n_splits=CV_SPLITS,
            class_weight=weighted,
            seed=seed,
            cache_dir=str(CV_CACHE_DIR),
        )

    print("Tuning ComplementNB alpha with 5-fold member GroupKFold...")
    nb_search = tune_text_complement_nb(
        train_texts,
        train_y,
        train_groups,
        feature_type=FEATURE_TYPE,
        variant=TEXT_VARIANT,
        min_df=MIN_DF,
        n_splits=CV_SPLITS,
        cache_dir=str(CV_CACHE_DIR),
    )
    settings = {
        "linear_svc_c": searches["linear_svc_unweighted"].best_params_["classifier__C"],
        "linear_svc_balanced_c": searches["linear_svc_balanced"].best_params_["classifier__C"],
        "logistic_c": searches["logistic_regression_unweighted"].best_params_["classifier__C"],
        "logistic_balanced_c": searches["logistic_regression_balanced"].best_params_["classifier__C"],
        "complement_alpha": nb_search.best_params_["classifier__alpha"],
    }

    # Refit selected settings on the full training partition. The RBF SVM alone
    # uses a fixed, stratified 5k subsample from training data for tractability.
    print("Fitting the model zoo on all training members...")
    models = fit_text_model_zoo(
        train_texts,
        train_y,
        feature_type=FEATURE_TYPE,
        variant=TEXT_VARIANT,
        min_df=MIN_DF,
        seed=seed,
        rbf_max_samples=rbf_max_samples,
        **settings,
    )
    scores = evaluate_model_zoo(models, test_texts, test_y)

    models_dir.mkdir(parents=True, exist_ok=True)
    model_files = {}
    for name, model in models.items():
        path = models_dir / f"{name}.joblib"
        joblib.dump(model, path)
        model_files[name] = str(path.relative_to(config.ROOT))

    figure_path = figure_dir / "model_zoo.png"
    _plot_scores(scores, figure_path)
    baseline = {
        "always_predict": "D",
        "accuracy": scores["majority_baseline"]["majority_baseline_accuracy"],
        "macro_f1": scores["majority_baseline"]["majority_baseline_macro_f1"],
    }
    result = {
        "dataset": "congresstweets with House member party labels",
        "feature_type": FEATURE_TYPE,
        "text_variant": TEXT_VARIANT,
        "min_df": MIN_DF,
        "split": {
            "method": "GroupShuffleSplit",
            "group": "member_id",
            "test_size": TEST_SIZE,
            "seed": seed,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_members": int(train["member_id"].nunique()),
            "test_members": int(test["member_id"].nunique()),
            "train_party_counts": _class_counts(train_y),
            "test_party_counts": _class_counts(test_y),
            "full_party_counts": _class_counts(data["party"]),
        },
        "tuning": {
            "method": "GroupKFold",
            "group": "member_id",
            "folds": CV_SPLITS,
            "training_only": True,
            "selection_metric": "macro_f1",
            "C_grid": list(c_values),
            "selected": {name: float(value) for name, value in settings.items()},
            "paper_C": 0.06,
            "paper_test_set_tuning_leakage": True,
        },
        "majority_baseline": baseline,
        "scores": scores,
        "models": model_files,
        "figure": str(figure_path.relative_to(config.ROOT)),
        "rbf_svm_training_subsample": min(rbf_max_samples, len(train)),
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    all_results = {}
    if results_path.exists():
        all_results = json.loads(results_path.read_text(encoding="utf-8"))
    all_results["model_zoo"] = result
    results_path.write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
    return result
