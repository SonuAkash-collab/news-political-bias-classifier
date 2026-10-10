"""Extension 3: in-domain training on AllSides news articles.

Measures the domain gap between social media text (tweets) and journalistic prose
(news articles).

Experiments:
1. Binary classification: left (D) vs right (R) on the media-based split.
   Models evaluated: Bernoulli NB, Multinomial NB, LinearSVC (tuned C=0.1),
   LinearSVC (balanced).
2. Three-class classification: left vs center vs right on the media-based split.
   Models evaluated: Multinomial NB, LinearSVC (tuned C=0.1).
3. Export the best binary in-domain model to models/ext3_in_domain.joblib
   (with size verified < 20 MB) for use in the live demo.
4. Save metrics to results/ext3.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from polibias import config
from polibias.data.articles import binary_articles
from polibias.evaluation import relative_path, save_figure, style_axes
from polibias.features import binary_vectorizer, count_vectorizer
from polibias.models.naive_bayes import BernoulliNB, MultinomialNB

RESULTS_PATH = config.RESULTS_DIR / "ext3.json"
MODEL_SAVE_PATH = config.MODELS_DIR / "ext3_in_domain.joblib"

TEXT_VARIANT = "no_punct"
MIN_DF = 2
C_TUNED = 0.10
MAX_MODEL_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB limit


def _wilson_ci(n_correct: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score 95% confidence interval."""
    if n == 0:
        return [0.0, 0.0]
    p = n_correct / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def run_ext3(
    articles: pd.DataFrame,
    *,
    results_path: Path = RESULTS_PATH,
    model_save_path: Path = MODEL_SAVE_PATH,
    figure_dir: Path = config.FIGURES_DIR,
    seed: int = config.SEED,
) -> dict[str, Any]:
    """Execute Extension 3: in-domain article training and testing.

    Args:
        articles: Processed articles DataFrame (all splits).
        results_path: Destination for ext3.json.
        model_save_path: Destination for best binary in-domain model.
        figure_dir: Directory for figures.
        seed: Random seed.
    """
    # -------------------------------------------------------------------------
    # 1. Binary task: left (D) vs right (R)
    # -------------------------------------------------------------------------
    bin_arts = binary_articles(articles)
    bin_train = bin_arts[bin_arts["split"] == "train"].dropna(subset=["content"]).reset_index(drop=True)
    bin_test = bin_arts[bin_arts["split"] == "test"].dropna(subset=["content"]).reset_index(drop=True)

    X_train_bin_texts = bin_train["content"].astype(str).tolist()
    y_train_bin = np.asarray(bin_train["party"].tolist())

    X_test_bin_texts = bin_test["content"].astype(str).tolist()
    y_test_bin = np.asarray(bin_test["party"].tolist())

    # Fit feature extractors on article training data
    vec_bin = binary_vectorizer(variant=TEXT_VARIANT, min_df=MIN_DF)
    X_tr_b = vec_bin.fit_transform(X_train_bin_texts)
    X_te_b = vec_bin.transform(X_test_bin_texts)

    vec_cnt = count_vectorizer(variant=TEXT_VARIANT, min_df=MIN_DF)
    X_tr_c = vec_cnt.fit_transform(X_train_bin_texts)
    X_te_c = vec_cnt.transform(X_test_bin_texts)

    # Models
    bnb = BernoulliNB().fit(X_tr_b, y_train_bin)
    mnb = MultinomialNB().fit(X_tr_c, y_train_bin)

    svc_tuned = LinearSVC(C=C_TUNED, random_state=seed, max_iter=10_000)
    svc_tuned.fit(X_tr_c, y_train_bin)

    svc_balanced = LinearSVC(C=C_TUNED, class_weight="balanced", random_state=seed, max_iter=10_000)
    svc_balanced.fit(X_tr_c, y_train_bin)

    binary_models = {
        "bernoulli_nb": (bnb, X_te_b, vec_bin),
        "multinomial_nb": (mnb, X_te_c, vec_cnt),
        "linear_svc": (svc_tuned, X_te_c, vec_cnt),
        "linear_svc_balanced": (svc_balanced, X_te_c, vec_cnt),
    }

    binary_scores: dict[str, dict[str, Any]] = {}
    best_bin_model_name = "linear_svc"
    best_bin_acc = -1.0

    majority_label = "R" if (y_test_bin == "R").sum() >= (y_test_bin == "D").sum() else "D"
    majority_acc = float((y_test_bin == majority_label).mean())

    for name, (model, X_te, vec) in binary_models.items():
        preds = model.predict(X_te)
        acc = float(accuracy_score(y_test_bin, preds))
        mf1 = float(f1_score(y_test_bin, preds, average="macro", zero_division=0))
        ci = _wilson_ci(int((y_test_bin == preds).sum()), len(y_test_bin))
        cm = confusion_matrix(y_test_bin, preds, labels=["D", "R"]).tolist()
        binary_scores[name] = {
            "accuracy": acc,
            "macro_f1": mf1,
            "accuracy_ci": ci,
            "accuracy_delta_vs_majority": acc - majority_acc,
            "confusion_matrix": cm,
        }
        if acc > best_bin_acc:
            best_bin_acc = acc
            best_bin_model_name = name

    # -------------------------------------------------------------------------
    # 2. Three-class task: left vs center vs right
    # -------------------------------------------------------------------------
    tri_train = articles[articles["split"] == "train"].dropna(subset=["content", "bias_text"]).reset_index(drop=True)
    tri_test = articles[articles["split"] == "test"].dropna(subset=["content", "bias_text"]).reset_index(drop=True)

    # Valid three classes
    valid_biases = {"left", "center", "right"}
    tri_train = tri_train[tri_train["bias_text"].isin(valid_biases)].reset_index(drop=True)
    tri_test = tri_test[tri_test["bias_text"].isin(valid_biases)].reset_index(drop=True)

    X_train_tri_texts = tri_train["content"].astype(str).tolist()
    y_train_tri = np.asarray(tri_train["bias_text"].tolist())

    X_test_tri_texts = tri_test["content"].astype(str).tolist()
    y_test_tri = np.asarray(tri_test["bias_text"].tolist())

    vec_cnt_tri = count_vectorizer(variant=TEXT_VARIANT, min_df=MIN_DF)
    X_tr_tri_c = vec_cnt_tri.fit_transform(X_train_tri_texts)
    X_te_tri_c = vec_cnt_tri.transform(X_test_tri_texts)

    mnb_tri = MultinomialNB().fit(X_tr_tri_c, y_train_tri)
    svc_tri = LinearSVC(C=C_TUNED, random_state=seed, max_iter=10_000).fit(X_tr_tri_c, y_train_tri)

    three_class_models = {
        "multinomial_nb": mnb_tri,
        "linear_svc": svc_tri,
    }

    three_class_scores: dict[str, dict[str, Any]] = {}
    tri_labels = ["left", "center", "right"]
    for name, model in three_class_models.items():
        preds = model.predict(X_te_tri_c)
        acc = float(accuracy_score(y_test_tri, preds))
        mf1 = float(f1_score(y_test_tri, preds, average="macro", zero_division=0))
        ci = _wilson_ci(int((y_test_tri == preds).sum()), len(y_test_tri))
        cm = confusion_matrix(y_test_tri, preds, labels=tri_labels).tolist()
        three_class_scores[name] = {
            "accuracy": acc,
            "macro_f1": mf1,
            "accuracy_ci": ci,
            "confusion_matrix": cm,
        }

    # -------------------------------------------------------------------------
    # 3. Save best binary model for the demo (under 20 MB)
    # -------------------------------------------------------------------------
    best_model_obj, _, best_vec = binary_models[best_bin_model_name]
    best_bundle = {
        "model_name": best_bin_model_name,
        "task": "binary_left_right",
        "classes": ["D", "R"],
        "vectorizer": best_vec,
        "classifier": best_model_obj,
    }

    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_bundle, model_save_path, compress=3)
    model_size_bytes = model_save_path.stat().st_size
    model_size_mb = model_size_bytes / (1024 * 1024)

    if model_size_bytes > MAX_MODEL_SIZE_BYTES:
        raise RuntimeError(
            f"Saved model exceeds 20 MB: {model_size_mb:.2f} MB at {model_save_path}"
        )

    # -------------------------------------------------------------------------
    # 4. Generate in-domain diagnostic figures
    # -------------------------------------------------------------------------
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figure_dir / "ext3_in_domain.png"
    _plot_ext3_results(binary_scores, three_class_scores, fig_path)

    # -------------------------------------------------------------------------
    # 5. Save results to results/ext3.json
    # -------------------------------------------------------------------------
    results_data = {
        "description": (
            "Extension 3: In-domain training on AllSides news articles (media-based split). "
            "Evaluates binary (left vs right) and 3-class (left vs center vs right) accuracy "
            "to quantify the cost of the tweet-to-article domain gap."
        ),
        "split_type": "media-based",
        "binary_task": {
            "train_samples": len(bin_train),
            "test_samples": len(bin_test),
            "scores": binary_scores,
            "best_model": best_bin_model_name,
            "best_accuracy": best_bin_acc,
        },
        "three_class_task": {
            "train_samples": len(tri_train),
            "test_samples": len(tri_test),
            "scores": three_class_scores,
        },
        "saved_model": {
            "path": relative_path(model_save_path),
            "model_name": best_bin_model_name,
            "size_mb": round(model_size_mb, 2),
            "size_bytes": model_size_bytes,
        },
        "figures": {
            "in_domain_summary": relative_path(fig_path)
        },
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results_data, indent=2) + "\n", encoding="utf-8")
    return results_data


def _plot_ext3_results(
    binary_scores: dict[str, dict[str, Any]],
    three_class_scores: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate summary chart for in-domain binary and 3-class performance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    fig.patch.set_facecolor(config.LIGHT)

    # Binary plot
    style_axes(ax1)
    b_names = list(binary_scores.keys())
    b_labels = [n.replace("_", "\n") for n in b_names]
    b_accs = [binary_scores[n]["accuracy"] for n in b_names]
    bars1 = ax1.bar(b_labels, b_accs, color=config.OCEAN_TEAL, width=0.5)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel("Accuracy", fontsize=10)
    ax1.set_title("In-Domain Binary (Left vs Right)", fontsize=11)
    ax1.axhline(0.5, color=config.MUTED, linestyle="--", linewidth=1.0, alpha=0.7)
    for b in bars1:
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.1%}", ha="center", fontsize=9)

    # Three-class plot
    style_axes(ax2)
    t_names = list(three_class_scores.keys())
    t_labels = [n.replace("_", "\n") for n in t_names]
    t_accs = [three_class_scores[n]["accuracy"] for n in t_names]
    bars2 = ax2.bar(t_labels, t_accs, color=config.GOLD_DARK, width=0.4, edgecolor=config.MIDNIGHT, linewidth=0.6)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("Accuracy", fontsize=10)
    ax2.set_title("In-Domain 3-Class (Left / Center / Right)", fontsize=11)
    ax2.axhline(0.333, color=config.MUTED, linestyle="--", linewidth=1.0, alpha=0.7, label="Chance (33.3%)")
    ax2.legend(frameon=False, loc="upper right")
    for b in bars2:
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.02, f"{h:.1%}", ha="center", fontsize=9)

    save_figure(fig, output_path)


def main() -> dict[str, Any]:
    from polibias.data.articles import load_processed as load_processed_articles
    from polibias.experiments.ext_comparison import generate_comparison_figure

    config.set_seed()
    articles = load_processed_articles()
    result = run_ext3(
        articles,
        results_path=config.RESULTS_DIR / "ext3.json",
        model_save_path=config.MODELS_DIR / "ext3_in_domain.joblib",
        figure_dir=config.FIGURES_DIR,
    )
    try:
        generate_comparison_figure(
            results_dir=config.RESULTS_DIR,
            output_path=config.FIGURES_DIR / "extension_comparison.png",
        )
    except Exception as exc:
        print(f"Note: Comparison figure generation skipped: {exc}")

    print(f"Saved {config.RESULTS_DIR / 'ext3.json'}, {config.MODELS_DIR / 'ext3_in_domain.joblib'}, and figures")
    return result


if __name__ == "__main__":
    main()
