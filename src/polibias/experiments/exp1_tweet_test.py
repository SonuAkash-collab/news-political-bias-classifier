"""Experiment 1: accuracy on held-out tweets, with the paper's random split and our grouped split.

Paper (Crain and Erhan 2021, Section 4.1 and Figure 1): Bernoulli NB, Multinomial NB and LinearSVC
(C = 0.06) on a random 90/10 split of tweets, for the three vocabulary variants.

We train the same three models on two 80/20 splits of the same tweets (see polibias.evaluation):
    random   tweets shuffled individually, so every author is in train and in test (the paper's setup)
    grouped  whole congresspeople held out, so the test authors were never seen in training
The gap between the two is how much the random split inflates accuracy.

C = 0.06 is the paper's value, kept on purpose to reproduce the paper; it was tuned on the paper's
test tweets. Our own tuned LinearSVC is in polibias.experiments.model_zoo. No model files are saved.

Usage: python -m polibias.experiments.exp1_tweet_test
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.svm import LinearSVC

from polibias import config
from polibias.data.tweets import load_processed
from polibias.evaluation import (
    MODEL_COLORS,
    bar_style,
    grouped_split,
    majority_baseline,
    new_figure,
    random_split,
    save_figure,
    scores,
    split_summary,
    write_results,
)
from polibias.features import binary_vectorizer, count_vectorizer
from polibias.models.naive_bayes import BernoulliNB, MultinomialNB

RESULTS_PATH = config.RESULTS_DIR / "exp1.json"
FIGURE_PATH = config.FIGURES_DIR / "exp1_tweet_test.png"

VARIANTS = ("all_words", "no_punct", "no_markers")
VARIANT_LABELS = {"all_words": "All words", "no_punct": "No punctuation", "no_markers": "No punct., # or @"}
MODELS = ("bernoulli_nb", "multinomial_nb", "linear_svc")
MODEL_LABELS = {"bernoulli_nb": "Bernoulli NB", "multinomial_nb": "Multinomial NB", "linear_svc": "LinearSVC"}
SPLITS = {"random": random_split, "grouped": grouped_split}
SPLIT_LABELS = {"random": "Random split (paper's method)", "grouped": "Grouped by congressperson (ours)"}
PAPER_C = 0.06
# Paper, Section 4.1 (90/10 random split). Only Bernoulli NB is given as numbers; Figure 1 shows the rest.
PAPER_BERNOULLI_ACCURACY = {"all_words": 0.8596, "no_punct": 0.8674, "no_markers": 0.8206}


def make_models() -> dict:
    """Each model with the vectoriser it uses: word existence for Bernoulli NB, counts for the others."""
    return {
        "bernoulli_nb": (binary_vectorizer, BernoulliNB()),
        "multinomial_nb": (count_vectorizer, MultinomialNB()),
        "linear_svc": (count_vectorizer, LinearSVC(C=PAPER_C, random_state=config.SEED, max_iter=10_000)),
    }


def run_split(train, test, variant: str) -> dict:
    """Fit every model on train with one vocabulary variant and score it on test."""
    out = {}
    for name, (vectorizer, model) in make_models().items():
        vec = vectorizer(variant)  # min_df = 1: every training word, as in the paper
        X_train = vec.fit_transform(train["text"])
        model.fit(X_train, train["party"].to_numpy())
        predictions = model.predict(vec.transform(test["text"]))
        out[name] = scores(test["party"].to_numpy(), predictions)
    return out


def inflation(results: dict) -> dict:
    """Accuracy points (percentage points) that the random split adds over the grouped split."""
    points = {
        variant: {
            model: 100 * (results["random"][variant][model]["accuracy"] - results["grouped"][variant][model]["accuracy"])
            for model in MODELS
        }
        for variant in VARIANTS
    }
    values = [v for per_model in points.values() for v in per_model.values()]
    return {"points": points, "mean_points": float(np.mean(values)), "min_points": min(values), "max_points": max(values)}


def paper_comparison(results: dict) -> list[dict]:
    rows = []
    for variant in VARIANTS:
        rows.append(
            {
                "variant": variant,
                "paper_bernoulli_nb_random_90_10": PAPER_BERNOULLI_ACCURACY[variant],
                "ours_bernoulli_nb_random_80_20": results["random"][variant]["bernoulli_nb"]["accuracy"],
                "ours_bernoulli_nb_grouped_80_20": results["grouped"][variant]["bernoulli_nb"]["accuracy"],
            }
        )
    return rows


def plot(results: dict, baselines: dict, path) -> str:
    """Grouped bars like the paper's Figure 1, one panel per split, accuracy in percent."""
    fig, axes = new_figure(width=13, height=5.6, ncols=2, sharey=True)
    x = np.arange(len(VARIANTS))
    width = 0.26
    for ax, split in zip(axes, SPLITS):
        for i, model in enumerate(MODELS):
            values = [100 * results[split][v][model]["accuracy"] for v in VARIANTS]
            bars = ax.bar(x + (i - 1) * width, values, width * 0.92, label=MODEL_LABELS[model], **bar_style(MODEL_COLORS[model]))
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8.5, color=config.MIDNIGHT)
        base = 100 * baselines[split]["accuracy"]
        ax.axhline(base, color=config.MUTED, linestyle="--", linewidth=1.2, label="Always predict D (majority baseline)")
        ax.set_xticks(x, [VARIANT_LABELS[v] for v in VARIANTS])
        ax.set_title(f"{SPLIT_LABELS[split]}\nalways-D baseline {base:.1f}%", fontsize=12, fontfamily="serif", pad=10)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Test accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    fig.suptitle("Held-out tweet accuracy: random vs grouped 80/20 split", fontsize=14, fontfamily="serif", color=config.MIDNIGHT)
    return save_figure(fig, path)


def main() -> dict:
    config.set_seed()
    tweets = load_processed()
    results, baselines, split_info = {}, {}, {}
    for split, split_fn in SPLITS.items():
        train, test = split_fn(tweets)
        split_info[split] = split_summary(train, test)
        baselines[split] = majority_baseline(train["party"], test["party"])
        results[split] = {}
        for variant in VARIANTS:
            start = time.perf_counter()
            results[split][variant] = run_split(train, test, variant)
            line = "  ".join(f"{MODEL_LABELS[m]} {100 * results[split][variant][m]['accuracy']:.2f}" for m in MODELS)
            print(f"{split:8} {variant:11} {line}  ({time.perf_counter() - start:.0f}s)")

    gap = inflation(results)
    figure = plot(results, baselines, FIGURE_PATH)
    summary = (
        f"Across 3 models and 3 vocabulary variants, the random split gives "
        f"{gap['mean_points']:.1f} more accuracy points on average than the grouped split "
        f"(range {gap['min_points']:.1f} to {gap['max_points']:.1f})."
    )
    result = {
        "description": "Paper Experiment 1 on held-out tweets, random vs member-grouped 80/20 split",
        "data": {"tweets": len(tweets), "members": int(tweets["member_id"].nunique())},
        "settings": {
            "test_size": 0.2,
            "seed": config.SEED,
            "min_df": 1,
            "linear_svc_C": PAPER_C,
            "linear_svc_C_note": "paper's value, chosen by the paper on its test tweets; not re-tuned here",
            "features": {"bernoulli_nb": "binary", "multinomial_nb": "count", "linear_svc": "count"},
            "paper_split": "random 90/10",
        },
        "splits": split_info,
        "majority_baseline": baselines,
        "scores": results,
        "random_minus_grouped_accuracy": gap,
        "paper_comparison": paper_comparison(results),
        "summary": summary,
        "figure": figure,
    }
    write_results(result, RESULTS_PATH)

    print("\nPaper vs ours, Bernoulli NB accuracy (%)")
    print("| Variant | Paper (random 90/10) | Ours (random 80/20) | Ours (grouped 80/20) |")
    print("|---|---|---|---|")
    for row in result["paper_comparison"]:
        print(
            f"| {VARIANT_LABELS[row['variant']]} | {100 * row['paper_bernoulli_nb_random_90_10']:.2f} "
            f"| {100 * row['ours_bernoulli_nb_random_80_20']:.2f} | {100 * row['ours_bernoulli_nb_grouped_80_20']:.2f} |"
        )
    for split in SPLITS:
        print(f"Majority baseline ({split}): always {baselines[split]['always_predict']}, accuracy {100 * baselines[split]['accuracy']:.2f}")
    print(summary)
    print(f"Saved {RESULTS_PATH} and {FIGURE_PATH}")
    return result


if __name__ == "__main__":
    main()
