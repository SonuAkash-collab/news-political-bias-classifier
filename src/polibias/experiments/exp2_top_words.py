"""Experiment 2: the words most indicative of each party, as in the paper's Table 1.

For every vocabulary variant, fit our Bernoulli NB on all tweets and rank words by the paper's log
ratio log p(word | D) / p(word | R) (top 10 for D) and its negative (top 10 for R). This describes
the training data; nothing here is evaluated, so using every tweet does not touch any test set.

Next to each word we store how many D and R tweets contain it and how many different members used
it. A word used by one or two members is a personal marker (a handle, a district name), not a party
signal, which is why the grouped split in Experiment 1 scores lower than the random one.

Usage: python -m polibias.experiments.exp2_top_words
"""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Patch

from polibias import config
from polibias.data.tweets import load_processed
from polibias.evaluation import bar_style, new_figure, save_figure, write_results
from polibias.experiments.exp1_tweet_test import VARIANT_LABELS, VARIANTS
from polibias.features import binary_vectorizer
from polibias.models.naive_bayes import BernoulliNB

RESULTS_PATH = config.RESULTS_DIR / "exp2.json"
FIGURE_PATH = config.FIGURES_DIR / "exp2_top_words.png"
TOP_N = 10


def top_words(tweets, variant: str, n: int = TOP_N) -> dict:
    """{party: [{word, log_ratio, d_tweets, r_tweets, members}, ...]} for one vocabulary variant."""
    vec = binary_vectorizer(variant)
    X = vec.fit_transform(tweets["text"]).tocsc()
    y = tweets["party"].to_numpy()
    model = BernoulliNB().fit(X, y)
    vocab = vec.get_feature_names_out()
    index = {word: j for j, word in enumerate(vocab)}
    members = tweets["member_id"].to_numpy()
    d_row, r_row = list(model.classes_).index("D"), list(model.classes_).index("R")

    out = {}
    for party, ranked in model.top_words(n, vocab).items():
        rows = []
        for word, ratio in ranked:
            j = index[word]
            docs = X[:, j].nonzero()[0]
            rows.append(
                {
                    "word": word,
                    "log_ratio": round(ratio, 4),
                    "d_tweets": int(model.feature_count_[d_row, j]),
                    "r_tweets": int(model.feature_count_[r_row, j]),
                    "members": int(len(np.unique(members[docs]))),
                }
            )
        out[str(party)] = rows
    return out


def plot(results: dict, path) -> str:
    """One panel per variant: top D words (turquoise) above top R words (gold), bar = log ratio."""
    fig, axes = new_figure(width=16, height=8.5, ncols=len(VARIANTS))
    top = max(row["log_ratio"] for variant in VARIANTS for party in ("D", "R") for row in results[variant][party])
    for ax, variant in zip(axes, VARIANTS):
        d_rows, r_rows = results[variant]["D"], results[variant]["R"]
        rows = [("D", row) for row in d_rows] + [("R", row) for row in r_rows]
        y = np.arange(len(rows))[::-1].astype(float)
        y[len(d_rows):] -= 0.8  # small gap between the two parties
        values = [row["log_ratio"] for _, row in rows]
        colors = [config.TURQUOISE if party == "D" else config.GOLD for party, _ in rows]
        for yi, value, color in zip(y, values, colors):
            ax.barh(yi, value, height=0.75, **bar_style(color))
        ax.set_yticks(y, [row["word"] for _, row in rows], fontsize=10.5)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, top * 1.45)  # same scale in every panel, room for the labels
        for yi, value, (_, row) in zip(y, values, rows):
            label = f"{value:.1f}  ({row['members']} member{'s' if row['members'] != 1 else ''})"
            ax.text(value + top * 0.02, yi, label, va="center", fontsize=8.5, color=config.MUTED)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=config.MUTED, alpha=0.25, linewidth=0.8)
        ax.set_xlabel("Log ratio (in brackets: members who used the word)", fontsize=9.5)
        ax.set_title(VARIANT_LABELS[variant], fontsize=13, fontfamily="serif", pad=8)
    handles = [
        Patch(**bar_style(config.TURQUOISE), label="Top 10 Democratic words: log p(w|D) / p(w|R)"),
        Patch(**bar_style(config.GOLD), label="Top 10 Republican words: log p(w|R) / p(w|D)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False, fontsize=10.5)
    fig.suptitle(
        "Most indicative words per party, Bernoulli NB on all tweets (paper Table 1)",
        fontsize=14,
        fontfamily="serif",
        color=config.MIDNIGHT,
    )
    return save_figure(fig, path)


def main() -> dict:
    config.set_seed()
    tweets = load_processed()
    results = {variant: top_words(tweets, variant) for variant in VARIANTS}
    figure = plot(results, FIGURE_PATH)
    result = {
        "description": "Paper Experiment 2 (Table 1): top words per party by Bernoulli NB log ratio",
        "data": {"tweets": len(tweets), "members": int(tweets["member_id"].nunique()), "trained_on": "all tweets"},
        "settings": {"top_n": TOP_N, "min_df": 1, "alpha": 1.0},
        "columns": {
            "log_ratio": "natural log of p(word | party) / p(word | other party)",
            "d_tweets": "D tweets containing the word",
            "r_tweets": "R tweets containing the word",
            "members": "different members who used the word",
        },
        "top_words": results,
        "figure": figure,
    }
    write_results(result, RESULTS_PATH)

    for variant in VARIANTS:
        print(f"\n{VARIANT_LABELS[variant]}")
        print(f"  {'Democratic':<28}{'Republican':<28}")
        for d, r in zip(results[variant]["D"], results[variant]["R"]):
            left = f"{d['word']} ({d['members']}m)"
            right = f"{r['word']} ({r['members']}m)"
            print(f"  {left:<28}{right:<28}")
    print(f"\nSaved {RESULTS_PATH} and {FIGURE_PATH}")
    return result


if __name__ == "__main__":
    main()
