"""Generate a unified benchmark figure comparing all models and extensions.

Compares:
1. Paper LinearSVC 63% (Crain & Erhan 2021)
2. Our Exp 3 baseline (tweet-trained LinearSVC tested on 1,001 AllSides articles)
3. Ext 1 (from results/ext1.json if present)
4. Ext 2 (Sentence-chunk voting)
5. Ext 3 (In-domain training on AllSides articles)

Uses the exact project palette from polibias.config:
- Background: config.LIGHT (#F2EFE6)
- Bars: config.OCEAN_TEAL, config.DEEP_TEAL, config.TURQUOISE, config.GOLD_DARK
- Annotations: config.MIDNIGHT, config.MUTED
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from polibias import config
from polibias.evaluation import save_figure, style_axes

FIGURE_PATH = config.FIGURES_DIR / "extension_comparison.png"


def generate_comparison_figure(
    results_dir: Path = config.RESULTS_DIR,
    output_path: Path = FIGURE_PATH,
) -> Path:
    """Read experiment results and generate the unified comparison figure.

    Comments and design choices explain why each method helps (or does not):
    - Paper LinearSVC (63%): trained on tweets, tested on 100 poll-labelled articles.
      Small test sample with high variance (95% CI: [53.2%, 71.8%]).
    - Exp 3 Baseline (58.4%): evaluated on 1,001 test articles from AllSides.
      Suffers from severe domain shift (vocabulary, style, length mismatch) and tweet class imbalance.
    - Ext 1 (if available): TF-IDF + Logistic Regression downweights omnipresent words and catches phrases.
    - Ext 2 (Chunk Voting): breaks 500+ word articles into ~30 word tweet-sized chunks,
      bringing the length and feature distribution closer to the tweet domain.
    - Ext 3 (In-Domain Training): eliminates the cross-domain gap completely by training directly
      on news articles, setting the true upper bound for bag-of-words models on this task.
    """
    exp3_file = results_dir / "exp3.json"
    ext1_file = results_dir / "ext1.json"
    ext2_file = results_dir / "ext2.json"
    ext3_file = results_dir / "ext3.json"

    categories = ["Paper LinearSVC\n(100 articles)"]
    accuracies = [0.630]
    ci_lows = [0.532]
    ci_highs = [0.718]
    bar_colors = [config.MUTED]

    # Exp 3 Baseline
    if exp3_file.exists():
        exp3_data = json.loads(exp3_file.read_text(encoding="utf-8"))
        best_svc_key = "linear_svc_tuned_c0.1"
        if best_svc_key in exp3_data.get("scores", {}):
            s = exp3_data["scores"][best_svc_key]
        else:
            s = list(exp3_data["scores"].values())[0]
        categories.append("Exp 3: Tweet LinearSVC\n(Whole Article)")
        accuracies.append(s["accuracy"])
        ci = s.get("accuracy_ci", [s["accuracy"], s["accuracy"]])
        ci_lows.append(ci[0])
        ci_highs.append(ci[1])
        bar_colors.append(config.OCEAN_TEAL)

    # Ext 1 (if present)
    if ext1_file.exists():
        try:
            ext1_data = json.loads(ext1_file.read_text(encoding="utf-8"))
            acc1 = ext1_data.get("accuracy") or ext1_data.get("test_results", {}).get("accuracy")
            if acc1:
                categories.append("Ext 1: TF-IDF + LR\n(Tweet-trained)")
                accuracies.append(float(acc1))
                ci_lows.append(float(acc1))
                ci_highs.append(float(acc1))
                bar_colors.append(config.DEEP_TEAL)
        except Exception:
            pass

    # Ext 2 (Chunk Voting)
    if ext2_file.exists():
        ext2_data = json.loads(ext2_file.read_text(encoding="utf-8"))
        svc_chunk = ext2_data.get("test_results", {}).get("linear_svc", {}).get("chunk_voting", {})
        if svc_chunk:
            sz = svc_chunk.get("chunk_size", 30)
            categories.append(f"Ext 2: Chunk Voting\n({sz}-word chunks)")
            accuracies.append(svc_chunk["accuracy"])
            ci = svc_chunk.get("accuracy_ci", [svc_chunk["accuracy"], svc_chunk["accuracy"]])
            ci_lows.append(ci[0])
            ci_highs.append(ci[1])
            bar_colors.append(config.TURQUOISE)

    # Ext 3 (In-Domain Training)
    if ext3_file.exists():
        ext3_data = json.loads(ext3_file.read_text(encoding="utf-8"))
        best_in_domain = ext3_data.get("binary_task", {})
        best_m = best_in_domain.get("best_model", "linear_svc")
        s3 = best_in_domain.get("scores", {}).get(best_m, {})
        if s3:
            categories.append("Ext 3: In-Domain Article\n(LinearSVC)")
            accuracies.append(s3["accuracy"])
            ci = s3.get("accuracy_ci", [s3["accuracy"], s3["accuracy"]])
            ci_lows.append(ci[0])
            ci_highs.append(ci[1])
            bar_colors.append(config.GOLD_DARK)

    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    fig.patch.set_facecolor(config.LIGHT)
    style_axes(ax)

    x = np.arange(len(categories))
    yerr = [
        [acc - lo for acc, lo in zip(accuracies, ci_lows)],
        [hi - acc for acc, hi in zip(accuracies, ci_highs)],
    ]

    bars = ax.bar(
        x,
        accuracies,
        width=0.52,
        color=bar_colors,
        yerr=yerr,
        capsize=4,
        error_kw={"ecolor": config.MIDNIGHT, "elinewidth": 1.2, "alpha": 0.8},
        edgecolor=config.MIDNIGHT,
        linewidth=0.6,
    )

    ax.set_ylabel("Binary Test Accuracy (Left vs Right)", fontsize=11)
    ax.set_title(
        "Political Bias Classification Across Approaches & Extensions",
        fontsize=13,
        pad=14,
        weight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5)
    ax.set_ylim(0, 0.95)
    ax.axhline(0.5, color=config.MUTED, linestyle="--", linewidth=1.1, alpha=0.7, label="Chance Baseline (50%)")

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.025,
            f"{h:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
            color=config.MIDNIGHT,
        )

    note = (
        "Why each helps / does not help:\n"
        "- Exp 3 vs Paper: 1,001 AllSides articles show paper's 63% on 100 articles had wide variance (95% CI [53%, 72%]).\n"
        "- Ext 2 (Chunking): Aligns ~30-word tweet length with article text; mitigates word-count distribution shifts.\n"
        "- Ext 3 (In-Domain): Exposes true domain cost. Training on articles eliminates social media vocabulary mismatch."
    )
    ax.text(
        0.02,
        0.04,
        note,
        transform=ax.transAxes,
        fontsize=8.2,
        color=config.MIDNIGHT,
        bbox=dict(boxstyle="round,pad=0.5", facecolor=config.LIGHT, edgecolor=config.MUTED, alpha=0.8),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, output_path)
    return output_path
