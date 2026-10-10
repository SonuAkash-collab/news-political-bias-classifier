"""Extension 2: sentence-chunk voting for news articles.

Tweets are roughly 30 words, whereas news articles are hundreds of words long.
When tweet-trained bag-of-words models score a full news article at once, word counts
scale up dramatically, shifting feature distributions far outside the tweet training
domain.

In Extension 2, we:
1. Split each article into tweet-sized chunks (target size ~30 words, respecting sentence
   boundaries where possible).
2. Score every chunk individually with the tweet-trained models (Bernoulli NB, Multinomial NB,
   LinearSVC).
3. Average the class probabilities across chunks for each article. For LinearSVC, decision
   scores are converted to calibrated class probabilities via a sigmoid transform before averaging.
4. Tune the chunk size exclusively on the article VALIDATION split (evaluating candidate sizes
   such as 15, 30, 50, 100 words).
5. Compare chunk voting vs whole-article scoring on the article TEST split.
6. Write results to results/ext2.json and figures to results/figures/.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from polibias import config
from polibias.data.articles import binary_articles
from polibias.evaluation import relative_path, save_figure, style_axes
from polibias.features import binary_vectorizer, count_vectorizer
from polibias.models.naive_bayes import BernoulliNB, MultinomialNB

RESULTS_PATH = config.RESULTS_DIR / "ext2.json"

TEXT_VARIANT = "no_punct"
MIN_DF = 2
C_TUNED = 0.10
CANDIDATE_CHUNK_SIZES = (15, 30, 50, 100)


def chunk_text(text: str, target_words: int = 30) -> list[str]:
    """Split text into chunks of approximately target_words respecting sentence boundaries.

    Sentence boundaries are determined using common punctuation ([.!?]).
    If an individual sentence is exceedingly long (>= target_words * 2), it is word-sliced
    so chunks remain tweet-sized. Empty chunks are discarded.
    """
    if not text or not str(text).strip():
        return []
    raw_sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current_sentences: list[str] = []
    current_word_count = 0

    for sent in sentences:
        words = sent.split()
        sent_words = len(words)
        if sent_words == 0:
            continue

        if sent_words >= target_words * 2:
            if current_sentences:
                chunks.append(" ".join(current_sentences))
                current_sentences = []
                current_word_count = 0
            for i in range(0, len(words), target_words):
                chunk = " ".join(words[i : i + target_words])
                if chunk:
                    chunks.append(chunk)
            continue

        if current_word_count + sent_words > target_words and current_sentences:
            chunks.append(" ".join(current_sentences))
            current_sentences = [sent]
            current_word_count = sent_words
        else:
            current_sentences.append(sent)
            current_word_count += sent_words

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


def _predict_proba_svc(vec: CountVectorizer, clf: LinearSVC, texts: list[str]) -> np.ndarray:
    """Predict class probabilities [P(D), P(R)] for LinearSVC via sigmoid calibration."""
    if not texts:
        return np.empty((0, 2))
    X = vec.transform(texts)
    scores = clf.decision_function(X)
    # classes_ are sorted: ["D", "R"]. positive score -> "R"
    p_R = 1.0 / (1.0 + np.exp(-scores))
    p_D = 1.0 - p_R
    return np.column_stack([p_D, p_R])


def score_article_chunks(
    article_chunks: list[str],
    predict_proba_fn: Callable[[list[str]], np.ndarray],
    default_prob: np.ndarray,
) -> np.ndarray:
    """Score all chunks of a single article and return the averaged class probability vector."""
    if not article_chunks:
        return default_prob
    chunk_probs = predict_proba_fn(article_chunks)
    if len(chunk_probs) == 0:
        return default_prob
    return np.mean(chunk_probs, axis=0)


def predict_by_chunk_voting(
    articles_text: list[str],
    predict_proba_fn: Callable[[list[str]], np.ndarray],
    target_words: int,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Score articles via chunk voting.

    Returns:
        (predicted_labels, mean_probabilities)
    """
    default_prob = np.ones(len(classes)) / len(classes)
    article_probs = []

    for text in articles_text:
        chunks = chunk_text(text, target_words=target_words)
        mean_p = score_article_chunks(chunks, predict_proba_fn, default_prob)
        article_probs.append(mean_p)

    probs_arr = np.array(article_probs)
    preds = classes[np.argmax(probs_arr, axis=1)]
    return preds, probs_arr


def _wilson_ci(n_correct: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score 95% confidence interval."""
    if n == 0:
        return [0.0, 0.0]
    p = n_correct / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def run_ext2(
    tweets: pd.DataFrame,
    articles: pd.DataFrame,
    *,
    candidate_chunk_sizes: tuple[int, ...] = CANDIDATE_CHUNK_SIZES,
    results_path: Path = RESULTS_PATH,
    figure_dir: Path = config.FIGURES_DIR,
    seed: int = config.SEED,
) -> dict[str, Any]:
    """Execute Extension 2: sentence-chunk voting.

    Args:
        tweets: Processed tweet training dataframe with 'text', 'party'.
        articles: Processed articles dataframe with 'split', 'bias_text', 'content'.
        candidate_chunk_sizes: Chunk word sizes to evaluate on the validation set.
        results_path: Destination for ext2.json.
        figure_dir: Directory to save diagnostic plots.
        seed: Random seed.
    """
    # 1. Prepare tweet training data
    train = tweets[["text", "party"]].dropna().copy()
    train["text"] = train["text"].astype(str)
    train["party"] = train["party"].astype(str)
    train = train[train["party"].isin(["D", "R"])].reset_index(drop=True)

    train_texts = train["text"].tolist()
    train_y = np.asarray(train["party"].tolist())

    # 2. Prepare article validation & test data (binary left/right)
    bin_arts = binary_articles(articles)
    val_df = bin_arts[bin_arts["split"] == "validation"].reset_index(drop=True)
    test_df = bin_arts[bin_arts["split"] == "test"].reset_index(drop=True)

    val_texts = val_df["content"].fillna("").astype(str).tolist()
    val_y = np.asarray(val_df["party"].tolist())

    test_texts = test_df["content"].fillna("").astype(str).tolist()
    test_y = np.asarray(test_df["party"].tolist())

    # 3. Train models on tweets
    bin_vec = binary_vectorizer(variant=TEXT_VARIANT, min_df=MIN_DF)
    X_train_bin = bin_vec.fit_transform(train_texts)
    bnb = BernoulliNB().fit(X_train_bin, train_y)

    cnt_vec = count_vectorizer(variant=TEXT_VARIANT, min_df=MIN_DF)
    X_train_cnt = cnt_vec.fit_transform(train_texts)
    mnb = MultinomialNB().fit(X_train_cnt, train_y)

    svc = LinearSVC(C=C_TUNED, random_state=seed, max_iter=10_000)
    svc.fit(X_train_cnt, train_y)

    classes = np.array(["D", "R"])

    # Probability prediction closures for chunk scoring
    def bnb_proba(chunks: list[str]) -> np.ndarray:
        return bnb.predict_proba(bin_vec.transform(chunks))

    def mnb_proba(chunks: list[str]) -> np.ndarray:
        return mnb.predict_proba(cnt_vec.transform(chunks))

    def svc_proba(chunks: list[str]) -> np.ndarray:
        return _predict_proba_svc(cnt_vec, svc, chunks)

    models_to_eval = [
        ("bernoulli_nb", bnb_proba, lambda txts: bnb.predict(bin_vec.transform(txts))),
        ("multinomial_nb", mnb_proba, lambda txts: mnb.predict(cnt_vec.transform(txts))),
        ("linear_svc", svc_proba, lambda txts: svc.predict(cnt_vec.transform(txts))),
    ]

    # 4. Tune chunk size on the VALIDATION split only
    # Note: validation split has class imbalance (mostly Left in media split),
    # so we monitor both macro_f1 and accuracy. We use validation macro_f1
    # across models to pick the best chunk size.
    val_tuning_results: dict[str, dict[int, dict[str, float]]] = {}
    size_mean_f1: dict[int, float] = {sz: 0.0 for sz in candidate_chunk_sizes}

    for model_name, proba_fn, whole_pred_fn in models_to_eval:
        val_tuning_results[model_name] = {}
        for sz in candidate_chunk_sizes:
            preds, _ = predict_by_chunk_voting(val_texts, proba_fn, target_words=sz, classes=classes)
            acc = float(accuracy_score(val_y, preds))
            mf1 = float(f1_score(val_y, preds, average="macro", zero_division=0))
            val_tuning_results[model_name][sz] = {"accuracy": acc, "macro_f1": mf1}
            size_mean_f1[sz] += mf1 / len(models_to_eval)

    # Pick best chunk size based on validation set
    best_chunk_size = max(size_mean_f1.keys(), key=lambda sz: size_mean_f1[sz])

    # 5. Evaluate on TEST split: Whole Article vs Chunk Voting (at best_chunk_size and other sizes)
    test_results: dict[str, Any] = {}
    for model_name, proba_fn, whole_pred_fn in models_to_eval:
        # Whole article baseline
        whole_preds = whole_pred_fn(test_texts)
        whole_acc = float(accuracy_score(test_y, whole_preds))
        whole_mf1 = float(f1_score(test_y, whole_preds, average="macro", zero_division=0))
        whole_ci = _wilson_ci(int((test_y == whole_preds).sum()), len(test_y))

        # Chunk voting at best_chunk_size
        chunk_preds, _ = predict_by_chunk_voting(
            test_texts, proba_fn, target_words=best_chunk_size, classes=classes
        )
        chunk_acc = float(accuracy_score(test_y, chunk_preds))
        chunk_mf1 = float(f1_score(test_y, chunk_preds, average="macro", zero_division=0))
        chunk_ci = _wilson_ci(int((test_y == chunk_preds).sum()), len(test_y))

        # Also collect scores across all candidate chunk sizes on test for full inspection
        all_sizes_test: dict[int, dict[str, float]] = {}
        for sz in candidate_chunk_sizes:
            p_sz, _ = predict_by_chunk_voting(test_texts, proba_fn, target_words=sz, classes=classes)
            all_sizes_test[sz] = {
                "accuracy": float(accuracy_score(test_y, p_sz)),
                "macro_f1": float(f1_score(test_y, p_sz, average="macro", zero_division=0)),
            }

        test_results[model_name] = {
            "whole_article": {
                "accuracy": whole_acc,
                "macro_f1": whole_mf1,
                "accuracy_ci": whole_ci,
            },
            "chunk_voting": {
                "chunk_size": best_chunk_size,
                "accuracy": chunk_acc,
                "macro_f1": chunk_mf1,
                "accuracy_ci": chunk_ci,
                "accuracy_delta": chunk_acc - whole_acc,
                "macro_f1_delta": chunk_mf1 - whole_mf1,
            },
            "test_by_chunk_size": all_sizes_test,
        }

    # 6. Generate comparison plot for Ext 2
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figure_dir / "ext2_chunk_voting.png"
    _plot_ext2_comparison(test_results, best_chunk_size, fig_path)

    # 7. Write results/ext2.json
    results_data = {
        "description": (
            "Extension 2: Sentence-chunk voting for news articles. Each article is split "
            "into tweet-sized chunks, scored with tweet-trained models, and probabilities averaged. "
            "Chunk size tuned on validation split only, then evaluated on test split."
        ),
        "validation_tuning": {
            "candidate_chunk_sizes": list(candidate_chunk_sizes),
            "chosen_chunk_size": best_chunk_size,
            "mean_validation_macro_f1_per_size": {str(k): v for k, v in size_mean_f1.items()},
            "details": {
                m: {str(sz): metrics for sz, metrics in val_tuning_results[m].items()}
                for m in val_tuning_results
            },
        },
        "test_results": test_results,
        "figures": {
            "chunk_voting_comparison": relative_path(fig_path)
        },
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results_data, indent=2) + "\n", encoding="utf-8")
    return results_data


def _plot_ext2_comparison(
    test_results: dict[str, Any],
    best_chunk_size: int,
    output_path: Path,
) -> None:
    """Plot bar comparison between whole article scoring and chunk voting."""
    models = list(test_results.keys())
    labels = [m.replace("_", " ").title() for m in models]
    x = np.arange(len(models))
    width = 0.35

    whole_accs = [test_results[m]["whole_article"]["accuracy"] for m in models]
    chunk_accs = [test_results[m]["chunk_voting"]["accuracy"] for m in models]

    fig, ax = plt.subplots(figsize=(8.5, 5), constrained_layout=True)
    fig.patch.set_facecolor(config.LIGHT)
    style_axes(ax)

    b1 = ax.bar(x - width / 2, whole_accs, width, label="Whole Article", color=config.OCEAN_TEAL)
    b2 = ax.bar(
        x + width / 2,
        chunk_accs,
        width,
        label=f"Chunk Voting ({best_chunk_size} words)",
        color=config.GOLD_DARK,
        edgecolor=config.MIDNIGHT,
        linewidth=0.6,
    )

    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_title(
        f"Extension 2: Whole Article vs Chunk Voting ({best_chunk_size}-word Chunks)",
        fontsize=12,
        pad=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 0.8)
    ax.axhline(0.5, color=config.MUTED, linestyle="--", linewidth=1.0, alpha=0.7, label="50% Chance")
    ax.legend(frameon=False, loc="upper right")

    for bar in list(b1) + list(b2):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.015,
            f"{height:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=config.MIDNIGHT,
        )

    save_figure(fig, output_path)
