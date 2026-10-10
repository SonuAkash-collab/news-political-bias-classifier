"""Extension 1: TF-IDF weighted word unigrams and bigrams with logistic regression.

Why this should help (the hypothesis, written before the results):
    TF-IDF   Every congressperson writes "today", "great", "house" and "thank". Counts and word
             existence give these words as much weight as rare, telling words. TF-IDF multiplies each
             count by log(number of tweets / tweets containing the word), so words used everywhere
             count for little and distinctive words count for more. sublinear_tf replaces a count c
             by 1 + log(c), so a word repeated three times is not three times the evidence.
    Bigrams  Pairs of adjacent words catch phrases whose meaning differs from their parts:
             "tax cuts", "voting rights", "border security", "for the people". Single words
             ("tax", "rights") are used by both parties.
    LR       Logistic regression gives calibrated probabilities (the demo shows them) and, like
             LinearSVC, handles hundreds of thousands of sparse features.

What we found: see "verdict" in results/ext1.json and the RESULTS comment at the end of this file.

Protocol. Tweets are split with evaluation.grouped_split (same 90 test members as Exp 1 and the
model zoo). C, min_df and class_weight are chosen with 5-fold GroupKFold over member_id on the
training members only, selecting by macro F1; the vectoriser is inside the Pipeline, so each fold
learns its own vocabulary and IDF weights. The test members are used once, for the final score.
For the articles, the selected settings are refit on all tweets and scored on the left/right
articles of the AllSides media-based test split (left = D, right = R). That refit is the saved
model, models/ext1_tfidf_lr.joblib, used by the demo.

Usage: python -m polibias.experiments.ext1_tfidf_lr   (after exp1, so results/exp1.json exists)
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from polibias import config
from polibias.data.articles import binary_articles
from polibias.data.tweets import load_processed
from polibias.evaluation import (
    bar_style,
    grouped_split,
    majority_baseline,
    new_figure,
    relative_path,
    save_figure,
    scores,
    split_summary,
    write_results,
)
from polibias.text import no_punct

RESULTS_PATH = config.RESULTS_DIR / "ext1.json"
FIGURE_PATH = config.FIGURES_DIR / "ext1_tfidf_lr.png"
MODEL_PATH = config.MODELS_DIR / "ext1_tfidf_lr.joblib"
EXP1_PATH = config.RESULTS_DIR / "exp1.json"
EXP3_PATH = config.RESULTS_DIR / "exp3.json"

VARIANT = "no_punct"
CV_FOLDS = 5
C_GRID = (0.3, 1.0, 3.0, 10.0)
MIN_DF_GRID = (2, 5, 10)
CLASS_WEIGHT_GRID = (None, "balanced")
QUICK_GRID = {"C": (1.0, 10.0), "min_df": (2,), "class_weight": (None, "balanced")}
MAX_MODEL_MB = 20.0
PAPER_C = 0.06
EXP1_LABELS = {"bernoulli_nb": "Bernoulli NB", "multinomial_nb": "Multinomial NB", "linear_svc": "LinearSVC (C=0.06)"}


def make_pipeline(C: float = 1.0, min_df: int = 2, class_weight: str | None = None, ngram_max: int = 2) -> Pipeline:
    """TF-IDF over no_punct word n-grams (1 to ngram_max), then logistic regression."""
    # A tokenizer (not an analyzer) is passed so that scikit-learn still builds the n-grams.
    features = TfidfVectorizer(
        tokenizer=no_punct,
        lowercase=False,  # no_punct lowercases
        token_pattern=None,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        sublinear_tf=True,
        dtype=np.float32,
    )
    classifier = LogisticRegression(
        C=C, class_weight=class_weight, solver="liblinear", max_iter=2_000, random_state=config.SEED
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def tune(train: pd.DataFrame) -> GridSearchCV:
    """Grid search over C, min_df and class_weight with member-grouped folds on training data."""
    grid = QUICK_GRID if config.QUICK else {"C": C_GRID, "min_df": MIN_DF_GRID, "class_weight": CLASS_WEIGHT_GRID}
    cache = tempfile.mkdtemp(prefix="ext1_cv_")
    try:
        pipeline = make_pipeline()
        # Fold-local vectorising is the slow part; the cache reuses it across C and class_weight.
        pipeline.memory = Memory(location=cache, verbose=0)
        search = GridSearchCV(
            pipeline,
            {
                "features__min_df": list(grid["min_df"]),
                "classifier__C": list(grid["C"]),
                "classifier__class_weight": list(grid["class_weight"]),
            },
            cv=GroupKFold(n_splits=CV_FOLDS),
            scoring="f1_macro",
            refit=False,
            n_jobs=4,
            verbose=1,
        )
        with warnings.catch_warnings():
            # joblib warns about two pipeline helpers with the same name; the cache is still correct.
            warnings.filterwarnings("ignore", message="Possible name collisions")
            search.fit(train["text"], train["party"], groups=train["member_id"])
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    return search


def cv_table(search: GridSearchCV) -> list[dict]:
    rows = []
    cv = search.cv_results_
    for i, params in enumerate(cv["params"]):
        rows.append(
            {
                "C": params["classifier__C"],
                "min_df": params["features__min_df"],
                "class_weight": params["classifier__class_weight"],
                "mean_macro_f1": float(cv["mean_test_score"][i]),
                "std_macro_f1": float(cv["std_test_score"][i]),
                "rank": int(cv["rank_test_score"][i]),
            }
        )
    return sorted(rows, key=lambda row: row["rank"])


def best_settings(search: GridSearchCV) -> dict:
    params = search.best_params_
    return {"C": params["classifier__C"], "min_df": params["features__min_df"], "class_weight": params["classifier__class_weight"]}


def fit_score(pipeline: Pipeline, train: pd.DataFrame, test_texts, test_labels) -> tuple[Pipeline, dict]:
    pipeline.fit(train["text"], train["party"])
    return pipeline, scores(np.asarray(test_labels), pipeline.predict(test_texts))


def load_article_test() -> pd.DataFrame:
    """Left/right articles of the AllSides media-based test split, with text = title + body."""
    if not config.ARTICLES_CSV.exists():
        raise FileNotFoundError(f"Missing {config.ARTICLES_CSV}. Run scripts/download_articles.py first.")
    articles = pd.read_csv(config.ARTICLES_CSV, dtype=str, keep_default_na=False)
    test = binary_articles(articles[articles["split"] == "test"])
    test["text"] = (test["title"] + " " + test["content"]).str.strip()
    if config.QUICK:
        test = test.groupby("party", group_keys=False).sample(frac=config.SAMPLE_FRACTION, random_state=config.SEED)
    return test.reset_index(drop=True)


def strip_for_saving(pipeline: Pipeline) -> Pipeline:
    """Drop the vectoriser's list of discarded terms; it is only for inspection and can be huge."""
    features = pipeline.named_steps["features"]
    if hasattr(features, "stop_words_"):
        features.stop_words_ = None
    return pipeline


def save_model(pipeline: Pipeline, tweets: pd.DataFrame, min_df: int) -> tuple[Pipeline, int, float]:
    """Save the all-tweets model; refit with a larger min_df until the file is under MAX_MODEL_MB."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    while True:
        joblib.dump(strip_for_saving(pipeline), MODEL_PATH, compress=3)
        size_mb = MODEL_PATH.stat().st_size / 2**20
        if size_mb <= MAX_MODEL_MB:
            return pipeline, min_df, size_mb
        min_df *= 2
        print(f"Model is {size_mb:.1f} MB; refitting with min_df = {min_df}")
        pipeline.set_params(features__min_df=min_df)
        pipeline.fit(tweets["text"], tweets["party"])


def exp1_reference() -> dict | None:
    """Exp 1 grouped-split scores for the same vocabulary variant (no_punct), if available."""
    if not EXP1_PATH.exists():
        return None
    exp1 = json.loads(EXP1_PATH.read_text(encoding="utf-8"))
    per_model = exp1["scores"]["grouped"][VARIANT]
    return {name: {"accuracy": s["accuracy"], "macro_f1": s["macro_f1"]} for name, s in per_model.items()}


def exp3_reference() -> list[dict] | None:
    """Every accuracy / macro F1 pair found in results/exp3.json, with its path, if the file exists.

    Exp 3 is written by Person B; reading it generically keeps this module independent of its layout.
    """
    if not EXP3_PATH.exists():
        return None
    found = []

    def walk(node, path):
        if isinstance(node, dict):
            if isinstance(node.get("accuracy"), (int, float)):
                found.append({"path": "/".join(path), "accuracy": node["accuracy"], "macro_f1": node.get("macro_f1")})
            for key, value in node.items():
                walk(value, path + [str(key)])

    walk(json.loads(EXP3_PATH.read_text(encoding="utf-8")), [])
    return found


def paper_svc_on_articles(tweets: pd.DataFrame, articles: pd.DataFrame) -> dict:
    """Reference on the same articles: the paper's setup (word counts, LinearSVC C = 0.06) on all tweets."""
    model = Pipeline(
        [
            ("features", CountVectorizer(analyzer=no_punct)),
            ("classifier", LinearSVC(C=PAPER_C, random_state=config.SEED, max_iter=10_000)),
        ]
    )
    _, result = fit_score(model, tweets, articles["text"], articles["party"])
    return result


def verdict(tweet: dict, articles: dict) -> str:
    """One honest sentence per test set, comparing Extension 1 with the best earlier model."""
    parts = []
    ext, best_name, best = tweet["ext1"], tweet["best_previous_name"], tweet["best_previous"]
    if best is not None:
        delta = 100 * (ext["macro_f1"] - best["macro_f1"])
        word = "better than" if delta >= 0.5 else "worse than" if delta <= -0.5 else "about the same as"
        parts.append(
            f"Held-out members: TF-IDF + bigrams + LR reaches macro F1 {ext['macro_f1']:.3f} "
            f"(accuracy {ext['accuracy']:.3f}), {word} the best earlier model {best_name} "
            f"({best['macro_f1']:.3f}, {delta:+.1f} points)."
        )
    ref = articles["paper_linear_svc_counts"]
    ext_a = articles["ext1"]
    delta = 100 * (ext_a["macro_f1"] - ref["macro_f1"])
    word = "better than" if delta >= 0.5 else "worse than" if delta <= -0.5 else "about the same as"
    parts.append(
        f"Articles: macro F1 {ext_a['macro_f1']:.3f} (accuracy {ext_a['accuracy']:.3f}), {word} the "
        f"paper's count LinearSVC on the same articles ({ref['macro_f1']:.3f}, {delta:+.1f} points)."
    )
    base = articles["article_majority_baseline"]
    side = "above" if ext_a["accuracy"] > base["accuracy"] else "not above"
    parts.append(
        f"Its article accuracy is {side} always predicting {base['always_predict']} "
        f"({base['accuracy']:.3f}, the article test majority)."
    )
    return " ".join(parts)


def plot(tweet_rows: list[tuple[str, dict]], article_rows: list[tuple[str, dict]]) -> str:
    """Accuracy and macro F1 per model, on held-out tweets (left) and on articles (right).

    The majority baseline is the first group in each panel, so no separate baseline line is drawn.
    """
    fig, axes = new_figure(width=14, height=6.2, ncols=2, sharey=True,
                           gridspec_kw={"width_ratios": [len(tweet_rows), max(len(article_rows), 2)]})
    for ax, rows, title in (
        (axes[0], tweet_rows, "Held-out congresspeople (grouped tweet test)"),
        (axes[1], article_rows, "AllSides left/right articles (media-based test)"),
    ):
        x = np.arange(len(rows))
        width = 0.38
        for offset, metric, color, label in ((-0.5, "accuracy", config.OCEAN_TEAL, "Accuracy"), (0.5, "macro_f1", config.GOLD, "Macro F1")):
            values = [100 * row[metric] for _, row in rows]
            bars = ax.bar(x + offset * width, values, width * 0.94, label=label, **bar_style(color))
            ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8.5, color=config.MIDNIGHT)
        ax.set_xticks(x, [name for name, _ in rows], fontsize=9)
        ax.set_title(title, fontsize=12, fontfamily="serif", pad=10)
        ax.set_ylim(0, 100)
        # Extension 1 is the last group; mark it so it stands out without a new colour.
        ax.get_xticklabels()[-1].set_fontweight("bold")
    axes[0].set_ylabel("Score (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    fig.suptitle("Extension 1: TF-IDF unigrams + bigrams with logistic regression",
                 fontsize=14, fontfamily="serif", color=config.MIDNIGHT)
    return save_figure(fig, FIGURE_PATH)


def main() -> dict:
    config.set_seed()
    tweets = load_processed()
    train, test = grouped_split(tweets)

    start = time.perf_counter()
    search = tune(train)
    settings = best_settings(search)
    print(f"Selected {settings} (CV macro F1 {search.best_score_:.4f}, {time.perf_counter() - start:.0f}s)")

    _, ext1_tweet = fit_score(make_pipeline(**settings), train, test["text"], test["party"])
    # Ablation with the same settings but unigrams only, to see what the bigrams add.
    _, unigram_tweet = fit_score(make_pipeline(**settings, ngram_max=1), train, test["text"], test["party"])
    tweet_base = majority_baseline(train["party"], test["party"])

    exp1 = exp1_reference()
    best_name, best = None, None
    if exp1:
        best_key = max(exp1, key=lambda name: exp1[name]["macro_f1"])
        best_name, best = EXP1_LABELS.get(best_key, best_key), exp1[best_key]
    tweet_result = {
        "ext1": ext1_tweet,
        "tfidf_unigrams_only": unigram_tweet,
        "majority_baseline": tweet_base,
        "exp1_grouped_no_punct": exp1,
        "best_previous_name": best_name,
        "best_previous": best,
    }
    print(f"Grouped tweet test: accuracy {ext1_tweet['accuracy']:.4f}, macro F1 {ext1_tweet['macro_f1']:.4f}")

    # Final model: all tweets, selected settings. Scored on articles and saved for the demo.
    articles = load_article_test()
    final = make_pipeline(**settings)
    final.fit(tweets["text"], tweets["party"])
    final, saved_min_df, size_mb = save_model(final, tweets, settings["min_df"])
    ext1_articles = scores(articles["party"].to_numpy(), final.predict(articles["text"]))
    article_result = {
        "ext1": ext1_articles,
        "paper_linear_svc_counts": paper_svc_on_articles(tweets, articles),
        # Always predict the training majority (D, from the tweets) ...
        "majority_baseline": majority_baseline(tweets["party"], articles["party"]),
        # ... and always predict the article test majority (R: 599 right vs 402 left). This one is
        # read off the test labels, so it is only a reference line, never used for any choice.
        "article_majority_baseline": majority_baseline(articles["party"], articles["party"]),
        "exp3": exp3_reference(),
    }
    print(f"Articles: accuracy {ext1_articles['accuracy']:.4f}, macro F1 {ext1_articles['macro_f1']:.4f}")

    tweet_rows = [("Always D", tweet_base)]
    if exp1:
        tweet_rows += [(EXP1_LABELS.get(name, name), exp1[name]) for name in exp1]
    tweet_rows += [("TF-IDF LR\nunigrams", unigram_tweet), ("Ext 1: TF-IDF LR\nuni + bigrams", ext1_tweet)]
    article_rows = [
        ("Always D", article_result["majority_baseline"]),
        ("Always R\n(article majority)", article_result["article_majority_baseline"]),
        ("LinearSVC (C=0.06)\ncounts", article_result["paper_linear_svc_counts"]),
        ("Ext 1: TF-IDF LR\nuni + bigrams", ext1_articles),
    ]
    figure = plot(tweet_rows, article_rows)

    result = {
        "description": "Extension 1: TF-IDF word unigrams + bigrams (no_punct) with logistic regression",
        "quick_run": config.QUICK,
        "data": {
            "tweets": len(tweets),
            "members": int(tweets["member_id"].nunique()),
            "article_test": len(articles),
            "article_test_party_counts": articles["party"].value_counts().to_dict(),
        },
        "split": split_summary(train, test),
        "settings": {
            "variant": VARIANT,
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "classifier": "LogisticRegression(solver='liblinear')",
            "article_text": "title + content",
        },
        "tuning": {
            "method": "GroupKFold",
            "group": "member_id",
            "folds": CV_FOLDS,
            "training_only": True,
            "selection_metric": "macro_f1",
            "grid": QUICK_GRID if config.QUICK else {"C": C_GRID, "min_df": MIN_DF_GRID, "class_weight": CLASS_WEIGHT_GRID},
            "selected": settings,
            "best_cv_macro_f1": float(search.best_score_),
            "cv_results": cv_table(search),
        },
        "tweet_test": tweet_result,
        "article_test": article_result,
        "model": {
            "path": relative_path(MODEL_PATH),
            "trained_on": "all tweets",
            "min_df": saved_min_df,
            "size_mb": round(size_mb, 2),
            "vocabulary_size": len(final.named_steps["features"].vocabulary_),
        },
        "verdict": verdict(tweet_result, article_result),
        "figure": figure,
    }
    write_results(result, RESULTS_PATH)
    print(result["verdict"])
    print(f"Saved {RESULTS_PATH}, {FIGURE_PATH} and {MODEL_PATH} ({size_mb:.1f} MB)")
    return result


if __name__ == "__main__":
    main()
