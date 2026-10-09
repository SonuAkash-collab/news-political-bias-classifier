"""Linear SVM and comparison models for tweet-based political leaning.

This module keeps model construction and grouped hyperparameter selection
separate from feature extraction so the same estimators can use the project's
binary or count vectorizers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from joblib import Memory
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

from polibias.features import binary_vectorizer, count_vectorizer


DEFAULT_C_GRID = (0.01, 0.06, 0.1, 1.0)


def make_model_zoo(
    *,
    seed: int = 42,
    linear_svc_c: float = 0.06,
    logistic_c: float = 1.0,
    complement_alpha: float = 1.0,
    rbf_c: float = 1.0,
    rbf_gamma: str | float = "scale",
) -> dict[str, Any]:
    """Create baseline and comparison estimators with explicit settings.

    Both weighted variants are included because tweets are imbalanced while the
    paper's article evaluation is approximately balanced.
    """
    return {
        "majority_baseline": DummyClassifier(strategy="constant", constant="D"),
        "linear_svc": LinearSVC(C=linear_svc_c, random_state=seed, max_iter=10_000),
        "linear_svc_balanced": LinearSVC(
            C=linear_svc_c, class_weight="balanced", random_state=seed, max_iter=10_000
        ),
        "logistic_regression": LogisticRegression(
            C=logistic_c, solver="liblinear", random_state=seed, max_iter=2_000
        ),
        "logistic_regression_balanced": LogisticRegression(
            C=logistic_c,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
            max_iter=2_000,
        ),
        "complement_nb": ComplementNB(alpha=complement_alpha),
        "rbf_svm": SVC(C=rbf_c, gamma=rbf_gamma, kernel="rbf"),
    }


def _vectorizer(feature_type: str, variant: str, min_df: int | float):
    """Build features only through the shared project vectorizer module."""
    factories = {"binary": binary_vectorizer, "count": count_vectorizer}
    try:
        factory = factories[feature_type]
    except KeyError:
        raise ValueError("feature_type must be 'binary' or 'count'") from None
    return factory(variant=variant, min_df=min_df)


def _pipeline_memory(cache_dir: str | None):
    """Optionally reuse fold-local vectorization across hyperparameter searches."""
    return Memory(location=cache_dir, verbose=0) if cache_dir else None


def make_text_model_zoo(
    *,
    feature_type: str = "count",
    variant: str = "no_punct",
    min_df: int | float = 2,
    seed: int = 42,
    linear_svc_c: float = 0.06,
    linear_svc_balanced_c: float | None = None,
    logistic_c: float = 1.0,
    logistic_balanced_c: float | None = None,
    complement_alpha: float = 1.0,
    rbf_c: float = 1.0,
    rbf_gamma: str | float = "scale",
) -> dict[str, Pipeline]:
    """Build text pipelines; each CV fold fits its own shared vectorizer."""
    models = make_model_zoo(
        seed=seed,
        linear_svc_c=linear_svc_c,
        logistic_c=logistic_c,
        complement_alpha=complement_alpha,
        rbf_c=rbf_c,
        rbf_gamma=rbf_gamma,
    )
    if linear_svc_balanced_c is not None:
        models["linear_svc_balanced"] = LinearSVC(
            C=linear_svc_balanced_c,
            class_weight="balanced",
            random_state=seed,
            max_iter=10_000,
        )
    if logistic_balanced_c is not None:
        models["logistic_regression_balanced"] = LogisticRegression(
            C=logistic_balanced_c,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
            max_iter=2_000,
        )
    return {
        name: Pipeline(
            [
                ("features", _vectorizer(feature_type, variant, min_df)),
                ("classifier", estimator),
            ]
        )
        for name, estimator in models.items()
    }


def tune_linear_svc(
    X: Any,
    y: Any,
    groups: Any,
    *,
    c_values: tuple[float, ...] = DEFAULT_C_GRID,
    n_splits: int = 5,
    class_weight: str | None = None,
    seed: int = 42,
    scoring: str = "accuracy",
) -> GridSearchCV:
    """Tune LinearSVC using GroupKFold, keeping each member in one fold.

    Call this with training data only. ``groups`` must contain a stable
    congressperson/member identifier, not an account handle or row identifier.
    The returned fitted search object contains the selected C and fold scores.
    """
    group_values = np.asarray(groups)
    if len(group_values) != len(y):
        raise ValueError("groups and y must have the same number of rows")
    distinct_groups = np.unique(group_values)
    if len(distinct_groups) < n_splits:
        raise ValueError(
            f"GroupKFold with {n_splits} splits needs at least {n_splits} distinct groups; "
            f"got {len(distinct_groups)}"
        )
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("c_values must contain positive C values")

    estimator = LinearSVC(
        random_state=seed,
        class_weight=class_weight,
        max_iter=10_000,
    )
    search = GridSearchCV(
        estimator,
        {"C": list(c_values)},
        cv=GroupKFold(n_splits=n_splits),
        scoring=scoring,
        refit=True,
        n_jobs=2,
        pre_dispatch=2,
        verbose=1,
        return_train_score=False,
    )
    search.fit(X, y, groups=group_values)
    return search


def tune_text_linear_svc(
    texts: Any,
    y: Any,
    groups: Any,
    *,
    feature_type: str = "count",
    variant: str = "no_punct",
    min_df: int | float = 2,
    c_values: tuple[float, ...] = DEFAULT_C_GRID,
    n_splits: int = 5,
    class_weight: str | None = None,
    seed: int = 42,
    scoring: str = "f1_macro",
    cache_dir: str | None = None,
) -> GridSearchCV:
    """Tune text LinearSVC with fold-local project features and member GroupKFold.

    Since the vectorizer is in the pipeline, its vocabulary is learned only from
    each fold's training members. Use this on the training partition only.
    """
    group_values = np.asarray(groups)
    if len(group_values) != len(y):
        raise ValueError("groups and y must have the same number of rows")
    if len(np.unique(group_values)) < n_splits:
        raise ValueError(f"GroupKFold with {n_splits} splits needs at least {n_splits} groups")
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("c_values must contain positive C values")

    estimator = Pipeline(
        [
            ("features", _vectorizer(feature_type, variant, min_df)),
            (
                "classifier",
                LinearSVC(
                    class_weight=class_weight,
                    random_state=seed,
                    max_iter=10_000,
                ),
            ),
        ],
        memory=_pipeline_memory(cache_dir),
    )
    search = GridSearchCV(
        estimator,
        {"classifier__C": list(c_values)},
        cv=GroupKFold(n_splits=n_splits),
        scoring=scoring,
        refit=True,
        n_jobs=2,
        pre_dispatch=2,
        verbose=1,
        return_train_score=False,
    )
    search.fit(texts, y, groups=group_values)
    return search


def tune_text_logistic_regression(
    texts: Any,
    y: Any,
    groups: Any,
    *,
    feature_type: str = "count",
    variant: str = "no_punct",
    min_df: int | float = 2,
    c_values: tuple[float, ...] = DEFAULT_C_GRID,
    n_splits: int = 5,
    class_weight: str | None = None,
    seed: int = 42,
    scoring: str = "f1_macro",
    cache_dir: str | None = None,
) -> GridSearchCV:
    """Tune logistic-regression C with the same member-grouped, training-only CV."""
    group_values = np.asarray(groups)
    if len(group_values) != len(y):
        raise ValueError("groups and y must have the same number of rows")
    if len(np.unique(group_values)) < n_splits:
        raise ValueError(f"GroupKFold with {n_splits} splits needs at least {n_splits} groups")
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("c_values must contain positive C values")

    estimator = Pipeline(
        [
            ("features", _vectorizer(feature_type, variant, min_df)),
            (
                "classifier",
                LogisticRegression(
                    class_weight=class_weight,
                    solver="liblinear",
                    random_state=seed,
                    max_iter=2_000,
                ),
            ),
        ],
        memory=_pipeline_memory(cache_dir),
    )
    search = GridSearchCV(
        estimator,
        {"classifier__C": list(c_values)},
        cv=GroupKFold(n_splits=n_splits),
        scoring=scoring,
        refit=True,
        n_jobs=2,
        pre_dispatch=2,
        verbose=1,
        return_train_score=False,
    )
    search.fit(texts, y, groups=group_values)
    return search


def tune_text_complement_nb(
    texts: Any,
    y: Any,
    groups: Any,
    *,
    feature_type: str = "count",
    variant: str = "no_punct",
    min_df: int | float = 2,
    alpha_values: tuple[float, ...] = (0.1, 1.0, 5.0),
    n_splits: int = 5,
    scoring: str = "f1_macro",
    cache_dir: str | None = None,
) -> GridSearchCV:
    """Tune ComplementNB smoothing with member-grouped training-only CV."""
    group_values = np.asarray(groups)
    if len(group_values) != len(y):
        raise ValueError("groups and y must have the same number of rows")
    if len(np.unique(group_values)) < n_splits:
        raise ValueError(f"GroupKFold with {n_splits} splits needs at least {n_splits} groups")
    if not alpha_values or any(value <= 0 for value in alpha_values):
        raise ValueError("alpha_values must contain positive smoothing values")
    estimator = Pipeline(
        [
            ("features", _vectorizer(feature_type, variant, min_df)),
            ("classifier", ComplementNB()),
        ],
        memory=_pipeline_memory(cache_dir),
    )
    search = GridSearchCV(
        estimator,
        {"classifier__alpha": list(alpha_values)},
        cv=GroupKFold(n_splits=n_splits),
        scoring=scoring,
        refit=True,
        n_jobs=2,
        pre_dispatch=2,
        verbose=1,
        return_train_score=False,
    )
    search.fit(texts, y, groups=group_values)
    return search


def fit_model_zoo(
    X: Any,
    y: Any,
    *,
    seed: int = 42,
    rbf_max_samples: int = 10_000,
    **model_settings: Any,
) -> dict[str, Any]:
    """Fit the model zoo; fit the RBF SVM on a deterministic stratified subset.

    The caller should pass only the training fold here. Subsampling controls the
    quadratic memory and time cost of the kernel SVM and does not use labels from
    any held-out evaluation set.
    """
    if rbf_max_samples < 2:
        raise ValueError("rbf_max_samples must be at least 2")
    y_array = np.asarray(y)
    if len(y_array) != X.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    models = make_model_zoo(seed=seed, **model_settings)
    for name, model in models.items():
        if name == "rbf_svm" and X.shape[0] > rbf_max_samples:
            indices = np.arange(X.shape[0])
            try:
                fit_idx, _ = train_test_split(
                    indices,
                    train_size=rbf_max_samples,
                    random_state=seed,
                    stratify=y_array,
                )
            except ValueError as exc:
                raise ValueError(
                    "Cannot make a stratified RBF-SVM subsample; provide enough training "
                    "examples in every class or raise rbf_max_samples"
                ) from exc
            model.fit(X[fit_idx], y_array[fit_idx])
        else:
            model.fit(X, y_array)
    return models


def fit_text_model_zoo(
    texts: Any,
    y: Any,
    *,
    feature_type: str = "count",
    variant: str = "no_punct",
    min_df: int | float = 2,
    seed: int = 42,
    rbf_max_samples: int = 10_000,
    **model_settings: Any,
) -> dict[str, Pipeline]:
    """Fit all text models on a training partition; subsample only the RBF fit."""
    y_array = np.asarray(y)
    if len(y_array) != len(texts):
        raise ValueError("texts and y must have the same number of rows")
    pipelines = make_text_model_zoo(
        feature_type=feature_type,
        variant=variant,
        min_df=min_df,
        seed=seed,
        **model_settings,
    )
    for name, model in pipelines.items():
        if name == "rbf_svm" and len(y_array) > rbf_max_samples:
            indices = np.arange(len(y_array))
            try:
                fit_idx, _ = train_test_split(
                    indices,
                    train_size=rbf_max_samples,
                    random_state=seed,
                    stratify=y_array,
                )
            except ValueError as exc:
                raise ValueError(
                    "Cannot make a stratified RBF-SVM subsample; provide enough training "
                    "examples in every class or raise rbf_max_samples"
                ) from exc
            model.fit(np.asarray(texts, dtype=object)[fit_idx], y_array[fit_idx])
        else:
            model.fit(texts, y_array)
    return pipelines


def evaluate_model_zoo(models: dict[str, Any], X: Any, y: Any) -> dict[str, dict[str, float]]:
    """Return accuracy/macro-F1 and repeat the constant-D baseline beside each score."""
    y_true = np.asarray(y)
    baseline_predictions = np.full(len(y_true), "D", dtype=object)
    baseline = {
        "accuracy": float(accuracy_score(y_true, baseline_predictions)),
        "macro_f1": float(f1_score(y_true, baseline_predictions, average="macro")),
    }
    scores = {}
    for name, model in models.items():
        predictions = model.predict(X)
        accuracy = float(accuracy_score(y_true, predictions))
        macro_f1 = float(f1_score(y_true, predictions, average="macro"))
        scores[name] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "majority_baseline_accuracy": baseline["accuracy"],
            "majority_baseline_macro_f1": baseline["macro_f1"],
            "accuracy_delta_vs_baseline": accuracy - baseline["accuracy"],
            "macro_f1_delta_vs_baseline": macro_f1 - baseline["macro_f1"],
        }
    return scores
