"""Linear SVM and comparison models for tweet-based political leaning.

This module keeps model construction and grouped hyperparameter selection
separate from feature extraction so the same estimators can use the project's
binary or count vectorizers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC, SVC


DEFAULT_C_GRID = (0.001, 0.01, 0.06, 0.1, 0.5, 1.0, 10.0)


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
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
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
        n_jobs=1,
        return_train_score=False,
    )
    search.fit(X, y, groups=group_values)
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


def evaluate_model_zoo(models: dict[str, Any], X: Any, y: Any) -> dict[str, dict[str, float]]:
    """Return accuracy and macro-F1 for each fitted estimator."""
    y_true = np.asarray(y)
    scores = {}
    for name, model in models.items():
        predictions = model.predict(X)
        scores[name] = {
            "accuracy": float(accuracy_score(y_true, predictions)),
            "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        }
    return scores
