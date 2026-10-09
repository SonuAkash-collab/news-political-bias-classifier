"""Toy-data tests for model construction and member-grouped tuning."""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import GroupKFold

from polibias.features import binary_vectorizer, count_vectorizer
from polibias.models.svm import (
    evaluate_model_zoo,
    fit_model_zoo,
    make_model_zoo,
    make_text_model_zoo,
    tune_linear_svc,
    tune_text_linear_svc,
    tune_text_logistic_regression,
)


def _toy_data():
    # Eight members, each represented by several rows. Feature 0 is predictive,
    # and member-specific features make row-wise validation misleading.
    rng = np.random.default_rng(23)
    groups = np.repeat(np.arange(8), 8)
    y = np.tile(np.array(["D", "R"] * 4), 8)
    X = np.zeros((len(y), 10), dtype=float)
    X[:, 0] = np.where(y == "D", 1.0, 0.0) + rng.uniform(0, 0.15, len(y))
    X[np.arange(len(y)), 2 + groups] = 1.0
    X += rng.uniform(0, 0.05, X.shape)
    return csr_matrix(X), y, groups


def test_model_zoo_includes_weighted_models_and_scores_toy_data():
    X, y, _ = _toy_data()
    models = fit_model_zoo(X, y, rbf_max_samples=24, linear_svc_c=0.06)

    assert "majority_baseline" in models
    assert models["linear_svc_balanced"].class_weight == "balanced"
    assert models["logistic_regression_balanced"].class_weight == "balanced"
    assert models["rbf_svm"].n_support_.sum() <= 24

    scores = evaluate_model_zoo(models, X, y)
    assert set(scores) == set(models)
    assert all(0.0 <= score["accuracy"] <= 1.0 for score in scores.values())
    assert all(0.0 <= score["macro_f1"] <= 1.0 for score in scores.values())
    expected_baseline = float(np.mean(y == "D"))
    assert all(
        score["majority_baseline_accuracy"] == expected_baseline
        for score in scores.values()
    )


def test_linear_svc_tuning_uses_grouped_folds_and_returns_selected_model():
    X, y, groups = _toy_data()
    search = tune_linear_svc(X, y, groups, c_values=(0.01, 0.06, 1.0), n_splits=4)

    assert search.best_params_["C"] in (0.01, 0.06, 1.0)
    assert search.best_estimator_.C == search.best_params_["C"]
    folds = list(GroupKFold(n_splits=4).split(X, y, groups))
    for train_idx, valid_idx in folds:
        assert set(groups[train_idx]).isdisjoint(groups[valid_idx])


def test_model_factory_weighted_and_unweighted_estimators_are_distinct():
    models = make_model_zoo()

    assert models["linear_svc"].class_weight is None
    assert models["linear_svc_balanced"].class_weight == "balanced"
    assert models["logistic_regression"].class_weight is None
    assert models["logistic_regression_balanced"].class_weight == "balanced"


def test_text_pipelines_use_shared_project_vectorizers():
    binary = make_text_model_zoo(feature_type="binary", variant="no_punct")
    counts = make_text_model_zoo(feature_type="count", variant="all_words")

    binary_features = binary["linear_svc"].named_steps["features"]
    count_features = counts["linear_svc"].named_steps["features"]
    assert type(binary_features) is type(binary_vectorizer())
    assert type(count_features) is type(count_vectorizer())
    assert binary_features.binary is True
    assert count_features.binary is False
    assert binary_features.min_df == count_features.min_df == 2


def test_text_models_tune_with_group_folds():
    _, y, groups = _toy_data()
    texts = ["dem voting rights" if label == "D" else "rep tax cuts" for label in y]
    groups = groups.copy()

    svc_search = tune_text_linear_svc(
        texts, y, groups, c_values=(0.06, 1.0), n_splits=4, feature_type="binary"
    )
    lr_search = tune_text_logistic_regression(
        texts, y, groups, c_values=(0.06, 1.0), n_splits=4, feature_type="count"
    )

    assert svc_search.best_params_["classifier__C"] in (0.06, 1.0)
    assert lr_search.best_params_["classifier__C"] in (0.06, 1.0)
    assert "features" in svc_search.best_estimator_.named_steps
