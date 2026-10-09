"""Our Naive Bayes must give the same answers as scikit-learn's with alpha = 1."""

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import naive_bayes as sk

from polibias.features import binary_vectorizer, count_vectorizer
from polibias.models.naive_bayes import BernoulliNB, MultinomialNB

TOL = 1e-6


def count_data(n_classes: int, seed: int = 42):
    """Small random word-count matrices with labels, train and test."""
    rng = np.random.default_rng(seed)
    labels = np.array(["D", "R", "center"][:n_classes])
    X_train = rng.poisson(0.4, size=(120, 50))
    X_test = rng.poisson(0.4, size=(40, 50))
    y_train = rng.choice(labels, size=120)
    return X_train, y_train, X_test


@pytest.mark.parametrize("ours_cls, ref_cls", [(BernoulliNB, sk.BernoulliNB), (MultinomialNB, sk.MultinomialNB)])
@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("sparse", [False, True])
def test_matches_sklearn(ours_cls, ref_cls, n_classes, sparse):
    X_train, y_train, X_test = count_data(n_classes)
    if sparse:
        X_train, X_test = sp.csr_matrix(X_train), sp.csr_matrix(X_test)

    ours = ours_cls().fit(X_train, y_train)
    ref = ref_cls(alpha=1.0).fit(X_train, y_train)

    assert list(ours.classes_) == list(ref.classes_)
    np.testing.assert_allclose(ours.class_log_prior_, ref.class_log_prior_, rtol=0, atol=TOL)
    np.testing.assert_allclose(ours.feature_log_prob_, ref.feature_log_prob_, rtol=0, atol=TOL)
    np.testing.assert_allclose(ours.predict_proba(X_test), ref.predict_proba(X_test), rtol=0, atol=TOL)
    assert (ours.predict(X_test) == ref.predict(X_test)).all()


@pytest.mark.parametrize(
    "ours_cls, ref_cls, make_vec",
    [(BernoulliNB, sk.BernoulliNB, binary_vectorizer), (MultinomialNB, sk.MultinomialNB, count_vectorizer)],
)
def test_on_tweets_through_vectoriser(ours_cls, ref_cls, make_vec):
    train = [
        "Protect voting rights #ForThePeople",
        "Voting rights and health care for all",
        "Cut taxes and secure the border",
        "Secure the border now @RepBob",
        "Health care is a right",
        "Lower taxes, more jobs",
    ]
    y = ["D", "D", "R", "R", "D", "R"]
    test = ["voting rights now", "cut taxes", "border health"]

    vec = make_vec("no_punct")
    X_train, X_test = vec.fit_transform(train), vec.transform(test)
    ours = ours_cls().fit(X_train, y)
    ref = ref_cls(alpha=1.0).fit(X_train, y)
    np.testing.assert_allclose(ours.predict_proba(X_test), ref.predict_proba(X_test), rtol=0, atol=TOL)
    assert (ours.predict(X_test) == ref.predict(X_test)).all()
    assert list(ours.predict(X_test[:2])) == ["D", "R"]


@pytest.mark.parametrize("model_cls", [BernoulliNB, MultinomialNB])
def test_probabilities_sum_to_one_and_wrong_width_is_rejected(model_cls):
    X_train, y_train, X_test = count_data(2)
    model = model_cls().fit(X_train, y_train)
    np.testing.assert_allclose(model.predict_proba(X_test).sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        model.predict(X_test[:, :10])
