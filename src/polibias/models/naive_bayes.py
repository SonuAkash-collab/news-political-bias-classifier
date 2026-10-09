"""Bernoulli and Multinomial Naive Bayes from scratch, following Crain and Erhan (2021), Section 3.1.

Rows of X are documents and columns are vocabulary words (the output of polibias.features), as a
NumPy array or a SciPy sparse matrix. All parameters are stored as logs and every score is a sum
of logs, so the product over thousands of words never underflows to zero.

Attribute names follow scikit-learn (classes_, class_log_prior_, feature_log_prob_), so the
tests can compare both implementations directly.
"""

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp


class _NaiveBayes:
    """Shared fitting and prediction. Subclasses define the event model."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha  # Laplace smoothing: alpha = 1 adds one fake count, as in the paper

    def fit(self, X, y):
        X = self._prepare(X)
        y = np.asarray(y)
        if X.shape[0] != len(y):
            raise ValueError(f"X has {X.shape[0]} rows but y has {len(y)} labels")

        # One-hot labels: Y[i, c] = 1{y_i = class c}.
        self.classes_, y_index = np.unique(y, return_inverse=True)
        Y = np.zeros((len(y), len(self.classes_)))
        Y[np.arange(len(y)), y_index] = 1.0

        # class_count_[c] = sum_i 1{y_i = c}, and phi_y = class_count_ / n.
        self.class_count_ = Y.sum(axis=0)
        self.class_log_prior_ = np.log(self.class_count_) - np.log(len(y))
        # feature_count_[c, j] = sum_i 1{y_i = c} * x_ij, for all classes and words at once.
        self.feature_count_ = np.asarray((X.T @ Y).T)
        self.n_features_in_ = X.shape[1]
        self._fit_feature_probs()
        return self

    def predict_log_proba(self, X) -> np.ndarray:
        """log p(y | x) for every row of X and every class in classes_."""
        jll = self._joint_log_likelihood(self._prepare(X, fitting=False))
        # Bayes' rule: divide by p(x) = sum over classes of p(x | y) p(y), done in log space.
        return jll - logsumexp(jll, axis=1, keepdims=True)

    def predict_proba(self, X) -> np.ndarray:
        """p(y | x) for every row of X; column c is the probability of classes_[c]."""
        return np.exp(self.predict_log_proba(X))

    def predict(self, X) -> np.ndarray:
        """The most probable class for every row of X."""
        jll = self._joint_log_likelihood(self._prepare(X, fitting=False))
        return self.classes_[np.argmax(jll, axis=1)]

    def top_words(self, n: int, feature_names) -> dict:
        """The n words most indicative of each class, by the paper's log ratio (Section 4.1).

        For two classes a and b (classes_ in sorted order, so "D" and "R"), word j scores
        log p(w_j | a) / p(w_j | b) for class a and the negative of that for class b.
        feature_names is the vectoriser's get_feature_names_out(); the model never sees the words.
        Returns {class: [(word, log ratio), ...]}, highest ratio first.
        """
        if len(self.classes_) != 2:
            raise ValueError("top_words needs exactly two classes")
        feature_names = np.asarray(feature_names)
        if len(feature_names) != self.n_features_in_:
            raise ValueError(f"Got {len(feature_names)} feature names for {self.n_features_in_} features")
        log_ratio = self.feature_log_prob_[0] - self.feature_log_prob_[1]
        result = {}
        for label, score in zip(self.classes_, (log_ratio, -log_ratio)):
            best = np.argsort(-score, kind="stable")[:n]  # stable: ties keep vocabulary order
            result[label] = [(str(feature_names[j]), float(score[j])) for j in best]
        return result

    def _prepare(self, X, fitting: bool = True) -> sp.csr_matrix:
        X = sp.csr_matrix(X, dtype=np.float64)
        if X.nnz and X.data.min() < 0:
            raise ValueError("Naive Bayes features must be non-negative")
        if not fitting and X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} columns, the model was fitted on {self.n_features_in_}")
        return X

    def _fit_feature_probs(self) -> None:
        raise NotImplementedError

    def _joint_log_likelihood(self, X: sp.csr_matrix) -> np.ndarray:
        raise NotImplementedError


class BernoulliNB(_NaiveBayes):
    """Naive Bayes with the Bernoulli event model: features are word existence, x_j in {0, 1}.

    Any count above zero is treated as 1, so count features can be passed in as well.
    After fit, feature_log_prob_[c, j] = log p(x_j = 1 | y = c).
    """

    def _prepare(self, X, fitting: bool = True) -> sp.csr_matrix:
        X = super()._prepare(X, fitting)
        X.data = (X.data > 0).astype(np.float64)
        return X

    def _fit_feature_probs(self) -> None:
        # phi_{j|y} = (alpha + #docs of class y containing word j) / (2 alpha + #docs of class y)
        numerator = self.feature_count_ + self.alpha
        denominator = self.class_count_[:, None] + 2 * self.alpha
        self.feature_log_prob_ = np.log(numerator) - np.log(denominator)
        # 1 - phi_{j|y} = (alpha + #docs of class y without word j) / (same denominator),
        # and that numerator equals denominator - numerator. Computed from counts, not as
        # log(1 - phi), to keep full precision when phi is close to 1.
        self.neg_feature_log_prob_ = np.log(denominator - numerator) - np.log(denominator)

    def _joint_log_likelihood(self, X: sp.csr_matrix) -> np.ndarray:
        # log p(x | y) = sum_j [ x_j log phi_{j|y} + (1 - x_j) log(1 - phi_{j|y}) ]
        #             = sum_j x_j (log phi_{j|y} - log(1 - phi_{j|y})) + sum_j log(1 - phi_{j|y})
        # The second form only touches the words present in x, which keeps X sparse.
        log_odds = self.feature_log_prob_ - self.neg_feature_log_prob_
        return X @ log_odds.T + self.neg_feature_log_prob_.sum(axis=1) + self.class_log_prior_


class MultinomialNB(_NaiveBayes):
    """Naive Bayes with the multinomial event model: features are word counts.

    After fit, feature_log_prob_[c, j] = log p(word = j | y = c), the probability that a single
    word drawn from a document of class c is word j.
    """

    def _fit_feature_probs(self) -> None:
        # phi_{j|y} = (alpha + count of word j in class y) / (alpha |V| + total words in class y).
        # The total words in class y is sum_i 1{y_i = y} d_i, the paper's denominator, counting
        # only words that are in the vocabulary so that the phi_{j|y} sum to 1 over j.
        numerator = self.feature_count_ + self.alpha
        denominator = self.feature_count_.sum(axis=1, keepdims=True) + self.alpha * self.n_features_in_
        self.feature_log_prob_ = np.log(numerator) - np.log(denominator)

    def _joint_log_likelihood(self, X: sp.csr_matrix) -> np.ndarray:
        # log p(x | y) = sum_j x_j log phi_{j|y}, plus a multinomial coefficient that is the same
        # for every class and cancels in predict_proba, so it is left out.
        return X @ self.feature_log_prob_.T + self.class_log_prior_
