"""Vectorisers that turn texts into word-feature matrices. Every model in the project uses these.

    from polibias.features import binary_vectorizer, count_vectorizer

    vec = binary_vectorizer("no_punct", min_df=2)
    X_train = vec.fit_transform(train_texts)  # sparse matrix: one row per text, one column per word
    X_test = vec.transform(test_texts)        # same columns; words not seen in training are ignored
    words = vec.get_feature_names_out()       # column j of X is the word words[j]

binary_vectorizer gives word existence (1 if the word occurs in the text, else 0), the features
of Bernoulli Naive Bayes. count_vectorizer gives how often each word occurs, the features of
Multinomial Naive Bayes. Both return an unfitted scikit-learn CountVectorizer, so they work in a
Pipeline, inside cross-validation and with joblib.dump.

Fit on the training texts only, then transform test texts with the same fitted vectoriser.
The variant names are those of polibias.text: "all_words", "no_punct", "no_markers".
"""

from sklearn.feature_extraction.text import CountVectorizer

from polibias.text import get_variant


def binary_vectorizer(variant: str = "all_words", min_df: int | float = 1) -> CountVectorizer:
    """Word existence features: entry (i, j) is 1 if word j occurs in text i, else 0.

    variant: vocabulary variant, one of "all_words", "no_punct", "no_markers".
    min_df: keep only words that occur in at least this many training texts (an int), or in at
        least this fraction of them (a float in (0, 1]). The default 1 keeps every word, as the
        paper does.
    """
    return _make(variant, min_df, binary=True)


def count_vectorizer(variant: str = "all_words", min_df: int | float = 1) -> CountVectorizer:
    """Word count features: entry (i, j) is how many times word j occurs in text i.

    Arguments are the same as for binary_vectorizer.
    """
    return _make(variant, min_df, binary=False)


def _make(variant: str, min_df: int | float, binary: bool) -> CountVectorizer:
    # A callable analyzer replaces CountVectorizer's own lowercasing and tokenising.
    return CountVectorizer(analyzer=get_variant(variant), min_df=min_df, binary=binary)
