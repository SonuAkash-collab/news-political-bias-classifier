"""Split text into tokens for the paper's three vocabulary variants.

Crain and Erhan (2021) build the vocabulary in three ways (Section 4.1, Table 1):

    all_words   the text as it is, only lowercased and split on whitespace
    no_punct    punctuation removed
    no_markers  punctuation, hashtags, @-mentions and URLs removed

Each function takes one string and returns a list of tokens, so it can be passed to
scikit-learn's CountVectorizer as the analyzer (polibias.features does this).
"""

import re
from collections.abc import Callable

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
# "#" or "@" only counts as a marker at the start of a word, so e-mail addresses are left alone.
_HASHTAG = re.compile(r"(?<!\w)#\w+")
_MENTION = re.compile(r"(?<!\w)@\w+")
# Every character that is not a letter, digit or whitespace. \w also matches "_", so add it back.
_PUNCT = re.compile(r"[^\w\s]|_")


def all_words(text: str) -> list[str]:
    """Lowercase and split on whitespace. Nothing is removed.

    Punctuation stays attached to words, so "rescue", "rescue," and "rescue." are three different
    tokens, and hashtags, @-mentions and URLs are kept whole ("#stopasianhate", "@repcasten").
    """
    return text.lower().split()


def no_punct(text: str) -> list[str]:
    """Lowercase, delete all punctuation, split on whitespace.

    Removes every character that is not a letter, digit or whitespace: punctuation marks, "#",
    "@", "_", quotes, apostrophes, dashes and emojis. Characters are deleted, not replaced by a
    space, so "#StopAsianHate" becomes "stopasianhate", "@RepCasten" becomes "repcasten" and
    "don't" becomes "dont", as in the paper's Table 1. Hashtag and mention words are kept.
    """
    return _PUNCT.sub("", text.lower()).split()


def no_markers(text: str) -> list[str]:
    """Remove URLs, hashtags and @-mentions as whole tokens, then punctuation as in no_punct.

    URLs (starting with http://, https:// or www.) are removed first, so their pieces do not
    survive as words. A hashtag or mention is removed together with its word ("#StopAsianHate"
    and "@RepCasten" both disappear), but only when "#" or "@" starts a word, so an e-mail address
    such as "info@house.gov" is kept as ordinary text ("infohousegov").
    """
    text = _URL.sub(" ", text)
    text = _HASHTAG.sub(" ", text)
    text = _MENTION.sub(" ", text)
    return no_punct(text)


VARIANTS: dict[str, Callable[[str], list[str]]] = {
    "all_words": all_words,
    "no_punct": no_punct,
    "no_markers": no_markers,
}


def get_variant(name: str) -> Callable[[str], list[str]]:
    """Return the tokenising function for a variant name ("all_words", "no_punct", "no_markers")."""
    try:
        return VARIANTS[name]
    except KeyError:
        raise ValueError(f"Unknown vocabulary variant {name!r}; choose one of {list(VARIANTS)}") from None
