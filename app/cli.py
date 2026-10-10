"""Command-line fallback for the political bias classifier demo.

Usage:
    python app/cli.py
    python app/cli.py --text "The president signed the bill into law."
    python app/cli.py --text "..." --model ext3

Models:
    ext3      LinearSVC in-domain articles (default, best on articles)
    ext1      TF-IDF + Logistic Regression (best tweet model)
    linear    LinearSVC tweet-trained (model zoo)
    logistic  Logistic Regression tweet-trained
    cnb       Complement NB tweet-trained

The CLI prints a plain-text result with an ASCII confidence bar. It is
intended as a backup during review sessions when the Streamlit server is
not available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polibias import config  # noqa: E402
from polibias.text import no_punct  # noqa: E402

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "ext3":     (config.MODELS_DIR / "ext3_in_domain.joblib",         "LinearSVC in-domain articles (Ext 3)"),
    "ext1":     (config.MODELS_DIR / "ext1_tfidf_lr.joblib",          "TF-IDF + LogisticRegression (Ext 1)"),
    "linear":   (config.MODELS_DIR / "linear_svc.joblib",             "LinearSVC tweet-trained"),
    "logistic": (config.MODELS_DIR / "logistic_regression.joblib",    "LogisticRegression tweet-trained"),
    "cnb":      (config.MODELS_DIR / "complement_nb.joblib",          "ComplementNB tweet-trained"),
}

DEFAULT_MODEL = "ext3"


# ---------------------------------------------------------------------------
# Model loading — handles both sklearn Pipeline and ext3 bundle dict
# ---------------------------------------------------------------------------

def _load_model(path: Path):
    """Load a model from disk. Returns a sklearn Pipeline.

    ext3_in_domain.joblib is saved as a dict bundle::

        {"vectorizer": <fitted vectorizer>, "classifier": <fitted model>, ...}

    All other models are direct sklearn Pipelines. This function wraps a
    bundle into a Pipeline so callers don't need to special-case it.
    """
    import joblib
    from sklearn.pipeline import Pipeline

    obj = joblib.load(path)
    if isinstance(obj, dict) and "vectorizer" in obj and "classifier" in obj:
        # Re-wrap into a Pipeline with the canonical step names.
        pipe = Pipeline([
            ("features", obj["vectorizer"]),
            ("classifier", obj["classifier"]),
        ])
        return pipe
    return obj

# Column widths for pretty printing
W_TOTAL = 68

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 40, fill: str = "#", empty: str = ".") -> str:
    """Simple ASCII progress bar from 0.0 to 1.0."""
    filled = round(value * width)
    return fill * filled + empty * (width - filled)


def _confidence(model, text: str) -> tuple[str, float, float]:
    """Return (predicted_label, p_left, p_right)."""
    if hasattr(model, "decision_function"):
        score = float(model.decision_function([text])[0])
        p_right = 1.0 / (1.0 + np.exp(-score))
    elif hasattr(model, "predict_proba"):
        proba = model.predict_proba([text])[0]
        classes = list(model.classes_)
        r_idx = classes.index("R") if "R" in classes else 1
        p_right = float(proba[r_idx])
    else:
        pred = str(model.predict([text])[0])
        p_right = 1.0 if pred == "R" else 0.0

    p_left = 1.0 - p_right
    pred = "R" if p_right >= 0.5 else "D"
    return pred, p_left, p_right


def _driving_words(model, text: str, predicted: str, n: int = 10) -> list[str]:
    """Top n vocabulary words present in text that drove the prediction."""
    try:
        if hasattr(model, "named_steps"):
            vec = model.named_steps.get("features") or model.named_steps.get("vectorizer")
            clf = model.named_steps.get("classifier")
        else:
            return []
        if vec is None or clf is None:
            return []

        vocab = np.array(vec.get_feature_names_out())
        tokens = set(no_punct(text))
        present = np.array([w in tokens for w in vocab])

        if hasattr(clf, "coef_"):
            coef = clf.coef_.ravel()
            classes = list(clf.classes_) if hasattr(clf, "classes_") else ["D", "R"]
            if len(classes) == 2 and predicted == classes[1]:
                feat_scores = coef * present
            else:
                feat_scores = -coef * present
        elif hasattr(clf, "feature_log_prob_"):
            classes = list(clf.classes_)
            pred_idx = classes.index(predicted)
            other_idx = 1 - pred_idx
            feat_scores = (clf.feature_log_prob_[pred_idx] - clf.feature_log_prob_[other_idx]) * present
        else:
            return []

        top_idx = np.argsort(-feat_scores)[:n]
        return [str(vocab[i]) for i in top_idx if feat_scores[i] > 0]
    except Exception:
        return []


def _sep(char: str = "-") -> str:
    return char * W_TOTAL


def classify(text: str, model_key: str = DEFAULT_MODEL) -> None:
    """Load model, classify text, print result."""
    model_key = model_key.lower()
    if model_key not in MODELS:
        print(f"Unknown model '{model_key}'. Choose from: {', '.join(MODELS)}")
        sys.exit(1)

    path, name = MODELS[model_key]
    if not path.exists():
        print(
            f"\n  Model not found: {path}\n"
            "  Run 'python scripts/run_all.py' to train and save all models.\n"
        )
        sys.exit(1)

    print(_sep("="))
    print(" Political Bias Classifier — command-line interface")
    print(_sep("="))
    print(f" Model : {name}")
    print(_sep())

    model = _load_model(path)
    pred, p_left, p_right = _confidence(model, text)

    label = "Left  (Democratic)" if pred == "D" else "Right (Republican)"
    conf = p_left if pred == "D" else p_right

    print(f" Prediction  : {label}")
    print(f" Confidence  : {conf:.1%}")
    print()
    print(f"  Left  {p_left:5.1%} [{_bar(p_left, 36)}]")
    print(f"  Right {p_right:5.1%} [{_bar(p_right, 36)}]")
    print()

    words = _driving_words(model, text, pred)
    if words:
        print(" Driving words:")
        print("  " + "  ".join(words))

    print(_sep("="))


def interactive(model_key: str = DEFAULT_MODEL) -> None:
    """Read-eval-print loop: classify one text at a time until the user quits."""
    path, name = MODELS.get(model_key, (None, None))
    if path is None or not path.exists():
        print(
            f"\n  Model not found for key '{model_key}'.\n"
            "  Run 'python scripts/run_all.py' to train and save all models.\n"
        )
        sys.exit(1)

    model = _load_model(path)
    print(_sep("="))
    print(f" Political Bias Classifier  |  {name}")
    print(" Type text and press Enter twice to classify. Type 'quit' to exit.")
    print(_sep("="))

    while True:
        lines: list[str] = []
        print("\nText (blank line to submit, 'quit' to exit):")
        while True:
            try:
                line = input()
            except EOFError:
                return
            if line.strip().lower() in ("quit", "exit", "q"):
                print("Goodbye.")
                return
            if line == "":
                break
            lines.append(line)

        text = " ".join(lines).strip()
        if not text:
            continue

        pred, p_left, p_right = _confidence(model, text)
        label = "Left  (Democratic)" if pred == "D" else "Right (Republican)"
        conf = p_left if pred == "D" else p_right

        print()
        print(_sep())
        print(f" Prediction  : {label}   ({conf:.1%} confidence)")
        print(f"  Left  {p_left:5.1%} [{_bar(p_left, 34)}]")
        print(f"  Right {p_right:5.1%} [{_bar(p_right, 34)}]")
        words = _driving_words(model, text, pred)
        if words:
            print(f" Driving words: {', '.join(words)}")
        print(_sep())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Political bias classifier — command-line interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python app/cli.py\n"
            "  python app/cli.py --text \"The president signed the energy bill.\"\n"
            "  python app/cli.py --model ext1 --text \"Gun control legislation passed today.\"\n"
            "\nModel keys:\n"
            + "\n".join(f"  {k:<10} {v[1]}" for k, v in MODELS.items())
        ),
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to classify. Omit for interactive mode.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MODELS),
        help=f"Model to use (default: {DEFAULT_MODEL}).",
    )
    args = parser.parse_args()

    if args.text is not None:
        classify(args.text.strip(), model_key=args.model)
    else:
        interactive(model_key=args.model)


if __name__ == "__main__":
    main()
