"""Streamlit demo for the political bias classifier.

Run: streamlit run app/streamlit_app.py

Loads saved models from models/ and the article test set from data/processed/articles.csv.
The app still starts if models or data are missing; it shows a clear message in that case.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polibias import config  # noqa: E402
from polibias.data.articles import binary_articles  # noqa: E402
from polibias.text import no_punct  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Political Bias Classifier",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling — team palette, Georgia headings, Calibri body, no emojis
# ---------------------------------------------------------------------------

PALETTE = dict(
    midnight="#051218",
    deep_teal="#053944",
    ocean_teal="#0A5C68",
    turquoise="#088192",
    gold="#BCAC79",
    gold_dark="#9C8A4F",
    light="#F2EFE6",
    muted="#5B6B70",
)

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IM+Fell+English&family=Lato:wght@300;400;600&display=swap');

      html, body, [class*="css"] {{
          background-color: {PALETTE["light"]};
          color: {PALETTE["midnight"]};
      }}

      /* Top header bar */
      .app-header {{
          background: {PALETTE["midnight"]};
          color: {PALETTE["light"]};
          padding: 1.4rem 2rem 1rem 2rem;
          margin: -1rem -1rem 2rem -1rem;
      }}
      .app-header h1 {{
          font-family: Georgia, serif;
          font-size: 1.65rem;
          font-weight: 400;
          letter-spacing: 0.04em;
          margin: 0;
          color: {PALETTE["light"]};
      }}
      .app-header p {{
          font-family: Calibri, sans-serif;
          font-size: 0.82rem;
          color: {PALETTE["muted"]};
          margin: 0.25rem 0 0 0;
          letter-spacing: 0.06em;
          text-transform: uppercase;
      }}

      /* Section labels */
      .section-label {{
          font-family: Calibri, sans-serif;
          font-size: 0.72rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: {PALETTE["muted"]};
          margin-bottom: 0.3rem;
      }}

      /* Result card */
      .result-card {{
          background: {PALETTE["midnight"]};
          border-radius: 6px;
          padding: 1.5rem 2rem;
          margin-top: 1rem;
      }}
      .result-label {{
          font-family: Calibri, sans-serif;
          font-size: 0.72rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: {PALETTE["muted"]};
          margin-bottom: 0.2rem;
      }}
      .result-prediction {{
          font-family: Georgia, serif;
          font-size: 2.2rem;
          font-weight: 400;
          margin: 0;
      }}
      .result-left  {{ color: {PALETTE["turquoise"]}; }}
      .result-right {{ color: {PALETTE["gold"]}; }}

      /* Confidence bar track */
      .bar-track {{
          background: {PALETTE["deep_teal"]};
          border-radius: 3px;
          height: 8px;
          margin: 0.6rem 0 1.2rem 0;
          position: relative;
      }}
      .bar-fill-left  {{ background: {PALETTE["turquoise"]}; border-radius: 3px; height: 8px; }}
      .bar-fill-right {{ background: {PALETTE["gold"]};     border-radius: 3px; height: 8px; }}

      /* Driving words */
      .word-chip {{
          display: inline-block;
          background: {PALETTE["deep_teal"]};
          color: {PALETTE["light"]};
          font-family: Calibri, monospace;
          font-size: 0.80rem;
          padding: 0.2rem 0.55rem;
          border-radius: 3px;
          margin: 0.2rem 0.2rem 0.2rem 0;
      }}

      /* Sidebar */
      section[data-testid="stSidebar"] {{
          background: {PALETTE["deep_teal"]};
      }}
      section[data-testid="stSidebar"] * {{
          color: {PALETTE["light"]} !important;
      }}
      section[data-testid="stSidebar"] .stSelectbox label,
      section[data-testid="stSidebar"] .stRadio label {{
          color: {PALETTE["light"]} !important;
          font-family: Calibri, sans-serif;
          font-size: 0.85rem;
      }}

      /* Text area */
      .stTextArea textarea {{
          background: white;
          border: 1px solid {PALETTE["muted"]};
          border-radius: 4px;
          font-family: Calibri, sans-serif;
          font-size: 0.95rem;
          color: {PALETTE["midnight"]};
      }}

      /* Streamlit button */
      .stButton > button {{
          background: {PALETTE["ocean_teal"]};
          color: {PALETTE["light"]};
          border: none;
          border-radius: 4px;
          font-family: Calibri, sans-serif;
          font-size: 0.9rem;
          letter-spacing: 0.04em;
          padding: 0.5rem 1.5rem;
          transition: background 0.18s;
      }}
      .stButton > button:hover {{
          background: {PALETTE["turquoise"]};
      }}

      div[data-testid="stMetricValue"] {{
          font-family: Georgia, serif;
          color: {PALETTE["midnight"]};
      }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="app-header">
      <h1>Political Bias Classifier</h1>
      <p>UE24CS352A Machine Learning &nbsp;|&nbsp; Team 16 &nbsp;|&nbsp;
         Sonu Akash Soundararajan &amp; Rahul Saran Sharma</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model registry: name -> (joblib path, type)
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    "LinearSVC — in-domain articles (Ext 3, best)": {
        "path": config.MODELS_DIR / "ext3_in_domain.joblib",
        "kind": "pipeline",
        "description": "Trained on 19,102 AllSides news articles. "
                       "Best article classifier: 66.5% accuracy (+6.7 pp over majority baseline).",
    },
    "TF-IDF + Logistic Regression (Ext 1)": {
        "path": config.MODELS_DIR / "ext1_tfidf_lr.joblib",
        "kind": "pipeline",
        "description": "Tweet-trained. TF-IDF unigrams + bigrams with logistic regression. "
                       "Best tweet model: 83.3% accuracy on held-out members.",
    },
    "LinearSVC — tweet-trained (model zoo)": {
        "path": config.MODELS_DIR / "linear_svc.joblib",
        "kind": "pipeline",
        "description": "Tweet-trained LinearSVC (C = 0.1, tuned by grouped cross-validation). "
                       "80.9% accuracy on held-out congress members.",
    },
    "LinearSVC balanced — tweet-trained": {
        "path": config.MODELS_DIR / "linear_svc_balanced.joblib",
        "kind": "pipeline",
        "description": "Same as above with class_weight='balanced'. "
                       "Better macro F1 (0.799) at roughly the same accuracy.",
    },
    "Logistic Regression — tweet-trained": {
        "path": config.MODELS_DIR / "logistic_regression.joblib",
        "kind": "pipeline",
        "description": "Tweet-trained logistic regression (C = 1.0, count features). "
                       "80.9% accuracy, comparable to LinearSVC.",
    },
    "Complement NB — tweet-trained": {
        "path": config.MODELS_DIR / "complement_nb.joblib",
        "kind": "pipeline",
        "description": "Complement Naive Bayes (alpha = 0.1, tuned by grouped CV). "
                       "80.1% accuracy, lightest model.",
    },
}


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    import joblib
    from sklearn.pipeline import Pipeline as _Pipeline

    obj = joblib.load(path)
    if isinstance(obj, dict) and "vectorizer" in obj and "classifier" in obj:
        return _Pipeline([
            ("features", obj["vectorizer"]),
            ("classifier", obj["classifier"]),
        ])
    return obj


@st.cache_data(show_spinner=False)
def load_article_test() -> pd.DataFrame | None:
    if not config.ARTICLES_CSV.exists():
        return None
    articles = pd.read_csv(config.ARTICLES_CSV, dtype=str, keep_default_na=False)
    test = binary_articles(articles[articles["split"] == "test"])
    test["text"] = (test["title"].fillna("") + " " + test["content"].fillna("")).str.strip()
    return test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Driving words — works for pipelines with a vectorizer + linear classifier
# ---------------------------------------------------------------------------

def top_driving_words(model, text: str, predicted: str, n: int = 10) -> list[str]:
    """Return up to n words from text that most pushed the prediction toward `predicted`."""
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
            # For LinearSVC/LR: positive coef pushes toward classes_[1] (usually "R")
            classes = list(clf.classes_) if hasattr(clf, "classes_") else ["D", "R"]
            if len(classes) == 2 and predicted == classes[1]:
                scores = coef * present
            else:
                scores = -coef * present
        elif hasattr(clf, "feature_log_prob_"):
            classes = list(clf.classes_)
            pred_idx = classes.index(predicted)
            other_idx = 1 - pred_idx
            scores = (clf.feature_log_prob_[pred_idx] - clf.feature_log_prob_[other_idx]) * present
        else:
            return []

        top_idx = np.argsort(-scores)[:n]
        return [str(vocab[i]) for i in top_idx if scores[i] > 0]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Confidence from decision score / probability
# ---------------------------------------------------------------------------

def predict_with_confidence(model, text: str) -> tuple[str, float, float]:
    """Returns (predicted_label, p_left, p_right)."""
    try:
        if hasattr(model, "decision_function"):
            score = float(model.decision_function([text])[0])
            p_right = 1.0 / (1.0 + np.exp(-score))
        elif hasattr(model, "predict_proba"):
            proba = model.predict_proba([text])[0]
            classes = list(model.classes_) if hasattr(model, "classes_") else ["D", "R"]
            d_idx = classes.index("D") if "D" in classes else 0
            r_idx = classes.index("R") if "R" in classes else 1
            p_right = float(proba[r_idx])
            _ = float(proba[d_idx])
        else:
            pred = model.predict([text])[0]
            return str(pred), 0.5, 0.5

        p_left = 1.0 - p_right
        pred = "R" if p_right >= 0.5 else "D"
        return pred, p_left, p_right
    except Exception:
        pred = str(model.predict([text])[0])
        p_right = 1.0 if pred == "R" else 0.0
        return pred, 1.0 - p_right, p_right


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"<p style='font-family:Georgia,serif;font-size:1.05rem;color:{PALETTE['light']};margin-bottom:0.2rem;'>Model</p>",
        unsafe_allow_html=True,
    )
    selected_name = st.selectbox(
        "Select model",
        options=list(MODEL_REGISTRY.keys()),
        index=0,
        label_visibility="collapsed",
    )
    meta = MODEL_REGISTRY[selected_name]
    st.markdown(
        f"<p style='font-family:Calibri,sans-serif;font-size:0.80rem;color:{PALETTE['muted']};margin-top:0.6rem;'>"
        f"{meta['description']}</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        f"<p style='font-family:Georgia,serif;font-size:1.05rem;color:{PALETTE['light']};margin-bottom:0.2rem;'>Input</p>",
        unsafe_allow_html=True,
    )
    input_mode = st.radio(
        "Input mode",
        ["Paste text", "Pick a test article"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f"<p style='font-family:Calibri,sans-serif;font-size:0.75rem;color:{PALETTE['muted']};'>"
        "Models trained on US House tweets (Jan–Apr 2021) and AllSides articles (media-based split). "
        "Left = Democratic, Right = Republican."
        "</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

model_path = str(meta["path"])
model_available = meta["path"].exists()

if not model_available:
    st.warning(
        f"Model file not found: {meta['path'].name}. "
        "Run `python scripts/run_all.py` to generate all models, then restart the app."
    )

model = None
if model_available:
    with st.spinner("Loading model..."):
        model = load_model(model_path)

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

articles_df = load_article_test()
input_text = ""
true_label: str | None = None

col_main, col_gap = st.columns([3, 1])

with col_main:
    if input_mode == "Paste text":
        st.markdown("<p class='section-label'>Article or tweet text</p>", unsafe_allow_html=True)
        input_text = st.text_area(
            "Text",
            height=240,
            placeholder="Paste a news article excerpt or any political text here.",
            label_visibility="collapsed",
        )
        run_btn = st.button("Classify")

    else:
        if articles_df is None:
            st.warning(
                "Article test set not found at data/processed/articles.csv. "
                "Run `python scripts/run_all.py` to download and process the data."
            )
            run_btn = False
        else:
            st.markdown("<p class='section-label'>Test article</p>", unsafe_allow_html=True)
            preview_df = articles_df[["title", "source", "party"]].copy()
            preview_df["label"] = preview_df["party"].map({"D": "Left", "R": "Right"})
            choices = [
                f"[{row.label}] {row.source} — {str(row.title)[:80]}"
                for row in preview_df.itertuples()
            ]
            pick = st.selectbox("Select article", choices, label_visibility="collapsed")
            idx = choices.index(pick)
            row = articles_df.iloc[idx]
            input_text = str(row.get("text", row.get("content", "")))
            true_label = str(row["party"])
            with st.expander("Article text"):
                st.write(input_text[:2000] + ("..." if len(input_text) > 2000 else ""))
            run_btn = st.button("Classify")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

if run_btn and model is not None:
    text = input_text.strip()
    if not text:
        st.warning("Enter some text before classifying.")
    else:
        pred, p_left, p_right = predict_with_confidence(model, text)
        label_str = "Left (Democratic)" if pred == "D" else "Right (Republican)"
        css_class = "result-left" if pred == "D" else "result-right"
        bar_class = "bar-fill-left" if pred == "D" else "bar-fill-right"
        conf = p_left if pred == "D" else p_right
        bar_w = round(conf * 100)

        # Result card
        st.markdown(
            f"""
            <div class="result-card">
              <p class="result-label">Predicted leaning</p>
              <p class="result-prediction {css_class}">{label_str}</p>
              <div class="bar-track">
                <div class="{bar_class}" style="width:{bar_w}%;"></div>
              </div>
              <p style="font-family:Calibri,sans-serif;font-size:0.82rem;color:{PALETTE['muted']};">
                Left &nbsp; {p_left:.1%} &nbsp;|&nbsp; Right &nbsp; {p_right:.1%}
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # True label (test set mode)
        if true_label is not None:
            true_str = "Left (Democratic)" if true_label == "D" else "Right (Republican)"
            match = pred == true_label
            colour = PALETTE["turquoise"] if match else PALETTE["gold_dark"]
            verdict = "Correct" if match else "Incorrect"
            st.markdown(
                f"<p style='font-family:Calibri,sans-serif;font-size:0.85rem;"
                f"color:{colour};margin-top:0.6rem;'>"
                f"{verdict} &mdash; true label: {true_str}</p>",
                unsafe_allow_html=True,
            )

        # Driving words
        words = top_driving_words(model, text, pred)
        if words:
            st.markdown(
                f"<p class='section-label' style='margin-top:1.4rem;'>Words that drove this prediction</p>",
                unsafe_allow_html=True,
            )
            chips = " ".join(f"<span class='word-chip'>{w}</span>" for w in words)
            st.markdown(chips, unsafe_allow_html=True)

elif run_btn and not model_available:
    st.error("Model not loaded. See the warning above.")

# ---------------------------------------------------------------------------
# Results reference table (collapsed by default)
# ---------------------------------------------------------------------------

with st.expander("Accuracy reference — all models on news articles (1,001 AllSides test articles)"):
    ref = {
        "Model": [
            "Always R (majority baseline)",
            "Bernoulli NB — tweet-trained",
            "Multinomial NB — tweet-trained",
            "LinearSVC C=0.06 — tweet-trained",
            "LinearSVC C=0.1 — tweet-trained",
            "TF-IDF + LR — tweet-trained (Ext 1)",
            "LinearSVC — in-domain articles (Ext 3)",
        ],
        "Test accuracy": [
            "59.8%", "60.6%", "60.5%", "57.7%", "58.4%", "59.2%", "66.5%",
        ],
        "Macro F1": [
            "0.374", "0.590", "0.554", "0.569", "0.574", "0.591", "0.658",
        ],
    }
    st.dataframe(pd.DataFrame(ref), use_container_width=True, hide_index=True)
