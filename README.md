# Categorizing News Articles on Political Bias

UE24CS352A Machine Learning, Mini-Project (Section D, Team 16, Problem 57)

| Member | SRN | GitHub |
|---|---|---|
| Sonu Akash Soundararajan | PES1UG24AM908 | [SonuAkash-collab](https://github.com/SonuAkash-collab) |
| Rahul Saran Sharma | PES1UG24AM218 | [ZapCannonYT](https://github.com/ZapCannonYT) |

## Problem

Given a news article, predict its political leaning. We follow the Stanford CS229 (Spring 2021) project
"Categorizing News Articles on Political Bias" by Crain and Erhan, which trains classifiers on tweets
from members of the US House of Representatives (labelled by the author's party) and then uses them to
label news articles as Democratic or Republican leaning.

Reference paper: https://cs229.stanford.edu/proj2021spr/report2/81996990.pdf

## Plan

1. Reproduce the paper: Bernoulli and Multinomial Naive Bayes (from scratch) and a linear SVM, trained on
   117th Congress tweets (January to April 2021), evaluated on held-out tweets and on news articles.
2. Evaluate fairly: split tweets by congressperson rather than by random rows, and tune hyperparameters
   with cross-validation on the training data only.
3. Extend the baseline with our own improvements and measure each one.

## Setup

Requires Python 3.10 or newer (tested with 3.13). All package versions are pinned in
`requirements.txt`. Run every command below from the repository root.

Windows (PowerShell):

```powershell
git clone https://github.com/SonuAkash-collab/news-political-bias-classifier.git
cd news-political-bias-classifier
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, first run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same window, or use
`.venv\Scripts\activate.bat` from Command Prompt instead.

Linux and macOS:

```bash
git clone https://github.com/SonuAkash-collab/news-political-bias-classifier.git
cd news-political-bias-classifier
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check the install by running the tests. They use small fake data and need no download:

```bash
python -m pytest
```

## Data

Raw and processed data are not stored in the repository (`data/raw/`, `data/interim/` and
`data/processed/` are in `.gitignore`). The commands below download everything into `data/raw/` and
write the cleaned tables to `data/processed/`. `scripts/run_all.py` also does this by itself when
the processed files are missing, so this section can be skipped if you go straight to
[How to run](#how-to-run).

Tweets and party labels:

```bash
python scripts/download_tweets.py
python scripts/run_all.py --only tweets
```

The first command downloads one file per day from 3 January 2021 to 1 April 2021 (the paper's date
range, 89 files) into `data/raw/tweets/`, plus the member list
`data/raw/historical-users-filtered.json`. The second keeps House members (`chamber == "house"`,
`type == "member"`) of party D or R, joins every tweet to its author's party through the account
handle, drops retweets and duplicate texts, and writes `data/processed/tweets.csv` with the columns
`text`, `screen_name`, `member_id`, `party` and `time`. `member_id` is the member's Bioguide ID, so
a member's office and campaign accounts count as one person when we split by congressperson.

News articles:

```bash
python scripts/download_articles.py
```

This downloads the Article-Bias-Prediction archive (about 135 MB), extracts the 37,554 article JSON files
and the published splits into `data/raw/articles/`, and writes `data/processed/articles.csv` with
the media-based split attached (`--split random` attaches the random split instead). The media-based
split assigns 30,246 of the 37,554 articles to train, validation or test (26,590 / 2,356 / 1,300);
the rest are not in any split and are left out. Both download
scripts skip files that are already present, so they can be re-run after an interruption.

Sources and citations:

- Tweets: Alex Litel, *congresstweets*, a daily archive of tweets by members of the US Congress.
  https://github.com/alexlitel/congresstweets (files `data/YYYY-MM-DD.json`).
- Party labels: Alex Litel, *congresstweets-automator*, file `data/historical-users-filtered.json`.
  https://github.com/alexlitel/congresstweets-automator
- News articles: Ramy Baly, Giovanni Da San Martino, James Glass and Preslav Nakov. "We Can Detect
  Your Bias: Predicting the Political Ideology of News Articles." In Proceedings of EMNLP 2020.
  https://github.com/ramybaly/Article-Bias-Prediction. Labels (left, center, right) come from
  AllSides. We use this dataset in place of the paper's 100 poll-labelled articles, which were never
  published.
- Reference paper: Dylan Crain and Yagmur Erhan. "Categorizing News Articles on Political Bias."
  Stanford CS229 project report, Spring 2021.
  https://cs229.stanford.edu/proj2021spr/report2/81996990.pdf

## How to run

Run everything in order (download and process the data if needed, every experiment, then combine
the results):

```bash
python scripts/run_all.py
```

With the data already downloaded, the full run takes about 15 minutes on a laptop. Each experiment
writes its own file in `results/` (for example `results/exp1.json`), its figures in
`results/figures/` and any trained model in `models/`. At the end `run_all.py` merges every
`results/*.json` into `results/results.json`, one key per experiment, and prints one summary line per
step. An experiment module that does not exist yet is reported as skipped. A step that fails prints
its traceback, the remaining steps still run, and the script exits with status 1.

For a fast end-to-end check, add `--quick`:

```bash
python scripts/run_all.py --quick
```

This keeps 10% of the congresspeople of each party (at least 10 per party) and 10% of the article
test set, and writes all outputs under `data/interim/quick_run/`, so the real results and models are
never overwritten. It still needs the downloaded data and takes a few minutes. Use
`--output-dir PATH` to send the outputs somewhere else.

To run one step on its own, give its name to `--only`. Several names can be given at once, and
`--only` can be combined with `--quick`, for example `python scripts/run_all.py --quick --only exp1 ext1`.

| Step | What it does | Output |
|---|---|---|
| `tweets` | Download and process the tweets if `data/processed/tweets.csv` is missing | `data/processed/tweets.csv` |
| `articles` | Download and process the articles if `data/processed/articles.csv` is missing | `data/processed/articles.csv` |
| `model_zoo` | LinearSVC, logistic regression and Complement NB tuned with grouped cross-validation, plus an RBF SVM and a majority baseline, on the grouped tweet split | `results/model_zoo.json`, `results/figures/model_zoo.png`, `models/*.joblib` |
| `exp1` | Paper Experiment 1: the three paper models on held-out tweets, random split against grouped split, for the three vocabulary variants | `results/exp1.json`, `results/figures/exp1_tweet_test.png` |
| `exp2` | Paper Experiment 2: the ten most indicative words per party for each vocabulary variant | `results/exp2.json`, `results/figures/exp2_top_words.png` |
| `exp3` | Paper Experiment 3: models trained on all tweets, tested on the left and right news articles | `results/exp3.json`, `results/figures/exp3_*.png` |
| `ext1` | Extension 1: TF-IDF word unigrams and bigrams with logistic regression, tuned with grouped cross-validation | `results/ext1.json`, `results/figures/ext1_tfidf_lr.png`, `models/ext1_tfidf_lr.joblib` |
| `ext2` | Extension 2: score each article in tweet-sized chunks and average the chunk probabilities | `results/ext2.json` |
| `ext3` | Extension 3: train and test on articles to measure the cost of the tweet-to-article domain gap | `results/ext3.json` |

Experiments 1 and 2 and Extension 1 can also be started as modules, with `src` on the import path:

```bash
# Linux and macOS
PYTHONPATH=src python -m polibias.experiments.exp1_tweet_test
```

```powershell
# Windows (PowerShell)
$env:PYTHONPATH = "src"
python -m polibias.experiments.exp1_tweet_test
```

All randomness uses seed 42 (`polibias.config.SEED`), so a rerun on the same data gives the same
numbers.

## Repository structure

Owner A is Sonu Akash Soundararajan and owner B is Rahul Saran Sharma. Each person writes the tests
for their own modules.

```
news-political-bias-classifier/
  README.md                          setup, data, how to run (A); demo and results (B)
  requirements.txt                   pinned package versions (A; both add pins)
  pyproject.toml                     pytest settings: puts src/ on the import path
  scripts/
    download_tweets.py               download the daily tweet files and the member list (A)
    download_articles.py             download and process the AllSides articles (B)
    run_svm_models.py                run the grouped model zoo on its own (B)
    run_all.py                       run every step in order and combine the results (A)
  src/polibias/
    config.py                        paths, seed, date range, source URLs, colour palette (A)
    text.py                          tokenisers for the three vocabulary variants (A)
    features.py                      binary and count vectorisers used by every model (A)
    evaluation.py                    random and grouped splits, metrics, plot helpers (A)
    data/tweets.py                   load tweets, attach the party, dedupe, save tweets.csv (A)
    data/articles.py                 load articles and their published splits, binary subset (B)
    data/download.py                 download and extract the article archive (B)
    models/naive_bayes.py            Bernoulli and Multinomial Naive Bayes from scratch (A)
    models/svm.py                    LinearSVC and comparison models, grouped CV tuning (B)
    experiments/model_zoo.py         grouped tweet evaluation of the model zoo (B)
    experiments/exp1_tweet_test.py   paper Experiment 1, random vs grouped split (A)
    experiments/exp2_top_words.py    paper Experiment 2, top words per party (A)
    experiments/exp3_articles.py     paper Experiment 3, tweets to articles, error analysis (B)
    experiments/ext1_tfidf_lr.py     Extension 1, TF-IDF with bigrams and logistic regression (A)
    experiments/ext2_chunk_vote.py   Extension 2, chunk voting on articles (B)
    experiments/ext3_in_domain.py    Extension 3, in-domain training on articles (B)
  app/                               Streamlit demo and command-line fallback (B)
  tests/                             pytest unit and smoke tests on small fake data (each owner)
  results/                           one JSON file per experiment, results.json, figures/
  models/                            saved models (.joblib) that the demo loads
  docs/writeup/                      two-page write-up (A: first half; B: second half and PDF)
  docs/slides/                       slide deck (shared, split by slide)
  data/                              raw/, processed/ and interim/ (not committed)
```

Modules in this list that are not in the repository yet are still being written; `run_all.py`
skips them until they exist.

## Demo

The demo loads trained models from `models/` and lets you paste an article or pick one from the
test set. It shows the predicted leaning (Left/Democratic or Right/Republican), a confidence
score derived from the model's decision boundary, and the vocabulary features that most pushed
the prediction in that direction.

**Prerequisites:** run `python scripts/run_all.py` at least once so that all `.joblib` files in
`models/` exist.

### Streamlit web interface (preferred)

```bash
streamlit run app/streamlit_app.py
```

Open the URL printed by Streamlit (default: http://localhost:8501) in a browser. The sidebar
lets you pick any of the six saved models and switch between free-text input and a dropdown of
the 1,001 article test set. The recommended default is the in-domain LinearSVC (Extension 3,
66.5% accuracy).

### Command-line fallback

Use this if the Streamlit server cannot start during the review session.

```bash
# Interactive mode — type text, press Enter twice to classify, type "quit" to exit.
python app/cli.py

# Single-shot mode.
python app/cli.py --text "The president signed the energy bill into law today."

# Choose a different model (default: ext3).
python app/cli.py --model ext1 --text "Gun rights are fundamental to American freedom."
```

Available model keys: `ext3` (in-domain LinearSVC, best on articles), `ext1` (TF-IDF + LR),
`linear` (LinearSVC tweet-trained), `logistic` (LR tweet-trained), `cnb` (Complement NB).

The CLI prints an ASCII confidence bar and the driving vocabulary words:

```
====================================================================
 Political Bias Classifier — command-line interface
====================================================================
 Model : LinearSVC in-domain articles (Ext 3)
--------------------------------------------------------------------
 Prediction  : Right (Republican)   (68.2% confidence)
  Left  31.8% [############............................]
  Right 68.2% [##########################..............]
 Driving words: energy  oil  gas  pipeline  federal  regulation
====================================================================
```

## Results

All numbers below are on the **media-based split** of 1,001 AllSides test articles (402 Left,
599 Right). Majority baseline always predicts Right (the larger class) and scores 59.8%.

### Tweet-trained models on news articles (Experiment 3)

| Model | Accuracy | Macro F1 | vs. baseline |
|---|---|---|---|
| Majority baseline (always Right) | 59.8% | 0.374 | — |
| Bernoulli NB | 60.6% | 0.590 | +0.8 pp |
| Multinomial NB | 60.5% | 0.554 | +0.7 pp |
| LinearSVC C=0.06 (paper's C) | 57.7% | 0.569 | −2.1 pp |
| LinearSVC C=0.1 (tuned) | 58.4% | 0.574 | −1.4 pp |

All tweet-trained models struggle on articles: the training domain (short, partisan social-media
posts) differs substantially from journalistic prose.

### Extensions

| Extension | Accuracy | Macro F1 | vs. baseline |
|---|---|---|---|
| Ext 1: TF-IDF + LR — tweet-trained | 59.2% | 0.591 | −0.6 pp |
| Ext 2: Chunk voting (50-word chunks, Bernoulli NB) | 60.8% | 0.571 | +1.0 pp |
| Ext 2: Chunk voting (50-word chunks, LinearSVC) | 51.8% | 0.513 | −8.0 pp |
| Ext 3: LinearSVC — in-domain articles | **66.5%** | **0.658** | **+6.7 pp** |

Extension 3 (in-domain training on AllSides articles with a media-based train/test split) gives
the largest improvement and is the model used by the demo.

### Tweet classifier on held-out congress members (model zoo)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Majority baseline (always Democratic) | 61.2% | 0.380 |
| Complement NB | 80.1% | 0.791 |
| LinearSVC | 80.9% | 0.792 |
| LinearSVC balanced | 80.8% | 0.799 |
| Logistic Regression | 80.9% | 0.793 |
| Logistic Regression balanced | 80.8% | 0.798 |
| Ext 1: TF-IDF + LR | 83.3% | 0.824 |

On the grouped split (each congressperson entirely in train or test) accuracy is 5.0 points lower
on average than on the paper's random split (range 2.1 to 6.8 points over the three paper models
and three vocabulary variants, `results/exp1.json`). The random split lets a model recognise
authors (handles, district names, pet phrases) instead of party language.
