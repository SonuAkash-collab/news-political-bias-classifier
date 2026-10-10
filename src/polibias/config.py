"""Shared settings: paths, random seed, tweet date range and colour palette."""

import os
import random
from datetime import date
from pathlib import Path

import numpy as np

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_TWEETS_DIR = RAW_DIR / "tweets"
RAW_ARTICLES_DIR = RAW_DIR / "articles"
USERS_FILE = RAW_DIR / "historical-users-filtered.json"
PROCESSED_DIR = DATA_DIR / "processed"
TWEETS_CSV = PROCESSED_DIR / "tweets.csv"
ARTICLES_CSV = PROCESSED_DIR / "articles.csv"
# Outputs. scripts/run_all.py --quick points POLIBIAS_OUTPUT_DIR somewhere else, so a fast check
# never overwrites the real results and models. Inputs (data/) are never redirected.
OUTPUT_DIR = Path(os.environ.get("POLIBIAS_OUTPUT_DIR", ROOT)).resolve()
RESULTS_DIR = OUTPUT_DIR / "results"
RESULTS_JSON = RESULTS_DIR / "results.json"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = OUTPUT_DIR / "models"

# Quick mode (scripts/run_all.py --quick): keep only this fraction of congresspeople and articles.
# 1.0 means the full data.
SAMPLE_FRACTION = float(os.environ.get("POLIBIAS_SAMPLE_FRACTION", "1.0"))
QUICK = SAMPLE_FRACTION < 1.0

# Reproducibility
SEED = 42

# Tweet date range used by the paper (117th Congress, both ends inclusive)
START_DATE = date(2021, 1, 3)
END_DATE = date(2021, 4, 1)

# Data sources
TWEETS_URL = "https://raw.githubusercontent.com/alexlitel/congresstweets/master/data/{day}.json"
USERS_URL = (
    "https://raw.githubusercontent.com/alexlitel/congresstweets-automator/"
    "master/data/historical-users-filtered.json"
)

# Colour palette (team palette from PROJECT_CONTEXT.md, "Visual style")
MIDNIGHT = "#051218"  # main dark colour, body text on light
DEEP_TEAL = "#053944"  # secondary dark
OCEAN_TEAL = "#0A5C68"  # mid-tone accents
TURQUOISE = "#088192"  # accent; Democratic / left
GOLD = "#BCAC79"  # highlight; Republican / right
GOLD_DARK = "#9C8A4F"  # gold that stays visible on light backgrounds
LIGHT = "#F2EFE6"  # off-white background
MUTED = "#5B6B70"  # muted text and grid lines
CENTER = "#8A949A"  # center grey
PARTY_COLORS = {"D": TURQUOISE, "R": GOLD, "left": TURQUOISE, "center": CENTER, "right": GOLD}


def set_seed(seed: int = SEED) -> None:
    """Seed Python and NumPy random generators."""
    random.seed(seed)
    np.random.seed(seed)
