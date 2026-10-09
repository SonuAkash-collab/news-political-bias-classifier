"""Shared settings: paths, random seed, tweet date range and colour palette."""

import random
from datetime import date
from pathlib import Path

import numpy as np

# Paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_TWEETS_DIR = RAW_DIR / "tweets"
USERS_FILE = RAW_DIR / "historical-users-filtered.json"
PROCESSED_DIR = DATA_DIR / "processed"
TWEETS_CSV = PROCESSED_DIR / "tweets.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "results.json"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

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

# Colour palette
INK = "#1C2230"
PAPER = "#F4F1EA"
ACCENT = "#C47F2C"
MUTED = "#6B7280"
DEM = "#2E5E8C"
CENTER = "#9AA0A6"
REP = "#A33B3B"
PARTY_COLORS = {"D": DEM, "R": REP, "left": DEM, "center": CENTER, "right": REP}


def set_seed(seed: int = SEED) -> None:
    """Seed Python and NumPy random generators."""
    random.seed(seed)
    np.random.seed(seed)
