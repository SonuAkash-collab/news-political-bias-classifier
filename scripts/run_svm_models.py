"""Run the grouped LinearSVC and model-zoo experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from polibias import config
from polibias.experiments.model_zoo import run_model_zoo


def main() -> None:
    if not config.TWEETS_CSV.exists():
        raise FileNotFoundError(
            f"Missing {config.TWEETS_CSV}. Run scripts/download_tweets.py and "
            "python -m polibias.data.tweets first."
        )
    tweets = pd.read_csv(config.TWEETS_CSV)
    result = run_model_zoo(tweets)
    print("\nGrouped test scores:")
    for name, score in result["scores"].items():
        print(
            f"{name:32} accuracy={score['accuracy']:.4f} "
            f"macro_f1={score['macro_f1']:.4f} "
            f"baseline_accuracy={score['majority_baseline_accuracy']:.4f} "
            f"baseline_macro_f1={score['majority_baseline_macro_f1']:.4f}"
        )
    print(f"Saved results to {config.RESULTS_JSON}")


if __name__ == "__main__":
    main()
