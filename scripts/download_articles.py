"""Download and prepare the AllSides Article-Bias-Prediction dataset.

The dataset was released with Baly et al. (2020), “We Can Detect Your Bias:
Predicting the Political Ideology of News Articles,” EMNLP 2020. The official
dataset repository is https://github.com/ramybaly/Article-Bias-Prediction.
This downloads one archive and extracts only its article JSONs and evaluation
splits. Raw and processed data remain local and are ignored by Git.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from polibias.data.articles import process_articles  # noqa: E402
from polibias.data.download import download_article_archive  # noqa: E402
from polibias import config  # noqa: E402

ROOT = config.ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=config.RAW_ARTICLES_DIR,
        help="Directory for the downloaded JSON articles and split files.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download and extract the archive again, replacing the local copy.",
    )
    parser.add_argument(
        "--split",
        choices=("media-based", "random"),
        default="media-based",
        help="Split assignments to attach to the processed CSV (default: media-based).",
    )
    args = parser.parse_args()

    download_article_archive(args.destination, force=args.force_download)
    frame = process_articles(
        data_dir=args.destination,
        split_type=args.split,
        output_path=config.ARTICLES_CSV,
    )
    print(f"Prepared {len(frame):,} articles at data/processed/articles.csv")


if __name__ == "__main__":
    main()
