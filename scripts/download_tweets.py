"""Download the congresstweets daily files and the member list into data/raw/.

Usage: python scripts/download_tweets.py
Files that already exist are skipped, so the script can be re-run after an interruption.
"""

import sys
import time
from datetime import timedelta
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polibias import config  # noqa: E402

RETRIES = 5
TIMEOUT = 60


def download(url: str, dest: Path, session: requests.Session) -> bool:
    """Download url to dest unless it exists. Returns True if a new file was written."""
    if dest.exists() and dest.stat().st_size > 0:
        return False
    for attempt in range(1, RETRIES + 1):
        try:
            response = session.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            # Write to a temporary file first so an interrupted download never looks complete.
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(response.content)
            tmp.replace(dest)
            return True
        except requests.RequestException as err:
            if attempt == RETRIES:
                raise RuntimeError(f"Failed to download {url}: {err}") from err
            time.sleep(2**attempt)
    return False


def tweet_days():
    day = config.START_DATE
    while day <= config.END_DATE:
        yield day
        day += timedelta(days=1)


def main() -> None:
    config.RAW_TWEETS_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    download(config.USERS_URL, config.USERS_FILE, session)

    days = list(tweet_days())
    new = 0
    for day in tqdm(days, desc="Tweet files", unit="day"):
        name = day.isoformat()
        url = config.TWEETS_URL.format(day=name)
        new += download(url, config.RAW_TWEETS_DIR / f"{name}.json", session)

    print(f"{len(days)} daily files in {config.RAW_TWEETS_DIR} ({new} downloaded, {len(days) - new} already present)")
    print(f"Member list: {config.USERS_FILE}")


if __name__ == "__main__":
    main()
