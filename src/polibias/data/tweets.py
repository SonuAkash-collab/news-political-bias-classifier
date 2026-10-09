"""Load House tweets, attach the author's party, clean, and save data/processed/tweets.csv.

Usage: python -m polibias.data.tweets   (run scripts/download_tweets.py first)
"""

import json
from pathlib import Path

import pandas as pd

from polibias import config

PARTIES = {"D", "R"}
OUTPUT_COLUMNS = ["text", "screen_name", "member_id", "party", "time"]
PAPER_TWEET_COUNT = 103_852


def load_accounts(users: list[dict]) -> pd.DataFrame:
    """One row per House D/R member account: handle (lowercase), account id, member_id, party.

    Old handles in prev_names get their own rows, so tweets posted before a rename still match.
    """
    rows = []
    for member in users:
        if member.get("chamber") != "house" or member.get("type") != "member":
            continue
        if member.get("party") not in PARTIES:
            continue
        # The Bioguide ID is the official, stable identifier for a member of Congress.
        member_id = (member.get("id") or {}).get("bioguide") or member["name"]
        for account in member.get("accounts", []):
            handles = [account["screen_name"]]
            for prev in account.get("prev_names", []):
                # A few entries store old handles as {member_id: handle} instead of a plain string.
                handles.extend(prev.values() if isinstance(prev, dict) else [prev])
            for handle in handles:
                rows.append(
                    {
                        "handle": handle.lower(),
                        "account_id": str(account.get("id", "")),
                        "member_id": member_id,
                        "party": member["party"],
                    }
                )
    return pd.DataFrame(rows, columns=["handle", "account_id", "member_id", "party"])


def label_tweets(tweets: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    """Attach member_id and party to each tweet. Unmatched tweets get NaN in both columns.

    Match on the handle first (case-insensitive). If the handle is unknown, fall back to the
    numeric Twitter user id, which does not change when an account is renamed.
    """
    by_handle = accounts.drop_duplicates("handle").set_index("handle")
    by_id = accounts[accounts["account_id"] != ""].drop_duplicates("account_id").set_index("account_id")

    out = tweets.copy()
    handles = out["screen_name"].str.lower()
    out["member_id"] = handles.map(by_handle["member_id"])
    out["party"] = handles.map(by_handle["party"])
    if "user_id" in out.columns:
        missing = out["member_id"].isna()
        ids = out.loc[missing, "user_id"].astype(str)
        out.loc[missing, "member_id"] = ids.map(by_id["member_id"])
        out.loc[missing, "party"] = ids.map(by_id["party"])
    return out


def is_retweet(text: pd.Series) -> pd.Series:
    return text.str.match(r"RT @")


def load_raw_tweets(tweets_dir: Path = config.RAW_TWEETS_DIR) -> pd.DataFrame:
    files = sorted(tweets_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No tweet files in {tweets_dir}. Run scripts/download_tweets.py first.")
    records = []
    for path in files:
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(records)


def build(users_file: Path = config.USERS_FILE, tweets_dir: Path = config.RAW_TWEETS_DIR):
    """Return the cleaned, labelled tweets and a dict of counts for each filtering step."""
    users = json.loads(Path(users_file).read_text(encoding="utf-8"))
    accounts = load_accounts(users)
    raw = load_raw_tweets(tweets_dir)
    stats = {"raw_tweets": len(raw)}

    labelled = label_tweets(raw, accounts)
    matched = labelled["party"].notna()
    stats["dropped_no_party_match"] = int((~matched).sum())
    stats["dropped_no_party_match_accounts"] = int(labelled.loc[~matched, "screen_name"].str.lower().nunique())
    df = labelled[matched]

    retweet = is_retweet(df["text"])
    stats["dropped_retweets"] = int(retweet.sum())
    df = df[~retweet]

    df = df.sort_values("time", kind="stable")
    duplicate = df["text"].duplicated(keep="first")
    stats["dropped_duplicates"] = int(duplicate.sum())
    df = df[~duplicate]

    df = df[OUTPUT_COLUMNS].reset_index(drop=True)
    stats["kept_tweets"] = len(df)
    stats["kept_members"] = int(df["member_id"].nunique())
    stats["kept_accounts"] = int(df["screen_name"].str.lower().nunique())
    stats["party_tweets"] = df["party"].value_counts().to_dict()
    stats["party_members"] = df.drop_duplicates("member_id")["party"].value_counts().to_dict()
    return df, stats


def print_summary(stats: dict) -> None:
    kept = stats["kept_tweets"]
    d, r = stats["party_tweets"].get("D", 0), stats["party_tweets"].get("R", 0)
    print(f"Raw tweets in date range:          {stats['raw_tweets']:>8,}")
    print(
        f"Dropped, account not a House D/R:  {stats['dropped_no_party_match']:>8,}"
        f"  ({stats['dropped_no_party_match_accounts']} accounts)"
    )
    print(f"Dropped, retweets:                 {stats['dropped_retweets']:>8,}")
    print(f"Dropped, exact duplicate text:     {stats['dropped_duplicates']:>8,}")
    print(f"Kept tweets:                       {kept:>8,}")
    print(
        f"Kept members:                      {stats['kept_members']:>8,}"
        f"  (D {stats['party_members'].get('D', 0)}, R {stats['party_members'].get('R', 0)};"
        f" {stats['kept_accounts']} accounts)"
    )
    print(f"Tweets by party:                   D {d:,} ({d / kept:.1%}), R {r:,} ({r / kept:.1%})")
    print(f"Paper reports {PAPER_TWEET_COUNT:,} tweets; difference {kept - PAPER_TWEET_COUNT:+,}")


def main() -> None:
    df, stats = build()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.TWEETS_CSV, index=False, encoding="utf-8")
    print_summary(stats)
    print(f"Saved {config.TWEETS_CSV}")


if __name__ == "__main__":
    main()
