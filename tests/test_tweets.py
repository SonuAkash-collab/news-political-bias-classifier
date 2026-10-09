"""Smoke tests for joining tweets to party labels, on a tiny fake sample."""

import json

import pandas as pd

from polibias.data.tweets import OUTPUT_COLUMNS, build, label_tweets, load_accounts

USERS = [
    {
        "name": "Alice Dem",
        "chamber": "house",
        "type": "member",
        "party": "D",
        "id": {"bioguide": "A000001", "govtrack": 1},
        "accounts": [
            {"id": "11", "screen_name": "RepAlice", "account_type": "office"},
            {"id": "12", "screen_name": "AliceForCongress", "account_type": "campaign",
             "prev_names": ["AliceOld"]},
        ],
    },
    {
        "name": "Bob Rep",
        "chamber": "house",
        "type": "member",
        "party": "R",
        "id": {"bioguide": "B000002", "govtrack": 2},
        "accounts": [{"id": "21", "screen_name": "RepBob", "account_type": "office",
                      "prev_names": [{"B000002": "BobOlder"}]}],
    },
    {"name": "Carol Sen", "chamber": "senate", "type": "member", "party": "D",
     "id": {"bioguide": "C000003"}, "accounts": [{"id": "31", "screen_name": "SenCarol"}]},
    {"name": "Dan Ind", "chamber": "house", "type": "member", "party": "I",
     "id": {"bioguide": "D000004"}, "accounts": [{"id": "41", "screen_name": "RepDan"}]},
    {"name": "Some Caucus", "chamber": "house", "type": "caucus", "party": "D",
     "accounts": [{"id": "51", "screen_name": "DemCaucus"}]},
]

TWEETS = [
    {"screen_name": "RepAlice", "user_id": "11", "time": "2021-01-04T10:00:00-05:00", "text": "Voting rights now"},
    {"screen_name": "alicforcongress_typo", "user_id": "12", "time": "2021-01-04T11:00:00-05:00", "text": "Renamed account"},
    {"screen_name": "aliceold", "user_id": "99", "time": "2021-01-04T12:00:00-05:00", "text": "Old handle"},
    {"screen_name": "RepBob", "user_id": "21", "time": "2021-01-05T10:00:00-05:00", "text": "Cut taxes"},
    {"screen_name": "BobOlder", "user_id": "98", "time": "2021-01-05T10:30:00-05:00", "text": "Older handle"},
    {"screen_name": "RepBob", "user_id": "21", "time": "2021-01-05T11:00:00-05:00", "text": "RT @someone hello"},
    {"screen_name": "RepBob", "user_id": "21", "time": "2021-01-06T10:00:00-05:00", "text": "Voting rights now"},
    {"screen_name": "SenCarol", "user_id": "31", "time": "2021-01-05T12:00:00-05:00", "text": "Senate tweet"},
    {"screen_name": "RepDan", "user_id": "41", "time": "2021-01-05T13:00:00-05:00", "text": "Independent tweet"},
    {"screen_name": "DemCaucus", "user_id": "51", "time": "2021-01-05T14:00:00-05:00", "text": "Caucus tweet"},
    {"screen_name": "Stranger", "user_id": "77", "time": "2021-01-05T15:00:00-05:00", "text": "Unknown account"},
]


def test_label_join():
    labelled = label_tweets(pd.DataFrame(TWEETS), load_accounts(USERS)).set_index("text")

    # Office, campaign and old handles all map to the same member and party.
    assert labelled.loc["Voting rights now", "member_id"].tolist() == ["A000001", "B000002"]
    assert labelled.loc["Renamed account", ["member_id", "party"]].tolist() == ["A000001", "D"]
    assert labelled.loc["Old handle", ["member_id", "party"]].tolist() == ["A000001", "D"]
    assert labelled.loc["Older handle", ["member_id", "party"]].tolist() == ["B000002", "R"]
    assert labelled.loc["Cut taxes", "party"] == "R"

    # Senators, non-D/R members, caucuses and unknown accounts get no label.
    for text in ["Senate tweet", "Independent tweet", "Caucus tweet", "Unknown account"]:
        assert pd.isna(labelled.loc[text, "party"])


def test_build_filters_and_columns(tmp_path):
    users_file = tmp_path / "users.json"
    users_file.write_text(json.dumps(USERS), encoding="utf-8")
    tweets_dir = tmp_path / "tweets"
    tweets_dir.mkdir()
    (tweets_dir / "2021-01-04.json").write_text(json.dumps(TWEETS), encoding="utf-8")

    df, stats = build(users_file, tweets_dir)

    assert list(df.columns) == OUTPUT_COLUMNS
    assert stats["dropped_no_party_match"] == 4
    assert stats["dropped_retweets"] == 1
    assert stats["dropped_duplicates"] == 1  # Bob's later copy of Alice's text
    assert len(df) == 5
    assert stats["kept_members"] == 2
    assert set(df["party"]) == {"D", "R"}
