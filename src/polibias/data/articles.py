"""Load Article-Bias-Prediction articles and their published data splits.

The source is the official AllSides dataset released by Baly et al. (2020),
“We Can Detect Your Bias: Predicting the Political Ideology of News Articles,”
EMNLP 2020: https://github.com/ramybaly/Article-Bias-Prediction.
"""

from __future__ import annotations

import csv
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from polibias import config

DEFAULT_DATA_DIR = config.RAW_ARTICLES_DIR
DEFAULT_OUTPUT = config.ARTICLES_CSV
LABELS = {0: "left", 1: "center", 2: "right"}
PARTY_LABELS = {"left": "D", "right": "R"}


def _first(record: dict[str, Any], *keys: str, default: Any = "") -> Any:
    lowered = {str(key).lower(): value for key, value in record.items()}
    for key in keys:
        value = lowered.get(key.lower())
        if value is not None:
            return value
    return default


def _label_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return LABELS.get(int(value), "")
    text = str(value).strip().lower()
    if text in LABELS.values():
        return text
    try:
        return LABELS.get(int(text), "")
    except ValueError:
        return ""


def _iter_split_rows(value: Any) -> Iterable[tuple[str, str]]:
    """Yield article IDs and optional labels from common JSON split structures."""
    if isinstance(value, dict):
        article_id = _first(value, "id", "article_id", "ID")
        if article_id:
            label = _first(value, "bias", "label", "bias_text", default="")
            yield str(article_id), _label_text(label)
            return
        for key, item in value.items():
            if isinstance(item, dict):
                item_id = _first(item, "id", "article_id", default=key)
                label = _first(item, "bias", "label", "bias_text", default="")
                yield str(item_id), _label_text(label)
            elif isinstance(item, (list, tuple)) and item:
                yield str(key), _label_text(item[-1])
            else:
                yield str(key), _label_text(item)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                article_id = _first(item, "id", "article_id", "ID")
                if article_id:
                    label = _first(item, "bias", "label", "bias_text", default="")
                    yield str(article_id), _label_text(label)
            elif isinstance(item, (list, tuple)) and item:
                label = item[1] if len(item) > 1 else ""
                yield str(item[0]), _label_text(label)
            elif item is not None:
                yield str(item), ""


def _read_split_file(path: Path) -> list[tuple[str, str]]:
    if path.suffix.lower() == ".json":
        with path.open(encoding="utf-8") as handle:
            return list(_iter_split_rows(json.load(handle)))

    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,; ")
            parsed_rows = csv.reader(handle, dialect)
            for row in parsed_rows:
                cells = [cell.strip() for cell in row if cell.strip()]
                if not cells or cells[0].startswith("#"):
                    continue
                first_cell = cells[0].lower()
                if first_cell in {"id", "article_id", "article", "articleid"}:
                    continue
                article_id = cells[0]
                label = _label_text(cells[1]) if len(cells) > 1 else ""
                rows.append((article_id, label))
        except csv.Error:
            for line in handle:
                cells = re.split(r"[\s,;]+", line.strip())
                if not cells or not cells[0] or cells[0].startswith("#"):
                    continue
                if cells[0].lower() in {"id", "article_id", "article", "articleid"}:
                    continue
                label = _label_text(cells[1]) if len(cells) > 1 else ""
                rows.append((cells[0], label))
    return rows


def _split_kind(path: Path) -> str | None:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(path).lower())
    if "media" in normalized or "source" in normalized or "outlet" in normalized:
        return "media-based"
    if "random" in normalized:
        return "random"
    return None


def _split_part(path: Path) -> str | None:
    name = re.sub(r"[^a-z0-9]+", " ", str(path).lower())
    if re.search(r"\b(train|training)\b", name):
        return "train"
    if re.search(r"\b(val|valid|validation|dev|development)\b", name):
        return "validation"
    if re.search(r"\b(test|testing)\b", name):
        return "test"
    return None


def _load_split_assignments(data_dir: Path, split_type: str) -> dict[str, tuple[str, str]]:
    split_dir = data_dir / "splits"
    files = sorted(path for path in split_dir.rglob("*") if path.is_file())
    assignments: dict[str, tuple[str, str]] = {}
    matching_files = [
        path
        for path in files
        if _split_kind(path.relative_to(data_dir)) == split_type
        and _split_part(path.relative_to(data_dir)) is not None
    ]
    if not matching_files:
        available = ", ".join(str(path.relative_to(split_dir)) for path in files[:12]) or "none"
        raise FileNotFoundError(
            f"No {split_type!r} train/validation/test files found under {split_dir}. "
            f"Files found: {available}"
        )

    for path in matching_files:
        relative = path.relative_to(data_dir)
        partition = _split_part(relative)
        assert partition is not None
        for article_id, label in _read_split_file(path):
            previous = assignments.get(article_id)
            if previous and previous[0] != partition:
                raise ValueError(
                    f"Article {article_id!r} appears in both {previous[0]} and {partition} "
                    f"in the {split_type} split."
                )
            assignments[article_id] = (partition, label)

    if not assignments:
        raise ValueError(f"Split files under {split_dir} contained no article IDs.")
    return assignments


def _article_record(path: Path) -> dict[str, Any] | None:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        return None
    article_id = _first(raw, "id", "article_id", "ID", default=path.stem)
    bias = _label_text(_first(raw, "bias_text", "bias", "label"))
    content = _first(raw, "content", "content_original", default="")
    if isinstance(content, list):
        content = " ".join(str(token) for token in content)
    elif content is None:
        content = ""
    return {
        "id": str(article_id),
        "source": str(_first(raw, "source", "media", default="")),
        "source_url": str(_first(raw, "source_url", "sourceurl", default="")),
        "date": str(_first(raw, "date", "published", default="")),
        "title": str(_first(raw, "title", default="")),
        "content": str(content),
        "bias_text": bias,
        "topic": str(_first(raw, "topic", default="")),
    }


def load_articles(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    *,
    split_type: str = "media-based",
) -> pd.DataFrame:
    """Load article JSON files and attach the selected published data split.

    Args:
        data_dir: Extracted dataset directory containing ``jsons/`` and ``splits/``.
        split_type: ``"media-based"`` (recommended) or ``"random"``.

    Returns:
        A DataFrame with article metadata, labels, and a train/validation/test column.
    """
    if split_type not in {"media-based", "random"}:
        raise ValueError("split_type must be 'media-based' or 'random'")
    data_dir = Path(data_dir).expanduser().resolve()
    json_dir = data_dir / "jsons"
    if not json_dir.is_dir():
        raise FileNotFoundError(
            f"Article JSON directory not found: {json_dir}. "
            "Run scripts/download_articles.py first."
        )

    assignments = _load_split_assignments(data_dir, split_type)
    def read_assigned(article_id: str) -> dict[str, Any] | None:
        path = json_dir / f"{article_id}.json"
        if not path.is_file():
            return None
        record = _article_record(path)
        if record is None:
            return None
        partition, split_label = assignments[article_id]
        if not record["bias_text"]:
            record["bias_text"] = split_label
        record["split"] = partition
        return record

    records: list[dict[str, Any]] = []
    article_ids = list(assignments)
    # The release has tens of thousands of small JSON files. A few workers
    # reduce per-file I/O overhead while keeping the work bounded.
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(read_assigned, article_id) for article_id in article_ids]
        for number, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            if record is not None:
                records.append(record)
            if number % 5000 == 0:
                print(f"Loaded {number:,}/{len(article_ids):,} split article records.", flush=True)

    if not records:
        raise ValueError(
            f"No article JSON files matched the {split_type} split IDs under {data_dir}."
        )
    if len(records) != len(assignments):
        raise ValueError(
            f"The {split_type} split lists {len(assignments):,} article IDs, but only "
            f"{len(records):,} matching JSON files were found. Re-run the downloader."
        )
    frame = pd.DataFrame.from_records(records).sort_values("id").reset_index(drop=True)
    frame = frame.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    columns = [
        "id",
        "source",
        "source_url",
        "date",
        "title",
        "content",
        "bias_text",
        "topic",
        "split",
    ]
    return frame[columns]


def binary_articles(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only left/right articles, mapped to D/R for the paper comparison."""
    binary = frame.loc[frame["bias_text"].isin(PARTY_LABELS)].copy()
    binary["party"] = binary["bias_text"].map(PARTY_LABELS)
    return binary.reset_index(drop=True)


def print_article_summary(frame: pd.DataFrame, *, split_type: str = "media-based") -> None:
    """Print class counts for each split and the article date range."""
    print(f"Article counts by {split_type} split and label:")
    counts = (
        frame.groupby(["split", "bias_text"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=["train", "validation", "test"], fill_value=0)
    )
    print(counts.to_string())
    dates = pd.to_datetime(frame["date"], errors="coerce", utc=True).dropna()
    if dates.empty:
        print("Article date range: unavailable (no parseable dates).")
    else:
        print(f"Article date range: {dates.min().date()} to {dates.max().date()}")
        current_year = pd.Timestamp.now(tz="UTC").year
        future_count = int((dates.dt.year > current_year).sum())
        if future_count:
            print(
                f"Date warning: {future_count:,} record(s) have a date after "
                f"{current_year}; verify these source values before temporal analysis."
            )


def process_articles(
    data_dir: Path | str = DEFAULT_DATA_DIR,
    *,
    split_type: str = "media-based",
    output_path: Path | str = DEFAULT_OUTPUT,
) -> pd.DataFrame:
    """Load articles, save the processed CSV, print dataset summary, and return it."""
    frame = load_articles(data_dir, split_type=split_type)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    print_article_summary(frame, split_type=split_type)
    print(f"Saved processed articles to {output_path}")
    return frame


def load_processed(path: Path | str = DEFAULT_OUTPUT) -> pd.DataFrame:
    """Load data/processed/articles.csv (or another path)."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Run scripts/download_articles.py first."
        )
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    if config.QUICK:
        parts = []
        for _, group in df.groupby(["split", "bias_text"]):
            n = min(len(group), max(5, round(config.SAMPLE_FRACTION * len(group))))
            parts.append(group.sample(n=n, random_state=config.SEED))
        df = pd.concat(parts, ignore_index=True) if parts else df
    return df
