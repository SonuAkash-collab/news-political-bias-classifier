"""Download the public Article-Bias-Prediction archive from GitHub."""

from __future__ import annotations

import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ARCHIVE_URL = (
    "https://codeload.github.com/ramybaly/Article-Bias-Prediction/zip/refs/heads/main"
)
CHUNK_SIZE = 1024 * 1024


def _has_dataset(destination: Path) -> bool:
    json_dir = destination / "jsons"
    split_dir = destination / "splits"
    return (
        (destination / ".dataset_download_complete").is_file()
        and
        json_dir.is_dir()
        and any(json_dir.glob("*.json"))
        and split_dir.is_dir()
        and any(path.is_file() for path in split_dir.rglob("*"))
    )


def _download_archive(archive_path: Path, retries: int = 3) -> None:
    request = urllib.request.Request(
        ARCHIVE_URL,
        headers={"User-Agent": "news-political-bias-classifier/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                total = int(response.headers.get("Content-Length", "0"))
                received = 0
                with archive_path.open("wb") as output:
                    while chunk := response.read(CHUNK_SIZE):
                        output.write(chunk)
                        received += len(chunk)
                        if total:
                            percent = min(100, received * 100 // total)
                            print(
                                f"\rDownloading archive: {percent:3d}% "
                                f"({received / (1024 * 1024):.1f} MiB)",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\rDownloaded {received / (1024 * 1024):.1f} MiB",
                                end="",
                                flush=True,
                            )
            print("\nArchive download complete.")
            return
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            archive_path.unlink(missing_ok=True)
            if attempt < retries:
                delay = 2 ** (attempt - 1)
                print(f"Download failed ({exc}); retrying in {delay}s.")
                time.sleep(delay)
    raise RuntimeError(f"Could not download the official dataset archive: {last_error}")


def _extract_dataset(archive_path: Path, destination: Path) -> None:
    """Extract only data/jsons and data/splits, with path traversal protection."""
    destination.mkdir(parents=True, exist_ok=True)
    wanted = ("data", "jsons")
    extracted_jsons = 0
    extracted_splits = 0

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        for index, member in enumerate(members, start=1):
            if member.is_dir():
                continue
            parts = PurePosixPath(member.filename).parts
            try:
                data_index = parts.index("data")
            except ValueError:
                continue
            relative_parts = parts[data_index + 1 :]
            if len(relative_parts) < 2 or relative_parts[0] not in {"jsons", "splits"}:
                continue
            if relative_parts[0] == "jsons" and not relative_parts[-1].endswith(".json"):
                continue

            relative = Path(*relative_parts)
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe path in dataset archive: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=CHUNK_SIZE)

            if relative_parts[0] == "jsons":
                extracted_jsons += 1
            else:
                extracted_splits += 1

            if index % 10000 == 0:
                print(f"Extracted {index:,}/{len(members):,} archive entries.", flush=True)

    if not extracted_jsons or not extracted_splits:
        raise RuntimeError(
            "The archive did not contain both data/jsons and data/splits; "
            f"found {extracted_jsons} article JSON files and {extracted_splits} split files."
        )
    print(
        f"Extracted {extracted_jsons:,} article JSON files and "
        f"{extracted_splits:,} split files into {destination}."
    )
    (destination / ".dataset_download_complete").write_text(
        f"{ARCHIVE_URL}\n", encoding="utf-8"
    )


def download_article_archive(destination: Path, *, force: bool = False) -> Path:
    """Download and unpack the official dataset unless it is already present.

    Args:
        destination: Directory that will contain ``jsons/`` and ``splits/``.
        force: Re-download the archive even when both data folders are present.

    Returns:
        The destination directory containing the extracted dataset.
    """
    destination = Path(destination).expanduser().resolve()
    if not force and _has_dataset(destination):
        print(f"Article data already exists at {destination}; skipping download.")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="article-bias-download-") as temp_dir:
        archive_path = Path(temp_dir) / "article-bias-prediction.zip"
        _download_archive(archive_path)
        if force and destination.exists():
            shutil.rmtree(destination)
        _extract_dataset(archive_path, destination)
    return destination
