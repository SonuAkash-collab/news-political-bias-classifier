import json
from pathlib import Path

from polibias.data.articles import binary_articles, load_articles, process_articles


def _write_article(path: Path, article_id: str, bias: str) -> None:
    path.write_text(
        json.dumps(
            {
                "id": article_id,
                "topic": "Politics",
                "source": "Example News",
                "source_url": "example.test",
                "date": "2021-04-03",
                "title": f"{bias.title()} article",
                "content": f"Body text for {article_id}",
                "bias_text": bias,
            }
        ),
        encoding="utf-8",
    )


def test_loads_media_split_and_maps_binary_labels(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw" / "articles"
    json_dir = data_dir / "jsons"
    split_dir = data_dir / "splits" / "media"
    json_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    _write_article(json_dir / "left1.json", "left1", "left")
    _write_article(json_dir / "center1.json", "center1", "center")
    _write_article(json_dir / "right1.json", "right1", "right")
    (split_dir / "train.txt").write_text("left1 0\n", encoding="utf-8")
    (split_dir / "valid.tsv").write_text("ID\tbias\ncenter1\t1\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("right1 2\n", encoding="utf-8")

    frame = load_articles(data_dir)
    binary = binary_articles(frame)

    assert list(frame["id"]) == ["center1", "left1", "right1"]
    assert dict(zip(frame["id"], frame["split"])) == {
        "center1": "validation",
        "left1": "train",
        "right1": "test",
    }
    assert dict(zip(binary["id"], binary["party"])) == {"left1": "D", "right1": "R"}


def test_process_saves_csv_and_reports_rows(tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "articles"
    json_dir = data_dir / "jsons"
    split_dir = data_dir / "splits" / "media"
    json_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    _write_article(json_dir / "story.json", "story", "left")
    (split_dir / "train.txt").write_text("story 0\n", encoding="utf-8")
    (split_dir / "valid.tsv").write_text("ID\tbias\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("", encoding="utf-8")

    output = tmp_path / "processed" / "articles.csv"
    frame = process_articles(data_dir, output_path=output)

    assert len(frame) == 1
    assert output.is_file()
    assert "Article counts by media-based split and label:" in capsys.readouterr().out
