"""Smoke test: scripts/run_all.py --quick runs every existing step end to end on a sample."""

import json
import os
import subprocess
import sys

import pytest

from polibias import config

ROOT = config.ROOT
# Inside the repository (and ignored by Git): the model zoo stores model paths relative to it.
OUTPUT = ROOT / ".pytest-tmp" / "run_all_quick"


@pytest.mark.skipif(
    not (config.TWEETS_CSV.exists() and config.ARTICLES_CSV.exists()),
    reason="needs data/processed/tweets.csv and articles.csv (run the download scripts first)",
)
def test_run_all_quick_end_to_end():
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    env.pop("POLIBIAS_OUTPUT_DIR", None)
    env.pop("POLIBIAS_SAMPLE_FRACTION", None)
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_all.py"), "--quick", "--output-dir", str(OUTPUT)],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=1800,
    )
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    assert "FAILED" not in completed.stdout

    combined = json.loads((OUTPUT / "results" / "results.json").read_text(encoding="utf-8"))
    for key in ("model_zoo", "exp1", "exp2", "ext1"):
        assert key in combined
    assert combined["ext1"]["quick_run"] is True
    assert (OUTPUT / "models" / "ext1_tfidf_lr.joblib").exists()
    assert (OUTPUT / "results" / "figures" / "ext1_tfidf_lr.png").exists()
    # Missing experiment modules are skipped with a message, not treated as failures.
    for module in ("exp3_articles", "ext2_chunk_vote", "ext3_in_domain"):
        if not (ROOT / "src" / "polibias" / "experiments" / f"{module}.py").exists():
            assert "SKIPPED" in completed.stdout
