"""Run the whole project in order and combine every results file into results/results.json.

    python scripts/run_all.py            full run (writes results/, results/figures/ and models/)
    python scripts/run_all.py --quick    fast check on a sample, written to data/interim/quick_run/

Steps, in order:
    tweets     download and process the tweets, only if data/processed/tweets.csv is missing
    articles   download and process the articles, only if data/processed/articles.csv is missing
    model_zoo  grouped model zoo (polibias.experiments.model_zoo)
    exp1 exp2 exp3 ext1 ext2 ext3   each polibias.experiments module's main()

An experiment module that does not exist yet is skipped with a message. A step that fails is
reported and the remaining steps still run; the script then exits with status 1.

--quick keeps 10% of the congresspeople of each party (at least 10 per party) and 10% of the article
test set, and writes every output under data/interim/quick_run/ (or --output-dir), so the real
results and models are never overwritten. Experiments read the sample size from polibias.config.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import runpy
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUICK_FRACTION = 0.1
QUICK_OUTPUT = ROOT / "data" / "interim" / "quick_run"
EXPERIMENTS = ("exp1_tweet_test", "exp2_top_words", "exp3_articles", "ext1_tfidf_lr", "ext2_chunk_vote", "ext3_in_domain")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every step of the project in order.")
    parser.add_argument("--quick", action="store_true", help="Subsample the data for a fast end-to-end check.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"Where results/ and models/ go (default: the repository, or {QUICK_OUTPUT.relative_to(ROOT)} with --quick).")
    parser.add_argument("--only", nargs="+", metavar="STEP", help="Run only these steps, for example --only exp1 ext1.")
    return parser.parse_args(argv)


def configure(args: argparse.Namespace) -> None:
    """Set the environment that polibias.config reads. Must run before polibias is imported."""
    output = args.output_dir or (QUICK_OUTPUT if args.quick else None)
    if output is not None:
        os.environ["POLIBIAS_OUTPUT_DIR"] = str(Path(output).resolve())
    if args.quick:
        os.environ["POLIBIAS_SAMPLE_FRACTION"] = str(QUICK_FRACTION)
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))


# Steps. Each returns a short status line, or raises.


def step_tweets() -> str:
    from polibias import config
    from polibias.data import tweets

    if config.TWEETS_CSV.exists():
        return f"{config.TWEETS_CSV.relative_to(config.ROOT)} exists, not rebuilt"
    if not any(config.RAW_TWEETS_DIR.glob("*.json")) or not config.USERS_FILE.exists():
        runpy.run_path(str(ROOT / "scripts" / "download_tweets.py"), run_name="__main__")
    tweets.main()
    return "processed tweets"


def step_articles() -> str:
    from polibias import config
    from polibias.data.articles import process_articles
    from polibias.data.download import download_article_archive

    if config.ARTICLES_CSV.exists():
        return f"{config.ARTICLES_CSV.relative_to(config.ROOT)} exists, not rebuilt"
    download_article_archive(config.RAW_ARTICLES_DIR)  # skips the download if the raw files are there
    process_articles(config.RAW_ARTICLES_DIR, output_path=config.ARTICLES_CSV)
    return "processed articles"


def step_model_zoo() -> str:
    if importlib.util.find_spec("polibias.experiments.model_zoo") is None:
        raise ModuleNotFoundError("polibias.experiments.model_zoo")
    from polibias.data.tweets import load_processed
    from polibias.experiments.model_zoo import run_model_zoo

    result = run_model_zoo(load_processed())  # writes the "model_zoo" key of results/results.json
    best = max((name for name in result["scores"] if name != "majority_baseline"),
               key=lambda name: result["scores"][name]["macro_f1"])
    return f"best macro F1: {best} {result['scores'][best]['macro_f1']:.3f}"


def experiment_step(module_name: str):
    def run() -> str:
        module = importlib.import_module(f"polibias.experiments.{module_name}")
        if not hasattr(module, "main"):
            raise AttributeError(f"polibias.experiments.{module_name} has no main() function")
        module.main()
        return "done"

    return run


def steps() -> dict:
    out = {"tweets": step_tweets, "articles": step_articles, "model_zoo": step_model_zoo}
    for module_name in EXPERIMENTS:
        out[module_name.split("_")[0]] = experiment_step(module_name)
    return out


def module_exists(step: str) -> bool:
    if step in ("tweets", "articles"):
        return True
    if step == "model_zoo":
        return importlib.util.find_spec("polibias.experiments.model_zoo") is not None
    module_name = next(name for name in EXPERIMENTS if name.split("_")[0] == step)
    return importlib.util.find_spec(f"polibias.experiments.{module_name}") is not None


def combine_results() -> Path:
    """Merge every results/*.json into results/results.json, one key per file (the file name).

    Keys already in results.json that have no file of their own (model_zoo writes its key there
    directly) are kept.
    """
    from polibias import config

    combined = {}
    if config.RESULTS_JSON.exists():
        combined = json.loads(config.RESULTS_JSON.read_text(encoding="utf-8"))
    for path in sorted(config.RESULTS_DIR.glob("*.json")):
        if path != config.RESULTS_JSON:
            combined[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    config.RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    config.RESULTS_JSON.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8", newline="\n")
    return config.RESULTS_JSON


def main(argv=None) -> int:
    # Tweets contain emojis; the default Windows console encoding (cp1252) cannot print them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    configure(args)
    from polibias import config

    all_steps = steps()
    selected = args.only or list(all_steps)
    unknown = set(selected) - set(all_steps)
    if unknown:
        print(f"Unknown step(s): {', '.join(sorted(unknown))}. Choose from: {', '.join(all_steps)}")
        return 2

    mode = f"QUICK (sample fraction {config.SAMPLE_FRACTION})" if config.QUICK else "FULL"
    print(f"Mode: {mode}. Outputs go to {config.OUTPUT_DIR}")
    summary, failed = [], False
    for name in selected:
        print(f"\n=== {name} ===", flush=True)
        if not module_exists(name):
            status = "SKIPPED (module not written yet)"
            print(status)
        else:
            start = time.perf_counter()
            try:
                status = f"ok, {all_steps[name]()}"
            except Exception:  # keep going so one broken step does not hide the others
                traceback.print_exc()
                status = "FAILED (traceback above)"
                failed = True
            status += f" [{time.perf_counter() - start:.0f}s]"
        summary.append((name, status))

    path = combine_results()
    keys = sorted(json.loads(path.read_text(encoding="utf-8")))
    print("\n=== summary ===")
    for name, status in summary:
        print(f"{name:10} {status}")
    print(f"Combined results: {path} (keys: {', '.join(keys)})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
