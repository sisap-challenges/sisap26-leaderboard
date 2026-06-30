#!/usr/bin/env python3
"""
Import published SISAP leaderboard CSVs for Task 1 and Task 2 and export them
to website/results/summary.parquet.

This importer is intentionally separate from export_results.py, which builds the
same parquet schema from local HDF5 submissions.

Usage:
    python import_csv_results.py
    python import_csv_results.py --output website/results/summary.parquet
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd


SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT = SCRIPT_DIR / "website" / "results" / "summary.parquet"


CSV_SOURCES = (
    {
        "task": "task1",
        "dataset": "wikipedia-eval",
        "threshold": 0.8,
        "n_queries": 6_400_000,
        "url": "https://files.webis.de/data-in-progress/data-research/sisap-2025/task-1-wikipedia-20260614-test.csv",
    }, {
        "task": "task1",
        "dataset": "wikipedia-small",
        "threshold": 0.8,
        # TODO
        "n_queries": 200_000,
        "url": "https://files.webis.de/data-in-progress/data-research/sisap-2025/task-1-wikipedia-small.csv",
    }, {
        "task": "task2",
        "dataset": "llama-eval",
        "threshold": 0.8,
        # TODO
        "n_queries": 10_000,
        "url": "https://files.webis.de/data-in-progress/data-research/sisap-2025/task-2-llama.csv",
    }, {
        "task": "task2",
        "dataset": "llama-pg174",
        "threshold": 0.8,
        # TODO
        "n_queries": 10_000,
        "url": "https://files.webis.de/data-in-progress/data-research/sisap-2025/task-2-llama-pg174.csv",
    }, {
        "task": "task2",
        "dataset": "llama-dev",
        "threshold": 0.8,
        # TODO
        "n_queries": 10_000,
        "url": "https://files.webis.de/data-in-progress/data-research/sisap-2025/task-2-llama-dev.csv",
    }, {
        "task": "task3",
        "dataset": "nq-eval",
        "threshold": 0.8,
        # TODO
        "n_queries": 10_000,
        "url": "https://files.webis.de/data-in-progress/data-research/sisap-2025/task-3-nq-20260610-test.csv",
    }, {
        "task": "task3",
        "dataset": "fiqa-dev",
        "threshold": 0.8,
        # TODO
        "n_queries": 10_000,
        "url": "https://files.webis.de/data-in-progress/data-research/sisap-2025/task-3-fiqa.csv",
    },
)

SUMMARY_COLUMNS = [
    "task",
    "dataset",
    "algo",
    "team",
    "repo",
    "is_baseline",
    "params",
    "buildtime",
    "querytime",
    "recall",
    "throughput",
    "n_queries",
    "threshold",
]


def clean_scalar(value: object) -> str:
    """Decode byte-like CSV values such as b'foo' into plain strings."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()

    text = str(value).strip()
    if len(text) >= 3 and text[0] == "b" and text[1] in {"'", '"'}:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
        if isinstance(parsed, bytes):
            return parsed.decode("utf-8", errors="replace").strip()
        return str(parsed).strip()
    return text


def parse_float(value: object, field_name: str, source_url: str) -> float:
    text = clean_scalar(value)
    if text == "":
        return 0
    if text == "TODO":
        return 0
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {field_name!r} in {source_url}: {text!r}") from exc


def load_csv_rows(url: str) -> list[dict[str, str]]:
    with urlopen(url) as response:
        payload = response.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(payload)))


def parse_software_blob(blob: str) -> dict:
    text = clean_scalar(blob)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def normalize_github_repo(url: str) -> str:
    url = clean_scalar(url)
    if not url:
        return ""

    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.removeprefix("git@github.com:")

    if url.endswith(".git"):
        url = url[:-4]

    if "/tree/" in url:
        url = url.split("/tree/", 1)[0]

    if "/commit/" in url:
        url = url.split("/commit/", 1)[0]

    if url.endswith("/"):
        url = url[:-1]

    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"} and parsed.netloc == "github.com":
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            return f"https://github.com/{parts[0]}/{parts[1]}"
    return url


def extract_repo_url(software: dict) -> str:
    remotes = software.get("source_code_remotes") or []
    for remote in remotes:
        repo = normalize_github_repo(remote.get("name", ""))
        if repo:
            return repo
        repo = normalize_github_repo(remote.get("href", ""))
        if repo:
            return repo
    return ""


def is_baseline(team: str, repo: str) -> bool:
    return team == "basel1nerz"


def compute_throughput(task: str, n_queries: int, buildtime: float, querytime: float) -> float:
    if task == "task1":
        total_time = buildtime + querytime
        return n_queries / total_time if total_time > 0 else 0.0
    return n_queries / querytime if querytime > 0 else 0.0


def import_source(source: dict) -> list[dict]:
    print(f"Fetching {source['url']} ...")
    rows = load_csv_rows(source["url"])
    print(f"  Loaded {len(rows)} CSV rows")

    imported_rows: list[dict] = []
    for index, row in enumerate(rows, start=1):
        software = parse_software_blob(row.get("software", ""))
        team = clean_scalar(software.get("vm_id") or software.get("display_name") or "unknown")
        repo = extract_repo_url(software)
        buildtime = parse_float(row.get("buildtime"), "buildtime", source["url"])
        querytime = parse_float(row.get("querytime"), "querytime", source["url"])
        recall = parse_float(row.get("recall"), "recall", source["url"])
        algo = clean_scalar(row.get("algo", ""))
        params = clean_scalar(row.get("params", ""))

        if not algo:
            raise ValueError(f"Missing algorithm name in {source['url']} row {index}")

        imported_rows.append(
            {
                "task": source["task"],
                "dataset": source["dataset"],
                "algo": algo,
                "team": team,
                "repo": repo,
                "is_baseline": is_baseline(team, repo),
                "params": params,
                "buildtime": buildtime,
                "querytime": querytime,
                "recall": recall,
                "throughput": compute_throughput(
                    source["task"], source["n_queries"], buildtime, querytime
                ),
                "n_queries": source["n_queries"],
                "threshold": source["threshold"],
            }
        )

    return imported_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output Parquet file path",
    )
    args = parser.parse_args()

    all_rows: list[dict] = []
    for source in CSV_SOURCES:
        all_rows.extend(import_source(source))

    if not all_rows:
        print("No rows imported from CSV sources.")
        sys.exit(1)

    df = pd.DataFrame(all_rows, columns=SUMMARY_COLUMNS)
    df.sort_values(["task", "dataset", "team", "algo", "recall"], inplace=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"\nWrote {len(df)} rows to {output_path}")
    summary = (
        df.groupby(["task", "dataset"], as_index=False)
        .agg(rows=("algo", "size"), teams=("team", "nunique"))
        .sort_values(["task", "dataset"])
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
