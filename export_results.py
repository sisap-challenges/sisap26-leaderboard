#!/usr/bin/env python3
"""
Export evaluation results to website/results/summary.parquet.

Scans team-results/<team>/**/*.h5 (or the path given by --results-root),
computes recall against ground-truth, and writes a single Parquet file
consumed by the Quarto website.

Team metadata (display name, repo URL, is_baseline flag) is read from
teams.yml in the same directory as this script.  Add one entry per
participant team before running.

Usage:
    python export_results.py
    python export_results.py --results-root /path/to/team-results
    python export_results.py --output website/results/summary.parquet
"""

import argparse
import glob
import sys
import h5py
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "team-results"
DEFAULT_OUTPUT       = SCRIPT_DIR / "website" / "results" / "summary.parquet"
TEAMS_FILE           = SCRIPT_DIR / "teams.yml"

# Ground-truth lives in the shared data directory next to this script.
DATA_DIR = SCRIPT_DIR / "data"

# ---------------------------------------------------------------------------
# Ground-truth configuration per (dataset, task)
# ---------------------------------------------------------------------------
DATASET_CFG: dict = {
    ("gooaq-small", "task1"): {
        "gt_path": DATA_DIR / "benchmark-dev-gooaq-small.h5",
        "gt_I":    lambda f: f["allknn"]["knns"],
        "k": 15,
        "n_vectors":  lambda f: len(f["train"]),
        "gt_1based": True,
    },
    ("wikipedia-small", "task1"): {
        "gt_path": DATA_DIR / "wikipedia-small" / "task1" / "wikipedia-small.h5",
        "gt_I":    lambda f: f["allknn"]["knns"],
        "k": 15,
        "n_vectors":  lambda f: len(f["train"]),
        "gt_1based": True,
    },
    ("llama-dev", "task2"): {
        "gt_path": DATA_DIR / "llama-dev.h5",
        "gt_I":    lambda f: np.array(f["test"]["knns"]),
        "k": 30,
        "n_queries": lambda f: np.array(f["test"]["knns"]).shape[0],
        "gt_1based": True,
    },
    ("fiqa-dev", "task3"): {
        "gt_path": DATA_DIR / "fiqa-dev.h5",
        "gt_I":    lambda f: np.array(f["otest"]["knns"]),
        "k": 30,
        "n_queries": lambda f: np.array(f["otest"]["knns"]).shape[0],
        "gt_1based": True,
    },
    ("nq", "task3"): {
        "gt_path": DATA_DIR / "nq" / "task3" / "nq.h5",
        "gt_I":    lambda f: f["otest"]["knns"],
        "k": 30,
        "n_queries": lambda f: f["otest"]["knns"].shape[0],
        "gt_1based": True,
    },
}

# ---------------------------------------------------------------------------
# Fallback (dataset, task) inference from path for submissions that do not
# write those attrs into their HDF5 files (e.g. the Julia baseline).
# Key: (team_dir, task_subdir) → (dataset, task)
# task_subdir is the directory name immediately containing the .h5 file.
# ---------------------------------------------------------------------------
PATH_INFERENCE: dict = {
    ("sisap2026-julia-example", "task2"): ("llama-dev",  "task2"),
    ("sisap2026-julia-example", "task3"): ("fiqa-dev",   "task3"),
}

TASK_THRESHOLDS = {"task1": 0.8, "task2": 0.8, "task3": 0.9}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_teams(path: Path) -> dict:
    """Load teams.yml → dict keyed by team directory name."""
    if not path.exists():
        print(f"Warning: {path} not found — using empty team registry.")
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return {entry["dir"]: entry for entry in data.get("teams", [])}


def get_recall(I: np.ndarray, gt: np.ndarray, k: int) -> float:
    assert k <= I.shape[1]
    n = len(I)
    return sum(len(set(I[i, :k]) & set(gt[i, :k])) for i in range(n)) / (n * k)


def process_result(h5path: Path, team_dir: str, teams: dict) -> dict | None:
    """Process one result HDF5 file; return summary row or None."""
    try:
        f = h5py.File(h5path, "r")
    except Exception as e:
        print(f"  SKIP (cannot open {h5path}): {e}")
        return None

    attrs = dict(f.attrs)
    required = {"algo", "task", "dataset", "buildtime", "querytime"}
    if not required.issubset(attrs):
        missing = required - set(attrs)
        # Try to infer dataset/task from path for submissions that omit them
        task_subdir = h5path.parent.name
        inferred = PATH_INFERENCE.get((team_dir, task_subdir))
        if inferred and "dataset" in missing and "task" in missing:
            attrs["dataset"], attrs["task"] = inferred
            missing -= {"dataset", "task"}
        if missing:
            print(f"  SKIP {h5path}: missing attrs {missing}")
            f.close()
            return None
    if "knns" not in f:
        print(f"  SKIP {h5path}: no 'knns' dataset")
        f.close()
        return None

    dataset    = attrs["dataset"].decode() if isinstance(attrs["dataset"], bytes) else str(attrs["dataset"])
    task       = attrs["task"].decode() if isinstance(attrs["task"], bytes) else str(attrs["task"])
    algo       = attrs["algo"].decode() if isinstance(attrs["algo"], bytes) else str(attrs["algo"])
    params     = attrs.get("params", "")
    params     = params.decode() if isinstance(params, bytes) else str(params)
    buildtime  = float(attrs["buildtime"])
    querytime  = float(attrs["querytime"])
    knns       = np.array(f["knns"])
    f.close()

    key = (dataset, task)
    if key not in DATASET_CFG:
        print(f"  SKIP {h5path}: unknown (dataset, task) = {key}")
        return None

    cfg     = DATASET_CFG[key]
    gt_path = cfg["gt_path"]
    if not gt_path.exists():
        print(f"  SKIP {h5path}: ground-truth not found at {gt_path}")
        return None

    gf   = h5py.File(gt_path, "r")
    gt_I = np.array(cfg["gt_I"](gf))
    if not cfg["gt_1based"]:
        gt_I = gt_I + 1

    k = cfg["k"]
    if task == "task1":
        n_total    = int(cfg["n_vectors"](gf))
        throughput = n_total / (buildtime + querytime) if (buildtime + querytime) > 0 else 0.0
        n_queries  = n_total
    else:
        n_queries  = int(cfg["n_queries"](gf))
        throughput = n_queries / querytime if querytime > 0 else 0.0
    gf.close()

    recall    = get_recall(knns, gt_I, k)
    threshold = TASK_THRESHOLDS.get(task, 0.8)

    # Team metadata: look up by directory name, fall back to sensible defaults
    team_meta = teams.get(team_dir, {})
    team      = team_meta.get("name", team_dir)
    repo      = team_meta.get("repo", "")
    is_base   = bool(team_meta.get("is_baseline", False))

    return {
        "task":        task,
        "dataset":     dataset,
        "algo":        algo,
        "team":        team,
        "repo":        repo,
        "is_baseline": is_base,
        "params":      params,
        "buildtime":   buildtime,
        "querytime":   querytime,
        "recall":      recall,
        "throughput":  throughput,
        "n_queries":   n_queries,
        "threshold":   threshold,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT),
                        help="Root directory containing one subdirectory per team")
    parser.add_argument("--output",       default=str(DEFAULT_OUTPUT),
                        help="Output Parquet file path")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_path  = Path(args.output)

    if not results_root.exists():
        print(f"Results root not found: {results_root}")
        sys.exit(1)

    teams = load_teams(TEAMS_FILE)
    print(f"Loaded {len(teams)} team(s) from {TEAMS_FILE}")

    # Discover all HDF5 files; the first path component under results_root is
    # the team directory name.
    h5files = sorted(results_root.glob("**/*.h5"))
    if not h5files:
        print(f"No .h5 files found under {results_root}")
        sys.exit(1)

    rows = []
    for h5path in h5files:
        # team_dir = first component below results_root
        rel       = h5path.relative_to(results_root)
        team_dir  = rel.parts[0]
        print(f"Processing [{team_dir}] {rel} …")
        row = process_result(h5path, team_dir, teams)
        if row is not None:
            rows.append(row)
            print(f"  {row['task']} | {row['dataset']} | {row['algo']} "
                  f"| recall={row['recall']:.4f} | throughput={row['throughput']:.1f}")

    if not rows:
        print("No valid results found.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nWrote {len(df)} rows → {output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
