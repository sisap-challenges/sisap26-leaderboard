#!/usr/bin/env python3
"""
Export results from ../sisap26-python-baseline/results/ to website/results/summary.parquet.

Computes recall against ground-truth and throughput for each result file.
"""

import glob
import os
import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASELINE_DIR = Path(__file__).parent.parent / "sisap26-python-baseline"
RESULTS_DIR = BASELINE_DIR / "results"
OUTPUT_DIR = Path(__file__).parent / "website" / "results"
OUTPUT_FILE = OUTPUT_DIR / "summary.parquet"

# Algorithms provided by the organisers as baselines
BASELINE_ALGOS = {"faissIVF", "ABS", "ExhaustiveSearch", "pytorch_sparse_mm"}

# Team metadata keyed by algo name.
# Add real participant entries here as submissions come in.
# "team"  : display name shown in the leaderboard
# "repo"  : GitHub (or other) URL for the submission; None = no link
TEAM_INFO = {
    "faissIVF":          {"team": "python-baseline", "repo": "https://github.com/sisap-challenges/sisap26-python-baseline"},
    "ABS":               {"team": "julia-baseline",  "repo": "https://github.com/sisap-challenges/sisap2026-julia-example"},
    "ExhaustiveSearch":  {"team": "julia-baseline",  "repo": "https://github.com/sisap-challenges/sisap2026-julia-example"},
    "pytorch_sparse_mm": {"team": "python-baseline", "repo": "https://github.com/sisap-challenges/sisap26-python-baseline"},
}

# Task thresholds (for reference; stored in output so the website can use them)
TASK_THRESHOLDS = {
    "task1": 0.8,
    "task2": 0.8,
    "task3": 0.9,
}

# Ground-truth configuration per (dataset, task)
# gt_I: lambda to extract the GT index array from the HDF5 file
# gt_path: path to the HDF5 file relative to BASELINE_DIR
# k: number of neighbours
# n_vectors: lambda to get the total number of database vectors (for task1 throughput)
# n_queries: lambda to get the number of query vectors (for task2/3 throughput)
# gt_1based: whether the GT indices are already 1-based (False → add 1 before recall)
DATASET_CFG = {
    ("gooaq-small", "task1"): {
        "gt_path": "data/benchmark-dev-gooaq-small.h5",
        "gt_I": lambda f: f["allknn"]["knns"],
        "k": 15,
        "n_vectors": lambda f: len(f["train"]),
        "gt_1based": True,
    },
    ("wikipedia-small", "task1"): {
        "gt_path": "data/wikipedia-small/task1/wikipedia-small.h5",
        "gt_I": lambda f: f["allknn"]["knns"],
        "k": 15,
        "n_vectors": lambda f: len(f["train"]),
        "gt_1based": True,
    },
    ("llama-dev", "task2"): {
        "gt_path": "data/llama-dev/task2/llama-dev.h5",
        "gt_I": lambda f: f["test"]["knns"],
        "k": 30,
        "n_queries": lambda f: len(f["test"]["queries"]),
        "gt_1based": False,   # 0-based → add 1
    },
    ("fiqa-dev", "task3"): {
        "gt_path": "data/fiqa-dev/task3/fiqa-dev.h5",
        "gt_I": lambda f: f["otest"]["knns"],
        "k": 30,
        "n_queries": lambda f: f["otest"]["knns"].shape[0],
        "gt_1based": True,
    },
    ("nq", "task3"): {
        "gt_path": "data/nq/task3/nq.h5",
        "gt_I": lambda f: f["otest"]["knns"],
        "k": 30,
        "n_queries": lambda f: f["otest"]["knns"].shape[0],
        "gt_1based": True,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_recall(I, gt, k):
    """Mean recall@k between result array I and ground-truth gt (both 1-based, N×k)."""
    assert k <= I.shape[1]
    n = len(I)
    recall = sum(
        len(set(I[i, :k]) & set(gt[i, :k]))
        for i in range(n)
    )
    return recall / (n * k)


def process_result(h5path: Path) -> dict | None:
    """Process a single result HDF5 file and return a summary dict, or None on error."""
    try:
        f = h5py.File(h5path, "r")
    except Exception as e:
        print(f"  SKIP (cannot open): {e}")
        return None

    attrs = dict(f.attrs)
    required = {"algo", "task", "dataset", "buildtime", "querytime"}
    if not required.issubset(attrs):
        print(f"  SKIP (missing attrs {required - set(attrs)})")
        f.close()
        return None
    if "knns" not in f:
        print(f"  SKIP (no knns dataset)")
        f.close()
        return None

    dataset = str(attrs["dataset"])
    task = str(attrs["task"])
    algo = str(attrs["algo"])
    params = str(attrs.get("params", ""))
    buildtime = float(attrs["buildtime"])
    querytime = float(attrs["querytime"])
    knns = np.array(f["knns"])
    f.close()

    key = (dataset, task)
    if key not in DATASET_CFG:
        print(f"  SKIP (unknown dataset/task {key})")
        return None

    cfg = DATASET_CFG[key]
    gt_path = BASELINE_DIR / cfg["gt_path"]
    if not gt_path.exists():
        print(f"  SKIP (GT not found: {gt_path})")
        return None

    gf = h5py.File(gt_path, "r")
    gt_I = np.array(cfg["gt_I"](gf))
    if not cfg["gt_1based"]:
        gt_I = gt_I + 1   # shift 0-based GT to 1-based to match result files

    k = cfg["k"]

    # Determine N (for throughput)
    if task == "task1":
        n_vectors = int(cfg["n_vectors"](gf))
        throughput = n_vectors / (buildtime + querytime)
        n_queries = n_vectors
    else:
        n_queries = int(cfg["n_queries"](gf))
        throughput = n_queries / querytime if querytime > 0 else 0.0
        n_vectors = None
    gf.close()

    recall = get_recall(knns, gt_I, k)
    threshold = TASK_THRESHOLDS.get(task, 0.8)

    team_info = TEAM_INFO.get(algo, {"team": algo, "repo": None})

    return {
        "task": task,
        "dataset": dataset,
        "algo": algo,
        "team": team_info["team"],
        "repo": team_info["repo"] or "",
        "is_baseline": algo in BASELINE_ALGOS,
        "params": params,
        "buildtime": buildtime,
        "querytime": querytime,
        "recall": recall,
        "throughput": throughput,
        "n_queries": n_queries,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pattern = str(RESULTS_DIR / "**" / "*.h5")
    h5files = sorted(glob.glob(pattern, recursive=True))
    if not h5files:
        print(f"No result files found under {RESULTS_DIR}")
        sys.exit(1)

    rows = []
    for h5path in h5files:
        p = Path(h5path)
        print(f"Processing {p.relative_to(BASELINE_DIR)} ...")
        row = process_result(p)
        if row is not None:
            rows.append(row)
            print(f"  {row['task']} | {row['dataset']} | {row['algo']} | params={row['params']} | recall={row['recall']:.4f} | throughput={row['throughput']:.1f}")

    if not rows:
        print("No valid results found.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nWrote {len(df)} rows to {OUTPUT_FILE}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
