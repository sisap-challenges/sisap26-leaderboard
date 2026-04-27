#!/usr/bin/env python3
"""
summarise-bench.py — summarise hforest benchmark results across nodes.

For each completed run under OUTPUT_BASE it reports:
  - node name and CPU model (from lscpu.txt)
  - SLURM wall time and peak RSS (from sacct CSV written by summarise-bench.sh)
  - hforest build time and query/graph time (from result HDF5 attributes)

Usage (via wrapper):
    sbatch ... hpc/summarise-bench.sh [--sort-by COLUMN]
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py")


OUTPUT_BASE_DEFAULT = "/home/maau/sisap26-baseline-dev/bench/"


def load_sacct_csv(path: Path) -> dict:
    """Load sacct CSV produced by summarise-bench.sh keyed by job_id string."""
    data = {}
    if path is None or not path.exists():
        return data
    with open(path) as f:
        for row in csv.DictReader(f):
            data[row["job_id"]] = {
                "state":   row.get("state", "n/a"),
                "elapsed": row.get("elapsed", "n/a"),
                "max_rss": row.get("max_rss", "n/a"),
            }
    return data


def cpu_model(lscpu_path: Path) -> str:
    """Extract 'Model name' line from lscpu.txt."""
    if not lscpu_path.exists():
        return "unknown"
    for line in lscpu_path.read_text().splitlines():
        if "Model name" in line:
            return line.split(":", 1)[-1].strip()
    return "unknown"


def find_result_h5(output_dir: Path):
    """Return the first HDF5 result file found under output_dir."""
    matches = list(output_dir.glob("gooaq/task2/hforest_*.h5"))
    return matches[0] if matches else None


def read_h5_attrs(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        return {
            "buildtime": float(f.attrs.get("buildtime", float("nan"))),
            "querytime": float(f.attrs.get("querytime", float("nan"))),
            "params":    str(f.attrs.get("params", "")),
        }


def parse_dir_name(name: str):
    """
    Directory names are <node>-<N>cpu-<jobid>, e.g. cn14-8cpu-74021.
    Returns (node, ncpus, job_id).
    """
    parts = name.rsplit("-", 2)
    if len(parts) == 3:
        node, cpupart, job_id = parts
        ncpus = cpupart.replace("cpu", "")
        return node, ncpus, job_id
    return name, "?", "?"


def collect(output_base: Path, sacct: dict) -> list:
    # First pass: collect all rows grouped by node so we can drop cancelled
    # duplicates (directories that have lscpu.txt but no h5 result file).
    by_node = {}
    for d in sorted(output_base.iterdir()):
        if not d.is_dir():
            continue
        node, ncpus, job_id = parse_dir_name(d.name)
        slurm = sacct.get(job_id, {"state": "n/a", "elapsed": "n/a", "max_rss": "n/a"})
        h5 = find_result_h5(d)
        row = {
            "node":      node,
            "cpus":      ncpus,
            "job_id":    job_id,
            "cpu_model": cpu_model(d / "lscpu.txt"),
            "state":     slurm["state"],
            "wall_time": slurm["elapsed"],
            "max_rss":   slurm["max_rss"],
            "has_result": h5 is not None,
            "build_s":   float("nan"),
            "query_s":   float("nan"),
            "total_s":   float("nan"),
            "params":    "",
        }
        if h5 is not None:
            attrs = read_h5_attrs(h5)
            row.update({
                "build_s": attrs["buildtime"],
                "query_s": attrs["querytime"],
                "total_s": attrs["buildtime"] + attrs["querytime"],
                "params":  attrs["params"],
            })
        by_node.setdefault(node, []).append(row)

    # Second pass: for each node prefer rows with results; only fall back to
    # no-result rows when there is nothing else for that node.
    rows = []
    for node, candidates in by_node.items():
        with_results = [r for r in candidates if r["has_result"]]
        rows.extend(with_results if with_results else candidates)

    return rows


def fmt(val, decimals=1):
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def print_table(rows: list, sort_by: str):
    sort_keys = {
        "node":  lambda r: r["node"],
        "total": lambda r: r["total_s"] if r["total_s"] == r["total_s"] else float("inf"),
        "query": lambda r: r["query_s"] if r["query_s"] == r["query_s"] else float("inf"),
        "wall":  lambda r: r["wall_time"],
    }
    rows = sorted(rows, key=sort_keys.get(sort_by, sort_keys["total"]))

    header = (
        f"{'Node':<12} {'CPUs':>4} {'JobID':>7}  "
        f"{'State':<12} {'Wall time':>10} {'MaxRSS':>8}  "
        f"{'Build(s)':>9} {'Query(s)':>9} {'Total(s)':>9}  "
        f"CPU model"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['node']:<12} {r['cpus']:>4} {r['job_id']:>7}  "
            f"{r['state']:<12} {r['wall_time']:>10} {r['max_rss']:>8}  "
            f"{fmt(r['build_s']):>9} {fmt(r['query_s']):>9} {fmt(r['total_s']):>9}  "
            f"{r['cpu_model']}"
        )
    print(sep)

    finished = [r for r in rows if r["total_s"] == r["total_s"]]
    if finished:
        best = min(finished, key=lambda r: r["total_s"])
        print(f"\nFastest: {best['node']} — {fmt(best['total_s'])}s total "
              f"(build {fmt(best['build_s'])}s + query {fmt(best['query_s'])}s)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-base", default=OUTPUT_BASE_DEFAULT,
                        help="Directory containing per-run result subdirectories")
    parser.add_argument("--sacct-csv", default=None,
                        help="CSV file with SLURM accounting data (written by summarise-bench.sh)")
    parser.add_argument("--sort-by", choices=["node", "total", "query", "wall"],
                        default="total",
                        help="Column to sort by (default: total)")
    args = parser.parse_args()

    base = Path(args.output_base)
    if not base.exists():
        sys.exit(f"Output base not found: {base}")

    sacct = load_sacct_csv(Path(args.sacct_csv) if args.sacct_csv else None)
    rows = collect(base, sacct)
    if not rows:
        sys.exit("No result directories found.")

    print_table(rows, args.sort_by)


if __name__ == "__main__":
    main()
