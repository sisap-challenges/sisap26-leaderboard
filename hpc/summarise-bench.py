#!/usr/bin/env python3
"""
summarise-bench.py — summarise hforest benchmark results across nodes.

For each completed run under OUTPUT_BASE it reports:
  - node name and CPU model (from lscpu.txt)
  - SLURM wall time and peak RSS (from sacct)
  - hforest build time and query/graph time (from result HDF5 attributes)

Usage:
    python summarise-bench.py [--output-base PATH] [--sort-by COLUMN]

OUTPUT_BASE defaults to /home/maau/sisap26-baseline-dev/bench/
"""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py")


OUTPUT_BASE_DEFAULT = "/home/maau/sisap26-baseline-dev/bench/"


def sacct_info(job_id: str) -> dict:
    """Return elapsed wall time and peak RSS for a SLURM job via sacct."""
    result = subprocess.run(
        [
            "sacct", "-j", job_id,
            "--format=Elapsed,MaxRSS,State",
            "--noheader", "--parsable2",
            # Use the batch step which carries RSS; fall back to job summary
            "--steps",
        ],
        capture_output=True, text=True,
    )
    best = {"elapsed": "n/a", "max_rss": "n/a", "state": "n/a"}
    for line in result.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 3:
            continue
        elapsed, rss, state = parts[0], parts[1], parts[2]
        if elapsed:
            best["elapsed"] = elapsed
        if rss:
            best["max_rss"] = rss
        if state:
            best["state"] = state
    return best


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


def collect(output_base: Path) -> list[dict]:
    rows = []
    for d in sorted(output_base.iterdir()):
        if not d.is_dir():
            continue
        node, ncpus, job_id = parse_dir_name(d.name)
        h5 = find_result_h5(d)
        if h5 is None:
            # Job may have failed or still running
            rows.append({
                "node": node, "cpus": ncpus, "job_id": job_id,
                "cpu_model": cpu_model(d / "lscpu.txt"),
                "state": sacct_info(job_id)["state"],
                "wall_time": "n/a", "max_rss": "n/a",
                "build_s": float("nan"), "query_s": float("nan"),
                "total_s": float("nan"), "params": "",
            })
            continue
        attrs = read_h5_attrs(h5)
        slurm = sacct_info(job_id)
        rows.append({
            "node":      node,
            "cpus":      ncpus,
            "job_id":    job_id,
            "cpu_model": cpu_model(d / "lscpu.txt"),
            "state":     slurm["state"],
            "wall_time": slurm["elapsed"],
            "max_rss":   slurm["max_rss"],
            "build_s":   attrs["buildtime"],
            "query_s":   attrs["querytime"],
            "total_s":   attrs["buildtime"] + attrs["querytime"],
            "params":    attrs["params"],
        })
    return rows


def fmt(val, decimals=1):
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def print_table(rows: list[dict], sort_by: str):
    sort_keys = {
        "node":    lambda r: r["node"],
        "total":   lambda r: r["total_s"] if r["total_s"] == r["total_s"] else float("inf"),
        "query":   lambda r: r["query_s"] if r["query_s"] == r["query_s"] else float("inf"),
        "wall":    lambda r: r["wall_time"],
    }
    rows = sorted(rows, key=sort_keys.get(sort_by, sort_keys["total"]))

    header = (
        f"{'Node':<12} {'CPUs':>4} {'JobID':>7}  "
        f"{'State':<10} {'Wall time':>10} {'MaxRSS':>8}  "
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
            f"{r['state']:<10} {r['wall_time']:>10} {r['max_rss']:>8}  "
            f"{fmt(r['build_s']):>9} {fmt(r['query_s']):>9} {fmt(r['total_s']):>9}  "
            f"{r['cpu_model']}"
        )
    print(sep)

    # Quick recommendation
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
    parser.add_argument("--sort-by", choices=["node", "total", "query", "wall"],
                        default="total",
                        help="Column to sort by (default: total)")
    args = parser.parse_args()

    base = Path(args.output_base)
    if not base.exists():
        sys.exit(f"Output base not found: {base}")

    rows = collect(base)
    if not rows:
        sys.exit("No result directories found.")

    print_table(rows, args.sort_by)


if __name__ == "__main__":
    main()
