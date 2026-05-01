"""
verify_output.py – Validate a SISAP 2026 result HDF5 file.

Usage:
    python verify_output.py <result.h5> [--dataset DATASET] [--compute-recall]

Checks performed:
  - Required datasets: knns (n×k int), dists (n×k float32)
  - Required root attributes: algo, task, buildtime, querytime, params
  - Shape consistency: knns.shape == dists.shape
  - 1-based indexing in knns (min >= 1)
  - No NaN/Inf in dists
  - File path follows results/<task>/<name>.h5 convention (warning only)
  - (Optional) recall computation against ground truth when --dataset is given

Exit codes:
    0  all checks passed
    1  one or more checks failed
"""

import argparse
import sys
import os
import h5py
import numpy as np


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"

_failures = []


def ok(msg):
    print(f"  [{PASS}] {msg}")


def fail(msg):
    print(f"  [{FAIL}] {msg}")
    _failures.append(msg)


def warn(msg):
    print(f"  [{WARN}] {msg}")


def check(condition, pass_msg, fail_msg):
    if condition:
        ok(pass_msg)
    else:
        fail(fail_msg)
    return condition


# --------------------------------------------------------------------------- #
# Task-level expectations (k per task)                                         #
# --------------------------------------------------------------------------- #

TASK_K = {
    "task1": 15,
    "task2": 30,
    "task3": 30,
}

# Recall thresholds per task
TASK_RECALL_THRESHOLD = {
    "task1": 0.8,
    "task2": 0.8,
    "task3": 0.9,
}


# --------------------------------------------------------------------------- #
# Checks                                                                       #
# --------------------------------------------------------------------------- #

def check_required_attrs(f):
    print("\n=== Required root attributes ===")
    required = ["algo", "task", "buildtime", "querytime", "params"]
    all_present = True
    for attr in required:
        if attr in f.attrs:
            ok(f"attr '{attr}' = {f.attrs[attr]!r}")
        else:
            fail(f"Missing required attribute '{attr}'")
            all_present = False
    return all_present


def check_required_datasets(f):
    print("\n=== Required datasets ===")
    knns_ok = "knns" in f
    dists_ok = "dists" in f
    check(knns_ok, "dataset 'knns' present", "dataset 'knns' missing")
    check(dists_ok, "dataset 'dists' present", "dataset 'dists' missing")
    return knns_ok, dists_ok


def check_shapes_and_dtypes(f, task_name):
    print("\n=== Shape and dtype checks ===")
    knns = f["knns"]
    dists = f["dists"]

    ok(f"knns: shape={knns.shape} dtype={knns.dtype}")
    ok(f"dists: shape={dists.shape} dtype={dists.dtype}")

    check(len(knns.shape) == 2,
          f"knns is 2-D",
          f"knns has {len(knns.shape)} dimensions, expected 2")
    check(len(dists.shape) == 2,
          f"dists is 2-D",
          f"dists has {len(dists.shape)} dimensions, expected 2")

    check(knns.shape == dists.shape,
          f"knns.shape == dists.shape ({knns.shape})",
          f"knns.shape {knns.shape} != dists.shape {dists.shape}")

    # dtype checks
    check(np.issubdtype(knns.dtype, np.integer),
          f"knns dtype {knns.dtype} is integer",
          f"knns dtype {knns.dtype} is not integer")
    check(np.dtype(dists.dtype) == np.dtype("float32"),
          f"dists dtype is float32",
          f"dists dtype is {dists.dtype}, expected float32")

    # k consistency with task
    if task_name in TASK_K:
        k_expected = TASK_K[task_name]
        k_actual = knns.shape[1]
        check(k_actual == k_expected,
              f"k={k_actual} matches expected k={k_expected} for {task_name}",
              f"k={k_actual} != expected k={k_expected} for {task_name}")

    return knns.shape  # (n, k)


def check_indexing_and_values(f, n, k):
    print("\n=== Value sanity checks ===")
    knns = f["knns"]
    dists = f["dists"]

    sample = min(n, 10_000)
    knns_sample = knns[:sample].astype(np.int64)
    dists_sample = dists[:sample].astype(np.float32)

    mn = int(knns_sample.min())
    mx = int(knns_sample.max())
    check(mn >= 1,
          f"knns min id={mn} >= 1 (1-based indexing)",
          f"knns min id={mn} < 1 – IDs must be 1-based")
    ok(f"knns max id={mx} (upper bound depends on dataset N)")

    has_nan = bool(np.any(np.isnan(dists_sample)))
    has_inf = bool(np.any(np.isinf(dists_sample)))
    check(not has_nan, f"dists: no NaN in first {sample} rows",
          f"dists: NaN values found in first {sample} rows")
    check(not has_inf, f"dists: no Inf in first {sample} rows",
          f"dists: Inf values found in first {sample} rows")


def check_file_path_convention(path, task_name):
    print("\n=== File path convention ===")
    parts = os.path.normpath(path).split(os.sep)
    # Accept both:
    #   results/<task>/<name>.h5            (SISAP spec)
    #   results/<dataset>/<task>/<name>.h5  (search.py actual output)
    results_idx = None
    for i, p in enumerate(parts):
        if p == "results":
            results_idx = i
    if results_idx is None:
        warn(f"'results/' not found in path – acceptable outside submission context")
        return
    tail = parts[results_idx + 1:]
    if len(tail) == 2:
        # results/<task>/<name>.h5
        task_dir = tail[0]
        if task_name and task_dir != task_name:
            warn(f"results/<task>/ dir is '{task_dir}', expected '{task_name}'")
        else:
            ok(f"Path follows results/<task>/<name>.h5 convention")
    elif len(tail) == 3:
        # results/<dataset>/<task>/<name>.h5
        task_dir = tail[1]
        if task_name and task_dir != task_name:
            warn(f"results/<dataset>/<task>/ task dir is '{task_dir}', expected '{task_name}'")
        else:
            ok(f"Path follows results/<dataset>/<task>/<name>.h5 convention")
    else:
        warn(f"Unexpected nesting depth under results/ in path '{path}'")


def compute_recall(res_knns, gt_knns, k, task_name):
    """Compute recall@k, removing self-references for Task 1."""
    assert len(res_knns) == len(gt_knns), \
        f"Result has {len(res_knns)} rows, GT has {len(gt_knns)} rows"

    n = len(res_knns)
    recall_sum = 0

    for i in range(n):
        res_set = set(res_knns[i, :k].tolist())
        gt_row = gt_knns[i]

        if task_name == "task1":
            # Remove self-reference (1-based: self = i+1)
            gt_row = [x for x in gt_row if x != i + 1]

        gt_set = set(gt_row[:k])
        recall_sum += len(res_set & gt_set)

    return recall_sum / (n * k)


def run_recall_check(f, dataset_name, task_name):
    print(f"\n=== Recall check: {dataset_name}/{task_name} ===")
    try:
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
        from datasets import DATASETS, get_fn, get_gt_fn, prepare
    except ImportError:
        warn("datasets.py not importable – skipping recall check")
        return

    if dataset_name not in DATASETS or task_name not in DATASETS[dataset_name]:
        fail(f"Unknown dataset/task: {dataset_name}/{task_name}")
        return

    prepare(dataset_name, task_name)
    cfg = DATASETS[dataset_name][task_name]
    gt_fn = get_gt_fn(dataset_name, task_name)

    try:
        f_gt = h5py.File(gt_fn, "r")
    except Exception as e:
        warn(f"Cannot open GT file '{gt_fn}': {e}\n"
             "    Tip: run this script from the directory that contains 'data/'")
        return

    with f_gt:
        try:
            gt_knns = np.array(cfg['gt_I'](f_gt), dtype=np.int64)
        except Exception as e:
            fail(f"cfg['gt_I'] raised {e}")
            return

    n_res = f["knns"].shape[0]
    n_gt = gt_knns.shape[0]
    if n_res != n_gt:
        fail(f"Result has {n_res} rows but GT has {n_gt} rows")
        return

    k = cfg.get('k', f["knns"].shape[1])
    res_knns = np.array(f["knns"], dtype=np.int64)

    recall = compute_recall(res_knns, gt_knns, k, task_name)
    threshold = TASK_RECALL_THRESHOLD.get(task_name, 0.8)

    ok(f"recall@{k} = {recall:.4f} (threshold={threshold})")
    check(recall >= threshold,
          f"recall {recall:.4f} >= threshold {threshold}",
          f"recall {recall:.4f} < threshold {threshold} – does not meet operating point")


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Verify a SISAP 2026 result HDF5 file.")
    ap.add_argument("file", help="Path to the result HDF5 file")
    ap.add_argument("--dataset", default=None,
                    help="Dataset name (e.g. gooaq-small) for recall computation")
    ap.add_argument("--compute-recall", action="store_true", default=False,
                    help="Compute recall against ground truth (requires --dataset)")
    args = ap.parse_args()

    print(f"Verifying result: {args.file}")

    try:
        f = h5py.File(args.file, "r")
    except Exception as e:
        print(f"[{FAIL}] Cannot open file: {e}")
        sys.exit(1)

    with f:
        attrs_ok = check_required_attrs(f)
        knns_ok, dists_ok = check_required_datasets(f)

        task_name = str(f.attrs.get("task", "")) if attrs_ok else ""

        if knns_ok and dists_ok:
            n, k = check_shapes_and_dtypes(f, task_name)
            check_indexing_and_values(f, n, k)
        else:
            fail("Cannot continue shape/value checks without both knns and dists")

        check_file_path_convention(args.file, task_name)

        if args.compute_recall:
            if not args.dataset:
                warn("--compute-recall requires --dataset; skipping recall check")
            elif not task_name:
                warn("Cannot determine task from file attributes; skipping recall check")
            else:
                run_recall_check(f, args.dataset, task_name)

    print()
    if _failures:
        print(f"Result: {len(_failures)} check(s) FAILED:")
        for msg in _failures:
            print(f"  - {msg}")
        sys.exit(1)
    else:
        print("Result: all checks PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
