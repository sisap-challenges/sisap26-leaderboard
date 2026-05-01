"""
verify_input.py – Validate a SISAP 2026 input HDF5 dataset file.

Usage:
    python verify_input.py <file.h5> [--dataset DATASET] [--task TASK]

Without --dataset / --task the script performs generic structural checks.
When a dataset name is given it additionally cross-checks shapes and GT
consistency against the datasets.py configuration.

Exit codes:
    0  all checks passed
    1  one or more checks failed
"""

import argparse
import sys
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
# Individual checks                                                            #
# --------------------------------------------------------------------------- #

def check_dense_dataset(f, path, expected_ndim=None, expected_dtype=None, label=""):
    """Check that a dense dataset exists and has the right shape/dtype."""
    tag = label or path
    if path not in f:
        fail(f"{tag}: dataset '{path}' not found")
        return None
    ds = f[path]
    ok(f"{tag}: found '{path}' shape={ds.shape} dtype={ds.dtype}")
    if expected_ndim is not None:
        check(len(ds.shape) == expected_ndim,
              f"{tag}: ndim={len(ds.shape)} == {expected_ndim}",
              f"{tag}: expected ndim={expected_ndim}, got {len(ds.shape)}")
    if expected_dtype is not None:
        check(np.dtype(ds.dtype) == np.dtype(expected_dtype),
              f"{tag}: dtype={ds.dtype} matches {expected_dtype}",
              f"{tag}: expected dtype={expected_dtype}, got {ds.dtype}")
    return ds


def check_no_nan_inf(ds, label=""):
    """Sample up to 100k rows and check for NaN/Inf."""
    tag = label or ds.name
    sample_rows = min(len(ds), 100_000)
    data = ds[:sample_rows]
    if not np.issubdtype(data.dtype, np.floating):
        ok(f"{tag}: dtype {ds.dtype} – NaN/Inf check skipped")
        return
    has_nan = np.any(np.isnan(data))
    has_inf = np.any(np.isinf(data))
    check(not has_nan, f"{tag}: no NaN values in first {sample_rows} rows",
          f"{tag}: NaN values found in first {sample_rows} rows")
    check(not has_inf, f"{tag}: no Inf values in first {sample_rows} rows",
          f"{tag}: Inf values found in first {sample_rows} rows")


def check_1based_indexing(ds, N, label=""):
    """Check that integer IDs are in [1, N]."""
    tag = label or ds.name
    sample_rows = min(len(ds), 10_000)
    data = ds[:sample_rows].astype(np.int64)
    mn, mx = int(data.min()), int(data.max())
    check(mn >= 1, f"{tag}: min id={mn} >= 1 (1-based indexing)",
          f"{tag}: min id={mn} < 1 – IDs must be 1-based")
    check(mx <= N, f"{tag}: max id={mx} <= N={N}",
          f"{tag}: max id={mx} > N={N} – out-of-range IDs")


def check_self_loop(ds, label=""):
    """Warn if the first column of each row is the row index (1-based self-loop)."""
    tag = label or ds.name
    sample_rows = min(len(ds), 1_000)
    data = ds[:sample_rows, 0].astype(np.int64)
    row_ids = np.arange(1, sample_rows + 1, dtype=np.int64)
    frac = float(np.mean(data == row_ids))
    if frac > 0.9:
        ok(f"{tag}: self-loop in col-0 for {frac*100:.0f}% of sampled rows (expected for Task 1 GT)")
    else:
        warn(f"{tag}: self-loop in col-0 for only {frac*100:.0f}% of sampled rows "
             "(expected ~100% for Task 1 all-kNN GT; may be intentional for other tasks)")


# --------------------------------------------------------------------------- #
# Generic structural checks (no dataset config needed)                         #
# --------------------------------------------------------------------------- #

def run_generic_checks(f, path):
    print("\n=== Generic structural checks ===")

    # --- train vectors -------------------------------------------------------
    train_ds = check_dense_dataset(f, "train", expected_ndim=2, label="train")
    if train_ds is None:
        fail("train: cannot continue without 'train' dataset")
        return

    N, d = train_ds.shape
    ok(f"train: N={N} vectors, d={d} dimensions")
    check_no_nan_inf(train_ds, label="train")

    # --- all-kNN GT (Task 1 layout: knns at root or under allknn/) ----------
    knns_paths = ["knns", "allknn/knns"]
    dists_paths = ["dists", "allknn/dists"]

    knns_ds = None
    for p in knns_paths:
        if p in f:
            knns_ds = check_dense_dataset(f, p, expected_ndim=2, label=f"GT({p})")
            break
    if knns_ds is None:
        warn("No 'knns' dataset found at root or under 'allknn/' – "
             "if this is a Task 2/3 file, GT lives under 'itest/' or 'otest/'")

    dists_ds = None
    for p in dists_paths:
        if p in f:
            dists_ds = check_dense_dataset(f, p, expected_ndim=2,
                                           expected_dtype="float32", label=f"GT dists({p})")
            break

    if knns_ds is not None:
        k_store = knns_ds.shape[1]
        check(knns_ds.shape[0] == N,
              f"GT knns: rows={knns_ds.shape[0]} == N={N}",
              f"GT knns: rows={knns_ds.shape[0]} != train N={N}")
        if dists_ds is not None:
            check(dists_ds.shape == knns_ds.shape,
                  f"GT dists shape {dists_ds.shape} == knns shape {knns_ds.shape}",
                  f"GT dists shape {dists_ds.shape} != knns shape {knns_ds.shape}")
        check_1based_indexing(knns_ds, N, label="GT knns")
        check_self_loop(knns_ds, label="GT knns")
        check_no_nan_inf(dists_ds, label="GT dists") if dists_ds is not None else None

    # --- Task 2/3: itest / otest ---------------------------------------------
    for split in ("itest", "otest"):
        if split in f:
            ok(f"Found '{split}' group (Task 2/3 layout)")
            for sub in ("queries", "knns"):
                p = f"{split}/{sub}"
                if p in f:
                    check_dense_dataset(f, p, label=p)


# --------------------------------------------------------------------------- #
# Dataset-config-aware checks                                                  #
# --------------------------------------------------------------------------- #

def run_dataset_checks(f, path, dataset_name, task_name):
    print(f"\n=== Dataset-config checks: {dataset_name}/{task_name} ===")
    try:
        from datasets import DATASETS, get_fn, get_gt_fn
    except ImportError:
        warn("datasets.py not importable from current directory – skipping config checks")
        return

    if dataset_name not in DATASETS:
        fail(f"Dataset '{dataset_name}' not in DATASETS")
        return
    if task_name not in DATASETS[dataset_name]:
        fail(f"Task '{task_name}' not in DATASETS['{dataset_name}']")
        return

    cfg = DATASETS[dataset_name][task_name]

    # Verify 'data' lambda
    try:
        data_ds = cfg['data'](f)
        ok(f"cfg['data'](f): shape={data_ds.shape} dtype={data_ds.dtype}")
        N = data_ds.shape[0]
    except Exception as e:
        fail(f"cfg['data'](f) raised {e}")
        return

    # Verify 'gt_I' lambda (only when GT is in the same file)
    gt_fn = get_gt_fn(dataset_name, task_name)
    if gt_fn == path:
        try:
            gt_ds = cfg['gt_I'](f)
            ok(f"cfg['gt_I'](f): shape={gt_ds.shape} dtype={gt_ds.dtype}")
            check(gt_ds.shape[0] == N,
                  f"GT rows={gt_ds.shape[0]} == N={N}",
                  f"GT rows={gt_ds.shape[0]} != N={N}")
            k_store = gt_ds.shape[1]
            k_req = cfg.get('k', None)
            if k_req is not None:
                check(k_store >= k_req,
                      f"GT k_store={k_store} >= required k={k_req}",
                      f"GT k_store={k_store} < required k={k_req}")
            check_1based_indexing(gt_ds, N, label="cfg gt_I")
        except Exception as e:
            fail(f"cfg['gt_I'](f) raised {e}")
    else:
        ok(f"GT lives in a separate file ({gt_fn}) – skipping GT checks here")

    # Verify 'queries' lambda for Task 2/3
    if 'queries' in cfg:
        try:
            q_ds = cfg['queries'](f)
            ok(f"cfg['queries'](f): shape={q_ds.shape} dtype={q_ds.dtype}")
        except Exception as e:
            fail(f"cfg['queries'](f) raised {e}")


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Verify a SISAP 2026 input HDF5 dataset.")
    ap.add_argument("file", help="Path to the HDF5 file to verify")
    ap.add_argument("--dataset", default=None,
                    help="Dataset name (e.g. gooaq-small) for config-aware checks")
    ap.add_argument("--task", default=None,
                    help="Task name (e.g. task1) for config-aware checks")
    args = ap.parse_args()

    print(f"Verifying: {args.file}")

    try:
        f = h5py.File(args.file, "r")
    except Exception as e:
        print(f"[{FAIL}] Cannot open file: {e}")
        sys.exit(1)

    with f:
        run_generic_checks(f, args.file)
        if args.dataset and args.task:
            run_dataset_checks(f, args.file, args.dataset, args.task)
        elif args.dataset or args.task:
            warn("Provide both --dataset and --task for config-aware checks; skipping.")

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
