"""
create-gooaq-small.py
=====================
Creates a small self-contained example file in the same HDF5 format as the
SISAP 2025 Task 2 GOOAQ dataset (all-kNN graph benchmark).

The output file can be used to test the sisap26-python-baseline pipeline
without downloading the full 4.8 GB dataset.

Two modes
---------
* Real data  – if data/benchmark-dev-gooaq.h5 exists and is non-empty, the
               first --n-vectors rows are extracted from it.
* Synthetic  – otherwise, unit-normalised Gaussian vectors are generated with
               the same dimensionality (384) and a fixed random seed.

Ground truth
------------
Exact k-NN (inner product) is computed with a FAISS IndexFlatIP index.
Self-loops are included in the result (distance ≈ 0, index = row+1) so that
the evaluation code can handle them the same way as the full dataset.

Output format
-------------
benchmark-dev-gooaq-small.h5
  /train          float32  (N, 384)   – embedding matrix (row-major, 1 row/vector)
  /allknn/knns    int32    (N, k_store) – 1-indexed neighbor ids
  /allknn/dists   float32  (N, k_store) – corresponding inner-product similarities
  Attributes: algo='exhaustive', n=N, d=384, k=k_store

Usage
-----
    python create-gooaq-small.py [options]

    -n / --n-vectors   Number of vectors to include  (default: 10000)
    -k / --k-store     Neighbors to store per vector (default: 32)
    --src              Path to the full gooaq HDF5 file
                       (default: data/benchmark-dev-gooaq.h5)
    --dst              Output HDF5 file
                       (default: data/benchmark-dev-gooaq-small.h5)
    --seed             RNG seed for synthetic data (default: 42)
"""

import argparse
import os
import time
from pathlib import Path

import faiss
import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unit_normalise(x: np.ndarray) -> np.ndarray:
    """L2-normalise each row of a 2-D float32 array in-place and return it."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)   # avoid division by zero
    x /= norms.astype(np.float32)
    return x


def load_real_vectors(src_path: str, n: int) -> np.ndarray:
    """
    Load the first *n* vectors from the full GOOAQ HDF5 file.

    The full file stores the embedding matrix under the key 'train'.
    The matrix may be row-major (N, d) or column-major (d, N) depending on
    how the file was created; this function handles both orientations.
    """
    with h5py.File(src_path, "r") as f:
        ds = f["train"]
        s = ds.shape
        if s[0] < s[1]:
            # Shape is (d, N) — take first n columns, then transpose
            print(f"  Detected transposed layout (d={s[0]}, N={s[1]}); "
                  f"reading first {n} columns.")
            data = ds[:, :n].T.astype(np.float32)
        else:
            # Shape is (N, d) — take first n rows directly
            print(f"  Detected row-major layout (N={s[0]}, d={s[1]}); "
                  f"reading first {n} rows.")
            data = ds[:n, :].astype(np.float32)
    return data


def make_synthetic_vectors(n: int, d: int, seed: int) -> np.ndarray:
    """
    Draw n unit-normalised float32 vectors from a standard Gaussian.

    This is a common approximation of real sentence-BERT distributions and
    gives a dataset with the same inner-product geometry.
    """
    rng = np.random.default_rng(seed)
    data = rng.standard_normal((n, d)).astype(np.float32)
    return unit_normalise(data)


def compute_exact_allknn(data: np.ndarray, k: int) -> tuple:
    """
    Compute the exact all-kNN graph with FAISS IndexFlatIP.

    Self-loops (each vector is its own nearest neighbour) are included so
    that the output matches the convention used by the full GOOAQ ground truth.
    Returns (distances, indices) both of shape (N, k) with 0-based indices.
    """
    n, d = data.shape
    # Normalise so inner product == cosine similarity
    data_norm = unit_normalise(data.copy())

    index = faiss.IndexFlatIP(d)
    index.add(data_norm)

    print(f"  Searching {n} queries (k={k}) against {index.ntotal} vectors …")
    t0 = time.time()
    D, I = index.search(data_norm, k)   # includes self (distance ≈ 1.0)
    print(f"  Exact search finished in {time.time() - t0:.2f}s")
    return D, I


def write_hdf5(dst: str, data: np.ndarray, D: np.ndarray, I: np.ndarray,
               k_store: int) -> None:
    """
    Save vectors and ground truth to an HDF5 file that mirrors the structure
    of the full GOOAQ benchmark files.

    Ground-truth indices are converted to **1-based** to match the convention
    used by the SISAP evaluation code (knns[i, j] == 1 means the nearest
    neighbour of vector i is vector 0+1=1).
    """
    n, d = data.shape
    os.makedirs(Path(dst).parent, exist_ok=True)
    with h5py.File(dst, "w") as f:
        # Vectors
        f.create_dataset("train", data=data, dtype=np.float32,
                         compression="gzip", compression_opts=4)
        # Ground truth (k_store neighbors per vector, 1-indexed)
        gt = f.create_group("allknn")
        gt.create_dataset("knns",  data=(I[:, :k_store] + 1).astype(np.int32),
                          compression="gzip", compression_opts=4)
        gt.create_dataset("dists", data=D[:, :k_store].astype(np.float32),
                          compression="gzip", compression_opts=4)
        # Metadata attributes
        f.attrs["algo"]    = "exhaustive"
        f.attrs["n"]       = n
        f.attrs["d"]       = d
        f.attrs["k_store"] = k_store
    print(f"  Written to {dst}  (train: {data.shape}, knns: {I[:, :k_store].shape})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Create a small GOOAQ-format all-kNN benchmark file."
    )
    ap.add_argument("-n", "--n-vectors", type=int, default=10_000,
                    help="Number of vectors (default: 10000)")
    ap.add_argument("-k", "--k-store", type=int, default=32,
                    help="Neighbors stored per vector, including self "
                         "(default: 32; must be >= the k used at evaluation)")
    ap.add_argument("--src", default="data/benchmark-dev-gooaq.h5",
                    help="Path to the full GOOAQ HDF5 file "
                         "(default: data/benchmark-dev-gooaq.h5)")
    ap.add_argument("--dst", default="data/benchmark-dev-gooaq-small.h5",
                    help="Output path (default: data/benchmark-dev-gooaq-small.h5)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for synthetic fallback (default: 42)")
    args = ap.parse_args()

    d = 384         # GOOAQ / sentence-BERT embedding dimension
    n = args.n_vectors
    k_store = args.k_store

    # ------------------------------------------------------------------
    # 1.  Load or generate vectors
    # ------------------------------------------------------------------
    src_ok = os.path.isfile(args.src) and os.path.getsize(args.src) > 0
    if src_ok:
        print(f"[1/3] Loading first {n} vectors from {args.src} …")
        data = load_real_vectors(args.src, n)
        source_tag = "real"
    else:
        if not src_ok:
            print(f"[1/3] {args.src} not found or empty – generating "
                  f"{n} synthetic unit-normalised vectors (d={d}, seed={args.seed}) …")
        data = make_synthetic_vectors(n, d, args.seed)
        source_tag = "synthetic"
    print(f"  Loaded {data.shape[0]} × {data.shape[1]} float32 vectors "
          f"({source_tag})")

    # ------------------------------------------------------------------
    # 2.  Compute exact all-kNN ground truth
    # ------------------------------------------------------------------
    # k_store includes the self-loop (distance ≈ 1 for unit vectors)
    print(f"[2/3] Computing exact all-kNN (k={k_store}, inner product) …")
    D, I = compute_exact_allknn(data, k=k_store)

    # Sanity check: first neighbour of each vector should be itself (0-based)
    n_self = np.sum(I[:, 0] == np.arange(n))
    print(f"  Self-loop sanity: {n_self}/{n} vectors have themselves as "
          f"nearest neighbour ({100*n_self/n:.1f}%)")

    # ------------------------------------------------------------------
    # 3.  Write HDF5
    # ------------------------------------------------------------------
    print(f"[3/3] Writing output to {args.dst} …")
    write_hdf5(args.dst, data, D, I, k_store)

    print("\nDone.  Summary:")
    print(f"  Source    : {source_tag}")
    print(f"  Vectors   : {n} × {d}  (float32)")
    print(f"  GT k      : {k_store}  (1-indexed, self-loop at position 0)")
    print(f"  Output    : {args.dst}")
    print()
    print("To use with the python baseline, add 'gooaq-small' to datasets.py")
    print("pointing to this file, or run eval.py directly after updating the")
    print("dataset registry.")


if __name__ == "__main__":
    main()
