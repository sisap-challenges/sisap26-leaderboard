# SISAP 2026 – Task 1 Format Reference

This document describes the **input dataset format**, how the **kNN-graph
baseline implementation** works, the **output result format**, and how
**recall is evaluated**.  A 10 000-vector example file is supplied so every
section can be explored without downloading the full 5 GB dataset.

---

## Quick start

```bash
pip install h5py numpy

# Inspect the small example dataset
python show_small_dataset.py

# Validate the example dataset (input format checker)
python verify_input.py ../../data/benchmark-dev-gooaq-small.h5 \
       --dataset gooaq-small --task task1

# Validate a result file (output format checker) + compute recall
python verify_output.py results/gooaq-small/task1/myresult.h5 \
       --dataset gooaq-small --compute-recall
```

---

## 1. Input dataset format

### File supplied: `data/benchmark-dev-gooaq-small.h5`

A small (15 MB, 10 000-vector) slice of the real GOOAQ dataset with an
exhaustively computed ground-truth kNN graph.  It is a self-contained HDF5
file with the following layout:

```
benchmark-dev-gooaq-small.h5
├── train          float32  (10000, 384)   – the vector corpus
├── allknn/
│   ├── knns       int32    (10000, 32)    – ground-truth neighbor IDs
│   └── dists      float32  (10000, 32)   – corresponding dot-product scores
└── attrs: algo='exhaustive', n=10000, d=384, k_store=32
```

### Key properties

| Property | Value |
|---|---|
| N (corpus size) | 10 000 (small) / 3 001 496 (full) |
| d (dimensions) | 384 |
| k stored in GT | 32 |
| Similarity metric | dot product (vectors are L2-normalised → dot product = cosine) |
| ID scheme | **1-based** — the first vector has ID 1 |
| Self-loop | `knns[i, 0] == i + 1` for every row (each point is its own nearest neighbour) |
| GT order | Neighbours sorted by descending dot-product score |

### Concrete example (row 0)

```
train[0, :5]  = [ 0.00648, -0.06333, -0.00724, -0.05695, -0.03529, … ]

knns[0, :]    = [1, 7997, 6227, 5979, 9841, 8076, 5337, 8956, 786, …]
                 ^--- self (1-based ID of row 0)

dists[0, :]   = [1.000, 0.464, 0.417, 0.411, 0.410, 0.410, 0.407, …]
                 ^--- dot product with itself = 1.0 (normalised)
```

### Full GOOAQ dataset

The full dataset is split into two files on HuggingFace
(`sadit/SISAP2025`):

| File | Size | Contents |
|---|---|---|
| `benchmark-dev-gooaq.h5` | ~4.8 GB | `train (3001496, 384) float32` |
| `allknn-benchmark-dev-gooaq.h5` | ~733 MB | `knns (3001496, 32) int32`, `dists (3001496, 32) float32` at root level |

---

## 2. How the baseline solves the task

The task is to build an **approximate k-nearest-neighbor graph** (k=15) over
all N vectors, using the **dot product** as the similarity measure.

The baseline (`search.py`, function `run_task1`) does the following:

### Step 1 – Load vectors

```python
data = np.array(f['train'], dtype=np.float32)   # shape (N, 384)
```

### Step 2 – Build a FAISS IVF index

```python
index = faiss.index_factory(384, "IVF1024,SQfp16",
                             faiss.METRIC_INNER_PRODUCT)
index.train(data)
index.add(data)        # adds all N vectors; FAISS assigns 0-based IDs
```

The index type `IVF1024,SQfp16` partitions the space into 1 024 Voronoi
cells and stores vectors in 16-bit scalar-quantised form — a good
memory/speed trade-off for 3 M × 384 float32 vectors (~4.3 GB uncompressed
→ ~2.2 GB in the index).

### Step 3 – Search (all vectors as queries)

```python
k_search = 15 + 1  # query for one extra to guarantee k non-self neighbours
index.nprobe = nprobe
D, I = index.search(data, k_search)  # D, I shape: (N, 16), IDs are 0-based

# Remove self-matches (I[i] == i) and FAISS sentinels (I[i] < 0),
# then keep exactly k=15 neighbours per row (pad if necessary).
I, D = remove_self_references(I, D, k=15)  # shape: (N, 15)

I = I + 1  # convert 0-based FAISS IDs to 1-based
```

This is repeated for `nprobe ∈ {1, 2, 5, 10, 100}`, producing five
operating points (higher nprobe = higher recall, slower search).

### Step 4 – Write results

```python
store_results("results/gooaq/task1/<identifier>.h5", ...)
```

---

## 3. Output result format

Each run produces one HDF5 file under `results/<dataset>/<task>/`.

### Layout

```
<identifier>.h5
├── knns    int32    (N, k)    – 1-based neighbor IDs, row-major
├── dists   float32  (N, k)   – dot-product scores, row-major
└── attrs:
      algo        = "faissIVF"
      dataset     = "gooaq"
      task        = "task1"
      buildtime   = <seconds>   (float)
      querytime   = <seconds>   (float)
      params      = "index=(IVF1024,SQfp16),query=(nprobe=10)"
```

### Required attributes (SISAP 2026 spec)

| Attribute | Type | Description |
|---|---|---|
| `algo` | str | Algorithm name |
| `task` | str | `task1`, `task2`, or `task3` |
| `buildtime` | float | Index construction time (seconds) |
| `querytime` | float | Total search time (seconds) |
| `params` | str | Human-readable parameter string |

### Notes

- `k = 15` for Task 1 (the output contains exactly 15 neighbours per query).
- IDs are **1-based** — consistent with the ground truth.
- The self-reference **must not be present** in the output (spec requirement:
  "k-nearest neighbor graph without self-references"). `search.py` strips it
  before writing results.

---

## 4. Recall computation

Recall@k measures the fraction of true nearest neighbours that appear in
the result, averaged over all queries.

### Definition

```
recall@k = (1/N) * Σ_i  |result_i ∩ gt_i| / k
```

where:
- `result_i` = set of IDs returned for query i (first k columns of `knns`), **no self-reference**
- `gt_i` = set of first k IDs from the ground-truth row (includes self at col 0; the
  self ID is never in `result_i` so it contributes 0 to the intersection)

### Task 1 operating point

The SISAP 2026 challenge requires **recall@15 ≥ 0.8**.

### Code (from `eval.py`)

```python
def get_recall(I, gt, k):
    n = len(I)
    recall = 0
    for i in range(n):
        recall += len(set(I[i, :k]) & set(gt[i, :k]))
    return recall / (n * k)
```

`I` is the result matrix (self-reference-free), `gt` is the ground-truth
matrix loaded directly from the HDF5 file (self-loop present in col 0).
Because the result never contains the self ID, the intersection never picks
it up — no explicit stripping needed in the eval loop.

### Running evaluation

```bash
# Evaluate all result files under results/ and write a CSV
python eval.py results.csv

# Validate + compute recall for one file
python verify_output.py results/gooaq/task1/myresult.h5 \
       --dataset gooaq --compute-recall
```

---

## 5. File map

| File | Purpose |
|---|---|
| `data/benchmark-dev-gooaq-small.h5` | 15 MB example dataset (10k real vectors + exhaustive GT) |
| `datasets.py` | Dataset registry: paths, GT lambdas, k values |
| `search.py` | Baseline runner: builds FAISS index, searches, writes results |
| `eval.py` | Batch recall evaluation → CSV |
| `verify_input.py` | Validates an input HDF5 (format, shapes, indexing) |
| `verify_output.py` | Validates a result HDF5 (format + optional recall check) |
| `show_small_dataset.py` | Prints a human-readable summary of the small example file |
