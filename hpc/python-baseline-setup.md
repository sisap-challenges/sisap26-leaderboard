# Python Baseline – GOOAQ (SISAP 2025 Task 2) HPC Setup

This guide explains how to run the `sisap26-python-baseline` on the GOOAQ
dataset (the SISAP 2025 Task 2 benchmark) under HPC / SLURM conditions that
mirror the official challenge constraints (8 CPUs, 16 GB RAM, 12-hour limit).

The approach exactly parallels the hforest setup documented in
`container_setup.md`.

---

## Overview

1. Download the GOOAQ datasets on the local machine (or directly on the HPC).
2. Build the Docker container locally and export it as a `.tar` archive.
3. Transfer the archive to the HPC cluster and convert it to an Apptainer
   `.sif` image.
4. Update path variables in the SLURM scripts and submit the job.

---

## Prerequisites

### Local machine
- Docker installed and running
- ~5 GB free disk space for the container image

### HPC cluster
- Apptainer (formerly Singularity) available
- SLURM job scheduler
- ~5 GB free space for the `.sif` file
- The two GOOAQ HDF5 files accessible (see step 1)

---

## Step 1 – Download the GOOAQ Data

The datasets can be fetched with the provided helper script:

```bash
# From the repository root
python setup-sisap25-task2-data.py --data-dir /path/to/sisap25/datasets
```

This downloads:
- `benchmark-dev-gooaq.h5`   – 3,012,496 vectors, 384 dimensions (~4.8 GB)
- `allknn-benchmark-dev-gooaq.h5` – ground-truth k=15 graph (~768 MB)

The SLURM script expects **both files** in the same directory and bind-mounts
that directory to `/app/data/gooaq/task1` inside the container.

---

## Step 2 – Build the Docker Container

```bash
cd sisap25-playground/sisap26-python-baseline

# Cross-compile for Linux x86_64 (standard HPC architecture)
docker build --platform linux/amd64 -t sisap25/python-baseline .

# Quick smoke test
docker run --rm sisap25/python-baseline python -c "import faiss; print('OK')"
```

---

## Step 3 – Export and Transfer

```bash
# Export to a tar archive
docker save sisap25/python-baseline -o python-baseline.tar

# Transfer to HPC (rsync supports resume if interrupted)
rsync -avz --progress python-baseline.tar \
    username@hpc-cluster:/path/to/containers/
```

---

## Step 4 – Convert to Apptainer on HPC

```bash
ssh username@hpc-cluster
cd /path/to/containers/

apptainer build python-baseline.sif docker-archive://python-baseline.tar

# Verify
apptainer exec python-baseline.sif python -c "import faiss; print('OK')"
```

---

## Step 5 – Update Paths in the SLURM Scripts

Edit `hpc/python-baseline-gooaq.slurm` and set:

```bash
CONTAINER_PATH="/path/to/containers/python-baseline.sif"
DATASET_PATH="/path/to/sisap25/datasets"   # directory with both .h5 files
OUTPUT_PATH="/path/to/results/python-baseline-gooaq-${SLURM_JOB_ID}"
```

Do the same for the three `# UPDATE` lines in `hpc/run-python-baseline.sh`.

---

## Step 6 – Submit the Job

```bash
cd hpc/
chmod +x run-python-baseline.sh

# Dry-run: prints the sbatch command without submitting
./run-python-baseline.sh

# Submit
./run-python-baseline.sh gooaq-task1
```

Or submit directly:

```bash
sbatch python-baseline-gooaq.slurm
```

Monitor progress:

```bash
squeue -u $USER
tail -f python-baseline-gooaq-<JOBID>.out
```

---

## What the Job Does

The container runs:

```
python search.py --dataset gooaq --task task1
```

This executes `run_task1` in `search.py`:

1. **Downloads** the dataset files if not already present (skipped when
   bind-mounted read-only).
2. **Loads** the full 3 M × 384 matrix, transposing from the on-disk
   (384, N) layout to (N, 384) for FAISS.
3. **Builds** a FAISS `IVF1024,SQfp16` index (inner product metric).
4. **Searches** the index against all vectors (all-kNN graph) for each
   `nprobe` in `[1, 2, 5, 10, 100]`, producing five operating points.
5. **Writes** results to `results/gooaq/task1/<identifier>.h5` with
   attributes `algo`, `dataset`, `task`, `buildtime`, `querytime`, `params`
   and datasets `knns` (1-indexed) and `dists`.

---

## Evaluating Results

After the job finishes, run `eval.py` to compute recall against the ground
truth:

```bash
# Inside the container or in a local venv with the same requirements
python eval.py results.csv

# Inspect the CSV
column -t -s, results.csv
```

The ground-truth file (`allknn-benchmark-dev-gooaq.h5`) is fetched
automatically by `eval.py` if it is not already present under
`data/gooaq/task1/allknn-gooaq.h5`.

---

## Resource Requirements

| Resource  | Limit   | Notes                              |
|-----------|---------|------------------------------------|
| CPUs      | 8       | SISAP 2025 challenge constraint    |
| Memory    | 16 GB   | SISAP 2025 challenge constraint    |
| Time      | 12 h    | SISAP 2025 challenge constraint    |
| Disk (in) | ~5.6 GB | Both .h5 dataset files             |
| Disk (out)| ~1 GB   | 5 result files (one per nprobe)    |

The FAISS index itself fits comfortably within 16 GB for the default
`IVF1024,SQfp16` configuration.

---

## File Map

| File | Purpose |
|------|---------|
| `sisap25-playground/sisap26-python-baseline/datasets.py` | Dataset registry; gooaq entry with transpose logic and separate GT URL |
| `sisap25-playground/sisap26-python-baseline/search.py` | Main runner; `--dataset gooaq --task task1` triggers all-kNN search |
| `sisap25-playground/sisap26-python-baseline/eval.py` | Recall evaluation; supports separate GT files via `get_gt_fn()` |
| `sisap25-playground/sisap26-python-baseline/Dockerfile` | Container definition |
| `hpc/python-baseline-gooaq.slurm` | SLURM job script (8 CPUs / 16 G / 12 h) |
| `hpc/run-python-baseline.sh` | Convenience launcher; wraps `sbatch` |
| `hpc/python-baseline-setup.md` | This document |

---

## Troubleshooting

```bash
# Verify container on HPC
apptainer exec python-baseline.sif python -c "import faiss; print(faiss.__version__)"

# Check bind-mount paths
apptainer exec \
    -B /path/to/datasets:/app/data/gooaq/task1:ro \
    python-baseline.sif ls /app/data/gooaq/task1/

# Inspect a finished job
sacct -j <JOBID> --format=JobID,JobName,MaxRSS,Elapsed,State
```
