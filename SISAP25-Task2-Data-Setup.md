# SISAP25 Challenge Task 2 Data Setup

This repository provides automated scripts to set up the data directory for **Task 2** of the SISAP25 Indexing Challenge.

## Task 2 Overview

**Objective**: Construct k=15 nearest neighbor graph for the GOOAQ dataset

**Specifications**:
- **Dataset**: GOOAQ (Google Questions & Answers)
- **Size**: 3,012,496 vectors (3M)
- **Dimensions**: 384 (sentence-BERT embeddings)
- **Similarity**: Cosine similarity / dot product
- **k-value**: 15 nearest neighbors per point
- **Target recall**: ≥ 0.8 for ranking
- **Resources**: 8 CPUs, 16GB RAM, 12-hour time limit

## Quick Start

### Option 1: Python Script (Recommended)

```bash
# Install dependencies
pip install requests h5py tqdm

# Basic setup - downloads required files (~5.6GB)
python setup-sisap25-task2-data.py

# Include evaluation dataset (additional 7.8GB)
python setup-sisap25-task2-data.py --include-eval

# Custom data directory
python setup-sisap25-task2-data.py --data-dir /path/to/sisap25/data

# Only verify existing files
python setup-sisap25-task2-data.py --verify-only
```

### Option 2: Bash Script

```bash
# Basic setup - downloads required files (~5.6GB)
./setup-sisap25-task2-data.sh

# Custom data directory
./setup-sisap25-task2-data.sh /path/to/data

# Include evaluation dataset
./setup-sisap25-task2-data.sh data true
```

## Downloaded Files

| File | Size | Description |
|------|------|-------------|
| `allknn-benchmark-dev-gooaq.h5` | 768MB | Task 2 gold standard (k=15 nearest neighbors) |
| `benchmark-dev-gooaq.h5` | 4.8GB | Full development dataset with train/test splits |
| `benchmark-eval-gooaq.h5` | 7.9GB | Final evaluation dataset (optional) |

## Data Structure

### Main Dataset (`benchmark-dev-gooaq.h5`)
```python
import h5py
with h5py.File('data/benchmark-dev-gooaq.h5', 'r') as f:
    train_data = f['train'][:]      # Shape: (384, 3012496) - Main dataset
    
    # Test sets
    itest_queries = f['itest/queries'][:]  # In-distribution test queries
    otest_queries = f['otest/queries'][:]  # Out-of-distribution test queries
    
    # Gold standard for test sets
    itest_knns = f['itest/knns'][:]    # Gold k-NN for in-distribution
    otest_knns = f['otest/knns'][:]    # Gold k-NN for out-of-distribution
```

### Task 2 Gold Standard (`allknn-benchmark-dev-gooaq.h5`)
```python
import h5py
with h5py.File('data/allknn-benchmark-dev-gooaq.h5', 'r') as f:
    gold_knns = f['knns'][:]    # Shape: (15, 3012496) - k=15 nearest neighbors
    gold_dists = f['dists'][:]  # Shape: (15, 3012496) - Corresponding distances
    
    # Note: Contains self-references that are ignored during evaluation
```

## Usage Example

```python
import h5py
import numpy as np

# Load the main dataset
with h5py.File('data/benchmark-dev-gooaq.h5', 'r') as f:
    train_data = f['train'][:]  # Shape: (384, 3012496)
    print(f"Dataset shape: {train_data.shape}")
    print(f"Number of vectors: {train_data.shape[1]:,}")
    print(f"Dimensions: {train_data.shape[0]}")

# Load Task 2 gold standard for evaluation
with h5py.File('data/allknn-benchmark-dev-gooaq.h5', 'r') as f:
    gold_knns = f['knns'][:]
    gold_dists = f['dists'][:]
    print(f"Gold standard k-NN shape: {gold_knns.shape}")
    print(f"k-value: {gold_knns.shape[0]}")

# Your k-NN graph construction algorithm goes here
# ...
# result_knns = your_algorithm(train_data, k=15)

# Evaluate against gold standard
# recall = evaluate_recall(result_knns, gold_knns)
```

## Integration with HPC Setup

After running the data setup script, use the HPC deployment approach:

```bash
# 1. Set up data directory
python setup-sisap25-task2-data.py --data-dir /path/to/hpc/sisap25/data

# 2. Update SLURM scripts to point to data directory
# In hforest-task2.slurm:
DATASET_PATH="/path/to/hpc/sisap25/data"

# 3. Run on HPC
sbatch hforest-task2.slurm
```

## Requirements

### Python Script
- Python 3.6+
- `requests`, `h5py`, `tqdm` packages

### Bash Script  
- `wget` or `curl`
- `bash` (compatible with older versions)

### System Requirements
- ~6GB free disk space (12GB with evaluation dataset)
- Internet connection for downloading

## Troubleshooting

### Download Issues
```bash
# If downloads fail, try manual download:
mkdir -p data
cd data
wget https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/allknn-benchmark-dev-gooaq.h5
wget https://huggingface.co/datasets/sadit/SISAP2025/resolve/main/benchmark-dev-gooaq.h5
```

### Verification Issues  
```bash
# Check file integrity
python setup-sisap25-task2-data.py --verify-only

# Or manually with h5py
python -c "import h5py; print(list(h5py.File('data/benchmark-dev-gooaq.h5', 'r').keys()))"
```

### HDF5 Issues
```bash
# Install h5py if missing
pip install h5py

# For system package managers:
# Ubuntu/Debian: apt-get install python3-h5py
# CentOS/RHEL: yum install python3-h5py
# macOS: brew install hdf5 && pip install h5py
```

## Expected Performance (hforest baseline)

Based on SISAP25 challenge results, the hforest algorithm achieved:

- **Rank**: 1st place (Winner) 🏆
- **Recall**: 0.8049 
- **Runtime**: 105 seconds total
- **Memory**: 7.1-7.4 GB peak usage
- **Container time**: 105 seconds

Your implementation should target similar or better performance!

## References

- [SISAP25 Challenge Overview](https://sisap-challenges.github.io/2025/)
- [GOOAQ Dataset](https://huggingface.co/datasets/sentence-transformers/gooaq/)
- [HuggingFace Dataset Repository](https://huggingface.co/datasets/sadit/SISAP2025)