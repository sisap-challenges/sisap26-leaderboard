# SISAP25 Container Setup for HPC Deployment

This guide covers the complete workflow for setting up and running SISAP25 hforest containers on HPC clusters using Apptainer.

## Overview

The sisap25-hforest implementation uses Docker containers for reproducible evaluation. Since HPC clusters typically use Apptainer (formerly Singularity) instead of Docker, we need to:

1. **Build Docker container locally** (with ARM64/x86_64 compatibility fixes)
2. **Convert to Apptainer format** (.sif file)
3. **Transfer to HPC cluster**
4. **Create SLURM job scripts**
5. **Run evaluation with proper resource constraints**

## Prerequisites

### Local Machine (Development)
- Docker installed and running
- Access to sisap25-hforest source code
- ~10GB free disk space for container images

### HPC Cluster
- Apptainer runtime available
- SLURM job scheduler
- Access to datasets (use data setup scripts)
- ~5GB free space for .sif container file

## Step 1: Build Docker Container Locally

### Navigate to Source Directory
```bash
cd sisap25-playground/sisap25-hforest
```

### Architecture Compatibility Fix

The original code has Intel-specific intrinsics that don't work on ARM64 (Apple Silicon). We've already applied this fix:

**File: `src/hforest.cpp`** (lines 26-30):
```cpp
// Third-party library headers
#ifdef __x86_64__
#include <immintrin.h>
#endif
#include <omp.h>
```

### Build Options

**Option A: Cross-compile for x86_64 (Recommended)**
```bash
# Build for Linux x86_64 (typical HPC architecture)
docker build --platform linux/amd64 -t sisap25/hforest .
```

**Option B: Native build** (if using x86_64 machine or after applying ARM64 fix):
```bash
docker build -t sisap25/hforest .
```

### Verify Build Success
```bash
# Check if image was created
docker images | grep sisap25/hforest

# Test basic functionality
docker run --rm sisap25/hforest python -c "import hforest; print('✓ Container works')"
```

## Step 2: Export and Transfer Container

### Export Docker Image
```bash
# Export to tar archive
docker save sisap25/hforest -o hforest.tar

# Verify export
ls -lh hforest.tar
```

### Transfer to HPC Cluster
```bash
# Transfer via SCP
scp hforest.tar username@hpc-cluster:/path/to/containers/

# Or use rsync for resume capability
rsync -avz --progress hforest.tar username@hpc-cluster:/path/to/containers/
```

## Step 3: Convert to Apptainer on HPC

### Log into HPC Cluster
```bash
ssh username@hpc-cluster
cd /path/to/containers/
```

### Convert Docker Archive to Apptainer SIF
```bash
# Convert to .sif format
apptainer build hforest.sif docker-archive://hforest.tar

# Verify conversion
ls -lh hforest.sif
apptainer exec hforest.sif python --version
```

### Test Container
```bash
# Basic test
# Note: hforest.so is built in-place under /app inside the container.
# Use --pwd /app so Python can find it, or use `apptainer run` which
# inherits the Dockerfile WORKDIR automatically.
apptainer exec --pwd /app hforest.sif python -c "
import sys
print(f'Python version: {sys.version}')
try:
    import hforest
    print('✓ hforest module imported successfully')
except ImportError as e:
    print(f'✗ hforest import failed: {e}')
"
```

## Step 4: Create SLURM Job Scripts

### Task 1 Script (`hforest-task1.slurm`)

```bash
#!/bin/bash

#SBATCH --job-name=sisap25-hforest-task1
#SBATCH --output=hforest-task1-%j.out
#SBATCH --error=hforest-task1-%j.err
#SBATCH --cpus-per-task=8                    # SISAP25 constraint: 8 CPUs
#SBATCH --mem=16G                           # SISAP25 constraint: 16GB RAM
#SBATCH --time=12:00:00                     # SISAP25 constraint: 12-hour limit
#SBATCH --partition=compute                 # Adjust to your cluster partitions

# Environment setup
echo "=== SISAP25 Task 1 Evaluation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo

# Paths (UPDATE THESE FOR YOUR HPC ENVIRONMENT)
CONTAINER_PATH="/path/to/containers/hforest.sif"
DATASET_PATH="/path/to/sisap25/datasets"           # Use data setup scripts
OUTPUT_PATH="/path/to/results/task1-${SLURM_JOB_ID}"

# Create output directory
mkdir -p $OUTPUT_PATH

# Set OpenMP threads to match allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Container: $CONTAINER_PATH"
echo "Dataset: $DATASET_PATH"  
echo "Output: $OUTPUT_PATH"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo

# Run Task 1 evaluation
echo "Starting Task 1 evaluation..."
apptainer exec --pwd /app \
    -B $DATASET_PATH:/app/data:ro \
    -B $OUTPUT_PATH:/app/results:rw \
    $CONTAINER_PATH python sisap2025.py task1

echo "Task 1 completed at $(date)"
echo "Results saved to: $OUTPUT_PATH"

# Show result summary
if [[ -f "$OUTPUT_PATH/task1_results.json" ]]; then
    echo "=== Results Summary ==="
    cat "$OUTPUT_PATH/task1_results.json"
fi
```

### Task 2 Script (`hforest-task2.slurm`)

```bash
#!/bin/bash

#SBATCH --job-name=sisap25-hforest-task2
#SBATCH --output=hforest-task2-%j.out
#SBATCH --error=hforest-task2-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=compute

echo "=== SISAP25 Task 2 Evaluation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo

# Paths (UPDATE THESE)
CONTAINER_PATH="/path/to/containers/hforest.sif"
DATASET_PATH="/path/to/sisap25/datasets"
OUTPUT_PATH="/path/to/results/task2-${SLURM_JOB_ID}"

mkdir -p $OUTPUT_PATH
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting Task 2 evaluation (k-NN graph construction)..."
apptainer exec --pwd /app \
    -B $DATASET_PATH:/app/data:ro \
    -B $OUTPUT_PATH:/app/results:rw \
    $CONTAINER_PATH python sisap2025.py task2

echo "Task 2 completed at $(date)"
echo "Results saved to: $OUTPUT_PATH"
```

### Launcher Script (`run-hforest.sh`)

```bash
#!/bin/bash

# SISAP25 HPC Launcher Script
# Usage: ./run-hforest.sh [task1|task2]

TASK=$1
CONTAINER_PATH="/path/to/containers/hforest.sif"    # UPDATE THIS PATH

if [[ -z "$TASK" ]]; then
    echo "Usage: $0 [task1|task2]"
    echo
    echo "Examples:"
    echo "  $0 task1    # Submit Task 1 (PUBMED23 dataset)"
    echo "  $0 task2    # Submit Task 2 (GOOAQ k-NN graph)"
    echo
    echo "Monitor jobs: squeue -u \$USER"
    exit 1
fi

if [[ ! -f "$CONTAINER_PATH" ]]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    echo "Please build and convert the container first:"
    echo "  1. docker build --platform linux/amd64 -t sisap25/hforest ."
    echo "  2. docker save sisap25/hforest -o hforest.tar"
    echo "  3. scp hforest.tar user@hpc:/path/"
    echo "  4. apptainer build hforest.sif docker-archive://hforest.tar"
    exit 1
fi

case $TASK in
    "task1")
        echo "🚀 Submitting SISAP25 Task 1 (PUBMED23 search)..."
        JOB_ID=$(sbatch --parsable hforest-task1.slurm)
        echo "✅ Job submitted with ID: $JOB_ID"
        echo "📊 Expected: ~43 min runtime, 0.70+ recall, 637+ q/s"
        ;;
    "task2")
        echo "🚀 Submitting SISAP25 Task 2 (GOOAQ k-NN graph)..."
        JOB_ID=$(sbatch --parsable hforest-task2.slurm)
        echo "✅ Job submitted with ID: $JOB_ID"  
        echo "🏆 Expected: ~105 sec runtime, 0.80+ recall (Winner performance!)"
        ;;
    *)
        echo "Error: Unknown task '$TASK'. Use 'task1' or 'task2'."
        exit 1
        ;;
esac

echo
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  scontrol show job $JOB_ID"
echo "  tail -f hforest-${TASK}-${JOB_ID}.out"
```

## Step 5: Update Paths and Execute

### Update Script Paths
Edit the SLURM scripts and update these paths for your HPC environment:

```bash
# In hforest-task1.slurm and hforest-task2.slurm:
CONTAINER_PATH="/path/to/containers/hforest.sif"           # Your container location
DATASET_PATH="/path/to/sisap25/datasets"                   # From data setup scripts  
OUTPUT_PATH="/path/to/results/task${N}-${SLURM_JOB_ID}"   # Results directory

# In run-hforest.sh:
CONTAINER_PATH="/path/to/containers/hforest.sif"           # Your container location
```

### Make Scripts Executable
```bash
chmod +x hforest-task1.slurm hforest-task2.slurm run-hforest.sh
```

### Submit Jobs
```bash
# Submit Task 1
./run-hforest.sh task1

# Submit Task 2  
./run-hforest.sh task2

# Monitor progress
squeue -u $USER
watch -n 30 squeue -u $USER
```

## Expected Performance

Based on SISAP25 challenge results:

### Task 1 (PUBMED23 Search)
- **Runtime**: ~43 minutes (2,594s container time)
- **Memory**: 12.2-16.0 GB peak usage
- **Recall**: ~0.7053 (exceeds 0.7 threshold)  
- **Throughput**: ~637 queries/second
- **Rank**: 4th place

### Task 2 (GOOAQ k-NN Graph) 🏆
- **Runtime**: ~1.75 minutes (105s total)
- **Memory**: 7.1-7.4 GB peak usage
- **Recall**: ~0.8049 (exceeds 0.8 threshold)
- **Performance**: Winner of Task 2!

## Troubleshooting

### Container Issues
```bash
# Test container locally before transfer
docker run --rm sisap25/hforest python -c "import hforest; print('OK')"

# Test on HPC after conversion  
apptainer exec --pwd /app hforest.sif python -c "import hforest; print('OK')"

# Check container contents
apptainer exec hforest.sif ls -la /app/
```

### Build Issues
```bash
# For ARM64 compatibility (Apple Silicon)
docker build --platform linux/amd64 -t sisap25/hforest .

# Check available platforms
docker buildx ls
```

### Path Issues
```bash
# Check dataset availability on HPC
ls -la /path/to/sisap25/datasets/
apptainer exec hforest.sif ls -la /app/data/

# Test bind mounts
apptainer exec -B /host/path:/container/path hforest.sif ls /container/path
```

### SLURM Issues
```bash
# Check job status
squeue -u $USER
scontrol show job JOBID

# View job output
tail -f hforest-task1-JOBID.out
tail -f hforest-task1-JOBID.err

# Check resource usage
sacct -j JOBID --format=JobID,JobName,MaxRSS,Elapsed
```

## Resource Requirements Summary

| Component | Task 1 | Task 2 | Notes |
|-----------|--------|--------|-------|
| **CPUs** | 8 | 8 | SISAP25 constraint |
| **Memory** | 16GB | 16GB | SISAP25 constraint |  
| **Time Limit** | 12h | 12h | SISAP25 constraint |
| **Expected Runtime** | ~43 min | ~2 min | Actual hforest performance |
| **Dataset Size** | ~35GB | ~7.4GB | PUBMED23 vs GOOAQ |
| **Container Size** | ~5GB | ~5GB | Same container for both tasks |

This setup provides a complete, production-ready workflow for running SISAP25 evaluations on HPC clusters while maintaining the exact same evaluation environment as the original challenge.