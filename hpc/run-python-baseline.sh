#!/bin/bash

# SISAP25 Python Baseline – HPC Launcher
# Usage: ./run-python-baseline.sh [task]
#
# Supported tasks:
#   gooaq-task1   – all-kNN graph on GOOAQ (SISAP25 Task 2, k=15, 3 M vectors)
#
# Prerequisites:
#   1. Build the Docker image and convert it to an Apptainer SIF (see
#      python-baseline-setup.md for detailed instructions).
#   2. Download the GOOAQ datasets with setup-sisap25-task2-data.py.
#   3. Update CONTAINER_PATH, DATASET_PATH, and OUTPUT_BASE below.

# ---- Configure these paths for your HPC environment --------------------
CONTAINER_PATH="/home/maau/sisap26-baseline-dev/python-sisap26-baseline.sif"  
DATASET_PATH="/home/maau/sisap26-baseline-dev/data"                 
OUTPUT_BASE="/home/maau/sisap26-baseline-dev/results"                         
# ------------------------------------------------------------------------

TASK=$1

if [[ -z "$TASK" ]]; then
    echo "Usage: $0 <task>"
    echo
    echo "Available tasks:"
    echo "  gooaq-task1   SISAP25 Task 2 – all-kNN graph on GOOAQ (k=15)"
    echo
    echo "Monitor jobs: squeue -u \$USER"
    exit 1
fi

if [[ ! -f "$CONTAINER_PATH" ]]; then
    echo "Error: container not found at $CONTAINER_PATH"
    echo "Build and convert it first (see hpc/python-baseline-setup.md):"
    echo "  1. cd sisap25-playground/sisap26-python-baseline"
    echo "  2. docker build --platform linux/amd64 -t sisap25/python-baseline ."
    echo "  3. docker save sisap25/python-baseline -o python-baseline.tar"
    echo "  4. scp python-baseline.tar user@hpc:/path/to/containers/"
    echo "  5. apptainer build python-baseline.sif docker-archive://python-baseline.tar"
    exit 1
fi

case $TASK in
    "gooaq-task1")
        echo "Submitting SISAP25 Python Baseline – GOOAQ all-kNN graph..."
        # Export paths so the SLURM script can pick them up if needed;
        # the script also defines sensible defaults.
        export CONTAINER_PATH DATASET_PATH OUTPUT_BASE
        JOB_ID=$(sbatch --parsable python-baseline-gooaq.slurm)
        echo "Job submitted with ID: $JOB_ID"
        echo "Expected: FAISS IVF1024,SQfp16 index, multiple nprobe sweeps"
        ;;
    *)
        echo "Error: unknown task '$TASK'."
        echo "Run $0 without arguments to see available tasks."
        exit 1
        ;;
esac

echo
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  scontrol show job $JOB_ID"
echo "  tail -f python-baseline-gooaq-${JOB_ID}.out"
