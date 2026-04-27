#!/bin/bash

#SBATCH --job-name=sisap25-python-baseline-gooaq
#SBATCH --output=python-baseline-gooaq-%j.out
#SBATCH --error=python-baseline-gooaq-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --partition=scavenge

echo "=== SISAP25 Python Baseline – GOOAQ Task 2 (all-kNN graph) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo

# Paths (UPDATE THESE FOR YOUR HPC ENVIRONMENT)
CONTAINER_PATH="/home/maau/sisap26-baseline-dev/python-sisap26-baseline.sif"  
DATASET_PATH="/home/maau/sisap26-baseline-dev/data" 
OUTPUT_PATH="/home/maau/sisap26-baseline-dev/results/python-baseline-gooaq-${SLURM_JOB_ID}"

mkdir -p "$OUTPUT_PATH"

# Honour the SISAP25 CPU constraint
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Container : $CONTAINER_PATH"
echo "Dataset   : $DATASET_PATH"
echo "Output    : $OUTPUT_PATH"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo

# The container's ENTRYPOINT is `python search.py`.
# --dataset gooaq selects the GOOAQ dataset; --task task1 runs the all-kNN
# graph construction (equivalent to SISAP25 Task 2).
# Data and results are bind-mounted so the container writes nothing to the
# image layer and the output survives after the job finishes.
echo "Starting all-kNN graph construction on GOOAQ..."
apptainer exec --pwd /app \
    -B "$DATASET_PATH":/app/data/gooaq/task1:ro \
    -B "$OUTPUT_PATH":/app/results:rw \
    "$CONTAINER_PATH" python search.py --dataset gooaq --task task1

echo "Job finished at $(date)"
echo "Results saved to: $OUTPUT_PATH"
