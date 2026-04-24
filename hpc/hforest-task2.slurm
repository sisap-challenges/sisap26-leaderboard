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
apptainer run \
    -B $DATASET_PATH:/app/data:ro \
    -B $OUTPUT_PATH:/app/results:rw \
    $CONTAINER_PATH python sisap2025.py task2

echo "Task 2 completed at $(date)"
echo "Results saved to: $OUTPUT_PATH"