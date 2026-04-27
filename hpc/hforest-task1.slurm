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
# sisap2025.py lives inside the container at /app/sisap2025.py (copied by the Dockerfile).
# Use `apptainer exec --pwd /app` rather than `apptainer run` because the container's
# CMD is `bash`; passing extra arguments to `apptainer run` would hand them to bash as
# a script path, not execute them as a command.
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