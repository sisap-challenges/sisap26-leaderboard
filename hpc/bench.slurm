#!/bin/bash
# Parametric benchmark job for hforest across nodes and thread counts.
#
# Do NOT submit this file directly. Use submit-bench.sh, which overrides
# --nodelist, --cpus-per-task, --mem, and --output on the sbatch command line
# for each (node, cpu_count) combination.
#
# Default values below act as a fallback / documentation only.

#SBATCH --job-name=hforest-bench
#SBATCH --output=bench-logs/bench-%j.out
#SBATCH --error=bench-logs/bench-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=scavenge
#SBATCH --exclusive          # one job per node so results are not contaminated

# ── Paths (update for your HPC environment) ─────────────────────────────────
CONTAINER_PATH="/path/to/containers/hforest.sif"
DATASET_PATH="/path/to/sisap25/datasets"
OUTPUT_BASE="/path/to/results/bench"
# ─────────────────────────────────────────────────────────────────────────────

NODE=$(hostname -s)
NCPUS=$SLURM_CPUS_PER_TASK
OUTPUT_PATH="${OUTPUT_BASE}/${NODE}-${NCPUS}cpu-${SLURM_JOB_ID}"

mkdir -p "$OUTPUT_PATH"

export OMP_NUM_THREADS=$NCPUS

echo "=== hforest benchmark ==="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $NODE"
echo "CPUs     : $NCPUS"
echo "Threads  : $OMP_NUM_THREADS"
echo "Started  : $(date)"
echo "Output   : $OUTPUT_PATH"
echo

# Record hardware context alongside the results
lscpu > "$OUTPUT_PATH/lscpu.txt"
numactl --hardware > "$OUTPUT_PATH/numa.txt" 2>/dev/null || true

echo "--- Task 2 (GOOAQ k-NN graph) ---"
apptainer exec --pwd /app \
    -B "$DATASET_PATH":/app/data:ro \
    -B "$OUTPUT_PATH":/app/results:rw \
    "$CONTAINER_PATH" python sisap2025.py task2

echo
echo "Completed: $(date)"
echo "Results saved to: $OUTPUT_PATH"
