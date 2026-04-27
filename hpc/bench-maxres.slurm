#!/bin/bash
# bench-maxres.slurm — full-resource hforest benchmark.
#
# Counterpart to bench.slurm (competition-spec, fixed 8 CPUs/16G).
# This script uses ALL allocated cores and runs every task2 quality
# level to produce a recall/speed tradeoff curve under full-node conditions.
#
# Do NOT submit directly. Use submit-bench-maxres.sh.

#SBATCH --job-name=hforest-maxres
#SBATCH --output=bench-logs/maxres-%j.out
#SBATCH --error=bench-logs/maxres-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=scavenge

# ── Paths ────────────────────────────────────────────────────────────────────
CONTAINER_PATH="/home/maau/sisap26-baseline-dev/hforest.sif"
DATASET_PATH="/home/maau/sisap26-baseline-dev/data/"
OUTPUT_BASE="/home/maau/sisap26-baseline-dev/bench-maxres"
# ─────────────────────────────────────────────────────────────────────────────

NODE=$(hostname -s)
NCPUS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$NCPUS
OUTPUT_PATH="${OUTPUT_BASE}/${NODE}-${NCPUS}cpu-${SLURM_JOB_ID}"

mkdir -p "$OUTPUT_PATH"

echo "=== hforest max-resource benchmark ==="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $NODE"
echo "Alloc CPUs : $NCPUS"
echo "OMP threads: $OMP_NUM_THREADS"
echo "Started    : $(date)"
echo "Output     : $OUTPUT_PATH"
echo

lscpu > "$OUTPUT_PATH/lscpu.txt"
numactl --hardware > "$OUTPUT_PATH/numa.txt" 2>/dev/null || true

# Run all five quality levels in sequence, one python invocation each.
# Results land in separate h5 files named by their hyperparameters.
# Recall/speed tradeoff: task2 (~81%) → task2:98 (~98%).
echo "--- Task 2 — all quality levels ---"
for TASK in task2 task2:85 task2:90 task2:95 task2:98; do
    echo "  Running $TASK at $(date)"
    apptainer exec --pwd /app \
        -B "$DATASET_PATH":/app/data:ro \
        -B "$OUTPUT_PATH":/app/results:rw \
        "$CONTAINER_PATH" python sisap2025.py "$TASK"
done

echo
echo "Completed: $(date)"
echo "Results saved to: $OUTPUT_PATH"
