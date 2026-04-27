#!/bin/bash
# submit-bench-maxres.sh — submit full-resource hforest benchmark jobs.
#
# Usage:
#   ./submit-bench-maxres.sh            # dry run
#   ./submit-bench-maxres.sh --submit   # submit to SLURM
#
# Each job uses ALL cores on the node and proportional RAM (capped at 90% of
# physical to leave headroom for the OS and Apptainer overhead).
# OMP_NUM_THREADS = all cores, so results are NOT comparable to the
# competition benchmark (bench.slurm / submit-bench.sh) — they are stored
# under bench-maxres/ to keep the two approaches fully isolated.
#
# Nodes selected from those currently free in the queue:
#   cn5  — 32c  / 512 GiB  / 100 Gbps IB  : high-memory IB node
#   cn9  — 192c / 250 GiB  / 10 Gbps SFP  : high core-count
#   cn10 — 48c  / 250 GiB  / 10 Gbps SFP  : new node, not in competition run
#   cn11 — 96c  / 1536 GiB / 10 Gbps SFP  : fastest in competition run
#   cn16 — 40c  / 256 GiB  / 1 Gbps Eth   : slow-network reference
#
# Time estimate: task2:98 takes ~611s at 8 threads; with N threads it scales
# roughly as 8/N, so e.g. cn11 at 96 threads ≈ ~51s. Worst case (cn5, 32c)
# ≈ ~153s per level × 5 levels ≈ 13 min. 2-hour limit is ample.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/bench-maxres.slurm"

SUBMIT=false
if [[ "${1:-}" == "--submit" ]]; then
    SUBMIT=true
fi

# ── Node matrix: total_cores  physical_ram_GiB ───────────────────────────────
declare -A NODE_CORES=( [cn5]=32  [cn9]=192 [cn10]=48 [cn11]=96  [cn16]=40  )
declare -A NODE_RAM=(   [cn5]=512 [cn9]=250 [cn10]=250 [cn11]=1536 [cn16]=256 )
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$SCRIPT_DIR/bench-logs"

echo "hforest max-resource benchmark submission"
echo "Script : $SLURM_SCRIPT"
echo "Mode   : $( $SUBMIT && echo 'SUBMIT' || echo 'DRY RUN (pass --submit to submit)' )"
echo

for NODE in cn5 cn9 cn10 cn11 cn16; do
    CORES=${NODE_CORES[$NODE]}
    RAM=${NODE_RAM[$NODE]}
    # 90% of physical RAM, rounded down to nearest GiB
    MEM=$(( RAM * 90 / 100 ))

    CMD=(
        sbatch
        --nodelist="$NODE"
        --partition=scavenge
        --cpus-per-task="$CORES"
        --mem="${MEM}G"
        --output="$SCRIPT_DIR/bench-logs/maxres-${NODE}-%j.out"
        --error="$SCRIPT_DIR/bench-logs/maxres-${NODE}-%j.err"
        "$SLURM_SCRIPT"
    )

    echo "  [${NODE}] ${CORES} CPUs, ${MEM}G RAM (90% of ${RAM} GiB)"
    echo "  ${CMD[*]}"
    if $SUBMIT; then
        JOB_ID=$("${CMD[@]}" --parsable)
        echo "    -> submitted job $JOB_ID"
    fi
    echo
done

echo
if ! $SUBMIT; then
    echo "Dry run complete. Re-run with --submit to submit all jobs."
else
    echo "All jobs submitted. Monitor with:"
    echo "  squeue -u \$USER"
fi
