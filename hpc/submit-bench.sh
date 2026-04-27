#!/bin/bash
# submit-bench.sh — submit one hforest benchmark job per node.
#
# Usage:
#   ./submit-bench.sh            # dry run: print sbatch commands, submit nothing
#   ./submit-bench.sh --submit   # actually submit to SLURM
#
# All jobs request exactly 8 CPUs (the SISAP25 competition constraint).
# Memory is capped at 16 GiB but may be lower on nodes where the cluster
# policy ties RAM allocation to the fraction of cores requested:
#   allowed_mem = floor(8 / total_cores * total_ram_GiB) GiB
# On cn8 (256c/256GiB) this yields 8 GiB; on cn9 (192c/250GiB) ~10 GiB.
#
# Node selection rationale — diverse across network tier, memory, and CPU arch:
#   desktop1  —  8c /  32 GiB / 1 Gbps Eth  : smallest machine, full RAM share
#   cn14      — 48c / 120 GiB / 1 Gbps Eth  : mid compute, slow network
#   cn16      — 40c / 256 GiB / 1 Gbps Eth  : larger RAM at same network tier
#   cn5       — 32c / 512 GiB / 100 Gbps IB : high-memory, fast IB
#   cn6       — 64c / 384 GiB / 100 Gbps IB : largest IB node
#   cn8       — 256c/ 256 GiB / 100 Gbps IB : highest core count (mem-limited)
#   cn9       — 192c/ 250 GiB / 10 Gbps SFP : high core count (mem-limited)
#   cn11      — 96c /1536 GiB / 10 Gbps SFP : largest RAM node
#
# cn15 is excluded (reserved partition pcn15).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPT="$SCRIPT_DIR/bench.slurm"

SUBMIT=false
if [[ "${1:-}" == "--submit" ]]; then
    SUBMIT=true
fi

# ── Node matrix: total_cores  total_ram_GiB ──────────────────────────────────
declare -A NODE_CORES=( [desktop1]=8  [cn14]=48 [cn16]=40 [cn5]=32  [cn6]=64  [cn8]=256 [cn9]=192 [cn11]=96  )
declare -A NODE_RAM=(   [desktop1]=32 [cn14]=120 [cn16]=256 [cn5]=512 [cn6]=384 [cn8]=256 [cn9]=250 [cn11]=1536 )
# ─────────────────────────────────────────────────────────────────────────────

FIXED_CPUS=8
MAX_MEM_GiB=16

mkdir -p "$SCRIPT_DIR/bench-logs"

echo "hforest benchmark submission"
echo "Script : $SLURM_SCRIPT"
echo "CPUs   : $FIXED_CPUS (fixed)"
echo "Mode   : $( $SUBMIT && echo 'SUBMIT' || echo 'DRY RUN (pass --submit to submit)' )"
echo

for NODE in desktop1 cn14 cn16 cn5 cn6 cn8 cn9 cn11; do
    TOTAL_CORES=${NODE_CORES[$NODE]}
    TOTAL_RAM=${NODE_RAM[$NODE]}

    # Proportional RAM: floor(8 / total_cores * total_ram), capped at MAX_MEM_GiB.
    PROP_MEM=$(( FIXED_CPUS * TOTAL_RAM / TOTAL_CORES ))
    MEM=$(( PROP_MEM < MAX_MEM_GiB ? PROP_MEM : MAX_MEM_GiB ))

    CMD=(
        sbatch
        --nodelist="$NODE"
        --partition=scavenge
        --cpus-per-task="$FIXED_CPUS"
        --mem="${MEM}G"
        --output="$SCRIPT_DIR/bench-logs/bench-${NODE}-%j.out"
        --error="$SCRIPT_DIR/bench-logs/bench-${NODE}-%j.err"
        "$SLURM_SCRIPT"
    )

    echo "  [${NODE}] ${FIXED_CPUS} CPUs, ${MEM}G RAM (proportional: ${PROP_MEM}G, node: ${TOTAL_CORES}c/${TOTAL_RAM}GiB)"
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
    echo "  watch -n 30 squeue -u \$USER"
fi


# ── Diverse node matrix ──────────────────────────────────────────────────────
# Format: "node  max_cores  mem_request"
# mem_request is kept safely below each node's physical RAM.
#
# Selection rationale:
#   desktop1  —  8c  / 32 GiB  / 1 Gbps   : smallest machine, competition spec
#   cn14      — 48c  / 120 GiB / 1 Gbps   : mid compute, slow network, no GPU
#   cn16      — 40c  / 250 GiB / 1 Gbps   : 40-core, larger RAM than cn14
#   cn5       — 32c  / 500 GiB / 100 Gbps : high-memory, fast IB
#   cn6       — 64c  / 370 GiB / 100 Gbps : largest IB node by core count
#   cn8       — 256c / 250 GiB / 100 Gbps : highest core count on the cluster
#   cn9       — 192c / 240 GiB / 10 Gbps  : high core count, SFP network
#   cn11      — 96c  / 1400 GiB/ 10 Gbps  : largest RAM node
# ─────────────────────────────────────────────────────────────────────────────
declare -A NODE_CORES=(
    [desktop1]=8
    [cn14]=48
    [cn16]=40
    [cn5]=32
    [cn6]=64
    [cn8]=256
    [cn9]=192
    [cn11]=96
)

declare -A NODE_MEM=(
    [desktop1]="28G"
    [cn14]="112G"
    [cn16]="240G"
    [cn5]="490G"
    [cn6]="370G"
    [cn8]="240G"
    [cn9]="235G"
    [cn11]="1400G"
)

mkdir -p "$SCRIPT_DIR/bench-logs"

echo "hforest benchmark submission"
echo "Script : $SLURM_SCRIPT"
echo "Mode   : $( $SUBMIT && echo 'SUBMIT' || echo 'DRY RUN (pass --submit to submit)' )"
echo

for NODE in desktop1 cn14 cn16 cn5 cn6 cn8 cn9 cn11; do
    MAX=${NODE_CORES[$NODE]}
    MEM=${NODE_MEM[$NODE]}

    # Thread counts to test: always include 8 (competition baseline) and max.
    # Add 32 as an intermediate point for nodes with enough cores.
    if (( MAX == 8 )); then
        THREAD_COUNTS=(8)
    elif (( MAX <= 32 )); then
        THREAD_COUNTS=(8 $MAX)
    else
        THREAD_COUNTS=(8 32 $MAX)
    fi

    for NCPUS in "${THREAD_COUNTS[@]}"; do
        CMD=(
            sbatch
            --nodelist="$NODE"
            --cpus-per-task=8 #"$NCPUS"
            #--mem="$MEM"
            --output="$SCRIPT_DIR/bench-logs/bench-${NODE}-${NCPUS}cpu-%j.out"
            --error="$SCRIPT_DIR/bench-logs/bench-${NODE}-${NCPUS}cpu-%j.err"
            "$SLURM_SCRIPT"
        )

        echo "  ${CMD[*]}"
        if $SUBMIT; then
            JOB_ID=$("${CMD[@]}" --parsable)
            echo "    -> submitted job $JOB_ID"
        fi
    done
done

echo
if ! $SUBMIT; then
    echo "Dry run complete. Re-run with --submit to submit all jobs."
else
    echo "All jobs submitted. Monitor with:"
    echo "  squeue -u \$USER"
    echo "  watch -n 30 squeue -u \$USER"
fi
