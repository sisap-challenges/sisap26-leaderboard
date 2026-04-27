#!/bin/bash
# summarise-bench.sh — collect SLURM accounting data on the host, then
# invoke summarise-bench.py inside the hforest container for the h5py parts.
#
# Usage:
#   sbatch --partition=scavenge --cpus-per-task=1 --mem=2G --time=00:05:00 \
#       --output=hpc/bench-logs/summarise-%j.out summarise-bench.sh [--sort-by COLUMN]

CONTAINER=/home/maau/sisap26-baseline-dev/hforest.sif
OUTPUT_BASE=/home/maau/sisap26-baseline-dev/bench
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SACCT_CSV=$(mktemp /tmp/sacct-bench-XXXXXX.csv)
trap "rm -f $SACCT_CSV" EXIT

echo "Collecting SLURM accounting data..."
echo "job_id,state,elapsed,max_rss" > "$SACCT_CSV"

for d in "$OUTPUT_BASE"/*/; do
    dir_name=$(basename "$d")
    # Directory names are <node>-<N>cpu-<jobid>
    job_id=$(echo "$dir_name" | rev | cut -d- -f1 | rev)
    [[ "$job_id" =~ ^[0-9]+$ ]] || continue

    # --steps gives the batch step which carries MaxRSS
    # Query the .batch step directly for MaxRSS; fall back to the job summary.
    # Avoid --steps which is not available on all SLURM versions.
    sacct -j "${job_id}.batch" --format=Elapsed,MaxRSS,State --noheader --parsable2 2>/dev/null \
    | awk -F'|' -v jid="$job_id" '
        NF >= 3 && $1 != "" { print jid "," $3 "," $1 "," $2; exit }
    ' >> "$SACCT_CSV"
    # If .batch produced nothing (e.g. job not yet in accounting), try the job record
    if ! grep -q "^${job_id}," "$SACCT_CSV" 2>/dev/null; then
        sacct -j "$job_id" --format=Elapsed,MaxRSS,State --noheader --parsable2 2>/dev/null \
        | awk -F'|' -v jid="$job_id" '
            NF >= 3 && $1 != "" { print jid "," $3 "," $1 "," $2; exit }
        ' >> "$SACCT_CSV"
    fi
done

echo "Invoking summarise-bench.py inside container..."
apptainer exec --pwd /app \
    -B /home/maau/sisap26-baseline-dev:/mnt \
    -B "$SACCT_CSV":/tmp/sacct.csv:ro \
    "$CONTAINER" \
    python3 /mnt/hpc/summarise-bench.py \
        --output-base /mnt/bench \
        --sacct-csv /tmp/sacct.csv \
        "$@"
