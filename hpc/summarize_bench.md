# Summarising Benchmark Results

Results from both benchmark approaches (competition-spec and max-resource) are
summarised using the same two scripts:

- `hpc/summarise-bench.sh` — runs on the host, collects SLURM accounting data
  via `sacct`, then calls the Python script inside the hforest container
- `hpc/summarise-bench.py` — reads result HDF5 files and produces the table;
  must run inside the container because it requires `h5py`

Because direct execution is not permitted on the login node, both must be
submitted as SLURM jobs.

## Competition-spec runs (`bench/`)

One quality level per node, fixed 8 CPUs / 16 GiB.

```bash
sbatch --partition=scavenge --cpus-per-task=1 --mem=2G --time=00:05:00 \
    --output=hpc/bench-logs/summarise-%j.out \
    hpc/summarise-bench.sh
```

## Max-resource runs (`bench-maxres/`)

Five quality levels per node (task2 → task2:98), all cores, 90% RAM.
Pass `--output-base` to point at the correct results directory.

```bash
sbatch --partition=scavenge --cpus-per-task=1 --mem=2G --time=00:05:00 \
    --output=hpc/bench-logs/summarise-maxres-%j.out \
    hpc/summarise-bench.sh --output-base /home/maau/sisap26-baseline-dev/bench-maxres
```

## Reading the output

Once the job completes:

```bash
cat hpc/bench-logs/summarise-<JOBID>.out
```

### Columns

| Column | Description |
|--------|-------------|
| Node | Hostname |
| CPUs | Allocated / OMP threads used |
| JobID | SLURM job ID |
| Task | Quality level (`task2` = 81% recall … `task2:98` = 98% recall) |
| State | SLURM job state (`COMPLETED`, `FAILED`, etc.) |
| Wall time | Elapsed wall-clock time from SLURM accounting |
| MaxRSS | Peak resident memory from SLURM accounting |
| Build(s) | Index build time in seconds (0 for task2 graph construction) |
| Query(s) | Search / graph construction time in seconds |
| Total(s) | Build + Query; primary column for comparison |
| CPU model | From `lscpu.txt` captured at job start |

Rows are sorted by `Total(s)` by default. Pass `--sort-by` to change:

```bash
hpc/summarise-bench.sh --sort-by node
hpc/summarise-bench.sh --sort-by query
```

Max-resource output groups rows by node (blank line between nodes) so the
recall/speed tradeoff curve for each node is easy to read.

## Troubleshooting

**`State` and accounting columns show `n/a`**
SLURM accounting (`sacct`) may not have ingested the job yet. Wait a few
minutes after job completion and resubmit the summary job.

**Row appears with `nan` times but no `Task`**
The result HDF5 file was not written — the job likely failed or was cancelled.
Check the corresponding error log:
```bash
cat hpc/bench-logs/bench-<NODE>-<JOBID>.err
```

**OOM on high core-count nodes (cn8, cn9, cn13)**
At 8 cores the proportional RAM allowance falls below 16 GiB on nodes with
many cores. Resubmit with more allocated cores to unlock the memory budget
while keeping `OMP_NUM_THREADS=8`:
```bash
sbatch --nodelist=cn8 --cpus-per-task=16 --mem=16G \
    --export=ALL,OMP_NUM_THREADS=8 \
    hpc/bench.slurm
```
