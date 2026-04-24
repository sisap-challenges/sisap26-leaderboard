#!/bin/bash

# SISAP25 HPC Launcher Script
# Usage: ./run-hforest.sh [task1|task2]

TASK=$1
CONTAINER_PATH="/path/to/containers/hforest.sif"    # UPDATE THIS PATH

if [[ -z "$TASK" ]]; then
    echo "Usage: $0 [task1|task2]"
    echo
    echo "Examples:"
    echo "  $0 task1    # Submit Task 1 (PUBMED23 dataset)"
    echo "  $0 task2    # Submit Task 2 (GOOAQ k-NN graph)"
    echo
    echo "Monitor jobs: squeue -u \$USER"
    exit 1
fi

if [[ ! -f "$CONTAINER_PATH" ]]; then
    echo "Error: Container not found at $CONTAINER_PATH"
    echo "Please build and convert the container first:"
    echo "  1. docker build --platform linux/amd64 -t sisap25/hforest ."
    echo "  2. docker save sisap25/hforest -o hforest.tar"
    echo "  3. scp hforest.tar user@hpc:/path/"
    echo "  4. apptainer build hforest.sif docker-archive://hforest.tar"
    exit 1
fi

case $TASK in
    "task1")
        echo "🚀 Submitting SISAP25 Task 1 (PUBMED23 search)..."
        JOB_ID=$(sbatch --parsable hforest-task1.slurm)
        echo "✅ Job submitted with ID: $JOB_ID"
        echo "📊 Expected: ~43 min runtime, 0.70+ recall, 637+ q/s"
        ;;
    "task2")
        echo "🚀 Submitting SISAP25 Task 2 (GOOAQ k-NN graph)..."
        JOB_ID=$(sbatch --parsable hforest-task2.slurm)
        echo "✅ Job submitted with ID: $JOB_ID"  
        echo "🏆 Expected: ~105 sec runtime, 0.80+ recall (Winner performance!)"
        ;;
    *)
        echo "Error: Unknown task '$TASK'. Use 'task1' or 'task2'."
        exit 1
        ;;
esac

echo
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  scontrol show job $JOB_ID"
echo "  tail -f hforest-${TASK}-${JOB_ID}.out"