#!/bin/bash

set -ex

# ---
# Job Runner Insert
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
scontrol show job --json $SLURM_JOB_ID > $OUTPUT_DIRECTORY/info.json
# ===

export PYTHON_VERSION='3.12'
export MILABENCH_GPU_ARCH="cuda"
export MILABENCH_WORDIR="$HOME/scratch/shared"
export MILABENCH_BRANCH="staging"

export MILABENCH_BASE="$MILABENCH_WORDIR"
export BENCHMARK_VENV="$MILABENCH_WORDIR/venv"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/scaling.yaml"
export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"

CONDA_EXEC="$(which conda)"
CONDA_BASE=$(dirname $CONDA_EXEC)
source $CONDA_BASE/../etc/profile.d/conda.sh

conda create --prefix $MILABENCH_ENV python=$PYTHON_VERSION -y
conda activate $MILABENCH_ENV

(
    cd $MILABENCH_SOURCE
    git checkout $MILABENCH_BRANCH
    git fetch origin
    git reset --hard origin/$MILABENCH_BRANCH
)

pip install -e $MILABENCH_SOURCE
pip install torch

module load cuda/12.6.0

(
    cd $MILABENCH_SOURCE
    milabench pin -c constraints/$MILABENCH_GPU_ARCH.txt --config config/standard.yaml --from-scratch
)

(
    cd $MILABENCH_SOURCE
    git checkout -b "update_pins_${SLURM_JOB_ID}"
    git add --all
    git commit -m "Pin Dependencies"
    git push origin "update_pins_${SLURM_JOB_ID}"
)