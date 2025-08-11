#!/bin/bash

set -ex

OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)

export MILABENCH_USE_UV=1
export MILABENCH_WORDIR="/tmp/$SLURM_JOB_ID"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"

export MILABENCH_VENV="$MILABENCH_WORDIR/env"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/scaling.yaml"

CONDA_EXEC="$(which conda)"
CONDA_BASE=$(dirname $CONDA_EXEC)
source $CONDA_BASE/../etc/profile.d/conda.sh

conda create --prefix $MILABENCH_WORDIR/env python='3.12' -y
conda activate $MILABENCH_WORDIR/env

(
    cd $MILABENCH_SOURCE
    git checkout staging
    git fetch origin
    git reset --hard origin/staging
)

pip install -e $MILABENCH_SOURCE
pip install torch

module load cuda/12.6.0

cd $MILABENCH_SOURCE
MILABENCH_GPU_ARCH=cuda milabench pin -c constraints/cuda.txt --config config/standard.yaml --from-scratch
MILABENCH_GPU_ARCH=rocm milabench pin -c constraints/rocm.txt --config config/standard.yaml --from-scratch

(
    cd $MILABENCH_SOURCE
    git checkout -b "update_pins_${SLURM_JOB_ID}"
    git push origin "update_pins_${SLURM_JOB_ID}"
)