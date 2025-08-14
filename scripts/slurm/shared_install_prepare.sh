#!/bin/bash

export MILABENCH_BRANCH=uv_compile_py3.12
export PYTHON_VERSION='3.12'
export MILABENCH_GPU_ARCH=cuda

set -ex

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
scontrol show job --json $SLURM_JOB_ID > $OUTPUT_DIRECTORY/info.json
# ===

CONDA_EXEC="$(which conda)"
CONDA_BASE=$(dirname $CONDA_EXEC)
source $CONDA_BASE/../etc/profile.d/conda.sh

export MILABENCH_WORDIR="$HOME/scratch/shared/$MILABENCH_GPU_ARCH"
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/scaling.yaml"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"

if [ -z "${MILABENCH_SOURCE}" ]; then
    export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/standard.yaml"
else
    export MILABENCH_CONFIG="$MILABENCH_SOURCE/config/standard.yaml"
fi

mkdir -p $MILABENCH_WORDIR
cd $MILABENCH_WORDIR

#
# Create Virtual Environment
#
if [ ! -d "$MILABENCH_ENV" ]; then
    conda create --prefix $MILABENCH_ENV python=$PYTHON_VERSION -y
fi

#
# Get Milabench or checkout branch
#
if [ -z "${MILABENCH_SOURCE}" ]; then
    if [ ! -d "$MILABENCH_WORDIR/milabench" ]; then
        git clone https://github.com/mila-iqia/milabench.git -b $MILABENCH_BRANCH
    fi
    export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
else
    (
        cd $MILABENCH_SOURCE
        git fetch origin
        git reset --hard origin/$MILABENCH_BRANCH
    )
fi

#
# Install Dependencies
#
conda activate $MILABENCH_ENV
pip install -e $MILABENCH_SOURCE[$MILABENCH_GPU_ARCH]

ARGS="$@"

milabench slurm_system > $MILABENCH_WORDIR/system.yaml
milabench install --system $MILABENCH_WORDIR/system.yaml $ARGS
milabench prepare --system $MILABENCH_WORDIR/system.yaml $ARGS


# ===
scontrol show job --json $SLURM_JOB_ID > $OUTPUT_DIRECTORY/info.json
# ===