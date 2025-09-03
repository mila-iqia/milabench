#!/bin/bash

export MILABENCH_BRANCH=staging
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=1
export MILABENCH_ARGS=""

set -ex

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
mkdir -p $OUTPUT_DIRECTORY/meta
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
touch $SLURM_SUBMIT_DIR/.no_report
# ===

CONDA_EXEC="$(which conda)"
CONDA_BASE=$(dirname $CONDA_EXEC)
source $CONDA_BASE/../etc/profile.d/conda.sh

export MILABENCH_SHARED="$HOME/scratch/shared/$MILABENCH_GPU_ARCH"
export MILABENCH_WORDIR="/tmp/$SLURM_JOB_ID/$MILABENCH_GPU_ARCH"  
export MILABENCH_ENV="$MILABENCH_SHARED/.env/$PYTHON_VERSION/"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/results/runs/scaling.yaml"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"


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

mkdir -p $MILABENCH_WORDIR
cd $MILABENCH_WORDIR

conda activate $MILABENCH_ENV

pip install -e $MILABENCH_SOURCE

mkdir -p $MILABENCH_BASE
MILABENCH_COPY_METHOD="rsync" milabench sharedsetup --network $MILABENCH_SHARED --local $MILABENCH_BASE
rm -rf $MILABENCH_BASE/*

mkdir -p $MILABENCH_BASE
export MILABENCH_COPY_NPROC=32
MILABENCH_COPY_METHOD="FIND_XARGS_RSYNC" milabench sharedsetup --network $MILABENCH_SHARED --local $MILABENCH_BASE
rm -rf $MILABENCH_BASE/*

mkdir -p $MILABENCH_BASE
export MILABENCH_COPY_NPROC=8
MILABENCH_COPY_METHOD="FIND_XARGS_RSYNC" milabench sharedsetup --network $MILABENCH_SHARED --local $MILABENCH_BASE
rm -rf $MILABENCH_BASE/*

mkdir -p $MILABENCH_BASE
export MILABENCH_COPY_NPROC=16
MILABENCH_COPY_METHOD="FIND_XARGS_RSYNC" milabench sharedsetup --network $MILABENCH_SHARED --local $MILABENCH_BASE
rm -rf $MILABENCH_BASE/*

mkdir -p $MILABENCH_BASE
MILABENCH_COPY_METHOD="untar" milabench sharedsetup --network $MILABENCH_SHARED --local $MILABENCH_BASE
rm -rf $MILABENCH_BASE/*

mkdir -p $MILABENCH_BASE
MILABENCH_COPY_METHOD="rsync_untar" milabench sharedsetup --network $MILABENCH_SHARED --local $MILABENCH_BASE
rm -rf $MILABENCH_BASE/*


# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===