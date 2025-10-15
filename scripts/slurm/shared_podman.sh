#!/bin/bash

export MILABENCH_BRANCH=master
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=1
export MILABENCH_ARGS="--select dinov2-giant-gpus"
export MILABENCH_IMAGE=ghcr.io/mila-iqia/milabench:${MILABENCH_GPU_ARCH}-stacc-v1


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

export MILABENCH_SHARED="$HOME/scratch/shared"
export MILABENCH_WORDIR="/tmp/$SLURM_JOB_ID/$MILABENCH_GPU_ARCH"
 
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/results/runs/scaling.yaml"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/standard.yaml"

mkdir -p $MILABENCH_WORDIR
cd $MILABENCH_WORDIR
conda create --prefix $MILABENCH_ENV python=$PYTHON_VERSION -y
conda activate $MILABENCH_ENV

cd $MILABENCH_WORDIR
git clone https://github.com/mila-iqia/milabench.git -b $MILABENCH_BRANCH
pip install -e $MILABENCH_SOURCE[$MILABENCH_GPU_ARCH]

mkdir -p $MILABENCH_SHARED/data
mkdir -p $MILABENCH_SHARED/venv
milabench sharedsetup --network $MILABENCH_SHARED --local $MILABENCH_BASE

podman pull $MILABENCH_IMAGE

mkdir -p $MILABENCH_BASE/runs
mkdir -p $MILABENCH_BASE/data
mkdir -p $MILABENCH_BASE/cache

podman run --rm --ipc=host                                  \
      --device nvidia.com/gpu=all                           \
      --security-opt=label=disable                          \
      -v $MILABENCH_BASE/runs:/milabench/envs/runs          \
      -v $MILABENCH_BASE/data:/milabench/envs/data          \
      -v $MILABENCH_BASE/cache:/milabench/envs/cache        \
      $MILABENCH_IMAGE                                      \
      /milabench/.env/bin/milabench run $MILABENCH_ARGS || :

rsync -az $MILABENCH_WORDIR/results/runs $OUTPUT_DIRECTORY

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===


