#!/bin/bash

#!/bin/bash

export MILABENCH_BRANCH=staging
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=1
export MILABENCH_ARGS=""
export MILABENCH_IMAGE=ghcr.io/mila-iqia/milabench:${MILABENCH_GPU_ARCH}-nightly


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


mkdir -p $MILABENCH_WORDIR
cd $MILABENCH_WORDIR

podman pull $MILABENCH_IMAGE

mkdir -p $MILABENCH_BASE/runs
mkdir -p $MILABENCH_SHARED/data
mkdir -p $MILABENCH_SHARED/cache

podman run --rm --ipc=host                                  \
      --device nvidia.com/gpu=all                           \
      --security-opt=label=disable                          \
      -v $MILABENCH_BASE/runs:/milabench/envs/runs          \
      -v $MILABENCH_SHARED/data:/milabench/envs/data        \
      -v $MILABENCH_SHARED/cache:/milabench/envs/cache      \
      $MILABENCH_IMAGE                                      \
      /milabench/.env/bin/milabench run  || :

rsync -az $MILABENCH_WORDIR/results/runs $OUTPUT_DIRECTORY

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===


