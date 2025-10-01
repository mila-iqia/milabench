#!/bin/bash

export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=1
export MILABENCH_ARGS=""
export MILABENCH_IMAGE=ghcr.io/mila-iqia/milabench:${MILABENCH_GPU_ARCH}-nightly
export MILABENCH_HF_TOKEN="-"
set -ex

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
mkdir -p $OUTPUT_DIRECTORY/meta
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
touch $SLURM_SUBMIT_DIR/.no_report
# ===

export MILABENCH_WORDIR="/tmp/$SLURM_JOB_ID/$MILABENCH_GPU_ARCH" 
export MILABENCH_BASE="$MILABENCH_WORDIR/results"

mkdir -p $MILABENCH_WORDIR
cd $MILABENCH_WORDIR

podman pull $MILABENCH_IMAGE

mkdir -p $MILABENCH_BASE/runs
mkdir -p $MILABENCH_BASE/data
mkdir -p $MILABENCH_BASE/cache

podman run --rm --ipc=host                                  \
      --device nvidia.com/gpu=all                           \
      --security-opt=label=disable                          \
      -e HF_TOKEN=$MILABENCH_HF_TOKEN                       \
      -e MILABENCH_HF_TOKEN=$MILABENCH_HF_TOKEN             \
      -v $MILABENCH_BASE/runs:/milabench/envs/runs          \
      -v $MILABENCH_BASE/data:/milabench/envs/data          \
      -v $MILABENCH_BASE/cache:/milabench/envs/cache        \
      $MILABENCH_IMAGE                                      \
      /milabench/.env/bin/milabench prepare
    
podman run --rm --ipc=host                                  \
      --device nvidia.com/gpu=all                           \
      --security-opt=label=disable                          \
      -e HF_HUB_OFFLINE=1                                   \
      -e HF_TOKEN=$MILABENCH_HF_TOKEN                       \
      -e MILABENCH_HF_TOKEN=$MILABENCH_HF_TOKEN             \
      -v $MILABENCH_BASE/runs:/milabench/envs/runs          \
      -v $MILABENCH_BASE/data:/milabench/envs/data          \
      -v $MILABENCH_BASE/cache:/milabench/envs/cache        \
      $MILABENCH_IMAGE                                      \
      /milabench/.env/bin/milabench run  || :

rsync -az $MILABENCH_WORDIR/results/runs $OUTPUT_DIRECTORY

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===


