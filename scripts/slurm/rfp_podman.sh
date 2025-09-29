#!/bin/bash

export MILABENCH_BRANCH=realtime_tracking
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=0
export MILABENCH_ARGS=""
export MILABENCH_IMAGE=ghcr.io/mila-iqia/milabench:cuda-nightly
export MILABENCH_HF_TOKEN="-"

set -ex

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
export JR_JOB_ID=$(basename "$OUTPUT_DIRECTORY")
mkdir -p $OUTPUT_DIRECTORY/meta
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
touch $SLURM_SUBMIT_DIR/.no_report
# ===

export MILABENCH_SHARED="$HOME/scratch/shared"
export MILABENCH_WORDIR="/tmp/$SLURM_JOB_ID/$MILABENCH_GPU_ARCH"  
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export MILABENCH_NODES=$(scontrol show hostnames $SLURM_NODELIST)
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/standard.yaml"

CONDA_EXEC="$(which conda)"
CONDA_BASE=$(dirname $CONDA_EXEC)
source $CONDA_BASE/../etc/profile.d/conda.sh

mkdir -p $MILABENCH_WORDIR
cd $MILABENCH_WORDIR
conda create --prefix $MILABENCH_ENV python=$PYTHON_VERSION -y
conda activate $MILABENCH_ENV

mkdir -p $MILABENCH_WORDIR
cd $MILABENCH_WORDIR
git clone https://github.com/mila-iqia/milabench.git -b $MILABENCH_BRANCH
pip install -e $MILABENCH_SOURCE[$MILABENCH_GPU_ARCH]

#
# SLURM BOILERPLATE ENDS
#

pip install -e $MILABENCH_SOURCE

milabench container                                         \
        --base $MILABENCH_WORDIR                            \
        --image $MILABENCH_IMAGE                            \
        --node $MILABENCH_NODES                             \
        --token $MILABENCH_HF_TOKEN                         \
        --sshkey ~/.ssh/id_rsa                              \
        --args "$MILABENCH_ARGS"

rsync -az $MILABENCH_WORDIR/runs $OUTPUT_DIRECTORY

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===
