#!/bin/bash


export MILABENCH_BRANCH=main
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=0
export MILABENCH_ARGS=""
export MILABENCH_CONFIG_NAME=all
export MILABENCH_REPO=https://github.com/milabench/milabench.git
export CUDA_VERSION="130"
export PYTORCH_VERSION="2.10.0"

export MILABENCH_PUBLISH_KEY="{{ secrets.MILABENCH_PUBLISH_KEY }}"
export HF_TOKEN="{{ secrets.HF_TOKEN }}"
export MILABENCH_HF_TOKEN=$HF_TOKEN

# Set it AFTER exporting the secrets :)
set -ex

# Fix
export PATH="$HOME/.bin/:$PATH"

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
export JR_JOB_ID=$(basename "$OUTPUT_DIRECTORY")
mkdir -p $OUTPUT_DIRECTORY/meta
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
touch $SLURM_SUBMIT_DIR/.no_report
# ===

# CONDA_EXEC="$(which conda)"
# CONDA_BASE=$(dirname $CONDA_EXEC)
# source $CONDA_BASE/../etc/profile.d/conda.sh
export UV=$HOME/.local/bin/uv

export MILABENCH_SHARED="$HOME/scratch/shared"
export MILABENCH_WORDIR="/tmp/$SLURM_JOB_ID/$MILABENCH_GPU_ARCH"  
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/results/runs/scaling.yaml"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/$MILABENCH_CONFIG_NAME.yaml"


mkdir -p $MILABENCH_WORDIR
cd $MILABENCH_WORDIR
git clone $MILABENCH_REPO -b $MILABENCH_BRANCH

# conda create --prefix $MILABENCH_ENV python=$PYTHON_VERSION -y
# conda activate $MILABENCH_ENV

$UV venv --python=$PYTHON_VERSION $MILABENCH_ENV
. $MILABENCH_ENV/bin/activate

mkdir -p $MILABENCH_WORDIR/results/runs
# python -u /home/mila/d/delaunap/beefgs.py --pipe > $MILABENCH_WORDIR/results/runs/stats.jsonl &
# BEEGFS_PID=$!

$UV pip install -e $MILABENCH_SOURCE[$MILABENCH_GPU_ARCH]
$UV pip install psycopg2-binary

# LOCAL_PORT=8123
# export MILABENCH_DB="--plugin term --plugin sql postgresql://milabench_write:1234@localhost:$LOCAL_PORT/milabench"
# milabench tools tunnel --local-port $LOCAL_PORT &
# TUNNEL_PID=$!

milabench data sharedsetup --network $MILABENCH_SHARED --local $MILABENCH_BASE

milabench slurm system > $MILABENCH_WORDIR/system.yaml
rm -rf $MILABENCH_WORDIR/results/venv

module load cuda/12.6.0

# $UV pip install torch
# milabench tools pin --variant cuda

milabench install --system $MILABENCH_WORDIR/system.yaml --set cuda=$CUDA_VERSION torch=$PYTORCH_VERSION $MILABENCH_ARGS

milabench tools patch --venv $BENCHMARK_VENV

milabench run --system $MILABENCH_WORDIR/system.yaml $MILABENCH_ARGS $MILABENCH_DB || :

rsync -az $MILABENCH_WORDIR/results/runs $OUTPUT_DIRECTORY

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===

# kill $BEEGFS_PID
# wait $BEEGFS_PID 2>/dev/null || :

# kill $TUNNEL_PID
# wait $TUNNEL_PID 2>/dev/null || :