#!/bin/bash

export MILABENCH_BRANCH=realtime_tracking
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=0

set -ex

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
mkdir -p $OUTPUT_DIRECTORY/meta
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===

CONDA_EXEC="$(which conda)"
CONDA_BASE=$(dirname $CONDA_EXEC)
source $CONDA_BASE/../etc/profile.d/conda.sh

export MILABENCH_SHARED="$HOME/scratch/shared"
export MILABENCH_WORDIR="/tmp/$SLURM_JOB_ID/$MILABENCH_GPU_ARCH" 
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/results/runs/scaling.yaml"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/standard.yaml"

export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"

mkdir -p $MILABENCH_WORDIR
cd $MILABENCH_WORDIR
conda create --prefix $MILABENCH_ENV python=$PYTHON_VERSION -y
conda activate $MILABENCH_ENV

mkdir -p $MILABENCH_WORDIR/results/runs
python -u /home/mila/d/delaunap/beefgs.py --pipe > $MILABENCH_WORDIR/results/runs/stats.jsonl &
BEEGFS_PID=$!

mkdir -p $MILABENCH_WORDIR
cd $MILABENCH_WORDIR
git clone https://github.com/mila-iqia/milabench.git -b $MILABENCH_BRANCH
pip install -e $MILABENCH_SOURCE[$MILABENCH_GPU_ARCH]

ARGS="$@"

milabench slurm_system > $MILABENCH_WORDIR/system.yaml
rm -rf  $MILABENCH_BASE/extra

milabench install --force --system $MILABENCH_WORDIR/system.yaml $ARGS

milabench patch --venv $BENCHMARK_VENV

milabench prepare --system $MILABENCH_WORDIR/system.yaml $ARGS

TAR_FLAGS="--sort name --mtime='UTC 2020-01-01' --owner=0 --group=0 --numeric-owner -cf"
cd $MILABENCH_WORDIR

# Tar Locally
tar $TAR_FLAGS $MILABENCH_WORDIR/data.tar -C $MILABENCH_WORDIR data
tar $TAR_FLAGS $MILABENCH_WORDIR/cache.tar -C $MILABENCH_WORDIR cache

# Copy to scratch
rsync --inplace $MILABENCH_WORDIR/data.tar $MILABENCH_SHARED/data.tar
rsync --inplace $MILABENCH_WORDIR/cache.tar $MILABENCH_SHARED/cache.tar

rsync -az $MILABENCH_WORDIR/results/runs $OUTPUT_DIRECTORY

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===

kill $BEEGFS_PID
wait $BEEGFS_PID 2>/dev/null || :