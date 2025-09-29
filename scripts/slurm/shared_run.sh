#!/bin/bash

export MILABENCH_BRANCH=realtime_tracking
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export PYTHONUNBUFFERED=0
export MILABENCH_ARGS=""
export MILABENCH_CONFIG_NAME=standard

set -ex

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
export JR_JOB_ID=$(basename "$OUTPUT_DIRECTORY")
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
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/results/runs/scaling.yaml"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/$MILABENCH_CONFIG_NAME.yaml"

mkdir -p $MILABENCH_WORDIR
cd $MILABENCH_WORDIR
git clone https://github.com/mila-iqia/milabench.git -b $MILABENCH_BRANCH

conda create --prefix $MILABENCH_ENV python=$PYTHON_VERSION -y
conda activate $MILABENCH_ENV

mkdir -p $MILABENCH_WORDIR/results/runs
python -u /home/mila/d/delaunap/beefgs.py --pipe > $MILABENCH_WORDIR/results/runs/stats.jsonl &
BEEGFS_PID=$!

pip install -e $MILABENCH_SOURCE[$MILABENCH_GPU_ARCH]

milabench sharedsetup --network $MILABENCH_SHARED --local $MILABENCH_BASE

milabench slurm_system > $MILABENCH_WORDIR/system.yaml
rm -rf $MILABENCH_WORDIR/results/venv

milabench install --system $MILABENCH_WORDIR/system.yaml $MILABENCH_ARGS

milabench patch --venv $BENCHMARK_VENV

milabench run --system $MILABENCH_WORDIR/system.yaml $MILABENCH_ARGS || :

rsync -az $MILABENCH_WORDIR/results/runs $OUTPUT_DIRECTORY

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===

kill $BEEGFS_PID
wait $BEEGFS_PID 2>/dev/null