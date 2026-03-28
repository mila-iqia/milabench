#!/bin/bash

export MILABENCH_BRANCH=cu130_x86_pins
export CONFIG=all.yaml
export PYTHON_VERSION=3.12
export MILABENCH_GPU_ARCH=cuda
export HF_TOKEN=""
export MILABENCH_ARGS=""

set -ex

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
mkdir -p $OUTPUT_DIRECTORY/meta
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===

module load cuda/12.6.0

export MILABENCH_WORDIR="/tmp/$SLURM_JOB_ID/$MILABENCH_GPU_ARCH"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/scaling.yaml"
export MILABENCH_HF_TOKEN="$HF_TOKEN"
export MILABENCH_SHARED="$HOME/scratch/shared"

UV=$HOME/.local/bin/uv

mkdir -p $MILABENCH_WORDIR

if [ -z "${MILABENCH_PREPARE}" ]; then
    export MILABENCH_PREPARE=0
fi

ARGS="$@"

install_prepare() {
    mkdir -p $MILABENCH_WORDIR
    cd $MILABENCH_WORDIR

    if [ ! -d "$MILABENCH_ENV" ]; then
        $UV venv $MILABENCH_ENV --python=$PYTHON_VERSION
    fi

    git clone https://github.com/milabench/milabench.git -b $MILABENCH_BRANCH
    export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"

    source $MILABENCH_ENV/bin/activate
    $UV pip install -e $MILABENCH_SOURCE[$MILABENCH_GPU_ARCH]
    
    milabench slurm_system > $MILABENCH_WORDIR/system.yaml

    export MILABENCH_CONFIG="$MILABENCH_SOURCE/config/$CONFIG"
    #
    # Install milabench's benchmarks in their venv
    #
    # pip install torch
    # milabench pin --variant cuda --from-scratch $ARGS 

    milabench install --variant cuda --system $MILABENCH_WORDIR/system.yaml $MILABENCH_ARGS

    # (
    #     . $BENCHMARK_VENV/bin/activate
    #     which pip
    #     pip install torch torchvision torchaudio

    #     # DALI stuff
    #     pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120
    #     pip install nvidia-pyindex
    #     pip install nvidia-nvjpeg-cu12
    # )

    # Copy cached datasets/models
    milabench sharedsetup --network $MILABENCH_SHARED --local $MILABENCH_BASE

    #   Generate/download/verify datasets, download models etc...
    milabench prepare --system $MILABENCH_WORDIR/system.yaml $MILABENCH_ARGS

    # Generate an archive that is cached in the network drive
    # milabench archive --network $MILABENCH_SHARED --local $MILABENCH_BASE
}

# module load cuda/12.3.2

if [ ! -d "$MILABENCH_WORDIR/env" ]; then
    install_prepare 
else
    echo "Reusing previous install"

    source $MILABENCH_ENV/bin/activate
    # . $MILABENCH_WORDIR/env/bin/activate
fi

if [ "$MILABENCH_PREPARE" -eq 0 ]; then
    cd $MILABENCH_WORDIR

    source $MILABENCH_ENV/bin/activate
    export MILABENCH_CONFIG="$MILABENCH_SOURCE/config/$CONFIG"

    # pip install torch
    # milabench pin --variant cuda --from-scratch 
    # rm -rf $MILABENCH_WORDIR/results/venv/
    # rm -rf $MILABENCH_WORDIR/results/extra
    # milabench install --system $MILABENCH_WORDIR/system.yaml
    # milabench prepare --system $MILABENCH_WORDIR/system.yaml $ARGS

    (
        . $BENCHMARK_VENV/bin/activate
        # pip uninstall torchao -y
        # pip install torchao --no-input
    )

    milabench run --system $MILABENCH_WORDIR/system.yaml $MILABENCH_ARGS

    #
    #   Display report
    milabench report --runs $MILABENCH_WORDIR/results/runs
fi

rsync -az $MILABENCH_WORDIR/results/runs $OUTPUT_DIRECTORY

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===
