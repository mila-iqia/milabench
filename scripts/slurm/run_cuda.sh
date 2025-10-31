#!/bin/bash


export MILABENCH_BRANCH=realtime_tracking
export CONFIG=inference.yaml


set -ex

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
mkdir -p $OUTPUT_DIRECTORY/meta
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===

CONDA_EXEC="$(which conda)"
CONDA_BASE=$(dirname $CONDA_EXEC)
source $CONDA_BASE/../etc/profile.d/conda.sh

export PYTHON_VERSION='3.12'
export MILABENCH_GPU_ARCH=cuda

export MILABENCH_WORDIR="/tmp/$SLURM_JOB_ID/$MILABENCH_GPU_ARCH"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"

export MILABENCH_ENV="$MILABENCH_WORDIR/.env/$PYTHON_VERSION/"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/scaling.yaml"

mkdir -p $MILABENCH_WORDIR

if [ -z "${MILABENCH_PREPARE}" ]; then
    export MILABENCH_PREPARE=0
fi

if [ -z "${MILABENCH_SOURCE}" ]; then
    export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/$CONFIG"
else
    export MILABENCH_CONFIG="$MILABENCH_SOURCE/config/$CONFIG"
fi

ARGS="$@"

install_prepare() {
    mkdir -p $MILABENCH_WORDIR
    cd $MILABENCH_WORDIR

    if [ ! -d "$MILABENCH_ENV" ]; then
        conda create --prefix $MILABENCH_ENV python=$PYTHON_VERSION -y
    fi

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
    
    conda activate $MILABENCH_ENV
    pip install -e $MILABENCH_SOURCE

    milabench slurm_system > $MILABENCH_WORDIR/system.yaml

    #
    # Install milabench's benchmarks in their venv
    #
    # pip install torch
    # milabench pin --variant cuda --from-scratch $ARGS 

    export MILABENCH_NO_BUILD_ISOLATION=1
    export MILABENCH_USE_UV=1
    milabench install --system $MILABENCH_WORDIR/system.yaml $ARGS

    which pip

    # (
    #     . $BENCHMARK_VENV/bin/activate
    #     which pip
    #     pip install torch torchvision torchaudio

    #     # DALI stuff
    #     pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120
    #     pip install nvidia-pyindex
    #     pip install nvidia-nvjpeg-cu12
    # )

    #
    #   Generate/download datasets, download models etc...
    milabench prepare --system $MILABENCH_WORDIR/system.yaml $ARGS
}

# module load cuda/12.3.2

if [ ! -d "$MILABENCH_WORDIR/env" ]; then
    install_prepare 
else
    echo "Reusing previous install"

    conda activate $MILABENCH_ENV
    # . $MILABENCH_WORDIR/env/bin/activate
fi

if [ "$MILABENCH_PREPARE" -eq 0 ]; then
    cd $MILABENCH_WORDIR

    conda activate $MILABENCH_ENV
    # . $MILABENCH_WORDIR/env/bin/activate

    # pip install torch
    # milabench pin --variant cuda --from-scratch 
    # rm -rf $MILABENCH_WORDIR/results/venv/
    # rm -rf $MILABENCH_WORDIR/results/extra
    # milabench install --system $MILABENCH_WORDIR/system.yaml
    # milabench prepare --system $MILABENCH_WORDIR/system.yaml $ARGS

    (
        . $BENCHMARK_VENV/bin/activate
        which pip
        # pip uninstall torchao -y
        # pip install torchao --no-input
    )

    milabench run --system $MILABENCH_WORDIR/system.yaml $ARGS

    #
    #   Display report
    milabench report --runs $MILABENCH_WORDIR/results/runs
fi

rsync -az $MILABENCH_WORDIR/results/runs $OUTPUT_DIRECTORY

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===
