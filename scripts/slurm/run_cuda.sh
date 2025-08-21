#!/bin/bash

set -ex

OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)

echo "OUTPUT is $OUTPUT_DIRECTORY"

CONDA_EXEC="$(which conda)"
CONDA_BASE=$(dirname $CONDA_EXEC)
source $CONDA_BASE/../etc/profile.d/conda.sh

export MILABENCH_GPU_ARCH=cuda
export MILABENCH_WORDIR="/tmp/$SLURM_JOB_ID/$MILABENCH_GPU_ARCH"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"

export MILABENCH_VENV="$MILABENCH_WORDIR/env"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/scaling.yaml"

mkdir -p $MILABENCH_WORDIR

if [ -z "${MILABENCH_PREPARE}" ]; then
    export MILABENCH_PREPARE=0
fi

if [ -z "${MILABENCH_SOURCE}" ]; then
    export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/standard.yaml"
else
    export MILABENCH_CONFIG="$MILABENCH_SOURCE/config/standard.yaml"
fi

ARGS="$@"

install_prepare() {
    mkdir -p $MILABENCH_WORDIR
    cd $MILABENCH_WORDIR

    if [ ! -d "$MILABENCH_WORDIR/env" ]; then
        conda create --prefix $MILABENCH_WORDIR/env python='3.10' -y
        # virtualenv $MILABENCH_WORDIR/env
    fi

    if [ -z "${MILABENCH_SOURCE}" ]; then
        if [ ! -d "$MILABENCH_WORDIR/milabench" ]; then
            git clone https://github.com/mila-iqia/milabench.git -b staging
        fi
        export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
    else
        (
            cd $MILABENCH_SOURCE
            git checkout staging
            git pull origin staging
        )
    fi
    
    . $MILABENCH_WORDIR/env/bin/activate

    pip install -e $MILABENCH_SOURCE

    milabench slurm_system > $MILABENCH_WORDIR/system.yaml

    #
    # Install milabench's benchmarks in their venv
    #
    # pip install torch
    # milabench pin --variant cuda --from-scratch $ARGS 
    milabench install --system $MILABENCH_WORDIR/system.yaml $ARGS

    which pip

    (
        . $BENCHMARK_VENV/bin/activate
        which pip
        pip install torch torchvision torchaudio

        # DALI stuff
        pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120
        pip install nvidia-pyindex
        pip install nvidia-nvjpeg-cu12
    )

    #
    #   Generate/download datasets, download models etc...
    milabench prepare --system $MILABENCH_WORDIR/system.yaml $ARGS
}

# module load cuda/12.3.2

if [ ! -d "$MILABENCH_WORDIR/env" ]; then
    install_prepare 
else
    echo "Reusing previous install"
    . $MILABENCH_WORDIR/env/bin/activate
fi

if [ "$MILABENCH_PREPARE" -eq 0 ]; then
    cd $MILABENCH_WORDIR

    . $MILABENCH_WORDIR/env/bin/activate

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
