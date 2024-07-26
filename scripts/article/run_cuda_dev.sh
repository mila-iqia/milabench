#!/bin/bash

set -ex

# export MILABENCH_SOURCE=$HOME/milabench
# mkdir /tmp/workspace && cd /tmp/workspace
# conda activate py310
#
#

export MILABENCH_GPU_ARCH=cuda
export MILABENCH_WORDIR="$(pwd)/$MILABENCH_GPU_ARCH"

export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/standard.yaml"
export MILABENCH_VENV="$MILABENCH_WORDIR/env"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"


if [ -z "${MILABENCH_PREPARE}" ]; then
    export MILABENCH_PREPARE=0
fi

if [ -z "${MILABENCH_SOURCE}" ]; then
    export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/standard.yaml"
else
    export MILABENCH_CONFIG="$MILABENCH_SOURCE/config/standard.yaml"
fi

install_prepare() {
    mkdir -p $MILABENCH_WORDIR
    cd $MILABENCH_WORDIR

    virtualenv $MILABENCH_WORDIR/env --reset-app-data

    if [ -z "${MILABENCH_SOURCE}" ]; then
        if [ ! -d "$MILABENCH_WORDIR/milabench" ]; then
            git clone https://github.com/mila-iqia/milabench.git
        fi
        export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
    else
        export MILABENCH_CONFIG="$MILABENCH_SOURCE/config/standard.yaml"
    fi

    if [ ! -d "$MILABENCH_WORDIR/voir" ]; then
        echo ""
        # git clone https://github.com/Delaunay/voir.git
        # git clone https://github.com/Delaunay/torchcompat.git
    fi

    . $MILABENCH_WORDIR/env/bin/activate
    pip install -e $MILABENCH_SOURCE

    # milabench pin --variant cuda "$@"

    #
    # Install milabench's benchmarks in their venv
    #
    milabench install "$@"

    which pip
    # pip install -e $MILABENCH_WORDIR/voir
    # pip install -e $MILABENCH_WORDIR/torchcompat

    (
        . $BENCHMARK_VENV/bin/activate
        which pip
        #pip install -e $MILABENCH_WORDIR/voir
        # pip install -e $MILABENCH_WORDIR/torchcompat
        # pip install torch torchvision torchaudio

        pip install fvcore xFormers

        # DALI stuff
        # pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda120
        # pip install nvidia-pyindex
        # pip install nvidia-nvjpeg-cu12
    )

    #
    #   Generate/download datasets, download models etc...
    milabench prepare "$@"
}

module load cuda/12.3.2

if [ ! -d "$MILABENCH_WORDIR/results/venv/torch" ]; then
    install_prepare 
else
    echo "Reusing previous install"
    . $MILABENCH_WORDIR/env/bin/activate
fi

if [ "$MILABENCH_PREPARE" -eq 0 ]; then
    cd $MILABENCH_WORDIR

    #
    #   Run the benchmakrs
    milabench run "$@"

    #
    #   Display report
    milabench report --runs $MILABENCH_WORDIR/results/runs
fi