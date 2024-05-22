#!/bin/bash

set -ex

export MILABENCH_GPU_ARCH=cuda
export MILABENCH_WORDIR="$(pwd)/$MILABENCH_GPU_ARCH"

export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/standard.yaml"
export MILABENCH_VENV="$MILABENCH_WORDIR/env"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"


install_prepare() {
    mkdir -p $MILABENCH_WORDIR
    cd $MILABENCH_WORDIR

    virtualenv $MILABENCH_WORDIR/env

    git clone https://github.com/mila-iqia/milabench.git -b intel
    git clone https://github.com/Delaunay/voir.git -b async_timer
    git clone https://github.com/Delaunay/torchcompat.git

    . $MILABENCH_WORDIR/env/bin/activate
    pip install -e $MILABENCH_WORDIR/milabench

    #
    # Install milabench's benchmarks in their venv
    #
    milabench install

    which pip
    pip install -e $MILABENCH_WORDIR/voir
    pip install -e $MILABENCH_WORDIR/torchcompat

    (
        . $BENCHMARK_VENV/bin/activate
        which pip
        pip install -e $MILABENCH_WORDIR/voir
        pip install -e $MILABENCH_WORDIR/torchcompat
        pip install torch torchvision torchaudio

        (
            cd $MILABENCH_WORDIR/milabench/benchmarks/timm/pytorch-image-models; 
            git fetch origin; 
            git checkout cb0e439
        )
    )

    #
    #   Generate/download datasets, download models etc...
    milabench prepare
}

if [ ! -d "$MILABENCH_WORDIR" ]; then
    install_prepare 
else
    echo "Reusing previous install"
    . $MILABENCH_WORDIR/env/bin/activate
fi

cd $MILABENCH_WORDIR


#
#   Run the benchmakrs
milabench run 

#
#   Display report
milabench report --runs $MILABENCH_WORDIR/results/runs
