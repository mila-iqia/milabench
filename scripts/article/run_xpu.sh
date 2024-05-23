#!/bin/bash

set -ex

export MILABENCH_GPU_ARCH=xpu
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

    # XPU manager is necessary
    wget -nv https://github.com/intel/xpumanager/releases/download/V1.2.36/xpumanager_1.2.36_20240428.081009.377f9162.u22.04_amd64.deb
    sudo dpkg -i xpumanager_1.2.36_20240428.081009.377f9162.u22.04_amd64.deb

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

        # Override dependencies for XPU
        pip install torch, torchvision torchaudio intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
    
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
