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

    (
        . $BENCHMARK_VENV/bin/activate
        which pip

        # Override dependencies for XPU
        pip uninstall torch torchvision torchaudio
        pip install torch torchvision torchaudio intel-extension-for-pytorch oneccl_bind_pt intel-extension-for-pytorch-deepspeed --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
    )

    #
    #   Generate/download datasets, download models etc...
    #
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
milabench run "$@"

#
#   Display report
milabench report --runs $MILABENCH_WORDIR/results/runs
