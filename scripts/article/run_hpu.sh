#!/bin/bash

set -ex

# Asssumes
# ./habanalabs-installer.sh install --type base
# pip install virtualen

export MILABENCH_GPU_ARCH=hpu
export MILABENCH_WORDIR="$(pwd)/$MILABENCH_GPU_ARCH"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_VENV="$MILABENCH_WORDIR/env"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"


if [ -z "${MILABENCH_SOURCE}" ]; then
    export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/standard.yaml"
else
    export MILABENCH_CONFIG="$MILABENCH_SOURCE/config/standard.yaml"
fi

if [ -z "${MILABENCH_PREPARE}" ]; then
    export MILABENCH_PREPARE=0
fi

ARGS="$@"

install_prepare() {
    mkdir -p $MILABENCH_WORDIR
    cd $MILABENCH_WORDIR

    virtualenv $MILABENCH_WORDIR/env

    if [ -z "${MILABENCH_SOURCE}" ]; then
        if [ ! -d "$MILABENCH_WORDIR/milabench" ]; then
            git clone https://github.com/mila-iqia/milabench.git
        fi
        export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
    fi

    git clone https://github.com/huggingface/optimum-habana.git -b v1.13.2

    # wget -nv https://vault.habana.ai/artifactory/gaudi-installer/1.15.1/habanalabs-installer.sh
    # wget -nv https://vault.habana.ai/artifactory/gaudi-installer/1.16.1/habanalabs-installer.sh
    wget -nv https://vault.habana.ai/artifactory/gaudi-installer/1.17.1/habanalabs-installer.sh
    chmod +x habanalabs-installer.sh

    . $MILABENCH_WORDIR/env/bin/activate
    pip install -e $MILABENCH_SOURCE

    which pip

    # Override dependencies for HPU
    # milabench needs pyhlml
    export HABANALABS_VIRTUAL_DIR=$MILABENCH_VENV
    ./habanalabs-installer.sh install -t dependencies --venv -y
    ./habanalabs-installer.sh install -t pytorch --venv -y

    #
    # Install milabench's benchmarks in their venv
    #
    # milabench pin --variant hpu --from-scratch $ARGS 
    milabench install $ARGS

    (
        . $BENCHMARK_VENV/bin/activate
        which pip
        pip install --no-deps -e $MILABENCH_WORDIR/optimum-habana 

        # Override dependencies for HPU
        # benchmarks need pytorch
        pip uninstall torch torchvision torchaudio -y
        export HABANALABS_VIRTUAL_DIR=$BENCHMARK_VENV
        ./habanalabs-installer.sh install -t dependencies --venv -y
        ./habanalabs-installer.sh install -t pytorch --venv -y

        if [ -z "${MILABENCH_HF_TOKEN}" ]; then
            echo "Missing token"
        else
            huggingface-cli login --token $MILABENCH_HF_TOKEN
        fi
    )

    #
    #   Generate/download datasets, download models etc...
    #
    milabench prepare $ARGS
}

if [ ! -d "$MILABENCH_WORDIR" ]; then
    install_prepare
else
    echo "Reusing previous install"
    . $MILABENCH_WORDIR/env/bin/activate
fi


if [ "$MILABENCH_PREPARE" -eq 0 ]; then
    cd $MILABENCH_WORDIR

    # python -c "import torch; print(torch.__version__)"
    # milabench prepare $ARGS

    #
    #   Run the benchmakrs
    milabench run $ARGS

    #
    #   Display report
    milabench report --runs $MILABENCH_WORDIR/results/runs
fi