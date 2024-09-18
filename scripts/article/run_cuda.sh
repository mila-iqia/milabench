#!/bin/bash

set -ex

export MILABENCH_GPU_ARCH=cuda
export MILABENCH_WORDIR="$(pwd)/$MILABENCH_GPU_ARCH"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"

export MILABENCH_VENV="$MILABENCH_WORDIR/env"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/scaling.yaml"


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
        virtualenv $MILABENCH_WORDIR/env
    fi

    if [ -z "${MILABENCH_SOURCE}" ]; then
        if [ ! -d "$MILABENCH_WORDIR/milabench" ]; then
            git clone https://github.com/mila-iqia/milabench.git
        fi
        export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
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

module load cuda/12.3.2

if [ ! -d "$MILABENCH_WORDIR/results" ]; then
    install_prepare 
else
    echo "Reusing previous install"
    . $MILABENCH_WORDIR/env/bin/activate
fi

if [ "$MILABENCH_PREPARE" -eq 0 ]; then
    cd $MILABENCH_WORDIR

    . $MILABENCH_WORDIR/env/bin/activate

    # milabench install --system $MILABENCH_WORDIR/system.yaml
    # milabench prepare --system $MILABENCH_WORDIR/system.yaml $ARGS

    # pip install torch
    milabench pin --variant cuda --from-scratch $ARGS 
    milabench install --system $MILABENCH_WORDIR/system.yaml --force $ARGS
    milabench prepare --system $MILABENCH_WORDIR/system.yaml $ARGS

    ARGS="--select resnet50-noio,brax,lightning,dinov2-giant-single,dinov2-giant-gpus,llm-lora-ddp-gpus,llm-lora-ddp-nodes,llm-lora-mp-gpus,llm-full-mp-gpus,llm-full-mp-nodes,dqn,ppo,dimenet,llava-single,rlhf-single,rlhf-gpus,vjepa-single,vjepa-gpus"

    MEMORY_CAPACITY=("4Go" "8Go" "16Go" "32Go" "64Go" "80Go")
    # MEMORY_CAPACITY=("2048" "4096" "8192")

    #   Run the benchmakrs 
    for CAPACITY in "${MEMORY_CAPACITY[@]}"; do
        export MILABENCH_SIZER_AUTO=1
        export MILABENCH_SIZER_MULTIPLE=8
        export MILABENCH_SIZER_CAPACITY=$CAPACITY
        # export MILABENCH_SIZER_BATCH_SIZE=$CAPACITY
        milabench run --run-name "bs$CAPACITY.{time}" --system $MILABENCH_WORDIR/system.yaml $ARGS|| true
    done

    #
    #   Display report
    milabench report --runs $MILABENCH_WORDIR/results/runs
fi