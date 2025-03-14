#!/bin/bash

set -ex

module load gcc/13.3
module load cuda/12.6


export NETWORK_BASE=/home/d/delaunay/links/scratch/shared
export PYTHONUNBUFFERED=1
export MILABENCH_CONFIG=$NETWORK_BASE/milabench/config/standard.yaml
export MILABENCH_GPU_ARCH=cuda
export MILABENCH_BASE=/tmp/workspace/


(
    export MAX_JOBS=24
    export TORCH_CUDA_ARCH_LIST="9.0"

    nvcc --version
    source $NETWORK_BASE/venv/torch/bin/activate

    pip install nvidia-ml-py


    # pip uninstall torch_cluster torch_scatter torch_sparse -y

    # FORCE_ONLY_CUDA=1 pip install -U -v --use-pep517 --no-build-isolation git+https://github.com/rusty1s/pytorch_cluster.git
    # FORCE_ONLY_CUDA=1 pip install -U -v --use-pep517 --no-build-isolation git+https://github.com/rusty1s/pytorch_scatter.git
    # FORCE_ONLY_CUDA=1 pip install -U -v --use-pep517 --no-build-isolation git+https://github.com/rusty1s/pytorch_sparse.git
)
