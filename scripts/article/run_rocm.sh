#!/bin/bash

set -ex

# sudo usermod -a -G render,video $LOGNAME
# sudo chmod u+s /opt/rocm-6.2.2/lib/llvm/bin/amdgpu-arch
# sudo apt-get install g++-12 libstdc++-12-dev


export MILABENCH_GPU_ARCH=rocm
export MILABENCH_WORDIR="$(pwd)/$MILABENCH_GPU_ARCH"
export ROCM_PATH="/opt/rocm"
export MILABENCH_BASE="$MILABENCH_WORDIR/results"
export MILABENCH_VENV="$MILABENCH_WORDIR/env"
export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"
export MILABENCH_SIZER_SAVE="$MILABENCH_WORDIR/scaling.yaml"

if [ -z "${MILABENCH_SOURCE}" ]; then
    export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/standard.yaml"
else
    export MILABENCH_CONFIG="$MILABENCH_SOURCE/config/standard.yaml"
fi

export GPU="$(/opt/rocm/lib/llvm/bin/amdgpu-arch | head -n 1)"
export TORCH_ROCM_ARCH_LIST="$GPU"
export ROCM_TARGETS="$GPU"
export PYTORCH_ROCM_ARCH="$GPU"
export MAX_JOBS=24


ARGS="$@"

#
# DOES NOT WORK with autotune
#
export XLA_FLAGS=--xla_gpu_autotune_level=0

install_jax() {
    (
        . $BENCHMARK_VENV/bin/activate

        # Jax 0.4.30
        # https://github.com/ROCm/jax/releases/tag/rocm-jaxlib-v0.4.30
        # pip install https://github.com/ROCm/jax/releases/download/rocm-jaxlib-v0.4.30/jaxlib-0.4.30+rocm611-cp310-cp310-manylinux2014_x86_64.whl
        # pip install https://github.com/ROCm/jax/archive/refs/tags/rocm-jaxlib-v0.4.30.tar.gz

        # Jax 0.5
        # https://github.com/ROCm/jax/releases/download/rocm-jax-v0.5.0/jaxlib-0.5.0-cp310-cp310-manylinux_2_28_x86_64.whl

        pip install https://github.com/ROCm/jax/archive/refs/tags/rocm-jax-v0.5.0.tar.gz

        pip install https://github.com/ROCm/jax/releases/download/rocm-jax-v0.5.0/jaxlib-0.5.0-cp310-cp310-manylinux_2_28_x86_64.whl
        pip install https://github.com/ROCm/jax/releases/download/rocm-jax-v0.5.0/jax_rocm60_pjrt-0.5.0-py3-none-manylinux_2_28_x86_64.whl \
                    https://github.com/ROCm/jax/releases/download/rocm-jax-v0.5.0/jax_rocm60_plugin-0.5.0-cp310-cp310-manylinux_2_28_x86_64.whl
                
        # pip install jax[rocm]
        # pip freeze | grep jax
    )
}

install_graph() {
    pip uninstall torch_cluster torch_scatter torch_sparse -y
    FORCE_ONLY_CUDA=1 pip install -U -v --use-pep517 --no-build-isolation git+https://github.com/rusty1s/pytorch_cluster.git
    FORCE_ONLY_CUDA=1 pip install -U -v --use-pep517 --no-build-isolation git+https://github.com/rusty1s/pytorch_scatter.git
    FORCE_ONLY_CUDA=1 pip install -U -v --use-pep517 --no-build-isolation git+https://github.com/rusty1s/pytorch_sparse.git
}


install_override() {
  (
        . $BENCHMARK_VENV/bin/activate

        pip install ninja

        if [ -z "${MILABENCH_HF_TOKEN}" ]; then
            echo "Missing token"
        else
            huggingface-cli login --token $MILABENCH_HF_TOKEN
        fi

        #
        # Override/add package to the benchmark venv here
        #
        which pip
        export MAX_JOBS=24

        install_jax

        install_graph

        # takes forever to compile
        # https://github.com/ROCm/xformers
        # pip uninstall xformers -y
        # pip install "torch<2.6" --index-url https://download.pytorch.org/whl/rocm6.2
        # pip install xformers==0.0.29 --index-url https://download.pytorch.org/whl/rocm6.2
        # pip install -v -U --no-build-isolation --no-deps git+https://github.com/ROCm/xformers.git@develop#egg=xformers
        # pip install -v -U --no-build-isolation --no-deps git+https://github.com/facebookresearch/xformers.git
        # pip install xformers -U --index-url https://download.pytorch.org/whl/rocm6.1

        pip uninstall -y flash-attention
        pip install -v -U --no-build-isolation --use-pep517 --no-deps git+https://github.com/ROCm/flash-attention.git 
        pip uninstall pynvml nvidia-ml-py -y
        pip install einops
    )
}

install_prepare() {
    mkdir -p $MILABENCH_WORDIR
    cd $MILABENCH_WORDIR

    virtualenv $MILABENCH_WORDIR/env

    if [ -z "${MILABENCH_SOURCE}" ]; then
        if [ ! -d "$MILABENCH_WORDIR/milabench" ]; then
            git clone https://github.com/mila-iqia/milabench.git -b rocm
        fi
        export MILABENCH_SOURCE="$MILABENCH_WORDIR/milabench"
    fi

    . $MILABENCH_WORDIR/env/bin/activate

    pip install -e $MILABENCH_SOURCE

    
    #
    # Install milabench's benchmarks in their venv
    #
    # pip install torch --index-url https://download.pytorch.org/whl/rocm6.2.4
    # milabench pin --variant rocm --from-scratch $ARGS 
    
    milabench install $ARGS 

    #
    # Override/add package to milabench venv here
    #
    which pip
    pip uninstall pynvml

    install_override

    pip uninstall pynvml nvidia-ml-py -y
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




(
    . $MILABENCH_WORDIR/env/bin/activate
    pip install git+https://github.com/breuleux/voir.git
    pip install -e /home/testroot/milabench/benchmate
    (
        . $BENCHMARK_VENV/bin/activate
        pip install git+https://github.com/breuleux/voir.git
        pip install -e /home/testroot/milabench/benchmate
    )

)

# (
#     # . $BENCHMARK_VENV/bin/activate
#     # pip install xformers --index-url https://download.pytorch.org/whl/rocm6.1
# )

# milabench prepare $ARGS # --system $MILABENCH_WORDIR/system.yaml

# milabench prepare $ARGS # --system $MILABENCH_WORDIR/system.yaml
# install_prepare

#
#   Run the benchmakrs
milabench run $ARGS # --system $MILABENCH_WORDIR/system.yaml


#
#   Display report
# milabench report --runs $MILABENCH_WORDIR/results/runs

# rocm