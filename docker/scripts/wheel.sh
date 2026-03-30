
PYTORCH_VERSION="2.10.0"
CUDA="cu130"

pip install --extra-index-url https://download.pytorch.org/whl/$CUDA torch==$PYTORCH_VERSION+$CUDA

CLUSTER_VERSION=1.6.3
SPARSE_VERSION=0.6.18
SCATTER_VERSION=2.1.2
TORCH_AO_VERSION=0.16.0
XFORMER_VERSION=0.0.35
PT_VER="pt210"

export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0;10.0;12.0;12.1"

pip install wheel setuptools cmake ninja

(
    git clone --recurse-submodules --branch v$XFORMER_VERSION https://github.com/facebookresearch/xformers.git
    cd xformers

    # No cuda+130 for this
    BUILD_VERSION="$XFORMER_VERSION" FORCE_CUDA=1 pip wheel . --no-cache-dir --no-deps --no-build-isolation -w ../wheels/
    cd ..
    rm -rf xformers
)

(
    git clone --recurse-submodules --branch $CLUSTER_VERSION --depth 1 https://github.com/rusty1s/pytorch_cluster.git
    cd pytorch_cluster
    sed -i "s/__version__ = '$CLUSTER_VERSION'/__version__ = '$CLUSTER_VERSION+${PT_VER}${CUDA}'/" setup.py

    FORCE_CUDA=1 pip wheel . --no-cache-dir --no-deps --no-build-isolation -w ../wheels/
    cd ..
    rm -rf pytorch_cluster
)

(
    # torch-sparse
    git clone --recurse-submodules --branch $SPARSE_VERSION --depth 1 https://github.com/rusty1s/pytorch_sparse.git
    cd pytorch_sparse
    sed -i "s/__version__ = '$SPARSE_VERSION'/__version__ = '$SPARSE_VERSION+${PT_VER}${CUDA}'/" setup.py
    FORCE_CUDA=1 pip wheel . --no-cache-dir --no-deps --no-build-isolation -w ../wheels/
    cd ..
    rm -rf pytorch_sparse
)

(
    # torch-scatter
    git clone --recurse-submodules --branch $SCATTER_VERSION --depth 1 https://github.com/rusty1s/pytorch_scatter.git
    cd pytorch_scatter
    sed -i "s/__version__ = '$SCATTER_VERSION'/__version__ = '$SCATTER_VERSION+${PT_VER}${CUDA}'/" setup.py
    FORCE_CUDA=1 pip wheel . --no-cache-dir --no-deps --no-build-isolation -w ../wheels/
    cd ..
    rm -rf pytorch_scatter
)

(
    # torchao
    git clone --recurse-submodules --branch v$TORCH_AO_VERSION --depth 1 https://github.com/pytorch/ao.git
    cd ao
    FORCE_CUDA=1 VERSION_SUFFIX=+cu130 pip wheel . --no-cache-dir --no-deps --no-build-isolation -w ../wheels/
    cd ..
    rm -rf ao
)


gh release upload cu130-wheels wheels/*.whl --repo milabench/wheels