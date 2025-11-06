# export CUDA_HOME=/usr/local/cuda
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export VENV_BIN=/milabench/env/bin/
export FORCE_CUDA=1
export GIT_ARGS="--depth 1 --recursive"
# -- XFORMERS

cd /tmp
git clone $GIT_ARGS https://github.com/facebookresearch/xformers.git
$VENV_BIN/pip install wheel setuptools cmake ninja

cd /tmp/xformers
$VENV_BIN/pip install --no-build-isolation --no-deps -v --force-reinstall .
$VENV_BIN/python -m xformers.info

rm -rf /tmp/xformers

# -- Pytorch GEOMETRIC
cd /tmp
git clone $GIT_ARGS https://github.com/pyg-team/pytorch_geometric.git
cd /tmp/pytorch_geometric

$VENV_BIN/pip install --no-build-isolation --no-deps -v --force-reinstall .

rm -rf /tmp/pytorch_geometric


# -- Pytorch SCATTER
cd /tmp
git clone $GIT_ARGS https://github.com/rusty1s/pytorch_scatter.git
cd /tmp/pytorch_scatter

$VENV_BIN/pip install --no-build-isolation --no-deps -v --force-reinstall .

rm -rf /tmp/pytorch_scatter

# -- Pytorch SPARSE
cd /tmp
git clone $GIT_ARGS https://github.com/rusty1s/pytorch_sparse.git
cd /tmp/pytorch_sparse

$VENV_BIN/pip install --no-build-isolation --no-deps -v --force-reinstall .

rm -rf /tmp/pytorch_sparse

# -- Pytorch CLUSTER
cd /tmp
git clone $GIT_ARGS https://github.com/rusty1s/pytorch_cluster.git
cd /tmp/pytorch_cluster

$VENV_BIN/pip install --no-build-isolation --no-deps -v --force-reinstall .

rm -rf /tmp/pytorch_cluster
