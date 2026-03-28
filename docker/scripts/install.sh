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


sudo apt-get install pybind11-dev ffmpeg
I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 pip install "git+https://github.com/pytorch/torchcodec.git@release/0.10" --no-build-isolation
