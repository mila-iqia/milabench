
podman run --rm --device nvidia.com/gpu=all --storage-opt ignore_chown_errors=true --security-opt=label=disable --ipc=host -it -e HOME=$HOME -e USER=$USER -v $HOME:$HOME  nvcr.io/nvidia/pytorch:24.02-py3


cd /home/delaunay/
rm -rf env
pip install virtualenv
virtualenv --system-site-packages env
source ./env/bin/activate
pip install -U pip-tools
pip install -e milabench/

export MILABENCH_BASE="/home/delaunay/results"
export MILABENCH_CONFIG="/home/delaunay/milabench/config/standard.yaml"
# MILABENCH_GPU_ARCH="cuda" milabench pin --from-scratch --variant cuda -c constraints/cuda.txt

MILABENCH_GPU_ARCH="cuda" milabench install --use-current-env
pip uninstall torch torchvision torchaudio
MILABENCH_GPU_ARCH="cuda" milabench prepare --use-current-env
MILABENCH_GPU_ARCH="cuda" milabench run --use-current-env



