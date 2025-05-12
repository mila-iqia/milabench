

sudo apt-get update -y
sudo apt-get install -y --no-install-recommends git build-essential curl python3 python-is-python3 python3-pip libgl1-mesa-glx libglib2.0-0 gcc g++ 

pip install -e milabench





#!/bin/bash


export MILABENCH_BASE=/home/milabench/workspace/results
export MILABENCH_CONFIG=/home/milabench/workspace/milabench/config/standard.yaml
export MILABENCH_GPU_ARCH=cuda
export MILABENCH_HF_TOKEN=REPLACE_ME

# CUDA versions to test
CUDA_VERSIONS=("cu126" "cu128" "cu118")

# PyTorch versions to test
PYTORCH_VERSIONS=("2.4.0" "2.5.0" "2.6.0" "2.7.0")

# Activate virtualenv path
VENV_PATH="$MILABENCH_BASE/venv/torch/bin/activate"


milabench install

milabench prepare 

for cuda in "${CUDA_VERSIONS[@]}"; do
  for torch_version in "${PYTORCH_VERSIONS[@]}"; do
    echo "=== Installing torch~=$torch_version with CUDA $cuda ==="
    (
      . "$VENV_PATH"
      # Use compatible release operator to get latest patch version of minor
      pip install "torch~=$torch_version" torchvision torchaudio --index-url "https://download.pytorch.org/whl/$cuda"
      pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-$torch_version+$cuda.html
    )

    VERSION=$( . "$VENV_PATH" && python -c "import torch; print(torch.__version__)" )
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    RUN_NAME="${VERSION}.${TIMESTAMP}"
    echo "=== Running milabench: $RUN_NAME ==="
    milabench run --name "$RUN_NAME" --select pna,dimenet,recursiongfn,dinov2-giant-single,dinov2-giant-gpus,diffusion-single,diffusion-gpus
  done
done
