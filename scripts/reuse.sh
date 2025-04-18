




mkdir -p /tmp/workspace

 module load cuda/12.6.0
# Does not work in a bash script
# conda activate py310

# Rsync data to local folder and link the venv
export MILABENCH_BASE=/tmp/workspace/
export MILABENCH_GPU_ARCH=cuda
export MILABENCH_CONFIG=~/scratch/shared/milabench/config/standard.yaml
export MILABENCH_MULTIRUN_CACHE=~/scratch/shared/progress.txt
export MILABENCH_DIR_DATA=/tmp/workspace/data
export XDG_CACHE_HOME=/tmp/workspace/cache
export BENCHMARKS=dimenet,pna,recursiongfn
# milabench sharedsetup --network /home/mila/d/delaunap/scratch/shared/ --local $MILABENCH_BASE

# Those 2 commands should execute faster now (it will only verify)
# MILABENCH_USE_UV=1 milabench install --force
# 

(
    source /tmp/workspace/venv/torch/bin/activate

    #pip install torch_geometric
    #pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
    # pip install rdkit
    pip install botorch
)

# milabench prepare --select $BENCHMARKS

# Runs
milabench run --select $BENCHMARKS --exclude nobatch --system /home/mila/d/delaunap/scratch/shared/system.yaml

rsync -av /tmp/workspace/runs/ /home/mila/d/delaunap/scratch/shared/runs