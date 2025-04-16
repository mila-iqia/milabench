




mkdir -p /tmp/workspace
# Does not work in a bash script
# conda activate py310

# Rsync data to local folder and link the venv
export MILABENCH_BASE=/tmp/workspace/
export MILABENCH_GPU_ARCH=cuda
export MILABENCH_CONFIG=~/scratch/shared/milabench/config/standard.yaml

milabench sharedsetup --network /home/mila/d/delaunap/scratch/shared/ --local $MILABENCH_BASE

# Those 2 commands should execute faster now (it will only verify)
# milabench install
# milabench prepare 

# Runs
milabench run --exclude nobatch --system /home/mila/d/delaunap/scratch/shared/system.yaml

rsync -av /tmp/workspace/runs/ /home/mila/d/delaunap/scratch/shared/runs