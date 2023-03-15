#!/bin/bash

#SBATCH --reservation=milabench
#SBATCH --job-name=milabench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=100Gb

set -ev

#
# Configuration
#
# Tweak those for your system
WORKING_DIR=${SLURM_TMPDIR:-/tmp/milabench/$$}
OUTPUT_DIR=${SCRATCH:-output/$$}
#
export MILABENCH_GPU_ARCH=cuda

MILABENCH_CONFIG=$WORKING_DIR/milabench/config/standard-cuda.yaml
MILABENCH_BASE=$WORKING_DIR/runs
MILABENCH_OUTPUT=$OUTPUT_DIR/runs
MILABENCH_ARGS=""

MILABENCH_REPO=git@github.com:mila-iqia/milabench.git
MILABENCH_BRANCH="master"

mkdir -p $WORKING_DIR

#
# Setup Python
#
CONDA_PATH=$WORKING_DIR/conda
cd $WORKING_DIR

#
#   Rust
#
if [ ! -d ~/.cargo/bin ]
then
    curl https://sh.rustup.rs -sSf | sh -s -- -y
fi
export PATH="~/.cargo/bin:${PATH}"
# <<<<

#
# Anaconda
# 
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh
chmod +x Miniconda3-py39_23.1.0-1-Linux-x86_64.sh
./Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -b -p $CONDA_PATH

ls $CONDA_PATH/bin
export PATH="$CONDA_PATH/bin:$PATH"
. "$CONDA_PATH/etc/profile.d/conda.sh"

conda create -n milabench -y
conda activate milabench
python -m pip install pip -U
# <<<<

# Install Mila bench
cd $WORKING_DIR
git clone --depth 1 $MILABENCH_REPO --branch $MILABENCH_BRANCH
python -m pip install -e milabench

#
# Run milabench
#

milabench install $MILABENCH_CONFIG --base $MILABENCH_BASE $MILABENCH_ARGS
milabench prepare $MILABENCH_CONFIG --base $MILABENCH_BASE $MILABENCH_ARGS
milabench run $MILABENCH_CONFIG --base $MILABENCH_BASE $MILABENCH_ARGS

milabench summary $WORKING_DIR/runs/runs/
milabench summary $WORKING_DIR/runs/runs/ -o $MILABENCH_OUTPUT/summary.json

# Save data
cp -r $WORKING_DIR/runs/runs $MILABENCH_OUTPUT
