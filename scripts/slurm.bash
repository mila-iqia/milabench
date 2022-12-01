#!/bin/bash

#SBATCH --job-name=milabench
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=100Gb

#
# Configuration
#
# Tweak those for your system
WORKING_DIR=$SLURM_TMPDIR
OUTPUT_DIR=$SCRATCH
#

MILABENCH_CONFIG=$WORKING_DIR/milabench/config/standard.yaml
MILABENCH_BASE=$WORKING_DIR/runs
MILABENCH_OUTPUT=$OUTPUT_DIR/runs
MILABENCH_ARGS=""

MILABENCH_REPO=git@github.com:mila-iqia/milabench.git
MILABENCH_BRANCH="master"


#
# Setup Python
#
CONDA_PATH=$WORKING_DIR/conda
cd $WORKING_DIR

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_PATH

ls $CONDA_PATH/conda/bin
export PATH="$CONDA_PATH/bin:$PATH"
conda init bash

__conda_setup="$(\"$CONDA_PATH/bin/conda\" 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
        . "$CONDA_PATH/etc/profile.d/conda.sh"
    else
        export PATH="$CONDA_PATH/conda/bin:$PATH"
    fi
fi
unset __conda_setup

conda create -n milabench -y
conda activate milabench
python -m pip install pip -U

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

# Cleanup
conda init bash --reverse
