#!/bin/bash

#SBATCH --job-name=milabench
#SBATCH --output=/network/scratch/d/delaunap/milabench.out
#SBATCH --error=/network/scratch/d/delaunap/milabench.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=100Gb


#
# Setup Python
#
CONDA_PATH=$SLURM_TMPDIR/conda

/home/mila/d/delaunap/Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_PATH

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

pip install -e /home/mila/d/delaunap/milabench

#
# Run milabench
#
MILABENCH_BASE=$SLURM_TMPDIR/runs

milabench install config/standard.yaml --base $MILABENCH_BASE

milabench prepare config/standard.yaml --base $MILABENCH_BASE

milabench run config/standard.yaml --base $MILABENCH_BASE

milabench summary $MILABENCH_BASE/data/

milabench summary $MILABENCH_BASE/data/ -o summary.json

cp -r $SLURM_TMPDIR/runs /network/scratch/d/delaunap/runs

conda init bash --reverse