#!/bin/bash

#SBATCH --job-name=milabench
#SBATCH --output=/network/scratch/d/delaunap/milabench.out
#SBATCH --error=/network/scratch/d/delaunap/milabench.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --mem=100Gb

MILABENCH_BASE=$SLURM_TMPDIR/runs

milabench install config/standard.yaml --base $MILABENCH_BASE

milabench prepare config/standard.yaml --base $MILABENCH_BASE

milabench run config/standard.yaml --base $MILABENCH_BASE

milabench summary $MILABENCH_BASE/data/

milabench summary $MILABENCH_BASE/data/ -o summary.json

cp -r $SLURM_TMPDIR/runs /network/scratch/d/delaunap/runs