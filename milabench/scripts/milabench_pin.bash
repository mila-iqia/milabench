#!/bin/bash

# CPU only
# 16Gb



MILABENCH_GPU_ARCH=cuda milabench pin -c constraints/cuda.txt --config config/standard.yaml 
MILABENCH_GPU_ARCH=rocm milabench pin -c constraints/rocm.txt --config config/standard.yaml 


cd $SLURM_TMPDIR/milabench
git add --all
git commit -m "milabench pin"
git push $ORIGIN $BRANCH