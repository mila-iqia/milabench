#!/bin/bash

# CPU only
# 16Gb



MILABENCH_GPU_ARCH=cuda milabench pin --config config/standard.yaml --from-scratch --base /tmp
MILABENCH_GPU_ARCH=rocm milabench pin --config config/standard.yaml --from-scratch --base /tmp


cd $SLURM_TMPDIR/milabench
git add --all
git commit -m "milabench pin"
git push $ORIGIN $BRANCH