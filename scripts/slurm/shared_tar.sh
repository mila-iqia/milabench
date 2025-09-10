#!/bin/bash

#
# Tar the data and cache directory for faster transfer to local disk
#

export MILABENCH_GPU_ARCH=cuda
export COMPRESSED=0

set -ex

# ===
OUTPUT_DIRECTORY=$(scontrol show job "$SLURM_JOB_ID" --json | jq -r '.jobs[0].standard_output' | xargs dirname)
mkdir -p $OUTPUT_DIRECTORY/meta
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===

CONDA_EXEC="$(which conda)"
CONDA_BASE=$(dirname $CONDA_EXEC)
source $CONDA_BASE/../etc/profile.d/conda.sh

export MILABENCH_WORDIR="$HOME/scratch/shared/$MILABENCH_GPU_ARCH"
export MILABENCH_BASE="$MILABENCH_WORDIR"

if [ $COMPRESSED -eq 1 ]; then
    tar_flags="-czvf"
    extension="tar.gz"
else
    tar_flags="-cf"
    extension="tar"
fi

tar $tar_flags $MILABENCH_WORDIR/data.$extension -C $MILABENCH_WORDIR data
tar $tar_flags $MILABENCH_WORDIR/cache.$extension -C $MILABENCH_WORDIR cache

# ===
scontrol show job --json $SLURM_JOB_ID | jq '.jobs[0]' > $OUTPUT_DIRECTORY/meta/info.json
# ===