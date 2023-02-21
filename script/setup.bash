#
#   Setup milabench
#
set -x

ARCH=${ARCH:-cuda}
CONFIG=${CONFIG:-standard-cuda.yaml}
MILABENCH_ARGS=${MILABENCH_ARGS:-""}
WORKDIR=${WORKDIR:-/milabench}
CONDA_PATH=${CONDA_PATH:-/opt/anaconda}

export CONDA_PATH="$CONDA_PATH"
export MILABENCH_GPU_ARCH="$ARCH"
export MILABENCH_CONFIG_NAME="$CONFIG"
export MILABENCH_CONFIG=$WORKDIR/milabench/config/$MILABENCH_CONFIG_NAME
export MILABENCH_BASE=$WORKDIR/envs
export MILABENCH_OUTPUT=$WORKDIR/results/
export MILABENCH_ARGS="$MILABENCH_ARGS"

SCRIPTDIR=$( dirname -- "${BASH_SOURCE[0]}" )

. $SCRIPTDIR/dependencies.bash

#
# Install Milabench
#
python -m pip install -e $WORKDIR/milabench/

#
#   Prepare Benchmarks
#   

milabench install $MILABENCH_CONFIG --base $MILABENCH_BASE $MILABENCH_ARGS  

milabench prepare $MILABENCH_CONFIG --base $MILABENCH_BASE $MILABENCH_ARGS

#
#   Override version
#
. $SCRIPTDIR/nightly.bash
