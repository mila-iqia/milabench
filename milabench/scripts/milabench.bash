#!/bin/bash

function usage() {
  echo "Usage: $0 [-m] [-p]"
  echo "  -h              Display this help message."
  echo "  -b BRANCH       Branch to checkout (default: master)"
  echo "  -o ORIGIN       Origin to use      (default: github/mila/milabench)"
  echo "  -c CONFIG       Configuration      (default: milabench/config/standard.yaml)"
  echo "  -e ENV          Environment        (default: ./env)"
  echo "  -p PYTHON       Python version     (default: 3.9)"
  echo "  ARGUMENT        Any additional argument you want to process."
  exit 1
}

PYTHON="3.9"
BRANCH="master"
ORIGIN="https://github.com/mila-iqia/milabench.git"
CONFIG="$SLURM_TMPDIR/milabench/config/standard.yaml"
BASE="$SLURM_TMPDIR/base"
ENV="./env"
REMAINING_ARGS=""

while getopts ":hm:p:e:b:o:c:" opt; do
  case $opt in
    h)
      usage
      ;;
    p)
        PYTHON="$OPTARG"
        ;;
    b)
        BRANCH="$OPTARG"
        ;;
    o)
        ORIGIN="$OPTARG"
        ;;
    c)
        CONFIG="$OPTARG"
        ;;
    e)
        ENV="$OPTARG"
        ;;
    :)
        echo "Option -$OPTARG requires an argument." >&2
        usage
        ;;
  esac
done

shift "$((OPTIND-1))"
REMAINING_ARGS="$@"

echo "  PYTHON: $PYTHON"
echo "  branch: $BRANCH"
echo "  origin: $ORIGIN"
echo "  config: $CONFIG"
echo "     env: $ENV"
echo "    args: $REMAINING_ARGS"
#
#   Fix problem with conda saying it is not "init properly"
#
CONDA_EXEC="$(which conda)"
CONDA_BASE=$(dirname $CONDA_EXEC)
source $CONDA_BASE/../etc/profile.d/conda.sh

#
#   Create a new environment
#
if [ ! -d "$ENV" ] && [ "$ENV" != "base" ] && [ ! -d "$CONDA_ENVS/$ENV" ]; then
     conda create --prefix $ENV python=$PYTHON -y
fi
conda activate $ENV

#
# Fetch the repo
#
cd $SLURM_TMPDIR
git clone --single-branch -d 1 -b $BRANCH $ORIGIN
python -m pip install milabench

SYSTEM="$SLURM_TMPDIR/system.yaml"

milabench slurm_system 
milabench slurm_system > $SYSTE

milabench install --config $CONFIG --system $SYSTEM --base $BASE $REMAINING_ARGS
milabench prepare --config $CONFIG --system $SYSTEM --base $BASE $REMAINING_ARGS
milabench run     --config $CONFIG --system $SYSTEM --base $BASE $REMAINING_ARGS

milabench summary $SLURM_TMPDIR/base/runs/
