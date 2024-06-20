#!/usr/bin/env bash

#
# Cannot be split in multiple files because slurm copy the bash file to be executed
#

function usage() {
  echo "Usage: $0 [-m] [-p]"
  echo "  -h              Display this help message."
  echo "  -b arch         GPU arch           (default: cuda)"
  echo "  -b BRANCH       Branch to checkout (default: master)"
  echo "  -o ORIGIN       Origin to use      (default: github/mila/milabench)"
  echo "  -c CONFIG       Configuration      (default: milabench/config/standard.yaml)"
  echo "  -e ENV          Environment        (default: ./env)"
  echo "  -p PYTHON       Python version     (default: 3.9)"
  echo "  ARGUMENT        Any additional argument you want to process."
  exit 1
}

function parse_args() {
  ARCH="cuda"
  PYTHON="3.9"
  BRANCH="master"
  ORIGIN="https://github.com/mila-iqia/milabench.git"
  LOC="$SLURM_TMPDIR"
  CONFIG="$LOC/milabench/config/standard.yaml"
  BASE="$LOC/base"
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
      a)
          ARCH="$OPTARG"
          ;;
      l)
          # FIX ME
          LOC="$OPTARG"
          CONFIG="$LOC/milabench/config/standard.yaml"
          BASE="$LOC/base"
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

}

function conda_env() {
  #
  #   Fix problem with conda saying it is not "init properly"
  #
  CONDA_EXEC="$(which conda)"
  CONDA_BASE=$(dirname $CONDA_EXEC)
  source $CONDA_BASE/../etc/profile.d/conda.sh

  if [ -e $HOME/.credentials.env ]; then
    source $HOME/.credentials.env
  fi

  cd $LOC
  #
  #   Create a new environment
  #
  if [ ! -d "$ENV" ] && [ "$ENV" != "base" ] && [ ! -d "$CONDA_ENVS/$ENV" ]; then
      conda create --prefix $ENV python=$PYTHON -y
  fi
  conda activate $ENV
}

function setup() {
  export HF_HOME=$BASE/cache
  export HF_DATASETS_CACHE=$BASE/cache
  export TORCH_HOME=$BASE/cache
  export XDG_CACHE_HOME=$BASE/cache
  export MILABENCH_GPU_ARCH=$ARCH

  export MILABENCH_DASH=no 
  export PYTHONUNBUFFERED=1
  export MILABENCH_BASE=$BASE
  export MILABENCH_CONFIG=$CONFIG
  #
  # Fetch the repo
  #
  git clone --single-branch --depth 1 -b $BRANCH $ORIGIN
  python -m pip install -e ./milabench

  SYSTEM="$LOC/system.yaml"
}

function pin() {
  parse_args

  conda_env

  setup

  MILABENCH_GPU_ARCH=cuda milabench pin --config config/standard.yaml --from-scratch --base /tmp
  MILABENCH_GPU_ARCH=rocm milabench pin --config config/standard.yaml --from-scratch --base /tmp
  MILABENCH_GPU_ARCH=xpu milabench pin --config config/standard.yaml --from-scratch --base /tmp
  MILABENCH_GPU_ARCH=hpu milabench pin --config config/standard.yaml --from-scratch --base /tmp

  cd $SLURM_TMPDIR/milabench
  git add --all
  git commit -m "milabench pin"
  git push $ORIGIN $BRANCH

}

function run() {
  parse_args

  conda_env

  setup

  echo ""
  echo "System"
  echo "------"

  milabench slurm_system 
  milabench slurm_system > $SYSTEM

  module load gcc/9.3.0 
  module load cuda/11.8

  echo ""
  echo "Install"
  echo "-------"
  milabench install --config $CONFIG --system $SYSTEM --base $BASE $REMAINING_ARGS

  echo ""
  echo "Prepare"
  echo "-------"
  milabench prepare --config $CONFIG --system $SYSTEM --base $BASE $REMAINING_ARGS

  echo ""
  echo "Run"
  echo "---"
  milabench run     --config $CONFIG --system $SYSTEM --base $BASE $REMAINING_ARGS

  echo ""
  echo "Report"
  echo "------"

  milabench write_report_to_pr --remote $ORIGIN --branch $BRANCH --config $CONFIG

  echo "----"
  echo "Done after $SECONDS"
  echo ""
}