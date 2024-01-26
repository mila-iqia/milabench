

WORKSPACE="$(pwd)/workspace"


ARCH="cuda"
PYTHON="3.9"
BRANCH="master"
ORIGIN="https://github.com/mila-iqia/milabench.git"
LOC="$SLURM_TMPDIR"
CONFIG="$(pwd)/config/standard.yaml"
BASE="$WORKSPACE"

export HF_HOME=$BASE/cache
export HF_DATASETS_CACHE=$BASE/cache
export TORCH_HOME=$BASE/cache
export XDG_CACHE_HOME=$BASE/cache

export MILABENCH_GPU_ARCH=$ARCH
export MILABENCH_DASH=no 
export PYTHONUNBUFFERED=1
export MILABENCH_BASE=$BASE
export MILABENCH_CONFIG=$CONFIG

# . scripts/schedule.sh && milabench run --select resnet50
