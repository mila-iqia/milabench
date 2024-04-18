

export MILABENCH_BASE="/home/delaunay/results"
export MILABENCH_CONFIG="/home/delaunay/milabench/config/standard.yaml"
MILABENCH_GPU_ARCH="cuda" milabench pin --from-scratch --variant cuda -c constraints/cuda.txt

MILABENCH_GPU_ARCH="cuda" milabench install
MILABENCH_GPU_ARCH="cuda" milabench prepare
MILABENCH_GPU_ARCH="cuda" milabench run

