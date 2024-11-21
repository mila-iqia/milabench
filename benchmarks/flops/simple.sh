



export MILABENCH_BASE="$(pwd)/dev"
export MILABENCH_CONFIG="$(pwd)/dev.yaml"


milabench install  --select fp32

milabench prepare  --select fp32

milabench run --select fp32
