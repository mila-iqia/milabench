
module load cuda/12.3.2


export MILABENCH_BASE=output/

MILABENCH_GPU_ARCH=cuda milabench pin -c constraints/cuda.txt --config config/standard.yaml --from-scratch
MILABENCH_GPU_ARCH=rocm milabench pin -c constraints/rocm.txt --config config/standard.yaml --from-scratch
MILABENCH_GPU_ARCH=xpu milabench pin -c constraints/xpu.txt --config config/standard.yaml --from-scratch
MILABENCH_GPU_ARCH=hpu milabench pin -c constraints/hpu.txt --config config/standard.yaml --from-scratch
