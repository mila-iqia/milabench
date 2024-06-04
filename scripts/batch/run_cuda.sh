


export MILABENCH_GPU_ARCH=cuda
export MILABENCH_WORDIR="$(pwd)/$MILABENCH_GPU_ARCH"
export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/resnet50.yaml"

CUDA_VISIBLE_DEVICES=0 bash $MILABENCH_WORDIR/milabench/scripts/article/run_cuda.sh --config $MILABENCH_CONFIG --select resnet

CUDA_VISIBLE_DEVICES=0,1 bash $MILABENCH_WORDIR/milabench/scripts/article/run_cuda.sh --config $MILABENCH_CONFIG --select resnet

CUDA_VISIBLE_DEVICES=0,1,2,3 bash $MILABENCH_WORDIR/milabench/scripts/article/run_cuda.sh --config $MILABENCH_CONFIG --select resnet

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash $MILABENCH_WORDIR/milabench/scripts/article/run_cuda.sh --config $MILABENCH_CONFIG --select resnet
