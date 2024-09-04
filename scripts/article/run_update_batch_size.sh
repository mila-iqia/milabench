



export MILABENCH_SIZER_AUTO=1
export MILABENCH_SIZER_BATCH_SIZE=1
FINAL_OUTPUT="$HOME/batch_x_worker"
export MILABENCH_SIZER_SAVE="$FINAL_OUTPUT/scaling.yaml"
milabench run --system $MILABENCH_WORDIR/system.yaml --exclude llama

export MILABENCH_SIZER_AUTO=1
export MILABENCH_SIZER_BATCH_SIZE=2
FINAL_OUTPUT="$HOME/batch_x_worker"
export MILABENCH_SIZER_SAVE="$FINAL_OUTPUT/scaling.yaml"
milabench run --system $MILABENCH_WORDIR/system.yaml --exclude llama

export MILABENCH_SIZER_AUTO=1
export MILABENCH_SIZER_BATCH_SIZE=4
FINAL_OUTPUT="$HOME/batch_x_worker"
export MILABENCH_SIZER_SAVE="$FINAL_OUTPUT/scaling.yaml"
milabench run --system $MILABENCH_WORDIR/system.yaml --exclude llama

export MILABENCH_SIZER_AUTO=1
export MILABENCH_SIZER_BATCH_SIZE=8
FINAL_OUTPUT="$HOME/batch_x_worker"
export MILABENCH_SIZER_SAVE="$FINAL_OUTPUT/scaling.yaml"
milabench run --system $MILABENCH_WORDIR/system.yaml --exclude llama

export MILABENCH_SIZER_AUTO=1
export MILABENCH_SIZER_BATCH_SIZE=16
FINAL_OUTPUT="$HOME/batch_x_worker"
export MILABENCH_SIZER_SAVE="$FINAL_OUTPUT/scaling.yaml"
milabench run --system $MILABENCH_WORDIR/system.yaml --exclude llama