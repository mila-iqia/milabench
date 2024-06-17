#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

WORKERS=("2" "4" "8" "16" "32")
MEMORY_CAPACITY=("4Go" "8Go" "16Go" "32Go" "64Go" "80Go")
DRY=0
FINAL_OUTPUT="$HOME/batch_x_worker"

export MILABENCH_SIZER_SAVE="$FINAL_OUTPUT/scaling.yaml"
mkdir -p $FINAL_OUTPUT

module load cuda/12.3.2

#
# Install
#
if [ "$DRY" -eq 0 ]; then
    export MILABENCH_PREPARE=1
    source $SCRIPT_DIR/run_cuda.sh
fi

source $MILABENCH_WORDIR/env/bin/activate

maybe_run() {
    local name=$1
    local first_part=$(echo "$name" | cut -d'.' -f1)

    if ls -d "$FINAL_OUTPUT/$first_part".* 1> /dev/null 2>&1; then 
        echo "Skipping because folder exists $FINAL_OUTPUT/$name"
    else
        if [ "$DRY" -eq 1 ]; then
            mkdir -p dry
            echo $name
            milabench matrix --base output --config config/standard.yaml > dry/$name.yaml
        else
            echo "running $name"
            milabench prepare
            milabench run --run-name $name
            mv $MILABENCH_BASE/runs/* $FINAL_OUTPUT/
        fi  
    fi
}

#
# Default everything
#
export MILABENCH_CPU_AUTO=0
export MILABENCH_SIZER_AUTO=0
maybe_run "wdef-cdef.{time}"

#
# Auto everything
#
export MILABENCH_CPU_AUTO=1
export MILABENCH_SIZER_AUTO=1
export MILABENCH_SIZER_MULTIPLE=8
maybe_run "wauto-m8-cauto.{time}"

#
#   Multiple of 8
#
for CAPACITY in "${MEMORY_CAPACITY[@]}"; do
    for WORKER in "${WORKERS[@]}"; do
        export MILABENCH_CPU_AUTO=1
        export MILABENCH_CPU_N_WORKERS="$WORKER"
        
        export MILABENCH_SIZER_AUTO=1
        export MILABENCH_SIZER_MULTIPLE=8
        export MILABENCH_SIZER_CAPACITY="$CAPACITY"

        maybe_run "w$WORKER-m8-c$CAPACITY.{time}"
    done
done

#
#   Multiple of 32
#
for CAPACITY in "${MEMORY_CAPACITY[@]}"; do
    for WORKER in "${WORKERS[@]}"; do
        export MILABENCH_CPU_AUTO=1
        export MILABENCH_CPU_N_WORKERS="$WORKER"
        
        export MILABENCH_SIZER_AUTO=1
        export MILABENCH_SIZER_MULTIPLE=32
        export MILABENCH_SIZER_CAPACITY="$CAPACITY"

        maybe_run "w$WORKER-m32-c$CAPACITY.{time}"
    done
done

#
#   Power of 2
#
for CAPACITY in "${MEMORY_CAPACITY[@]}"; do
    for WORKER in "${WORKERS[@]}"; do
        export MILABENCH_CPU_AUTO=1
        export MILABENCH_CPU_N_WORKERS="$WORKER"
        
        export MILABENCH_SIZER_AUTO=1
        export MILABENCH_SIZER_MULTIPLE=0
        export MILABENCH_SIZER_POWER=2
        export MILABENCH_SIZER_CAPACITY="$CAPACITY"

        maybe_run "w$WORKER-p2-c$CAPACITY.{time}"
    done
done

