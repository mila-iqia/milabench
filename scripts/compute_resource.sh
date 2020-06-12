#!/usr/bin/env bash

set -e

if [[ -z "${RESOURCE_COMPUTED}" ]]; then
    # System Stats
    export DEVICE_TOTAL=0
    export DEVICE_TOTAL=$(python -c "import torch; print(torch.cuda.device_count())")
    export CPU_TOTAL=$(nproc)
    export RAM_TOTAL=$(cat /proc/meminfo | head -n 1 | grep -oP '[0-9]*')
    export USE_CUDA=$(python -c "import torch; print(int(torch.cuda.is_available()))")
    # ---------------

    export ALT_CPU=$(cat /proc/cpuinfo | grep processor | wc -l)
    if [[ $CPU_TOTAL -eq 1 ]]; then
        export CPU_TOTAL=$ALT_CPU
    fi

    if [[ $DEVICE_TOTAL != 0 ]]; then
        export RAM_CONSTRAINT=$(($RAM_TOTAL / $DEVICE_TOTAL))
        # Some CPUs may be left out if there is not a whole number of
        # processors per GPU device, although the OS can make use of them.
        export CPU_COUNT=$(($CPU_TOTAL / $DEVICE_TOTAL))
    else
        export RAM_CONSTRAINT=$(($RAM_TOTAL / 8))
        export CPU_COUNT=$(($CPU_TOTAL / 2))
    fi

    export RESOURCE_COMPUTED=1
fi
