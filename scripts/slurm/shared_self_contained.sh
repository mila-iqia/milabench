#!/bin/bash

RUN_RESOURCES=""
SETUP_RESOURCES=""

check_dependencies() {

}

install_milabench() {
    pip install -e milabench[cuda]
}

install_benchmarks() {
    milabench install --config $CONFIG --system $SYSTEM --base $BASE $REMAINING_ARGS
}

prepare_benchmarks() {
    milabench prepare --config $CONFIG --system $SYSTEM --base $BASE $REMAINING_ARGS
}

setup() {
    install_milabench()
    install_benchmarks()
    prepare_benchmarks()
}

setup_and_run() {
    #
    #   Check the resources first
    #
    setup()

    #
    #   Launch the heavy work job now
    #
    run_jobid=$(sbatch $RUN_RESOURCES --dependency=afterok:$setup_jobid "$0" run)
    
    exit 0
}

run_benchmarks() {
    #
    #   The problem with this is that it runs with lots of resources to just fail
    #   maybe it waited a long time, to just fail an launch a fast job
    #   better to launch a fast job to check the setup and schedule the heavy job after
    #
    if ! check_dependencies; then
        # dependencies not met, schedule the setup step
        setup_jobid=$(sbatch $SETUP_RESOURCES --parsable "$0" setup)

        # dependencies fine, schedule the work step
        run_jobid=$(sbatch $RUN_RESOURCES --dependency=afterok:$setup_jobid "$0" run)

        exit 0
    fi

    milabench run --config $CONFIG --system $SYSTEM --base $BASE $REMAINING_ARGS
}


MODE=$1

case "$MODE" in
    setup)
        setup
        ;;
    run)
        run_benchmarks
        ;;
esac
        