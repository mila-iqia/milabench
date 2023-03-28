
milabench container
    * Install milabench
    * Setup the environment for each tests

usage:

    docker run -it --rm -v /opt/results/:/milabench/results milabench milabench --help
                ^^         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^ ^^^^^^^^^ ^^^^^^
                                    Volume Binding           Image     Command   Arguments

To build this image:

        sudo docker build --build-arg ARCH=cuda --build-arg CONFIG=standard-8G.yaml -t milabench .

    Builds the milabench container for cuda and prepare the benchmark using the `standard-8G.yaml`
    configuration.

Folders:

    /milabench/milabench    <= milabench code
    /milabench/envs         <= benchmark enviroments
    /milabench/results      <= benchmark results


Useful Commands:

    milabench run --config $MILABENCH_CONFIG --base $MILABENCH_BASE $MILABENCH_ARGS
    milabench summary $WORKING_DIR/results/runs/
    milabench summary $WORKING_DIR/results/runs/ -o $MILABENCH_OUTPUT/summary.json

