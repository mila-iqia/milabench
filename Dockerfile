#
#   milabench container
#       * Install milabench
#       * Setup the environment for each tests
#
#   usage:
#   
#       docker run -it --rm -v /opt/results/:/milabench/results milabench milabench --help
#                   ^^         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^ ^^^^^^^^^ ^^^^^^
#                                       Volune Binding           Image     Command   Arguments
#   
#   To build this image:
#
#           sudo docker build --build-arg ARCH=cuda --build-arg CONFIG=standard-8G.yaml -t milabench .
#
#       Builds the milabench container for cuda and prepare the benchmark using the `standard-8G.yaml`
#       configuration.
#
#   Folders:
#
#       /milabench/milabench    <= milabench code
#       /milabench/envs         <= benchmark enviroments
#       /milabench/results      <= benchmark results
#
#
#   Useful Commands:
#
#       milabench run $MILABENCH_CONFIG --base $MILABENCH_BASE $MILABENCH_ARGS
#       milabench summary $WORKING_DIR/results/runs/
#       milabench summary $WORKING_DIR/results/runs/ -o $MILABENCH_OUTPUT/summary.json
#
#
FROM ubuntu

# Arguments
# ---------

ARG ARCH=cuda
ENV MILABENCH_GPU_ARCH=$ARCH

ARG CONFIG=standard-8G.yaml
ENV MILABENCH_CONFIG_NAME=$CONFIG


# Paths
# -----

ENV MILABENCH_CONFIG=/milabench/milabench/config/$MILABENCH_CONFIG_NAME
ENV MILABENCH_BASE=/milabench/envs
ENV MILABENCH_OUTPUT=/milabench/results/
ENV MILABENCH_ARGS=""
ENV CONDA_PATH=/opt/anaconda
ENV WORKDIR="/milabench"

# Copy milabench
# --------------

WORKDIR $WORKDIR
COPY . $WORKDIR/milabench/

ENV PIP_DEFAULT_TIMEOUT=800

RUN apt update                                                      && \
    apt install -y wget git build-essential curl                    && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh &&\
    /bin/bash ~/miniconda.sh -b -p $CONDA_PATH                      && \
    rm ~/miniconda.sh                                               && \
    find $CONDA_PATH/ -follow -type f -name '*.a' -delete           && \
    find $CONDA_PATH/ -follow -type f -name '*.js.map' -delete      && \
    $CONDA_PATH/bin/conda clean -afy

ENV PATH="$CONDA_PATH/bin:$PATH"
RUN /bin/bash $WORKDIR/milabench/script/setup.bash
