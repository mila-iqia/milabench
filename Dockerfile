#
#   milabench container
#       * Install milabench
#       * Setup the environment for each tests
#
#   usage:
#   
#       docker run -it --rm -v /opt/results/:/milabench/results milabench milabench --help
#                   ^^         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ^^^^^^^^^ ^^^^^^^^^ ^^^^^^
#                                       Volume Binding           Image     Command   Arguments
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
#       milabench run --config $MILABENCH_CONFIG --base $MILABENCH_BASE $MILABENCH_ARGS
#       milabench summary $WORKING_DIR/results/runs/
#       milabench summary $WORKING_DIR/results/runs/ -o $MILABENCH_OUTPUT/summary.json
#
#
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

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


# Copy milabench
# --------------

WORKDIR /milabench
COPY . /milabench/milabench/


# Install Dependencies
# --------------------

#     curl | wget: used to download anaconda
#             git: used by milabench
#           rustc: used by BERT models inside https://pypi.org/project/tokenizers/
# build-essential: for rust
#          libgl1: learn to paint bench
#    libglib2.0-0: learn to paint bench
RUN apt update && apt install -y wget git build-essential curl libgl1 libglib2.0-0
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python
# --------------

# Install anaconda because milabench will need it later anyway
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_PATH

RUN rm ~/miniconda.sh
ENV PATH=$CONDA_PATH/bin:$PATH


# Install Milabench
# -----------------

RUN python -m pip install -U pip            &&\
    python -m pip install -U setuptools     &&\
    python -m pip install -U poetry         &&\
    python -m pip install -e /milabench/milabench/

# Prepare bench
# -------------

# pip times out often when downloading pytorch
ENV PIP_DEFAULT_TIMEOUT=800

RUN milabench install --config $MILABENCH_CONFIG --base $MILABENCH_BASE $MILABENCH_ARGS &&\
    milabench prepare --config $MILABENCH_CONFIG --base $MILABENCH_BASE $MILABENCH_ARGS


# CUDA-11.8 Fix
ENV LD_LIBRARY_PATH="/usr/local/cuda/targets/x86_64-linux/lib/:$LD_LIBRARY_PATH"
RUN ln /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.11.8.89 /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so


# Cleanup
# Remove PIP cache
# Remove APT unused packages
# CMD milabench
