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
#           sudo docker build --build-arg ARCH=cuda CONFIG=standard-8G.yaml -t milabench .
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
#       milabench summary $WORKING_DIR/runs/runs/
#       milabench summary $WORKING_DIR/runs/runs/ -o $MILABENCH_OUTPUT/summary.json
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


# Copy milabench
# --------------

WORKDIR /milabench
COPY . /milabench/milabench/


# Install Dependencies
# --------------------

#  wget: used to download anaconda
#   git: used by milabench
# rustc: used by BERT models inside https://pypi.org/project/tokenizers/
# 
RUN apt update && apt install -y wget git build-essential curl
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python
# --------------

# Install anaconda because milabench will need it later anyway
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_PATH

RUN rm ~/miniconda.sh
ENV PATH=$CONDA_PATH/bin:$PATH


# Install Milabench
# -----------------

RUN python -m pip install pip -U
RUN python -m pip install -e /milabench/milabench/

# Prepare bench
# -------------

# pip times out often when downloading pytorch
ENV PIP_DEFAULT_TIMEOUT=800

RUN milabench install $MILABENCH_CONFIG --base $MILABENCH_BASE $MILABENCH_ARGS
RUN milabench prepare $MILABENCH_CONFIG --base $MILABENCH_BASE $MILABENCH_ARGS

# Cleanup
# Remove PIP cache
# Remove APT unused packages
# CMD milabench
