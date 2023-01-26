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
#        sudo docker build -t milabench .
#
#
FROM ubuntu

# Paths
# -----
ENV MILABENCH_CONFIG=/milabench/milabench/config/standard-8G.yaml
ENV MILABENCH_BASE=/milabench/runs
ENV MILABENCH_OUTPUT=/milabench/results/
ENV MILABENCH_ARGS=""
ENV CONDA_PATH=/opt/anaconda


# Copy milabench
# --------------

WORKDIR /milabench
COPY . /milabench/milabench/


# Install Dependencies
# --------------------

RUN apt update && apt install -y wget git

# Install Python
# --------------

# Install anaconda because milabench will need it later anyway#
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_PATH

RUN rm ~/miniconda.sh
ENV PATH=$CONDA_PATH/bin:$PATH


# Install Milabench
# -----------------

RUN python -m pip install -e /milabench/milabench/




# Prepare bench environment ahead of time
#
#   we should download everything we need to run the benchmark right now
#

# RUN milabench install $MILABENCH_CONFIG --base $MILABENCH_BASE $MILABENCH_ARGS
# RUN milabench prepare $MILABENCH_CONFIG --base $MILABENCH_BASE $MILABENCH_ARGS

# CMD milabench

# 
# To use this container
#
# Replace <results> with a folder in your local machine
#
#   docker run -it -v "<results>:/milabench/results" milabench run
#   docker run -it -v "<results>:/milabench/results" milabench summary
#   docker run -it -v "<results>:/milabenchresults" milabench publish
#

# RUN milabench run $MILABENCH_CONFIG --base $MILABENCH_BASE $MILABENCH_ARGS
# RUN milabench summary $WORKING_DIR/runs/runs/
# RUN milabench summary $WORKING_DIR/runs/runs/ -o $MILABENCH_OUTPUT/summary.json