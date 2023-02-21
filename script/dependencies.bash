#
#   This needs to run inside milabench repository
#
#       - Install rust
#       - Update pip and poetry
#       - Install Milabench
#

CONDA_PATH=${CONDA_PATH:-/opt/anaconda}

# Rust
if [[ ! -d "~/.cargo/bin" ]]; then
    wget --no-check-certificate --secure-protocol=TLSv1_2 -qO- https://sh.rustup.rs | sh -s -- -y 
fi
export PATH="~/.cargo/bin:${PATH}"

# Anaconda
if ! command -v conda &> /dev/null; then
    if [[ ! -d "$CONDA_PATH" ]]; then
        wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
        /bin/bash ~/miniconda.sh -b -p $CONDA_PATH 
        rm ~/miniconda.sh
    fi
    export PATH="$CONDA_PATH/bin:$PATH"
fi

# Update
python -m pip install -U pip
python -m pip install -U poetry
