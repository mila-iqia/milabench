#
#   This needs to run inside milabench repository
#
#       - Install rust
#       - Update pip and poetry
#       - Install Milabench
#

if [[ ! -d "~/.cargo/bin" ]]; then
    wget --no-check-certificate --secure-protocol=TLSv1_2 -qO- https://sh.rustup.rs | sh -s -- -y 
fi
export PATH="~/.cargo/bin:${PATH}"

python -m pip install -U pip
python -m pip install -U poetry
poetry lock --no-update
poetry install
