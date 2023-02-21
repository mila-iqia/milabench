# Rust
if [[ ! -d "~/.cargo/bin" ]]; then
    wget --no-check-certificate --secure-protocol=TLSv1_2 -qO- https://sh.rustup.rs | sh -s -- -y
fi
export PATH="~/.cargo/bin:${PATH}"

# Poetry
python -m pip install -U pip
python -m pip install -U poetry
