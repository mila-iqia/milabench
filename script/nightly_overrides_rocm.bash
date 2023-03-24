milabench pip uninstall torch torchvision torchaudio torchtext -y
milabench pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
milabench pip install -U pytorch-lightning==1.9.4
milabench pip uninstall lightning-bolts -y
milabench pip install git+https://github.com/Delaunay/lightning-bolts.git@patch-1
