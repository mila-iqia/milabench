milabench pip uninstall torch torchvision torchaudio torchtext -y
milabench pip install torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/cu118
milabench pip install -U pytorch-lightning==1.9.4
milabench pip uninstall lightning-bolts -y
milabench pip install git+https://github.com/Delaunay/lightning-bolts.git@patch-1
