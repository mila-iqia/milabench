#
#
#   H100 Patch
#   
#       Replace pytorch with pytorch nightly
#
#   This needs to run after milabench prepare & Install
#
#

milabench pip uninstall torch torchvision torchaudio torchtext -y
milabench pip install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cu118
milabench pip install -U pytorch-lightning
milabench pip uninstall lightning-bolts -y
milabench pip install git+https://github.com/Delaunay/lightning-bolts.git@patch-1
