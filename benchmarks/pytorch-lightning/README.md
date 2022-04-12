# PyTorch-Lightning Benchmarks

This folder contains a customizable PyTorch-Lightning benchmark.
The two default configurations are:
- data-parallel: Training in a data-parallel fashion by using the "dp" strategy of PyTorch-Lightning;
- model-parallel: Fully-sharded Model-Parallel training using the "fsdp" strategy and fairscale.

The choice of backbone model can be customized via the `--backbone` command-line argument, and can
be set to the name of any classification model from `torchvision.models`.

Furthermore, any of the arguments to the `Trainer` constructor can also be customized by setting the 
command-line argument of the same name. For example, using `--precision 16` will run the benchmark
with mixed precision training.

For more info about the available options, run `python main.py --help` or check out https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html for a more detailed description of the features available in PyTorch-Lightning.

