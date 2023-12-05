
# Milabench

[Documentation](https://mila-iqia.github.io/milabench)

Benchmarking framework for Machine learning and Artificial Intelligence, geared toward
evaluating current and future hardware in a research environment.

* Simple
* Wide selection of models on diverse applications
  * Multi GPUs
  * Multi node
* Docker Container
* Works on slurm
* Automatic batch resize
* Focussed on training
* Ease of use
* Pytorch focused


## Getting Started

The easiest way to run milabbench is to run it with one of its docker image.
It will include all of the necessary data


    # Choose the image you want to use
    export MILABENCH_IMAGE=ghcr.io/mila-iqia/milabench:cuda-nightly

    # Pull the image we are going to run
    docker pull $MILABENCH_IMAGE

    # Run milabench
    docker run -it --rm --ipc=host --gpus=all      \
          -v $(pwd)/results:/milabench/envs/runs   \
          $MILABENCH_IMAGE                         \
          milabench run

    =================
    Benchmark results
    =================
                             fail   n       perf   sem%   std% peak_memory          score weight
    bert-fp16                   0   1      49.82   0.0%   0.2%       23952      49.815508   0.00
    bert-fp32                   0   1      20.78   0.0%   0.2%       30922      20.783989   0.00
    bert-tf32                   0   1      20.79   0.0%   0.2%       30922      20.787725   0.00
    bert-tf32-fp16              0   1      49.70   0.1%   0.3%       23952      49.697091   3.00
    bf16                        0   1       7.91   0.0%   0.1%        1140       7.910341   0.00
    convnext_large-fp16         0   1     123.77   2.5%  13.6%       26632     123.767014   0.00
    convnext_large-fp32         0   1      32.69   0.5%   2.6%       45356      32.687851   0.00
    convnext_large-tf32         0   1      32.64   0.5%   2.6%       45356      32.636185   0.00
    convnext_large-tf32-fp16    0   1     124.93   2.5%  13.4%       26632     124.930007   3.00
    davit_large                 0   1     114.54   1.3%   9.8%       32374     114.539282   1.00
    davit_large-multi           0   1     115.18   1.2%   9.3%       32374     115.176873   5.00
    dlrm                        0   1  255977.96   0.5%   4.0%        6354  255977.960840   1.00
    focalnet                    0   1     151.78   1.6%  12.4%       24098     151.775544   2.00
    fp16                        0   1     101.03   0.1%   0.6%        1142     101.025637   0.00
    fp32                        0   1      14.42   0.0%   0.2%        1524      14.418942   0.00
    reformer                    0   1      10.22   0.0%   0.1%       24756      10.222305   1.00
    regnet_y_128gf              0   1      30.52   0.3%   1.9%       30748      30.518845   2.00
    resnet152                   0   1     232.63   1.1%   8.1%       29904     232.629851   1.00
    resnet152-multi             0   1     232.14   1.0%   7.7%       30614     232.144301   5.00
    resnet50                    0   1     490.08   2.5%  19.0%        4166     490.076388   1.00
    rwkv                        0   1     109.45   0.3%   2.0%        4944     109.449712   1.00
    stargan                     0   1      11.40   4.2%  31.9%       35648      11.399463   1.00
    super-slomo                 0   1      11.46   0.1%   0.5%       36364      11.463760   1.00
    t5                          0   1      13.91   0.6%   4.5%       34794      13.913109   2.00
    tf32                        0   1      14.43   0.0%   0.2%        1524      14.430707   0.00
    whisper                     0   1      81.71   0.1%   0.6%       35968      81.705971   1.00

    Scores
    ------
    Failure rate:       0.00% (OK)
    Score:              10.68


## Details

The benchmark suite has been validated on the following configurations:

| Python version |          GPU           |   Configuration file |
|       -        |        -               |           -          |
| 3.9.12 (conda) | 4x NVIDIA A100 80GB    | config/standard.yaml |
| 3.9.12 (conda) | 4x NVIDIA RTX8000 48GB | config/standard.yaml |
| 3.9.16 (conda) | 2x NVIDIA K80          | config/ci.yaml       |
| 3.9.16 (conda) | 2x AMD MI100           | config/ci.yaml       |

We are working on validating it on more configurations and will update the above table as we do.



