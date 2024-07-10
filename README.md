
# Milabench

[Documentation](https://milabench.readthedocs.io/en/stable/)

Benchmarking framework for Machine learning and Artificial Intelligence, geared toward
evaluating current and future hardware in a research environment.

* Simple / Hands-off
* Wide selection of models on diverse applications
  * Multi GPUs
  * Multi node
  * nlp / transformer / llm / rl / rnn
  * vision / classification / convnet / resnet / transformer
  * audio
* Docker Container
* Works on slurm
* Automatic batch resize
* Focussed on training
* Ease of use
* Pytorch focused
* ROCm & NVIDIA
* Independent 

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
                             fail n       perf   sem%   std% peak_memory          score weight
    bert-fp16                   0 8     155.08   0.3%   4.3%       24552    1241.260310   0.00
    bert-fp32                   0 8      29.52   0.0%   0.5%       31524     236.337218   0.00
    bert-tf32                   0 8     120.46   0.4%   6.1%       31524     964.713297   0.00
    bert-tf32-fp16              0 8     154.76   0.3%   4.1%       24552    1238.477257   3.00
    convnext_large-fp16         0 8     337.48   0.9%  14.0%       27658    2741.604444   0.00
    convnext_large-fp32         0 8      44.61   0.8%  12.6%       49786     354.207225   0.00
    convnext_large-tf32         0 8     135.99   0.7%  11.2%       49786    1089.394916   0.00
    convnext_large-tf32-fp16    0 8     338.58   0.8%  13.0%       27658    2744.325170   3.00
    davit_large                 0 8     312.79   0.3%   6.7%       35058    2515.326450   1.00
    davit_large-multi           0 1    2401.65   1.0%   7.7%       42232    2401.651720   5.00
    dlrm                        0 1  188777.20   1.8%  14.0%        3194  188777.203190   1.00
    focalnet                    0 8     400.47   0.2%   5.4%       26604    3215.431924   2.00
    opt-1_3b                    0 1      26.71   0.1%   0.4%       44116      26.714365   5.00
    opt-1_3b-multinode          0 2      34.62   0.2%   1.0%       43552      34.618292  10.00
    opt-6_7b                    0 1      14.32   0.0%   0.1%       55750      14.319587   5.00
    opt-6_7b-multinode          0 2      10.79   0.1%   0.7%       49380      10.792595  10.00
    reformer                    0 8      61.70   0.0%   0.9%       25376     494.110834   1.00
    regnet_y_128gf              0 8      99.96   0.2%   5.0%       31840     803.012507   2.00
    resnet152                   0 8     710.18   0.3%   6.2%       36732    5710.828608   1.00
    resnet152-multi             0 1    5367.34   1.0%   8.1%       38638    5367.338469   5.00
    resnet50                    0 8     984.43   0.9%  19.1%        5026    7927.257351   1.00
    rwkv                        0 8     428.65   0.2%   3.8%        5546    3435.097716   1.00
    stargan                     0 8      51.32   1.8%  40.8%       37848     413.238870   1.00
    super-slomo                 0 8      41.63   0.1%   2.3%       34082     332.395065   1.00
    t5                          0 8      48.05   0.2%   3.9%       35466     384.317023   2.00
    whisper                     0 8     248.16   0.0%   0.6%       37006    1985.861017   1.00
    
    Scores
    ------
    Failure rate:       0.00% (PASS)
    Score:             219.06


## Details

The benchmark suite has been validated on the following configurations:

| Python version |          GPU                   |   Configuration file |
|       -        |        -                       |           -          |
| 3.11   (conda) | 2 node x 8xNVIDIA A100 80GB    | config/standard.yaml |
| 3.9.12 (conda) | 8x NVIDIA RTX8000 48GB         | config/standard.yaml |
| 3.9.16 (conda) | 2x NVIDIA K80                  | config/ci.yaml       |
| 3.9.16 (conda) | 2x AMD MI100                   | config/ci.yaml       |
| 3.9.16 (conda) | 4x AMD MI250                   | config/standard.yaml |

We are working on validating it on more configurations and will update the above table as we do.



