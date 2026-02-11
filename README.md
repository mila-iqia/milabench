
# Milabench

* [Stable Documentation](https://milabench.readthedocs.io/en/stable/)
* [Nightly Documentation](https://mila-iqia.github.io/milabench/)

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
* ROCm, NVIDIA, Intel OneAPI, Habana Gaudi (Synapse)
* Independent 

## Getting Started

  
    git clone https://github.com/mila-iqia/milabench.git

    export MILABENCH_GPU_ARCH=cuda

    pip install -e milabench[cuda]

    milabench install --base workspace --config milabench/config/standard.yaml --select fp32

    export MILABENCH_HF_TOKEN={your_token}
    milabench prepare --base workspace --config milabench/config/standard.yaml --select fp32
    
    milabench run --base workspace --config milabench/config/standard.yaml --select fp32


## Gated Models

Some benchmark use gated models or datasets, which requires the user to request permission to huggingface

1. Request permission

       llama
           url: https://huggingface.co/meta-llama/Llama-2-7b/tree/main
       
       llm-lora-single
       llm-lora-ddp-gpus
       llm-lora-ddp-nodes
           url: https://huggingface.co/meta-llama/Llama-3.1-8B
       
       llm-lora-mp-gpus
       llm-full-mp-gpus
       llm-full-mp-nodes
           url: https://huggingface.co/meta-llama/Llama-3.1-70B

3. Create a new token
   
       https://huggingface.co/settings/tokens/new?tokenType=read
    
4. Add your token to your environment

       export MILABENCH_HF_TOKEN={your_token}

Now you are ready to execute `milabench prepare`

## Report
  
    =================
    Benchmark results
    =================
  
    System
    ------
    cpu:      AMD EPYC 7742 64-Core Processor
    n_cpu:    128
    product:  NVIDIA A100-SXM4-80GB
    n_gpu:    8
    memory:   81920.0
  
    Breakdown
    ---------
    bench                    | fail |   n | ngpu |           perf |   sem% |   std% | peak_memory |           score | weight
    brax                     |    0 |   1 |    8 |      730035.71 |   0.1% |   0.4% |        2670 |       730035.71 |   1.00
    diffusion-gpus           |    0 |   1 |    8 |         117.67 |   1.5% |  11.7% |       59944 |          117.67 |   1.00
    diffusion-single         |    0 |   8 |    1 |          25.02 |   0.8% |  17.9% |       53994 |          202.10 |   1.00
    dimenet                  |    0 |   8 |    1 |         366.85 |   0.7% |  16.2% |        2302 |         2973.32 |   1.00
    dinov2-giant-gpus        |    0 |   1 |    8 |         445.68 |   0.4% |   3.0% |       69614 |          445.68 |   1.00
    dinov2-giant-single      |    0 |   8 |    1 |          53.54 |   0.4% |   9.5% |       74646 |          432.65 |   1.00
    dqn                      |    0 |   8 |    1 | 23089954554.91 |   1.1% |  89.9% |       62106 | 184480810548.20 |   1.00
    bf16                     |    0 |   8 |    1 |         293.43 |   0.2% |   6.3% |        1788 |         2361.16 |   0.00
    fp16                     |    0 |   8 |    1 |         289.26 |   0.1% |   3.6% |        1788 |         2321.65 |   0.00
    fp32                     |    0 |   8 |    1 |          19.14 |   0.0% |   0.7% |        2166 |          153.21 |   0.00
    tf32                     |    0 |   8 |    1 |         146.63 |   0.1% |   3.6% |        2166 |         1177.04 |   0.00
    bert-fp16                |    0 |   8 |    1 |         263.73 |   1.1% |  16.7% |         nan |         2165.37 |   0.00
    bert-fp32                |    0 |   8 |    1 |          44.84 |   0.6% |   9.6% |       21170 |          364.52 |   0.00
    bert-tf32                |    0 |   8 |    1 |         141.95 |   0.9% |  14.1% |        1764 |         1162.94 |   0.00
    bert-tf32-fp16           |    0 |   8 |    1 |         265.04 |   1.0% |  15.6% |         nan |         2175.59 |   3.00
    reformer                 |    0 |   8 |    1 |          62.29 |   0.3% |   6.0% |       25404 |          501.89 |   1.00
    t5                       |    0 |   8 |    1 |          51.40 |   0.5% |   9.9% |       34390 |          416.14 |   2.00
    whisper                  |    0 |   8 |    1 |         481.95 |   1.0% |  21.4% |        8520 |         3897.53 |   1.00
    lightning                |    0 |   8 |    1 |         680.22 |   1.0% |  22.7% |       27360 |         5506.90 |   1.00
    lightning-gpus           |    0 |   1 |    8 |        3504.74 |   7.9% |  62.9% |       28184 |         3504.74 |   1.00
    llava-single             |    1 |   8 |    1 |           2.28 |   0.4% |   9.6% |       72556 |           14.12 |   1.00
    llama                    |    0 |   8 |    1 |         484.86 |   4.4% |  80.0% |       27820 |         3680.86 |   1.00
    llm-full-mp-gpus         |    0 |   1 |    8 |         193.92 |   3.1% |  16.2% |       48470 |          193.92 |   1.00
    llm-lora-ddp-gpus        |    0 |   1 |    8 |       16738.58 |   0.4% |   2.0% |       36988 |        16738.58 |   1.00
    llm-lora-mp-gpus         |    0 |   1 |    8 |        1980.63 |   2.2% |  11.8% |       55972 |         1980.63 |   1.00
    llm-lora-single          |    0 |   8 |    1 |        2724.95 |   0.2% |   3.0% |       49926 |        21861.99 |   1.00
    ppo                      |    0 |   8 |    1 |     3114264.32 |   1.6% |  57.2% |       62206 |     24915954.98 |   1.00
    recursiongfn             |    0 |   8 |    1 |        7080.67 |   1.2% |  27.1% |       10292 |        57038.34 |   1.00
    rlhf-gpus                |    0 |   1 |    8 |        6314.94 |   2.1% |  11.2% |       21730 |         6314.94 |   1.00
    rlhf-single              |    0 |   8 |    1 |        1143.72 |   0.4% |   8.4% |       19566 |         9174.52 |   1.00
    focalnet                 |    0 |   8 |    1 |         375.07 |   0.7% |  14.9% |       23536 |         3038.83 |   2.00
    torchatari               |    0 |   8 |    1 |        5848.88 |   0.6% |  12.7% |        3834 |        46613.34 |   1.00
    convnext_large-fp16      |    0 |   8 |    1 |         330.93 |   1.5% |  22.9% |       27376 |         2711.46 |   0.00
    convnext_large-fp32      |    0 |   8 |    1 |          59.49 |   0.6% |   9.8% |       55950 |          483.84 |   0.00
    convnext_large-tf32      |    0 |   8 |    1 |         155.41 |   0.9% |  14.3% |       49650 |         1273.31 |   0.00
    convnext_large-tf32-fp16 |    0 |   8 |    1 |         322.28 |   1.6% |  24.5% |       27376 |         2637.88 |   3.00
    regnet_y_128gf           |    0 |   8 |    1 |         119.46 |   0.5% |  10.0% |       29762 |          966.96 |   2.00
    resnet152-ddp-gpus       |    0 |   1 |    8 |        3843.06 |   5.2% |  39.3% |       27980 |         3843.06 |   0.00
    resnet50                 |    0 |   8 |    1 |         932.95 |   2.4% |  52.2% |       14848 |         7524.25 |   1.00
    resnet50-noio            |    0 |   8 |    1 |        1163.88 |   0.3% |   6.7% |       27480 |         9385.35 |   0.00
    vjepa-gpus               |    0 |   1 |    8 |         130.13 |   5.9% |  46.8% |       64244 |          130.13 |   1.00
    vjepa-single             |    0 |   8 |    1 |          21.29 |   1.0% |  22.4% |       58552 |          172.11 |   1.00
  
    Scores
    ------
    Failure rate:       0.38% (PASS)
    Score:            4175.57

