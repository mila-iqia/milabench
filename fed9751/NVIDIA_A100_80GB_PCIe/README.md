```
=================
Benchmark results
=================

System
------
cpu:      AMD EPYC 7V13 64-Core Processor
n_cpu:    96
product:  NVIDIA A100 80GB PCIe
n_gpu:    4
memory:   81920.0

Breakdown
---------
bench                          | fail |   n | ngpu |       perf |   sem% |   std% | peak_memory |      score | weight
diffusion-gpus                 |    0 |   1 |    4 |      95.27 |   0.1% |   0.7% |       57311 |      95.27 |   1.00
dinov2-giant-gpus              |    0 |   1 |    4 |     200.47 |   0.3% |   2.0% |       71295 |     200.47 |   1.00
lightning-gpus                 |    0 |   1 |    4 |    2723.23 |   0.4% |   3.4% |       27113 |    2723.23 |   1.00
llm-full-mp-gpus               |    0 |   1 |    4 |      43.56 |   3.3% |  17.6% |       62103 |      43.56 |   1.00
llm-lora-ddp-gpus              |    0 |   1 |    4 |    5415.41 |   0.9% |   4.6% |       33011 |    5415.41 |   1.00
llm-lora-mp-gpus               |    0 |   1 |    4 |     430.76 |   2.0% |  10.8% |       65093 |     430.76 |   1.00
resnet152-ddp-gpus             |    0 |   1 |    4 |    2364.75 |   0.8% |   5.9% |       26911 |    2364.75 |   0.00

Scores
------
Failure rate:       0.00% (PASS)
Score:             417.36
```
