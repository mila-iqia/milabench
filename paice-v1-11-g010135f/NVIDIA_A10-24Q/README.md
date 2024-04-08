```
=================
Benchmark results
=================
                         fail n       perf   sem%   std% peak_memory          score weight
bert-fp16                   4 4        NaN    NaN    NaN       24000            NaN   0.00
bert-fp32                   4 4        NaN    NaN    NaN       23304            NaN   0.00
bert-tf32                   4 4        NaN    NaN    NaN       23304            NaN   0.00
bert-tf32-fp16              4 4        NaN    NaN    NaN       24000            NaN   3.00
bf16                        0 4      91.87   0.1%   1.4%        3098     183.777391   0.00
convnext_large-fp16         4 4        NaN    NaN    NaN       24394            NaN   0.00
convnext_large-fp32         4 4        NaN    NaN    NaN       24430            NaN   0.00
convnext_large-tf32         4 4        NaN    NaN    NaN       24430            NaN   0.00
convnext_large-tf32-fp16    4 4        NaN    NaN    NaN       24470            NaN   3.00
davit_large                 4 4        NaN    NaN    NaN       24438            NaN   1.00
davit_large-multi           2 2        NaN    NaN    NaN       24366            NaN   5.00
dlrm                        0 2  376081.29   0.1%   1.4%        5996  376081.290012   1.00
focalnet                    0 4     146.78   1.0%  15.0%       24468     293.712272   2.00
fp16                        0 4      92.92   0.1%   1.1%        3098     185.826273   0.00
fp32                        0 4      15.61   0.1%   1.4%        3476      31.219423   0.00
llama                       4 4        NaN    NaN    NaN          -1            NaN   1.00
reformer                    4 4        NaN    NaN    NaN       23556            NaN   1.00
regnet_y_128gf              4 4        NaN    NaN    NaN       24450            NaN   2.00
resnet152                   4 4        NaN    NaN    NaN       24458            NaN   1.00
resnet152-multi             2 2        NaN    NaN    NaN       24470            NaN   5.00
resnet50                    0 4     546.80   0.5%   8.1%        5838    1094.496142   1.00
rwkv                        4 4        NaN    NaN    NaN        3976            NaN   1.00
stargan                     4 4        NaN    NaN    NaN       24384            NaN   1.00
super-slomo                 4 4        NaN    NaN    NaN       24458            NaN   1.00
t5                          4 4        NaN    NaN    NaN       24098            NaN   2.00
tf32                        0 4      44.61   0.1%   1.0%        3476      89.225443   0.00
whisper                     4 4        NaN    NaN    NaN       23124            NaN   1.00

Scores
------
Failure rate:      74.51% (FAIL)
Score:               2.65
```
