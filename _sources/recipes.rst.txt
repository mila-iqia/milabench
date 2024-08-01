Running Milabench
=================

Base Setup
----------

.. code-block:: bash
  
   salloc  -w "cn-d[003-004]" --ntasks=1 --gpus-per-task=a100l:8 --exclusive --nodes=1 --cpus-per-task=128 --time=120:00:00 --ntasks-per-node=1 --mem=0
   cd /tmp/
   mkdir milabench
   cd milabench
   git clone https://github.com/mila-iqia/milabench.git

   conda activate base
   python --version
   Python 3.11.4

   virtualenv ./env
   source ./env/bin/activate
   pip install -e milabench/

   export MILABENCH_WORDIR="$(pwd)"
   export MILABENCH_BASE="$MILABENCH_WORDIR/results"
   export MILABENCH_CONFIG="$MILABENCH_WORDIR/milabench/config/standard.yaml"
   export BENCHMARK_VENV="$MILABENCH_WORDIR/results/venv/torch"

   module load cuda/12.3.2                                          # <= or set CUDA_HOME to the right spot
   
   milabench install
   milabench prepare
   milabench run

The current setup runs on 8xA100 SXM4 80Go.
Note that some benchmarks do require more than 40Go of VRAM.
One bench might be problematic; rwkv which requires nvcc but can be ignored.

Recipes
-------

Increase Runtime
^^^^^^^^^^^^^^^^

For profiling it might be useful to run the benchmark for longer than the default configuration.
You can update the yaml file (``config/base.yaml`` or ``config/standard.yaml``) to increase the runtime limits.
There is two values that govern the runtime of a benchmark ``max_duration`` which is a pure timeout to avoid benchmark hangs
and ``voir.options.stop`` which represent the target number of observations milabench will gather before stopping.

.. code-block:: yaml
  
   _defaults:
     max_duration: 600           # <= Maximum number of seconds the bench can run
     voir:                       # note that if this triggers the bench is marked as failed
       options:
         stop: 60                # <= Maximum number of observation we are gathering
         interval: "1s"          # This is usually what triggers the premature exit of the benchmark
                                 # an observation is usually a batch forward/backward/optimizer.step (i.e one train step)

One Env
^^^^^^^

If your are using a container with dependencies such as pytorch already installed,
you can force milabench to use a single environment for everything.

.. code-block:: bash

    milabench install --use-current-env
    milabench prepare --use-current-env
    milabench run --use-current-env --select bert-fp32 

Batch resizer
^^^^^^^^^^^^^

If the GPU you are using has lower VRAM automatic batch resizing could be enabled with the command below.
Note that will not impact benchmarks that already use a batch of one, such as opt-6_7b and possibly opt-1_3b.

.. code-block:: bash

   MILABENCH_SIZER_AUTO=True milabench run

Device Select
^^^^^^^^^^^^^

To run on a subset of GPUs (note that by default milabench will try to use all the GPUs all the time
which might make a run take a bit longer, reducing the number of visible devices to 2 might make experimentation faster)

.. code-block:: bash
  
   CUDA_VISIBLE_DEVICES=0,1,2,3 milabench run 

Update Package
^^^^^^^^^^^^^^

To update pytorch to use a newer version of cuda (milabench creates a separate environment for benchmarks)

.. code-block:: bash
  
   # can be executed after `milabench install` at the earliest
   source $BENCHMARK_VENV/bin/activate
   pip install -U torch torchvision torchaudio

Arguments
^^^^^^^^^

If environment variables are troublesome, the values can also be passed as arguments.

.. code-block:: bash
   
   milabench install --base $MILABENCH_BASE --config $MILABENCH_CONFIG
   milabench prepare --base $MILABENCH_BASE --config $MILABENCH_CONFIG
   milabench run --base $MILABENCH_BASE --config $MILABENCH_CONFIG

To help us troubleshoot future issues, you can forward your result directory.
It holds all the benchmark specific logs and metrics gathered by milabench.

.. code-block:: bash

  zip -r results.zip results

Example Reports
---------------

8xA100 SXM4 80Go
^^^^^^^^^^^^^^^^

.. code-block:: bash
  
   milabench run 
   =================
   Benchmark results
   =================
   bench                          | fail | n |       perf |   sem% |   std% | peak_memory |      score | weight
   bert-fp16                      |    0 | 8 |     154.92 |   0.3% |   4.5% |       28500 |    1240.06 |  0.00
   bert-fp32                      |    0 | 8 |      29.55 |   0.0% |   0.5% |       35464 |     236.54 |  0.00
   bert-tf32                      |    0 | 8 |     120.02 |   0.3% |   4.9% |       35466 |     960.04 |  0.00
   bert-tf32-fp16                 |    0 | 8 |     154.87 |   0.3% |   4.5% |       28500 |    1239.70 |  3.00
   bf16                           |    0 | 8 |     293.43 |   0.3% |   7.2% |        5688 |    2363.29 |  0.00
   convnext_large-fp16            |    0 | 8 |     247.31 |   2.4% |  37.6% |       31362 |    1986.27 |  0.00
   convnext_large-fp32            |    0 | 8 |      45.58 |   0.7% |  11.5% |       53482 |     360.90 |  0.00 ** High memory **
   convnext_large-tf32            |    0 | 8 |     117.54 |   1.2% |  18.8% |       53482 |     940.03 |  0.00 ** High memory **
   convnext_large-tf32-fp16       |    0 | 8 |     214.41 |   2.9% |  46.4% |       31362 |    1713.47 |  3.00
   davit_large                    |    0 | 8 |     308.33 |   0.3% |   7.3% |       37900 |    2475.47 |  1.00
   davit_large-multi              |    0 | 1 |    2242.69 |   2.0% |  15.2% |       45610 |    2242.69 |  5.00 ** High memory **
   dlrm                           |    0 | 1 |  398088.30 |   2.5% |  19.3% |        7030 |  398088.30 |  1.00
   focalnet                       |    0 | 8 |     391.21 |   0.3% |   6.8% |       29808 |    3143.46 |  2.00
   fp16                           |    0 | 8 |     289.62 |   0.2% |   4.8% |        5688 |    2327.60 |  0.00
   fp32                           |    0 | 8 |      19.13 |   0.0% |   1.3% |        6066 |     153.20 |  0.00
   llama                          |    0 | 8 |     496.84 |   4.4% |  79.2% |       32326 |    3778.17 |  1.00
   opt-1_3b                       |    0 | 1 |      28.23 |   0.1% |   0.4% |       45976 |      28.23 |  5.00 ** High memory **
   opt-6_7b                       |    0 | 1 |      14.22 |   0.0% |   0.1% |       57196 |      14.22 |  5.00 ** High memory **
   reformer                       |    0 | 8 |      61.45 |   0.0% |   1.0% |       29304 |     492.17 |  1.00
   regnet_y_128gf                 |    0 | 8 |      82.25 |   0.3% |   6.8% |       35454 |     658.46 |  2.00
   resnet152                      |    0 | 8 |     669.61 |   0.4% |   9.6% |       37878 |    5378.33 |  1.00
   resnet152-multi                |    0 | 1 |    5279.39 |   1.2% |   9.2% |       42532 |    5279.39 |  5.00 ** High memory **
   resnet50                       |    0 | 8 |     456.63 |   3.0% |  66.1% |        8630 |    3620.48 |  1.00
   rwkv                           |    8 | 8 |        nan |   nan% |   nan% |        5458 |        nan |  1.00
   stargan                        |    0 | 8 |      34.07 |   2.1% |  45.4% |       41326 |     271.44 |  1.00
   super-slomo                    |    0 | 8 |      35.55 |   1.4% |  30.7% |       37700 |     285.19 |  1.00
   t5                             |    0 | 8 |      47.77 |   0.2% |   4.0% |       39344 |     382.20 |  2.00
   tf32                           |    0 | 8 |     147.05 |   0.2% |   4.9% |        6066 |    1181.93 |  0.00
   whisper                        |    0 | 8 |     145.26 |   2.2% |  48.3% |       40624 |    1160.69 |  1.00
    
    Scores
    ------
    Failure rate:       4.06% (FAIL)
    Score:             567.57
    
    Errors
    ------
    8 errors, details in HTML report

4xA100 SXM4 80Go
^^^^^^^^^^^^^^^^

.. code-block:: bash
  
    CUDA_VISIBLE_DEVICES=0,1,2,3 milabench run 
    =================
    Benchmark results
    =================
    bench                          | fail | n |       perf |   sem% |   std% | peak_memory |      score | weight
    bert-fp16                      |    0 | 4 |     154.86 |   0.4% |   4.5% |       28500 |     619.75 |  0.00
    bert-fp32                      |    0 | 4 |      29.58 |   0.0% |   0.5% |       35464 |     118.38 |  0.00
    bert-tf32                      |    0 | 4 |     119.99 |   0.4% |   4.4% |       35466 |     480.05 |  0.00
    bert-tf32-fp16                 |    0 | 4 |     155.04 |   0.4% |   4.6% |       28500 |     620.50 |  3.00
    bf16                           |    0 | 4 |     293.40 |   0.3% |   6.6% |        5688 |    1180.12 |  0.00
    convnext_large-fp16            |    0 | 4 |     265.18 |   2.8% |  30.6% |       31362 |    1065.59 |  0.00
    convnext_large-fp32            |    0 | 4 |      46.35 |   1.3% |  14.2% |       53482 |     182.25 |  0.00  ** High memory **
    convnext_large-tf32            |    0 | 4 |     122.58 |   1.4% |  15.9% |       53482 |     490.00 |  0.00  ** High memory **
    convnext_large-tf32-fp16       |    0 | 4 |     295.47 |   2.1% |  22.8% |       31362 |    1191.62 |  3.00
    davit_large                    |    0 | 4 |     310.47 |   0.4% |   6.5% |       38144 |    1247.04 |  1.00
    davit_large-multi              |    0 | 1 |    1183.76 |   1.1% |   8.2% |       45336 |    1183.76 |  5.00 ** High memory **
    dlrm                           |    0 | 1 |  430871.61 |   2.6% |  20.2% |        7758 |  430871.61 |  1.00
    focalnet                       |    0 | 4 |     391.96 |   0.4% |   6.4% |       29812 |    1575.26 |  2.00
    fp16                           |    0 | 4 |     289.99 |   0.2% |   4.1% |        5688 |    1164.13 |  0.00
    fp32                           |    0 | 4 |      19.13 |   0.0% |   0.9% |        6066 |      76.58 |  0.00
    llama                          |    0 | 4 |     492.72 |   6.2% |  78.3% |       32326 |    1884.58 |  1.00
    opt-1_3b                       |    0 | 1 |      14.45 |   0.0% |   0.2% |       46016 |      14.45 |  5.00 ** High memory **
    opt-6_7b                       |    0 | 1 |       5.96 |   0.0% |   0.1% |       75444 |       5.96 |  5.00 ** High memory **
    reformer                       |    0 | 4 |      61.39 |   0.1% |   1.0% |       29304 |     245.83 |  1.00
    regnet_y_128gf                 |    0 | 4 |      82.67 |   0.3% |   5.1% |       35454 |     330.98 |  2.00
    resnet152                      |    0 | 4 |     672.09 |   0.4% |   6.9% |       39330 |    2694.83 |  1.00
    resnet152-multi                |    0 | 1 |    2470.38 |   1.5% |  11.2% |       47288 |    2470.38 |  5.00 ** High memory **
    resnet50                       |    0 | 4 |     454.49 |   3.2% |  50.5% |        8630 |    1800.61 |  1.00
    rwkv                           |    4 | 4 |        nan |   nan% |   nan% |        5458 |        nan |  1.00
    stargan                        |    0 | 4 |      42.30 |   1.9% |  29.9% |       53412 |     169.73 |  1.00 ** High memory **
    super-slomo                    |    0 | 4 |      40.67 |   0.8% |  13.1% |       37700 |     163.08 |  1.00
    t5                             |    0 | 4 |      47.74 |   0.3% |   3.9% |       39344 |     190.95 |  2.00
    tf32                           |    0 | 4 |     146.72 |   0.2% |   4.0% |        6066 |     588.99 |  0.00
    whisper                        |    0 | 4 |     207.47 |   1.0% |  15.4% |       40624 |     832.75 |  1.00
    
    Scores
    ------
    Failure rate:       3.96% (FAIL)
    Score:             300.23

4xA100 SXM4 80Go limited to 40Go of VRAM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: bash
  
   CUDA_VISIBLE_DEVICES=0,1,2,3 MILABENCH_SIZER_AUTO=True MILABENCH_SIZER_CAPACITY=40000Mo milabench run
    =================
    Benchmark results
    =================
                             fail n       perf   sem%   std% peak_memory          score weight
    bert-fp16                   0 4     147.52   0.2%   1.9%       41938     588.500016   0.00
    bert-fp32                   0 4      29.08   0.9%  10.3%       42138     116.083048   0.00
    bert-tf32                   0 4     117.82   0.1%   1.0%       42140     470.743584   0.00
    bert-tf32-fp16              0 4     147.67   0.2%   2.4%       41938     588.804052   3.00
    bf16                        0 4     293.92   0.3%   6.0%        5688    1181.627938   0.00
    convnext_large-fp16         0 4     269.92   2.9%  32.5%       42628    1085.129084   0.00
    convnext_large-fp32         0 4      50.31   0.7%   7.8%       42136     199.292499   0.00
    convnext_large-tf32         0 4     136.86   0.5%   5.0%       42138     549.100135   0.00
    convnext_large-tf32-fp16    0 4     266.48   3.1%  33.8%       42628    1071.146282   3.00
    davit_large                 0 4     300.29   0.5%   7.7%       41728    1203.538777   1.00
    davit_large-multi           0 1    1171.04   1.2%   9.3%       50030    1171.042025   5.00
    dlrm                        0 1  454625.69   2.1%  16.4%        7758  454625.687871   1.00
    focalnet                    0 4     391.81   0.3%   5.1%       41802    1569.986673   2.00
    fp16                        0 4     289.96   0.2%   3.9%        5688    1163.810339   0.00
    fp32                        0 4      19.14   0.0%   0.8%        6066      76.603551   0.00
    llama                       0 4     493.43   6.1%  78.2%       32326    1888.979344   1.00
    opt-1_3b                    0 1      14.52   0.1%   0.3%       45930      14.518303   5.00
    opt-6_7b                    0 1       5.96   0.0%   0.1%       75444       5.955118   5.00 ** High memory **
    reformer                    0 4      46.27   0.0%   0.3%       41986     185.104527   1.00
    regnet_y_128gf              0 4     105.08   0.7%  10.8%       42318     421.706539   2.00
    resnet152                   0 4     674.90   0.5%   7.3%       43688    2706.277411   1.00
    resnet152-multi             0 1    2350.25   2.2%  16.9%       52338    2350.245540   5.00
    resnet50                    0 4     420.09   5.8%  91.1%       42262    1653.944065   1.00
    rwkv                        4 4        NaN    NaN    NaN        5458            NaN   1.00
    stargan                     0 4      36.75   1.3%  20.5%       32310     147.651415   1.00
    super-slomo                 0 4      41.87   0.8%  12.0%       41986     167.928514   1.00
    t5                          0 4      49.55   0.3%   4.5%       41444     198.383370   2.00
    tf32                        0 4     146.74   0.2%   3.8%        6066     588.944520   0.00
    whisper                     0 4     209.19   0.7%  10.5%       42242     838.753126   1.00
    
    Scores
    ------
    Failure rate:       4.00% (FAIL)
    Score:             444.18
    
    Errors
    ------
    4 errors, details in HTML report.


Issues
------
.. code-block:: txt
  
    > Traceback (most recent call last):
    >   File "/gpfs/home3/pmorillas/mila/milabench/milabench/utils.py", line 69, in wrapped
    > 	return fn(*args, **kwargs)
    >   File "/gpfs/home3/pmorillas/mila/milabench/milabench/summary.py", line 50, in aggregate
    > 	assert config and start and end
    > AssertionError
    > Source: mila_installation/runs/

This indicates that the configuration might be missing or invalid.
It can happen when generating a report from an incomplete run as either the first metric entry (config) or the last config entry (end)
might be missing. It can be the symptom of another problem that caused benchmarks to fail to run successfully.

.. code-block:: txt

    >   File "/gpfs/home3/pmorillas/mila2/milabench/milabench/cli/run.py", line 82, in cli_run
    >     arch = next(iter(mp.packs.values())).config["system"]["arch"]
    >            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    > StopIteration

This indicates no bench were found to run; either the configuration was invalid or the `--select` filtered out all benchmarks.
