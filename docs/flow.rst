Milabench Setup Overview
------------------------




.. code-block:: txt

    MILABENCH_BASE=workdir

    workdir
    ├── milabench
    │   ├── benchmarks          # Each benchmark is inside a folder
    │   │   ├── torchvision
    |   |   |   ├── benchfile.py    # Benchmark configuration (source to checkout, script to runs etc...)
    |   |   |   ├── voirfile.py     # Instrumentation to insert timers
    |   |   |   ├── prepare.py      # Prepare script executed to fetch datasets, download pretrained models
    |   |   |   └── main.py         # benchmark script to be ran
    |   |   └── timm 
    |   |       ├── benchfile.py    # Benchmark configuration (source to checkout, script to runs etc...)
    |   |       ├── voirfile.py     # Instrumentation to insert timers
    |   |       ├── prepare.py      # Prepare script executed to fetch datasets, download pretrained models
    |   |       └── main.py         # benchmark script to be ran
    │   ├── benchmate           # benchmate module
    │   ├── milabench           # milabench module
    │   ├── constraints         # pip constraints for different vendors
    │   └── config              # benchmark suite configuration
    ├── env                     # virtual environment where milabench is installed
    └── results
        ├── data                # Datasets, pre-trained models
        ├── extra              
        ├── venv                # Benchmark virtual environments
        │    └── torch          # each benchmark can have their own environments
        └── runs                # Raw metrics gathered by milabench
            └── {runname}.{time}
                └── {benchname}.D{device_id}.data   # Stream of metrics for a benchmark



Benchmark configuration
-----------------------

milabench was created first as a procurement tool.
It was design to measure performance of a given system in order to measure its suitabilty given a specific purpose (suite of benchmark).

As such milabench benchmark suite is fully configurable. Users can create their own benchmark suite from it in order to 
ensure the systems are tested for their real use case.


Milabench is configured using a yaml file that specify where are the benchmark and how to install them.


.. code-block:: yaml

    _defaults:
        max_duration: 600                           # Bench time out
        voir:
            options:                                
                stop: 60                            # Bench stops after gathering 60 observations
                interval: "1s"                      # Gathering interval

        validation:                                 # Validation (disabled by default)
            usage:
                gpu_load_threshold: 0.5             # ensure GPU load is higher than 50%
                gpu_mem_threshold: 0.5              # ensure GPU memory is higher than 50%

    _torchvision:
        inherits: _defaults                         # base configuration
        definition: ../benchmarks/torchvision       # benchmark definition location
        group: torchvision                          
        install_group: torch                        # venv name to use for this benchmark
        plan:                                       # Specify how the benchmark is scheduled
            method: per_gpu                         # `per_gpu` means it will spawn one bench per GPU
        argv:                                       # arguments to forward
            --precision: 'tf32-fp16'
            --lr: 0.01
            --no-stdout: true
            --epochs: 50
            --num-workers: 8

    resnet50:                                           # benchmark name "_" are "private" and never run
        inherits: _torchvision
        tags:                                           # selection tags
            - vision
            - classification
            - convnet
            - resnet
        
        argv:
            --model: resnet50
            --batch-size: 64
            --num-workers: "{cpu_per_gpu}"              # Placeholder variable to be resolved


System Configuration
--------------------