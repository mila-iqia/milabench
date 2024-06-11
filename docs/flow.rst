Milabench Overview
------------------

.. code-block:: txt

    MILABENCH_BASE=workdir/results

    workdir
    ├── milabench
    │   ├── benchmarks          # Each benchmark is inside a folder
    │   │   ├── torchvision
    |   |   |   ├── benchfile.py             # Benchmark configuration (source to checkout, script to runs etc...)
    |   |   |   ├── voirfile.py              # Instrumentation to insert timers
    |   |   |   ├── prepare.py               # Prepare script executed to fetch datasets, download pretrained models
    |   |   |   ├── main.py                  # benchmark script to be ran
    |   |   |   ├── requirements.in          # base requirements
    |   |   |   ├── requirements.cuda.txt    # pinned requirements for cuda
    |   |   |   ├── requirements.rocm.txt    # pinned requirements for rocm
    |   |   |   └── requirements.xpu.txt     # pinned requirements for xpu
    |   |   └── timm 
    |   |       ├── benchfile.py    # Benchmark configuration (source to checkout, script to runs etc...)
    |   |       ├── voirfile.py     # Instrumentation to insert timers
    |   |       ├── prepare.py      # Prepare script executed to fetch datasets, download pretrained models
    |   |       └── main.py         # benchmark script to be ran
    │   ├── benchmate           # benchmate module
    │   ├── milabench           # milabench module
    │   ├── constraints         # pip constraints for different vendors
    │   └── config              # benchmark suite configurations
    │       └── standard.yaml   # <= MILABENCH_CONFIG
    ├── env                     # virtual environment where milabench is installed
    └── results                 # <= MILABENCH_BASE
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

   # you can include a previous configuration
   # and override its values
   include:
      - base.yaml

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
         method: per_gpu                          # `per_gpu` means it will spawn one bench per GPU
      argv:                                       # arguments to forward
         --precision: 'tf32-fp16'
         --lr: 0.01
         --no-stdout: true
         --epochs: 50
         --num-workers: 8

   resnet50:                                          # benchmark name "_" are "private" and never run
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

   # milabench can also define matrix job
   resnet-matrix-noio:                                
      matrix:  
         batch-size: [32, 64, 128, 256, 512, 1024]

      job:
         name: 'resnet50-noio-bs{batch-size}'
         inherits: _resnet50
         argv:
            --batch-size: '{batch-size}'
            --synthetic-data: true
            --fixed-batch: true


System Configuration
--------------------

milabench can run benchmarks across multiple nodes, to do so a system configuration needs to be provided.
This file will define all the nodes accessible to milabench.

.. code-block:: yaml

   system:
      arch: cuda                 # Default arch
      sshkey: ~/.ssh/id_ed25519  # sshkey used in remote milabench operations
      # Docker image to use
      docker_image: ghcr.io/mila-iqia/milabench:${system.arch}-nightly

      # Nodes list
      nodes:
         # Alias used to reference the node
         - name: manager
           ip: 192.168.11.11
           port: 5000
           main: true     # Use this node as the rank=0 node or not
           user: manager  # User to use in remote milabench operations

         - name: node1
           ip: 192.168.11.12
           main: false
           user: username

Multinode
*********

Milabench takes care of sending the commands to all the nodes when appropriate.


Methodology
-----------

.. code-block:: python

   for i in range(epoch):
      events = []
      
      # Creation of the iterator from the dataloader is time consuming
      # it would get amortized across many batch during real training
      # but we want benchmarking to be fast so it is something we cannot afford
      batch_iter = iter(loader)
      total_obs = 0
      
      # Avoid sync in the batch loop
      start = Event()
      start.record()

      for batch in batch_iter:
         pred = model(batch)
         loss = fn(pred, target)
      
         end = Event()                                           # +->
         end.record()                                            # |
         events.append((start, end, len(batch), loss.detach()))  # | Limited overhead
         if len(events) + total_obs >= 60:                       # | 
            break                                                # |
         start = end                                             # +->

      # Force sync at the end of the epoch                       # +->
      for start, end, bs, loss  in events:                       # | Timer is off does not impact perf measures
         end.wait()                                              # |
         log(loss=loss.item())                                   # |
         log(rate=bs / (end - start))                            # |
                                                                 # |
      total_obs += len(events)                                   # |
      if total_obs >= 60:                                        # |
         raise StopProgram()                                     # +->

Instrumentations
****************

To minimize code change, milabench use `ptera <https://github.com/breuleux/ptera>`_ to modify
the code that will be run and insert the necessary hooks to measure performance.

The hooks are defined inside the ``voirfile.py``.
The example below override the return value of the ``dataloader()`` function which is defined in the ``__main__`` module.
It wraps the original object with a custom wrapper that will time the time between ``__next__`` calls.

This allows milabench to integrate benchmarks from code coming from third parties without modifying the code directly.

.. code-block:: python

   def wrapper(loader):
      print("received loader obj")
      return Wrapper(loader)

   probe = ov.probe("//dataloader() as loader", overridable=True)
   probe['loader'].override(wrapper)


Execution Flow
--------------

* ``milabench install``
   * Creates virtual env for benchmarks and install their dependencies
   * Modify: ``$MILABENCH_BASE/venv/{bench}``

* ``milabench prepare``
   * Call the prepare script for each benchmarks to download/generate dataset and download pretrained models
   * Modify: ``$MILABENCH_BASE/data/{dataset}``

* ``milabench run``
   * Execute each benchmark
   * Modify: ``$MILABENCH_BASE/runs/{runame}.{time}``
   * Steps
      * **init**: Voir has initialized itself. You can add command-line arguments here.
      * **parse_args**: The command-line arguments have been parsed.
      * **load_script**: The script has been loaded: its imports have been done, its functions defined,
        but the top level statements have not been executed. You can perform some
        manipulations prior to the script running.
      * **run_script**: the script will start to run now
      * **finalize**: tearing down

How do I
--------

* I want to run a benchmark without milabench for debugging purposes
   * ``milabench dev {benchname}`` will open bash with the benchmark venv sourced
   * alternatively: ``source $MILABENCH_BASE/venv/torch/bin/activate``


