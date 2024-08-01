Plan
====

* ``per_gpu``: used for mono gpu benchmarks, spawn one process per gpu and run the same benchmark

.. code-block:: yaml

  _torchvision:
    inherits: _defaults
    definition: ../benchmarks/torchvision
    group: torchvision
    install_group: torch
    plan:
      method: per_gpu

* ``njobs``: used to launch a single jobs that can see all the gpus.

.. code-block:: yaml

  _torchvision_ddp:
    inherits: _defaults
    definition: ../benchmarks/torchvision_ddp
    group: torchvision
    install_group: torch
    plan:
      method: njobs
      n: 1


Milabench processes overview
----------------------------

* milabench main process
  * gather metrics 

* milabench launches a new benchmark process
  * milabench launch monitoring processes
  * torchrun will launch one process per GPU
