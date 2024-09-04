Milabench processes overview
============================

* milabench main process
  * gather metrics from benchmark processes, save them to file
  * manages the benchmarks (timeout etc...)

  * if ``per_gpu`` is used, milabench will launch one process per GPU (sets ``CUDA_VISIBLE_DEVCES``)
    * each processes log their GPU data
    * might spawn a monitor process
      * will init pynvml
    * dataloader will also spawn process workers
      * usually not using GPU

  * if ``njobs`` is used, milabench will launch a single process (torchrun)
    * torchrun in turn will spawn one process per GPU
      * RANK 0 is used for logging
      * RANK 0 might spawn a monitor process
        * will init pynvml
      * dataloader will also spawn process workers 
        * usually not using GPU

Plan
----

per_gpu
+++++++

``per_gpu``: used for mono gpu benchmarks, spawn one process per gpu and run the same benchmark

.. code-block:: yaml

   _torchvision:
     inherits: _defaults
     definition: ../benchmarks/torchvision
     group: torchvision
     install_group: torch
     plan:
       method: per_gpu

Milabench will essentially execute something akin to below. 

.. code-block:: bash

   echo "---"
   echo "fp16"
   echo "===="
   time (
     CUDA_VISIBLE_DEVICES=0 $SRC/milabench/benchmarks/flops/activator $BASE/venv/torch $SRC/milabench/benchmarks/flops/main.py --number 30 --repeat 90 --m 8192 --n 8192 --dtype fp16 &
     CUDA_VISIBLE_DEVICES=1 $SRC/milabench/benchmarks/flops/activator $BASE/venv/torch $SRC/milabench/benchmarks/flops/main.py --number 30 --repeat 90 --m 8192 --n 8192 --dtype fp16 &
     CUDA_VISIBLE_DEVICES=2 $SRC/milabench/benchmarks/flops/activator $BASE/venv/torch $SRC/milabench/benchmarks/flops/main.py --number 30 --repeat 90 --m 8192 --n 8192 --dtype fp16 &
     CUDA_VISIBLE_DEVICES=3 $SRC/milabench/benchmarks/flops/activator $BASE/venv/torch $SRC/milabench/benchmarks/flops/main.py --number 30 --repeat 90 --m 8192 --n 8192 --dtype fp16 &
     CUDA_VISIBLE_DEVICES=4 $SRC/milabench/benchmarks/flops/activator $BASE/venv/torch $SRC/milabench/benchmarks/flops/main.py --number 30 --repeat 90 --m 8192 --n 8192 --dtype fp16 &
     CUDA_VISIBLE_DEVICES=5 $SRC/milabench/benchmarks/flops/activator $BASE/venv/torch $SRC/milabench/benchmarks/flops/main.py --number 30 --repeat 90 --m 8192 --n 8192 --dtype fp16 &
     CUDA_VISIBLE_DEVICES=6 $SRC/milabench/benchmarks/flops/activator $BASE/venv/torch $SRC/milabench/benchmarks/flops/main.py --number 30 --repeat 90 --m 8192 --n 8192 --dtype fp16 &
     CUDA_VISIBLE_DEVICES=7 $SRC/milabench/benchmarks/flops/activator $BASE/venv/torch $SRC/milabench/benchmarks/flops/main.py --number 30 --repeat 90 --m 8192 --n 8192 --dtype fp16 &
     wait
   )

njobs
+++++

``njobs`` used to launch a single jobs that can see all the gpus.

.. code-block:: yaml

   _torchvision_ddp:
     inherits: _defaults
     definition: ../benchmarks/torchvision_ddp
     group: torchvision
     install_group: torch
     plan:
       method: njobs
       n: 1

Milabench will essentially execute something akin to below.

.. code-block:: bash

   echo "---"
   echo "lightning-gpus"
   echo "=============="
   time (
     $BASE/venv/torch/bin/benchrun --nnodes=1 --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:29400 --master-addr=127.0.0.1 --master-port=29400 --nproc-per-node=8 --no-python -- python $SRC/milabench/benchmarks/lightning/main.py --epochs 10 --num-workers 8 --loader pytorch --data $BASE/data/FakeImageNet --model resnet152 --batch-size 16 &
     wait
   )







