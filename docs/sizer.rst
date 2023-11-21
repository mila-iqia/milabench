Scaling
=======

Milabench is able to select a batch size depending on the
underlying GPU capacity.

The feature is drivent by the ``config/scaling.yaml`` file, 
which holds information about the memory usage of a given bench
given the batch size.


.. code-block:: yaml

   convnext_large-fp32:
     arg: --batch-size
     default: 128
     model:
       8: 5824.75 MiB
       16: 8774.75 MiB
       32: 14548.75 MiB
       64: 26274.75 MiB
       128: 49586.75 MiB


Auto Batch size
---------------

To enable batch resizing an environment variable can be specified.
It will use the capacity inside the `system.yaml` configurattion file.

.. code-block:: yaml

   system:
     arch: cuda
     gpu:
       capacity: 81920 MiB
     nodes: []


.. code-block:: bash
    
   MILABENCH_SIZER_AUTO=1 milabench run --system system.yaml


For better performance, a multiple constraint can be added.
This will force batch size to be a multiple of 8.

.. code-block:: bash
   
   MILABENCH_SIZER_MULTIPLE=8 milabench run


Batch size override
-------------------

The batch size can be globally overriden

.. code-block:: bash

   MILABENCH_SIZER_BATCH_SIZE=64 milabench run


Memory Usage Extractor
----------------------

To automate batch size ``<=>`` memory usage data gathering
a validation layer that retrieve the batch size and the memory usage
can be enabled.

In the example below, once milabench has finished running it will
generate a new scaling configuration with the data extracted from the run.


.. code-block:: bash

   export MILABENCH_SIZER_SAVE="newscaling.yaml"
   MILABENCH_SIZER_BATCH_SIZE=64 milabench run

