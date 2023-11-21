Scaling
=======

Milabench is able to select a batch size depending on the
underlying GPU capacity.


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