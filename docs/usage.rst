
Install and use
---------------

To install, clone the repo:

.. code-block:: bash

    # You may need to upgrade pip
    pip install pip -U
    git clone git@github.com:mila-iqia/milabench.git
    cd milabench
    # <Activate virtual environment>
    # Install in editable mode
    pip install -e .

This will install two commands, ``milabench`` and ``voir``.


Before running the benchmarks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Set the ``$MILABENCH_BASE`` environment variable to the base directory in which all the code, virtual environments and data should be put.

2. Set the ``$MILABENCH_CONFIG`` environment variable to the configuration file that corresponds to your platform:

  * ``config/standard-cuda.yaml`` for NVIDIA/CUDA platforms.
  * ``config/standard-rocm.yaml`` for AMD/ROCm platforms.

3. ``milabench install``: Install the individual benchmarks in virtual environments.

4. ``milabench prepare``: Download the datasets

If the machine has both NVIDIA/CUDA and AMD/ROCm GPUs, you may have to set the ``MILABENCH_GPU_ARCH`` environment variable as well, to either ``cuda`` or ``rocm``.


Run milabench
~~~~~~~~~~~~~

The following command will run the whole benchmark and will put the results in a new directory in ``$MILABENCH_BASE/runs`` (the path will be printed to stdout).

.. code-block:: bash

  milabench run
  
The standard tests currently include:

* ``resnet50``
* ``squeezenet1_1``
* ``efficientnet_b0``
* ``efficientnet_b4``
* ``efficientnet_b7``
* ``convnext_large``
* ``regnet_y_128gf``
* ``bert``
* ``hf_reformer``
* ``hf_t5``
* ``dlrm``
* ``soft_actor_critic``
* ``speech_transformer``
* ``super_slomo``
* ``stargan``
* ``learning_to_paint``
* ``ppo``
* ``td3``
* ``resnet152``
* ``vit_l_32``

Here are a few useful options for ``milabench run``:

.. code-block:: bash

  # Only run the bert benchmark
  milabench run --select bert

  # Run all benchmarks EXCEPT bert and stargan
  milabench run --exclude bert,stargan

  # Run the benchmark suite three times in a row
  milabench run --repeat 3


Reports
~~~~~~~

The following command will print out a report of the tests that ran, the metrics and if there were any failures. It will also produce an HTML report that contains more detailed information about errors if there are any.

.. code-block:: bash

    milabench report --runs $MILABENCH_BASE/runs/some_specific_run --html report.html

To get an overall score, you must specify a weights file that specifies how much each test weighs. Use ``weights/standard.json`` in the repo for this purpose:

.. code-block:: bash

    milabench report --runs $MILABENCH_BASE/runs/some_specific_run --weights weights/standard.json --html report.html
