
Install and use
---------------

.. note::

  You may use Docker to run the benchmarks, which will likely be easier. See the Docker section of this documentation for more information.


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

1. Create a system configuration

  Milabench can run on multiple accelerator and multiple node.
  The ``system.yaml`` sepecify setup specific values that will be used by milabench to run the benchmarks.
  You can find an example below.

  * ``sshkey``: path to a privatekey to use to ssh to the worker nodes
  * ``arch``: GPU accelerator kind, this is optional, if not specified milabench will try to deduce the backend
    It might be necessary if accelerators from multiple vendors are installed on the system.
  * ``docker_image``: docker image to load up on worker nodes for multi-node benchmarks
  * ``nodes``: A list of worker node that will run the benchmarks.

  .. notes::

    The main node is the node that will be in charge of managing the other worker nodes.

  .. code-block:: yaml

    system:
      sshkey: <privatekey>
      arch: cuda
      docker_image: ghcr.io/mila-iqia/milabench:${system.arch}-nightly

      nodes:
        - name: node1
          ip: 192.168.0.25
          main: true
          port: 8123
          user: <username>

        - name: node2
          ip: 192.168.0.26
          main: false
          user: <username>


1. Set the ``$MILABENCH_BASE`` environment variable to the base directory in which all the code,
   virtual environments and data should be put.

  .. code-block:: text

    base/                  # folder $MILABENCH_BASE
    ├── extra              # Cache and lock files
    ├── data               # Dataset used by benchmarks
    ├── venv/              # Virtual environment created for each benchmark
    │   └── torch
    └── runs/              # Benchmark metrics
          ├── run_name_1
          └── run_name_2


2. Set the ``$MILABENCH_CONFIG`` environment variable to the configuration file that represents the benchmark suite you want to run.
   Normally it should be set to ``config/standard.yaml``.

3. ``milabench install --system system.yaml``: Install the individual benchmarks in virtual environments.

4. ``milabench prepare --system system.yaml``: Download the datasets, weights, etc.

If the machine has both NVIDIA/CUDA and AMD/ROCm GPUs, you may have to set the
``MILABENCH_GPU_ARCH`` environment variable as well, to either ``cuda`` or ``rocm``.


Run milabench
~~~~~~~~~~~~~

The following command will run the whole benchmark and will put the results in a new directory in ``$MILABENCH_BASE/runs`` (the path will be printed to stdout).

.. code-block:: bash

  milabench run

Here are a few useful options for ``milabench run``:

.. code-block:: bash

  # Only run the bert benchmark
  milabench run --system system.yaml --select bert

  # Run all benchmarks EXCEPT bert and stargan
  milabench run --system system.yaml --exclude bert,stargan

  # Run the benchmark suite three times in a row
  milabench run --system system.yaml --repeat 3


Batch Resizing
^^^^^^^^^^^^^^

Milabench supports automatic batch resize to accomodate different GPU memory capacity.
The feature is disabled by default and can be enabled using the environment variable ``MILABENCH_SIZER_AUTO``.
Additional constraint on the memory usage can be set to test for different condition.

.. code-block:: text

   MILABENCH_SIZER_BATCH_SIZE int      # Override the batch size
   MILABENCH_SIZER_AUTO       False    # Enable autoscaling from the GPU max memory
   MILABENCH_SIZER_MULTIPLE   int      # Force the Batch size to be a multiple of something
   MILABENCH_SIZER_OPTIMIZED  int      # Use configured batch
   MILABENCH_SIZER_CAPACITY   str      # Override GPU max memory


We recommend the following constraint:

.. code-block:: text

   export MILABENCH_SIZER_AUTO=1
   export MILABENCH_SIZER_MULTIPLE=8


Reports
~~~~~~~

The following command will print out a report of the tests that ran, the metrics and if there were any failures. It will also produce an HTML report that contains more detailed information about errors if there are any.

.. code-block:: bash

    milabench report --runs $MILABENCH_BASE/runs/some_specific_run --html report.html

The report will also print out a score based on a weighting of the metrics, as defined in the file ``$MILABENCH_CONFIG`` points to.


.. code-block:: text

   =================
   Benchmark results
   =================
                           fail n       perf   sem%   std% peak_memory          score weight
   bert-fp16                   0 8     155.08   0.3%   4.3%       24552    1241.260310   0.00
   bert-fp32                   0 8      29.52   0.0%   0.5%       31524     236.337218   0.00
   bert-tf32                   0 8     120.46   0.4%   6.1%       31524     964.713297   0.00
   bert-tf32-fp16              0 8     154.76   0.3%   4.1%       24552    1238.477257   3.00
   convnext_large-fp16         0 8     337.48   0.9%  14.0%       27658    2741.604444   0.00
   convnext_large-fp32         0 8      44.61   0.8%  12.6%       49786     354.207225   0.00
   convnext_large-tf32         0 8     135.99   0.7%  11.2%       49786    1089.394916   0.00
   convnext_large-tf32-fp16    0 8     338.58   0.8%  13.0%       27658    2744.325170   3.00
   davit_large                 0 8     312.79   0.3%   6.7%       35058    2515.326450   1.00
   davit_large-multi           0 1    2401.65   1.0%   7.7%       42232    2401.651720   5.00
   dlrm                        0 1  188777.20   1.8%  14.0%        3194  188777.203190   1.00
   focalnet                    0 8     400.47   0.2%   5.4%       26604    3215.431924   2.00
   opt-1_3b                    0 1      26.71   0.1%   0.4%       44116      26.714365   5.00
   opt-1_3b-multinode          0 2      34.62   0.2%   1.0%       43552      34.618292  10.00
   opt-6_7b                    0 1      14.32   0.0%   0.1%       55750      14.319587   5.00
   opt-6_7b-multinode          0 2      10.79   0.1%   0.7%       49380      10.792595  10.00
   reformer                    0 8      61.70   0.0%   0.9%       25376     494.110834   1.00
   regnet_y_128gf              0 8      99.96   0.2%   5.0%       31840     803.012507   2.00
   resnet152                   0 8     710.18   0.3%   6.2%       36732    5710.828608   1.00
   resnet152-multi             0 1    5367.34   1.0%   8.1%       38638    5367.338469   5.00
   resnet50                    0 8     984.43   0.9%  19.1%        5026    7927.257351   1.00
   rwkv                        0 8     428.65   0.2%   3.8%        5546    3435.097716   1.00
   stargan                     0 8      51.32   1.8%  40.8%       37848     413.238870   1.00
   super-slomo                 0 8      41.63   0.1%   2.3%       34082     332.395065   1.00
   t5                          0 8      48.05   0.2%   3.9%       35466     384.317023   2.00
   whisper                     0 8     248.16   0.0%   0.6%       37006    1985.861017   1.00

   Scores
   ------
   Failure rate:       0.00% (PASS)
   Score:             219.06
