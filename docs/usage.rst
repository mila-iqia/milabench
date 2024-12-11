
Install and use
===============

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
-----------------------------

1. Set the ``$MILABENCH_BASE`` environment variable to the base directory in which all the code, virtual environments and data should be put.

2. Set the ``$MILABENCH_CONFIG`` environment variable to the configuration file that represents the benchmark suite you want to run. Normally it should be set to ``config/standard.yaml``.

3. ``milabench install``: Install the individual benchmarks in virtual environments.

4. ``milabench prepare``: Download the datasets, weights, etc.

If the machine has both NVIDIA/CUDA and AMD/ROCm GPUs, you may have to set the ``MILABENCH_GPU_ARCH`` environment variable as well, to either ``cuda`` or ``rocm``.


Run milabench
-------------

The following command will run the whole benchmark and will put the results in a new directory in ``$MILABENCH_BASE/runs`` (the path will be printed to stdout).

.. code-block:: bash

  milabench run

Here are a few useful options for ``milabench run``:

.. code-block:: bash

  # Only run the bert benchmark
  milabench run --select bert

  # Run all benchmarks EXCEPT bert and stargan
  milabench run --exclude bert,stargan

  # Run the benchmark suite three times in a row
  milabench run --repeat 3


Reports
-------

The following command will print out a report of the tests that ran, the metrics and if there were any failures. It will also produce an HTML report that contains more detailed information about errors if there are any.

.. code-block:: bash

    milabench report --runs $MILABENCH_BASE/runs/some_specific_run --html report.html

The report will also print out a score based on a weighting of the metrics, as defined in the file ``$MILABENCH_CONFIG`` points to.


Use milabench in a multi-nodes system
-------------------------------------


Create a system configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a configuration file which contains the following structure:

.. code-block:: yaml

  system:
    # sshkey used in remote milabench operations
    sshkey: ~/.ssh/id_ed25519

    # Nodes list
    nodes:
        # Alias used to reference the node
      - name: manager
        ip: 192.168.11.11
        port: 5000
        # Use this node as the master node or not
        main: true
        # User to use in remote milabench operations
        user: manager

      - name: node2
        ip: 192.168.11.12
        main: false
        user: username

      - name: node3
        ip: 192.168.11.13
        main: false 
        user: username

      - name: node4
        ip: 192.168.11.14
        main: false 
        user: username


Create a slurm configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

| If ``milabench`` is executed in a slurm system, the generation of the
  configuration file can be automated with the ``milabench slurm_system``
  command:
| ``milabench slurm_system > config/system.yaml``


Run milabench on a multi-nodes system
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| Set the ``$MILABENCH_SYSTEM`` environment variable or use the ``--system``
  argument to identify the the system configuration file:
| ``milabench [prepare|install|run] --system config/system.yaml``