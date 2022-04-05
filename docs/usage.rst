
Install
-------

To install for development, clone the repo and use branch ``v2``:

.. code-block:: bash

    # You may need to upgrade pip
    pip install pip -U
    git clone -b v2 git@github.com:mila-iqia/milabench.git
    cd milabench
    # <Activate virtual environment>
    # Install in editable mode
    pip install -e .

This will install two commands, ``milabench`` and ``voir``.


Using milabench
---------------


To use ``milabench``, you need:

* A YAML configuration file to define the benchmarks to install, prepare or run.
* The base directory for code, virtual environments, data and outputs, set either with the ``$MILABENCH_BASE`` environment variable or the ``--base`` option. The base directory will be automatically constructed by milabench and will be organized as follows:


.. code-block::

    $MILABENCH_BASE/
    |- venv/                            # Virtual environments and dependencies
    |  |- bench1/                       # venv for benchmark bench1
    |  |- ...                           # etc
    |- code/                            # Benchmark code
    |  |- bench1/                       # Code for benchmark bench1
    |  |- ...                           # etc
    |- data/                            # Datasets
    |  |- dataset1/                     # A dataset
    |  |- ...                           # etc
    |- runs/                            # Outputs of benchmark runs
       |- calimero.2022-03-30_15:00:00/ # Auto-generated run name
       |  |- bench1.0.json              # Output for the first run of bench1
       |  |- bench1.1.json              # Output for the second run of bench1
       |  |- ...                        # etc
       |- blah/                         # Can set name with --run

It is possible to change the structure in the YAML to e.g. force benchmarks to all use the same virtual environment.

Important options
~~~~~~~~~~~~~~~~~

* Use the ``--select`` option with a comma-separated list of benchmarks in order to only install/prepare/run these benchmarks (or use ``--exclude`` to run all benchmarks except a specific set).
* You may use ``--use-current-env`` to force the use the currently active virtual environment (useful for development).

milabench install
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    milabench install benchmarks.yaml --select mybench

* Copies the code for the benchmark (specified in the ``definition`` field of the benchmark's YAML, relative to the YAML file itself) into ``$MILABENCH_BASE/code/mybench``. Only files listed by the ``manifest`` file are copied.
* Creates/reuses a virtual environment in ``$MILABENCH_BASE/venv/mybench`` and installs all pip dependencies in it.
* Optionally extracts a shallow git clone of an external repository containing model code into ``$MILABENCH_BASE/code/mybench``.

milabench prepare
~~~~~~~~~~~~~~~~~


.. code-block:: bash

    milabench prepare benchmarks.yaml --select mybench


* Prepares data for the benchmark into ``$MILABENCH_BASE/data/dataset_name``. Multiple benchmarks can share the same data. Some benchmarks need no preparation, so the prepare step does nothing.

milabench run
~~~~~~~~~~~~~

.. code-block:: bash

    milabench run benchmarks.yaml --select mybench

* Creates a certain number of tasks from the benchmark using the ``plan`` defined in the YAML. For instance, one plan might be to run it in parallel on each GPU on the machine.
* For each task, runs the benchmark installed in ``$MILABENCH_BASE/code/mybench`` in the appropriate virtual environment.
* The benchmark is run from that directory using a command like ``voir [VOIR_OPTIONS] main.py [SCRIPT_OPTIONS]``
  * Both option groups are defined in the YAML.
  * The VOIR_OPTIONS determine which instruments to use and what data to forward to milabench.
  * The SCRIPT_OPTIONS are benchmark dependent.
* Standard output/error and other data (training rates, etc.) are forwarded to the main dispatcher process and saved into ``$MILABENCH_BASE/runs/run_name/mybench.run_number.json`` (the name of the directory is printed out for easy reference).

milabench report
~~~~~~~~~~~~~~~~

TODO.

.. code-block:: bash

    milabench report benchmarks.yaml --run <run_name>
