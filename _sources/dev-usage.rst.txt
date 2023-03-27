
Using milabench (DEVELOPERS)
----------------------------

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
       |  |- bench1.0.stdout            # Output for the first run of bench1
       |  |- bench1.0.stderr            # Stderr for the first run of bench1
       |  |- bench1.0.data              # Structured data for the first run of bench1
       |  |- bench1.1.stdout            # Output for the second run of bench1
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

    milabench install config/standard.yaml --select mybench

* Copies the code for the benchmark (specified in the ``definition`` field of the benchmark's YAML, relative to the YAML file itself) into ``$MILABENCH_BASE/code/mybench``. Only files listed by the ``manifest`` file are copied.
* Creates/reuses a virtual environment in ``$MILABENCH_BASE/venv/mybench`` and installs all pip dependencies in it.
* Optionally extracts a shallow git clone of an external repository containing model code into ``$MILABENCH_BASE/code/mybench``.

milabench prepare
~~~~~~~~~~~~~~~~~

.. code-block:: bash

    milabench prepare config/standard.yaml --select mybench

* Prepares data for the benchmark into ``$MILABENCH_BASE/data/dataset_name``. Multiple benchmarks can share the same data. Some benchmarks need no preparation, so the prepare step does nothing.

milabench run
~~~~~~~~~~~~~

.. code-block:: bash

    milabench run config/standard.yaml --select mybench

* Creates a certain number of tasks from the benchmark using the ``plan`` defined in the YAML. For instance, one plan might be to run it in parallel on each GPU on the machine.
* For each task, runs the benchmark installed in ``$MILABENCH_BASE/code/mybench`` in the appropriate virtual environment.
* The benchmark is run from that directory using a command like ``voir [VOIR_OPTIONS] main.py [SCRIPT_OPTIONS]``
  * Both option groups are defined in the YAML.
  * The VOIR_OPTIONS determine which instruments to use and what data to forward to milabench.
  * The SCRIPT_OPTIONS are benchmark dependent.
* Standard output/error and other data (training rates, etc.) are forwarded to the main dispatcher process and saved into ``$MILABENCH_BASE/runs/run_name/mybench.run_number.stdout`` (``.stderr`` / ``.data``) (the name of the directory is printed out for easy reference).

milabench pin
~~~~~~~~~~~~~

.. code-block:: bash

    milabench pin config/standard.yaml --select mybench --variant cuda --constraint constraints/cu118.txt

The basic idea behind ``milabench pin`` is to pin software versions for stability and reproducibility. Using the command above, the base requirements in ``benchmarks/mybench/requirements.in`` will be saved in ``requirements.cuda.txt``. If variant is not specified, the value of ``install_variant`` in the config file will be used (in ``standard.yaml``, which is ``install_value: "{{arch}}"``; that resolves to either "rocm" or "cuda" depending on the machine's architecture).

The constraints file specifies appropriate constraints for the architecture, CUDA version, or other constraints that are specific to the environment.

milabench report
~~~~~~~~~~~~~~~~

TODO.

.. code-block:: bash

    milabench report config/standard.yaml --runs <path_to_runs>

milabench compare
~~~~~~~~~~~~~~~~~

TODO.
