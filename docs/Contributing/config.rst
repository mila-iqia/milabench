
.. warning::

  Outdated information, do not put in toctree.


Benchmark configuration
-----------------------

The configuration has two sections:

* ``defaults`` defines a template for benchmarks.
* ``benchmarks`` defines the benchmarks. Each benchmark may include the defaults with the special directive ``<<< *defaults``. Note that the ``<<<`` operator performs a deep merge. For example:

.. code-block:: yaml

    defaults: &defaults
      plan:
        method: njobs
        n: 2

    benchmarks:
      test:
        <<<: *defaults
        plan:
          n: 3

is equivalent to:


.. code-block:: yaml

    benchmarks:
      test:
        plan:
          method: njobs
          n: 3

### Fields

Let's say you have the following ``benchmark.yaml`` configuration:

.. code-block:: yaml

    benchmarks:
      mnist:
        definition: ../benchmarks/mnist-pytorch-example
    
        dirs:
          code: code/{name}
          venv: venv/{name}
          data: data
          runs: runs
    
        plan:
          method: njobs
          n: 2
    
        voir:
          --stop: 200
          --forward:
            - "#stdout"
            - "#stderr"
            - loss
            - compute_rate
            - train_rate
            - loading_rate
          --compute-rate: true
          --train-rate: true
          --loading-rate: true
    
        argv:
          --batch-size: 64

* ``definition`` points to the *definition directory* for the benchmark (more on that later). Important note: the path is *relative to benchmark.yaml*.
* ``dirs`` defines the directories for the venv, code, data and runs. Normally, this is set in the defaults, but it is technically possible to override it for every benchmark. The paths are relative to ``$MILABENCH_BASE`` (or the argument to ``--base``) ``code/{name}`` expands to ``code/mnist``.
* ``plan`` describes the way tasks will be created for this benchmark. ``nruns`` just launches a fixed number of parallel processes.
* ``voir`` are the arguments given to the ``voir`` command when running a task. The ``--forward`` argument is important because it defines what will end up in the final ``json`` output saved to the disk. Some of them correspond to what other flags output.
* ``argv`` are the arguments given to the benchmark script.
