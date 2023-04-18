
Creating a new benchmark
------------------------

To define a new benchmark (let's assume it is called ``ornatebench``), make a copy of ``benchmarks/_template`` using ``cp-template``:

.. code-block:: bash

    cp-template benchmarks/_template/ benchmarks/ornatebench

You should see a directory with the following structure:

.. code-block::

    ornatebench
    |- README.md          # Document the benchmark here
    |- benchfile.py       # Benchmark definition file
    |- main.py            # Executed by milabench run
    |- prepare.py         # Executed by milabench prepare (EXECUTABLE)
    |- requirements.in    # Python requirements to install from pip
    |- voirfile.py        # Probes and extra instruments

Some of these files may be unnecessary depending on the benchmark.

First of all, if you want to verify that everything works, you can use the ``dev.yaml`` benchmark config that comes with the template:

.. code-block:: bash

    # You can also use --config
    export MILABENCH_CONFIG=benchmarks/ornatebench/dev.yaml

    milabench install
    milabench prepare
    milabench run


Overview
~~~~~~~~


benchfile.py
++++++++++++

``benchfile.py`` defines what to do on ``milabench install/prepare/run``. It is run from the benchmark directory directly, in the *current* virtual environment, but it can create *new processes* in the virtual environment of the benchmark.

By default it will dispatch to ``requirements.in`` for install requirements, ``prepare.py`` for prep work and downloading datasets, and ``main.py`` for running the actual benchmark. If that is suitable you may not need to change it at all.


requirements.in
+++++++++++++++

Write all of the benchmark's requirements in this file. Use ``milabench install --config benchmarks/ornatebench/dev.yaml`` to install them during development (add ``--force`` if you made changes and want to reinstall.)


prepare.py
++++++++++

This script is executed in the venv for the benchmark when you run ``milabench prepare``.

The purpose of ``prepare.py`` is to download and/or generate everything that is required by the main script, so that the main script does not need to use the network and can start training right away. In particular, it must:

* Download any datasets required by the main script into ``$MILABENCH_DATA_DIR``.
* Preprocess the data, if this must be done prior to training.
* Generate synthetic data into ``$MILABENCH_DATA_DIR`` (if needed).
* Download and cache pretrained model weights (if needed).
  * Weights should ideally go somewhere under ``$XDG_CACHE_HOME`` (which milabench sets to ``$MILABENCH_BASE/cache``).
  * Note that most frameworks already cache weights in subdirectories of ``$XDG_CACHE_HOME``, so it is usually sufficient to import the framework, load the model, and then quit without training it.

If no preparation is needed, this file should be removed.


main.py
+++++++

This is the main script that will be benchmarked when you run ``milabench run``. It is run as ``voir main.py ARGV...``

The template ``main.py`` demonstrates a simple loop that you can adapt to any script:

.. code-block:: python

    def main():
        for i in voir.iterate("train", range(100), report_batch=True, batch_size=64):
            give(loss=1/(i + 1))
            time.sleep(0.1)

* Wrap the training loop with ``voir.iterate``.
  * ``report_batch=True`` triggers the computation of the number of training samples per second.
  * Set ``batch_size`` to the batch_size. milabench can also figure it out automatically if you are iterating over the input batches (it will use the first number in the tensor's shape).
* ``give(loss=loss.item())`` will forward the value of the loss to milabench. Make sure the value is a plain Python ``float``.

If the script takes command line arguments, you can parse them however you like, for example with ``argparse.ArgumentParser``. Then, you can add an ``argv`` section in ``dev.yaml``, just like this:

.. code-block:: yaml

    trivial:
      inherits: _defaults
      definition: .

      ...

      # Pass arguments to main.py below
      argv:
        --batch-size: 64

``argv`` can also be an array if you need to pass positional arguments, but I recommend using named parameters only.


voirfile.py
+++++++++++

The voirfile contains instrumentation for the main script. You can usually just leave it as it is. By default, it will:

* Compute the train "rate" (number of samples per second) using events from ``voir.iterate``.
* Forcefully stop the program after a certain number of rate measurements.
* Monitor GPU usage.


Development
~~~~~~~~~~~

To develop the benchmark, first run ``milabench dev --config benchmarks/BENCHNAME/dev.yaml``. This will activate the benchmark's virtual environment and put you into a shell.

Then, try and run ``voir --dash main.py``. This should show you a little dashboard and display losses, train rate calculations and one or more progress bars.

From there, you can develop as you would any other Python program.


Integrating in base.yaml
~~~~~~~~~~~~~~~~~~~~~~~~

You can copy-paste the contents of ``dev.yaml`` into ``config/base.yaml``, you will only need to change:

* ``definition`` should be the relative path to the ``benchfile.py``.
* Remove ``install_variant: unpinned``
* If the benchmark's requirements are compatible with those of other benchmarks, you can set ``install_group`` to the same ``install_group`` as them. For example, ``install_group: torch``.

Then, run the following commands:

* ``milabench pin --select NAME_OR_INSTALL_GROUP --variant cuda``
* ``milabench pin --select NAME_OR_INSTALL_GROUP --variant rocm``

This will create ``requirements.<arch>.txt`` for these two architectures. These files must be checked in under version control.

.. note::

    ``--variant unpinned`` means installing directly from ``requirements.in``. This can be useful during development, but less stable over time since various dependencies may break.


.. Adapting existing code
.. ~~~~~~~~~~~~~~~~~~~~~~

.. Now, let's say you want to adapt code from the repo at ``https://github.com/snakeoilplz/agi``, more specifically the ``train.py`` script.

.. TODO
