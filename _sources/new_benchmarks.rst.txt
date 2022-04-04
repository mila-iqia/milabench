
Creating a new benchmark
------------------------

To define a new benchmark (let's assume it is called ``ornatebench``), make a copy of ``benchmarks/_template`` using ``cp-template``:

.. code-block:: bash

    cp-template benchmarks/_template benchmarks/ornatebench

You should see a directory with the following structure:

.. code-block::

    ornatebench
    |- README.md          # Document the benchmark here
    |- benchfile.py       # Benchmark definition file
    |- dev.yaml           # Bench file to use for development
    |- main.py            # Executed by milabench run
    |- manifest           # Lists the file milabench install should copy
    |- prepare.py         # Executed by milabench prepare (EXECUTABLE)
    |- requirements.txt   # Python requirements to install from pip
    |- voirfile.py        # Probes and extra instruments

Some of these files may be unnecessary depending on the benchmark. Others, like ``manifest``, you can just leave alone.

First of all, if you want to verify that everything works, you can use the ``dev.yaml`` benchmark config that comes with the template:

.. code-block:: bash

    milabench install benchmarks/ornatebench/dev.yaml --dev
    milabench prepare benchmarks/ornatebench/dev.yaml --dev
    milabench run benchmarks/ornatebench/dev.yaml --dev

.. tip::
    The ``--dev`` flag does a few things:

    * Forces use of the current virtual environment.
    * Syncs any of your changes to the copy of the code under ``$MILABENCH_BASE``.
    * Force the use of only one job.

The ``dev.yaml`` config is specially configured for development: the ``code`` directory is forced to be ``_dev_<benchdir>`` (relative to your current directory, so not necessarily in ``$MILABENCH_BASE``). So for example if you are writing a benchmark in ``benchmarks/ornatebench``, ``milabench install benchmarks/ornatebench/dev.yaml`` will install the code for the benchmark in ``_dev_ornatebench/``.


Overview
~~~~~~~~


The benchfile
+++++++++++++

``benchfile.py`` defines what to do on ``milabench install/prepare/run``. It is run from the benchmark directory directly, in the *current* virtual environment, but it can create *new processes* in the virtual environment of the benchmark.

By default it will dispatch to ``requirements.txt`` for install requirements, ``prepare.py`` for prep work and downloading datasets, and ``main.py`` for running the actual benchmark. If that is suitable you may not need to change it at all.


Prepare/main scripts
++++++++++++++++++++

The ``prepare.py`` and ``main.py`` scripts are run in the venv for the benchmark, which is isolated.

Moreover, they are **not** run from the benchmark directory, the reason being that in the course of the install process, the files in the benchmark directory may be layered over files pulled from GitHub or other places. For example, if you want to make a benchmark out of some script that is on GitHub called ``train.py``, you may clone it during install (in ``benchfile.py``), then in a ``voirfile.py`` you write instrumentation that will be copied over to the cloned code to extract the data you need. So the directory seen by ``milabench run`` may contain a lot more files than your benchmark directory, all depending.

This is why you should use the ``--dev`` flag during development, because that flag takes care of syncing your changes to the install location (without re-running the rest of the install process).


Benchmark from scratch
~~~~~~~~~~~~~~~~~~~~~~

If you are writing a benchmark from scratch, using pip installable libraries:

* Put the requirements in ``requirements.txt``. It's unlikely you have to do anything fancier than that, but if you do, you can run arbitrary commands in ``install`` in the benchfile.
* If you need to download or generate data, write the code in ``prepare.py``. Remember that the root data directory is in ``$MILABENCH_DIR_DATA``, but do not store data directly in there, store it in the appropriate subdirectory. Do not repeat work if the data is already there.
* Write the main code in ``main.py``.
* Write a ``voirfile.py`` that corresponds to the directions in :ref:`probingmb`.
* You can check the progress of your probing with ``voir --verify main.py``


Adapting existing code
~~~~~~~~~~~~~~~~~~~~~~

Now, let's say you want to adapt code from the repo at ``https://github.com/snakeoilplz/agi``, more specifically the ``train.py`` script. Requirements are listed in ``requirements.txt`` in the repo.

First, remove ``main.py`` and ``requirements.txt`` from the benchmark directory. You won't need them.

Your benchfile should look like this:

.. code-block:: python

    from milabench.pack import Package

    BRANCH = "master"

    class TheBenchmark(Package):
        main_script = "train.py"
        prepare_script = "prepare.py"

        def install(self):
            code = self.dirs.code
            code.clone_subtree("https://github.com/snakeoilplz/agi", BRANCH)
            self.pip_install("-r", code / "requirements.txt")

    __pack__ = TheBenchmark

Run ``milabench install dev.yaml`` to install. In ``_dev_agi`` (assuming the project name is ``agi``), you should have a checkout of the target repository.

Now, you can ``cd`` to that directory and see if ``python train.py`` actually works. If it doesn't, well, look at the ``README``, try to make it work, and add all the necessary steps to the ``install`` method.

If a dataset is needed on disk, write the code in ``prepare.py``. Remember that the root data directory is in ``$MILABENCH_DIR_DATA``, but do not store data directly in there, store it in the appropriate subdirectory. Do not repeat work if the data is already there. Make sure to have ``train.py`` look for the data at the right place, either through a flag or an environment variable.

Once it works, run it with ``voir --verify train.py``, which shouldn't give you much new information. But now you can follow these steps in a loop:

* Modify ``voirfile.py`` *in the benchmark directory* accordingly to the directions in :ref:`probingmb`.
* Run ``milabench install dev.yaml --sync`` to transfer your changes to the dev directory.
* Run ``voir --verify train.py`` again, until everything looks good.

Once it looks good, it's just a matter of putting the necessary arguments into the benchmark YAML. You can set environment variables in the benchfile with ``make_env()``.

Finally, try ``milabench install/prepare/run`` without the ``--dev`` flag.
