
Creating a new benchmark
------------------------

To define a new benchmark (let's assume it is called ``ornatebench``), 

.. code-block:: bash

    git clone https://github.com/mila-iqia/milabench.git
    git checkout -b ornatebench

    pip install -e milabench/
    milabench new --name ornatebench


You should see a directory with the following structure:

.. code-block::

    milabench
    └── benchmarks
        └── ornatebench
            ├── README.md          # Document the benchmark here
            ├── benchfile.py       # Benchmark definition file
            ├── main.py            # Executed by milabench run
            ├── prepare.py         # Executed by milabench prepare (EXECUTABLE)
            ├── requirements.in    # Python requirements to install from pip
            └── voirfile.py        # Probes and extra instruments

Some of these files may be unnecessary depending on the benchmark.

First of all, if you want to verify that everything works, you can use the ``dev.yaml`` benchmark config that comes with the template:

.. code-block:: bash

    cd milabench/benchmarks/ornatebench

    milabench install --config dev.yaml --base .

    milabench prepare --config dev.yaml --base .

    milabench run     --config dev.yaml --base .


Overview
~~~~~~~~

benchfile.py
++++++++++++

``benchfile.py`` defines what to do on ``milabench install/prepare/run``. 
It is run from the benchmark directory directly, in the *current* virtual environment, 
but it can create *new processes* in the virtual environment of the benchmark.

By default it will dispatch to ``requirements.in`` for install requirements, 
``prepare.py`` for prep work and downloading datasets, and
``main.py`` for running the actual benchmark. 
If that is suitable you may not need to change it at all.


requirements.in
+++++++++++++++

Write all of the benchmark's requirements in this file. 
Use ``milabench install --config benchmarks/ornatebench/dev.yaml`` 
to install them during development (add ``--force`` if you made changes and want to reinstall.)


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
        observer = BenchObserver(batch_size_fn=lambda batch: 1)
        criterion = observer.criterion(criterion)
        optimizer = observer.optimizer(optimizer)

        for epoch in range(10):
            for i in observer.iterate(dataloader):
                # ...
                time.sleep(0.1)

* Create a new bench observer, this class is used to time the benchmark and measure batch times.
    * Set ``batch_size_fn`` to provide a function to compute the right batch size given a batch.
* ``observer.criterion(criterion)`` will wrap the criterion function so the loss will be reported automatically.
* ``observer.optimizer(optimizer)`` will wrap the optimizer so device that need special handling can have their logic executed there
* Wrap the batch loop with ``observer.iterate``, it will take care of timing the body of the loop and handle early stopping if necessary

.. note::

   Avoid calls to ``.item()``, ``torch.cuda`` and ``torch.cuda.synchronize()``.
   To access ``cuda`` related features use ``accelerator`` from torchcompat.
   ``accelerator`` is a light wrapper around ``torch.cuda`` to allow a wider range of devices to be used.

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

* Forcefully stop the program after a certain number of rate measurements.
* Monitor GPU usage.


Development
~~~~~~~~~~~

To develop the benchmark, first run ``milabench dev --config benchmarks/BENCHNAME/dev.yaml``. 
This will activate the benchmark's virtual environment and put you into a shell.

Then, try and run ``voir --dash main.py``. This should show you a little dashboard and display losses, 
train rate calculations and one or more progress bars.

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


Adapting existing repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To simplify the creation of benchmarks, milabench can use ptera and voir to override or wrap code from a third party
without modifying the third party code.

.. code-block:: bash

    git clone https://github.com/mila-iqia/milabench.git
    git checkout -b ornatebench

    pip install -e milabench/
    milabench new --name ornatebench --repo-url https://github.com/Delaunay/extern_example.git


The instrumentation is inserted inside ``voirfile.py`` using the ``Overseer.probe``, examples can 
be found in ptera `documentation <https://ptera.readthedocs.io/en/latest/guide.html#probing>`_


Wrap a return value
+++++++++++++++++++


.. code-block:: python

    class Wrapper:
        def __init__(self, value):
            self.value = value

    def wrap(original):
        return Wrapper(original)

    probe = ov.probe("//my_optimizer_creator() as optimizer", overridable=True)
    probe['optimizer'].override(wrap)


* ``//my_optimizer_creator() as optimizer``: get the return value of a function inside the main script
* ``/module.path.function() as optimizer``: get the return value of a function inside a module
* ``/module.path.function > loss_fn``: get a variable inside a function inside a module

