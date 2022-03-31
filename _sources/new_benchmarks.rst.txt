
Creating a new benchmark
------------------------

To define a new benchmark, create a directory with roughly the following files:

.. code-block::

    mybench
    |- manifest        # Lists the file milabench install should copy (accepts wildcards)
    |- benchfile.py    # Benchmark definition file
    |- voirfile.py     # Probes and extra instruments
    |- prepare.py      # Executed by milabench prepare
    |- main.py         # Executed by milabench run
    |- dev.yaml        # Bench file to use for development

TODO: rest of the instructions
