
.. warning::

  Outdated information, do not put in toctree.


Instrumenting code
------------------

Prior to creating a new benchmark, let us see how to instrument existing code in order to extract the desired metrics.

Suppose you have this simple script in ``main.py``:

.. code-block:: python

    import torch
    import torch.nn as nn
    import torchvision.datasets as datasets
    import torchvision.models as tvmodels
    import torchvision.transforms as transforms

    def train_epoch(model, criterion, optimizer, loader, device):
        for inp, target in loader:
            inp = inp.to(device)
            target = target.to(device)
            output = model(inp)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def main():
        torch.cuda.manual_seed(1234)
        device = torch.device("cuda")

        model = tvmodels.resnet18()
        model.to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), 0.1)
        train = datasets.ImageFolder("some_data_folder")
        train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

        for epoch in range(10):
            train_epoch(model, criterion, optimizer, train_loader, device)

    if __name__ == "__main__":
        main()


Probe example
~~~~~~~~~~~~~

To instrument this code, simply create a file named ``voirfile.py`` in the current directory. In that file, you can write *instruments* which will be activated by running ``voir <VOIR_OPTIONS> main.py <SCRIPT_OPTIONS>``. For example, this instrument will print out the losses as they are calculated:

.. code-block:: python

    def instrument_probes(ov):
        yield ov.phases.load_script
        ov.probe("//train_epoch > loss").display()

**Explanation:**

* An instrument is a function or generator defined in ``voirfile.py`` with a name that starts with ``instrument_``.
* It takes one parameter, an ``Overseer``.
* By yielding ``ov.phases.load_script``, the instrument asks the overseer to resume the instrument's execution once the script is loaded, but before it is executed.

  * The following phases are available:
  * ``ov.phases.init``: When the overseer is created, but before the arguments are parsed. You can add new options to ``voir`` in ``ov.argparser``.
  * ``ov.phases.parse_args``: voir's command line arguments have been parsed and are accessible in ``ov.options``.
  * ``ov.phases.load_script``: The script has been loaded.
  * ``ov.phases.run_script``: The script has been executed.
  * ``ov.phases.finalize``: After ``ov.given`` has been terminated (so reductions etc have been run).
* ``ov.probe`` sets a probe in the script's code (and/or in any module's code)

  * ``//train_epoch > loss`` is shorthand for ``/__main__/train_epoch > loss`` and it means: "instrument the loss variable inside the train_epoch function that's in the module with ``__name__ == "__main__"``.
  * For more information about probing, see `ptera <https://ptera.readthedocs.io/en/latest/guide.html>`_.
  * For more information about "absolute" probes (with a leading ``/``), see `this section <https://ptera.readthedocs.io/en/latest/guide.html#absolute-references>`_.


Giving and using data and metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ov.given`` is an instance of `Given <https://giving.readthedocs.io/en/latest/ref-gvn.html#giving.gvn.Given>`_ and is where various instruments contribute data and read data from other instruments.

**Contributing data**

* Calling ``ov.give(key=value, ...)`` in an instrument.
* Calling ``ov.probe(...).give()`` in an instrument.
* Calling ``from giving import give; give(key=value, ...)`` in a script.

**Using data**

* Using ``ov.given``'s various methods. 

**Example**

This example probes the loss using one instrument and then displays the minimum loss using a second instrument.

.. code-block:: python

    def instrument_probes(ov):
        yield ov.phases.load_script
        ov.probe("//train_epoch > loss").give()

    def instrument_display_min(ov):
        yield ov.phases.init
        ov.given["?loss"].min().print(f"Minimum loss: {}")

You can run the instruments with ``voir main.py``. In addition to that, ``voir --dash main.py`` will display everything that is given, so you will see the values of ``loss`` (as well as anything you give) change in real time.


.. _probingmb:

Probing for milabench
~~~~~~~~~~~~~~~~~~~~~

Milabench adds a few instruments to ``voir`` that can be enabled using flags, such as ``--train-rate``. To turn them all on and see whether you are instrumenting what needs to be instrumented for a benchmark, run:

.. code-block::

    voir --verify main.py

You typically don't need to do much except write a few probes. The various instruments provided by ``milabench`` will look for data with specific names and types.


Essential probes
++++++++++++++++

The **essential** probes to provide are those that allow the computation of the training rate.

* ``loss``: The current training loss. This is used as a sanity check, to verify that the algorithm is doing something.

* ``step + batch``: Both should be probed/given at the same time.

  * ``batch`` should be a tensor such that the first element of its shape is the batch size.
  * ``step`` can be anything, typically just the boolean ``True``. It merely indicates that this is the end of a training step.

* ``use_cuda``: True if using cuda for training, False otherwise. If using CUDA, cuda.synchronize will be periodically called to get more accurate readings.


Additional readings
+++++++++++++++++++

These readings can be very useful, but they are not strictly speaking necessary and may not always be available or easy to extract.

* ``options``: The parsed command line options. If the script is using the standard argparse module, you can import the already-made instrumenter in the voirfile: ``from milabench.opt import instrument_argparse``. Since the name starts with ``instrument_``, it is sufficient to import it to activate it.
* ``model``: The model object (PyTorch/etc.) that contains the parameters.
* ``loader``: The DataLoader that is being iterated on during training. The ``--loading-rate`` flag will attempt to instrument it, although that may not always work.
* ``batch + compute_start + compute_end``: This one is a bit trickier and uses `the wrapper probe feature <https://ptera.readthedocs.io/en/latest/guide.html#wrapper-probe>`_ in ptera. The ``--compute-rate`` flag uses this to calculate the time spent between the beginning and end of a loop iteration.


Example
+++++++

This is how you would provide this information for the example ``main.py`` above:

.. note::
    The special marker ``#endloop_X`` can be used in a probe to denote the end of a for loop that has iteration variable ``X``. ``#loop_X`` can be used to tag the beginning.

.. code-block:: python

    # Instrument argparse
    from milabench.opt import instrument_argparse

    def instrument_probes(ov):
        yield ov.phases.load_script

        # Give the loss
        ov.probe("//train_epoch > loss").give()

        # Give batch + step
        ov.probe("//train_epoch(inp as batch) > #endloop_inp as step").give()

        # Always use CUDA
        ov.give(use_cuda=True)

        # Give the model
        ov.probe("//main > model").give()

        # Give the loader
        ov.probe("//main > train_loader as loader").give()

        # Compute start/end
        # This basically creates a probe that has two trigger points, one
        # for ! and another for !!.
        ov.probe("//train_epoch(inp as batch, !#loop_inp as compute_start, !!#endloop_inp as compute_end)").give()


.. note::
    If, instead of being in the main script at ``main.py``, the ``train_epoch`` function is in ``somepackage/train.py``, you can simply use ``/somepackage.train/train_epoch`` instead of ``//train_epoch``.

    The syntax can be a bit tricky, so if you want you can also import the function from inside the instrumenter and use its name directly without slashes at the beginning. You can also print out the result of ``ptera.refstring(f)`` to get the string to use for the probe.
