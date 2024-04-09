
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

1. Set the ``$MILABENCH_BASE`` environment variable to the base directory in which all the code, virtual environments and data should be put.

2. Set the ``$MILABENCH_CONFIG`` environment variable to the configuration file that represents the benchmark suite you want to run. Normally it should be set to ``config/standard.yaml``.

3. ``milabench install``: Install the individual benchmarks in virtual environments.

4. ``milabench prepare``: Download the datasets, weights, etc.

If the machine has both NVIDIA/CUDA and AMD/ROCm GPUs, you may have to set the ``MILABENCH_GPU_ARCH`` environment variable as well, to either ``cuda`` or ``rocm``.


Run milabench
~~~~~~~~~~~~~

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
~~~~~~~

The following command will print out a report of the tests that ran, the metrics and if there were any failures. It will also produce an HTML report that contains more detailed information about errors if there are any.

.. code-block:: bash

    milabench report --runs $MILABENCH_BASE/runs/some_specific_run --html report.html

The report will also print out a score based on a weighting of the metrics, as defined in the file ``$MILABENCH_CONFIG`` points to.


Use milabench on the cloud
~~~~~~~~~~~~~~~~~~~~~~~~~~


Setup Terraform and a free Azure account
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. | Install azure cli (it does not need to be in the same environment than
     milabench)
   | ``pip install azure-cli``

2. Setup a free account on
   `azure.microsoft.com <https://azure.microsoft.com/en-us/free/>`_

3. Follow instructions in the
   `azurerm documentation <https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/service_principal_client_secret#creating-a-service-principal-using-the-azure-cli>`_
   to generate a ``ARM_CLIENT_ID`` as well as a ``ARM_CLIENT_SECRET``. If you
   don't have the permissions to create / assign a role to a service principal,
   you can ignore the ``az ad sp create-for-rbac`` command to work directly with
   your ``ARM_TENANT_ID`` and ``ARM_SUBSCRIPTION_ID``

4. `Install Terraform <https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli>`_

5. Configure the ``azurerm`` Terraform provider by
   `exporting the environment variables <https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs/guides/service_principal_client_secret#configuring-the-service-principal-in-terraform>`_


Create a cloud system configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add a ``cloud_profiles`` section to the ``system`` configuration which lists the
supported cloud profiles.

.. notes::

  Nodes that should be created on the cloud should have the ``1.1.1.1`` ip
  address placeholder. Other ip addresses will be used as-is and no cloud
  instance will be created for that node

.. notes::

  A cloud profile entry needs to start with a covalent plugin (e.g. `azure`). To
  define multiple profiles on the same cloud platform, use the form
  ``{PLATFORM}__{PROFILE_NAME}`` (e.g. ``azure__profile``). All cloud profile
  attributes will be used as is as argument for the target covalent plugin

.. code-block:: yaml

  system:
    nodes:
      - name: manager
        # Use 1.1.1.1 as an ip placeholder
        ip: 1.1.1.1
        main: true
        user: <username>
      - name: node1
        ip: 1.1.1.1
        main: false
        user: <username>
  
    # Cloud instances profiles
    cloud_profiles:
      # The cloud platform to use in the form of {PLATFORM} or
      # {PLATFORM}__{PROFILE_NAME}
      azure__free:
        # covalent-azure-plugin args
        username: ubuntu
        size: Standard_B2ats_v2
        location: eastus2


Run milabench on the cloud
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. | Initialize the cloud instances
   | ``milabench cloud --system {{SYSTEM_CONFIG.YAML}} --setup --run-on {{PROFILE}} >{{SYSTEM_CLOUD_CONFIG.YAML}}``

2. | Prepare, install and run milabench
   | ``milabench [prepare|install|run] --system {{SYSTEM_CLOUD_CONFIG.YAML}}``
