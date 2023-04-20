Docker
======

`Docker Images <https://github.com/mila-iqia/milabench/pkgs/container/milabench>`_ are created for each release. They come with all the benchmarks installed and the necessary datasets. No additional downloads are necessary.

CUDA
----

Requirements
^^^^^^^^^^^^

* NVIDIA driver
* `docker-ce <https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository>`_
* `nvidia-docker <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>`_


Usage
^^^^^

The commands below will download the lastest cuda container and run milabench right away,
storing the results inside the ``results`` folder on the host machine:

.. code-block:: bash

   # Choose the image you want to use
   export MILABENCH_IMAGE=ghcr.io/mila-iqia/milabench:cuda-nightly

   # Pull the image we are going to run
   docker pull $MILABENCH_IMAGE

   # Run milabench
   docker run -it --rm --ipc=host --gpus=all      \
         -v $(pwd)/results:/milabench/envs/runs   \
         $MILABENCH_IMAGE                         \
         milabench run

``--ipc=host`` removes shared memory restrictions, but you can also set ``--shm-size`` to a high value instead (at least ``8G``, possibly more).

Each run should store results in a unique directory under ``results/`` on the host machine. To generate a readable report of the results you can run:

.. code-block:: bash

   # Show Performance Report
   docker run -it --rm                             \
         -v $(pwd)/results:/milabench/envs/runs    \
         $MILABENCH_IMAGE                          \
         milabench report --runs /milabench/envs/runs


ROCM
----

Requirements
^^^^^^^^^^^^

* rocm
* docker

Usage
^^^^^

For ROCM the usage is similar to CUDA, but you must use a different image and the Docker options are a bit different:

.. code-block:: bash

   # Choose the image you want to use
   export MILABENCH_IMAGE=ghcr.io/mila-iqia/milabench:rocm-nightly

   # Pull the image we are going to run
   docker pull $MILABENCH_IMAGE

   # Run milabench
   docker run -it --rm  --ipc=host                           \
         --device=/dev/kfd --device=/dev/dri                 \
         --security-opt seccomp=unconfined --group-add video \
         -v $(pwd)/results:/milabench/envs/runs              \
         $MILABENCH_IMAGE                                    \
         milabench run

For the performance report, it is the same command:

.. code-block:: bash

   # Show Performance Report
   docker run -it --rm                             \
         -v $(pwd)/results:/milabench/envs/runs    \
         $MILABENCH_IMAGE                          \
         milabench report --runs /milabench/envs/runs


Multi-node benchmark
^^^^^^^^^^^^^^^^^^^^

There is currently one multi-node benchmark, ``opt-2_7b-multinode``. Here is how to run it:

* Provided you have the following two machines that can see each other on the network:
  * ``manager-node``
  * ``worker-node``
* ``docker pull`` the image on both nodes.
* Prior to running the benchmark, create a SSH key pair on ``manager-node`` and set up public key authentication to ``worker-node``.
* On ``manager-node``, execute ``milabench run`` via Docker.
  * Mount the private key at ``/milabench/id_milabench`` in the container
  * Use ``--override`` statements as shown below to tell milabench about both nodes

The command should look something like this:

.. code-block:: bash

    # On manager-node:

    # Modify these variables to match your setup
    export SSH_KEY=$HOME/.ssh/id_rsa
    export NODE1=manager-node
    export NODE2=worker-node
    export NUM_MACHINES=2
    export MILABENCH_USER=$USER  # The user on worker-node that public key auth is set up for

    docker run -it --rm --gpus all --network host --ipc=host --privileged \
      -v $SSH_KEY:/milabench/id_milabench \
      -v $(pwd)/results:/milabench/envs/runs \
      $MILABENCH_IMAGE \
      milabench run \
      --override opt-2_7b-multinode.docker_image='"'$MILABENCH_IMAGE'"' \
      --override opt-2_7b-multinode.manager_addr='"'$NODE1'"' \
      --override opt-2_7b-multinode.worker_addrs='["'$NODE2'"]' \
      --override opt-2_7b-multinode.worker_user='"'$MILABENCH_USER'"' \
      --override opt-2_7b-multinode.num_machines=$NUM_MACHINES \
      --capabilities nodes=$NUM_MACHINES \
      --select opt-2_7b-multinode

The last line (``--select opt-2_7b-multinode``) specifically selects the multi-node benchmark. Omit that line to run all benchmarks.

For 4 nodes, use ``--override opt-2_7b-multinode.worker_addrs='["'$NODE2'","'$NODE3'","'$NODE4'"]'`` (and of course ``NUM_MACHINES=4``).

.. note::
      The multi-node benchmark is sensitive to network performance. If the mono-node benchmark ``opt-2_7b`` is significantly faster than ``opt-2_7b-multinode``, this likely indicates that Infiniband is either not present or not used.

      Even if Infiniband is properly configured, the benchmark may fail to use it unless the ``--privileged`` flag is set when running the container.


Building images
---------------

Images can be built locally for prototyping and testing.

.. code-block::

   docker build -f docker/Dockerfile-cuda -t milabench:cuda-nightly --build-arg CONFIG=standard.yaml .

Or for ROCm:

.. code-block::

   docker build -f docker/Dockerfile-rocm -t milabench:rocm-nightly --build-arg CONFIG=standard.yaml .
