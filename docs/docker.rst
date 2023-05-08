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
   docker run -it --rm  --ipc=host                                                  \
         --device=/dev/kfd --device=/dev/dri                                        \
         --security-opt seccomp=unconfined --group-add video                        \
         -v /opt/amdgpu/share/libdrm/amdgpu.ids:/opt/amdgpu/share/libdrm/amdgpu.ids \
         -v /opt/rocm:/opt/rocm                                                     \
         -v $(pwd)/results:/milabench/envs/runs                                     \
         $MILABENCH_IMAGE                                                           \
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

There are currently two multi-node benchmarks, ``opt-1_3b-multinode`` (data-parallel) and ``opt-6_7b-multinode`` (model-parallel, that model is too large to fit on a single GPU). Here is how to run them:

1. Set up two or more machines that can see each other on the network. Suppose there are two and their addresses are:
  * ``manager-node`` ⬅ this is the node you will launch the job on
  * ``worker-node``
2. ``docker pull`` the image on both nodes.
3. Prior to running the benchmark, create a SSH key pair on ``manager-node`` and set up public key authentication to the other nodes (in this case, ``worker-node``).
4. Write an override file that will tell milabench about the network (see below). Note that you will need to copy/paste the same configuration for both multinode tests.
5. On ``manager-node``, execute ``milabench run`` via Docker.
  * Mount the private key at ``/milabench/id_milabench`` in the container
  * Use ``--override "$(cat overrides.yaml)"`` to pass the overrides

Example YAML configuration (``overrides.yaml``):

.. code-block:: yaml

    # Name of the benchmark. You can also override values in other benchmarks.
    opt-6_7b-multinode:

      # Docker image to use on the worker nodes (should be same as the manager)
      docker_image: "ghcr.io/mila-iqia/milabench:cuda-nightly"

      # The user on worker-node that public key auth is set up for
      worker_user: "username"

      # Address of the manager node from the worker nodes
      manager_addr: "manager-node"

      # Addresses of the worker nodes (do not include the manager node,
      # although it is also technically a worker node)
      worker_addrs:
        - "worker-node"

      # Make sure that this is equal to length(worker_addrs) + 1
      num_machines: 2

      capabilities:
        # Make sure that this is ALSO equal to length(worker_addrs) + 1
        nodes: 2

    opt-1_3b-multinode:
      # Copy the contents of the opt-6_7b-multinode section without any changes.
      docker_image: "ghcr.io/mila-iqia/milabench:cuda-nightly"
      worker_user: "username"
      manager_addr: "manager-node"
      worker_addrs:
        - "worker-node"
      num_machines: 2
      capabilities:
        nodes: 2


Then, the command should look like this:

.. code-block:: bash

    # On manager-node:

    # Change if needed
    export SSH_KEY_FILE=$HOME/.ssh/id_rsa

    docker run -it --rm --gpus all --network host --ipc=host --privileged \
      -v $SSH_KEY_FILE:/milabench/id_milabench \
      -v $(pwd)/results:/milabench/envs/runs \
      $MILABENCH_IMAGE \
      milabench run --override "$(cat overrides.yaml)" \
      --select multinode

The last line (``--select multinode``) specifically selects the multi-node benchmarks. Omit that line to run all benchmarks.

If you need to use more than two nodes, edit or copy ``overrides.yaml`` and simply add the other nodes' addresses in ``worker_addrs`` and adjust ``num_machines`` and ``capabilities.nodes`` accordingly. For example, for 4 nodes:

.. code-block:: yaml

    opt-6_7b-multinode:
      docker_image: "ghcr.io/mila-iqia/milabench:cuda-nightly"
      worker_user: "username"
      manager_addr: "manager-node"
      worker_addrs:
        - "worker-node1"
        - "worker-node2"
        - "worker-node3"
      num_machines: 4
      capabilities:
        nodes: 4

.. note::
      The multi-node benchmark is sensitive to network performance. If the mono-node benchmark ``opt-6_7b`` is significantly faster than ``opt-6_7b-multinode`` (e.g. processes more than twice the items per second), this likely indicates that Infiniband is either not present or not used. (It is not abnormal for the multinode benchmark to perform *a bit* worse than the mono-node benchmark since it has not been optimized to minimize the impact of communication costs.)

      Even if Infiniband is properly configured, the benchmark may fail to use it unless the ``--privileged`` flag is set when running the container.


Building images
---------------

Images can be built locally for prototyping and testing.

.. code-block::

   docker build -f docker/Dockerfile-cuda -t milabench:cuda-nightly --build-arg CONFIG=standard.yaml .

Or for ROCm:

.. code-block::

   docker build -f docker/Dockerfile-rocm -t milabench:rocm-nightly --build-arg CONFIG=standard.yaml .
