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

There are currently two multi-node benchmarks, ``opt-1_3b-multinode`` (data-parallel) and 
``opt-6_7b-multinode`` (model-parallel, that model is too large to fit on a single GPU). Here is how to run them:

0. Make sure the machine can ssh between each other without passwords
1. Pull the milabench docker image you would like to run on all machines
  - ``docker pull``
1. Create the output directory
  - ``mkdir -p results``
2. Create a list of nodes that will participate in the benchmark inside a ``results/system.yaml`` file (see example below)
  - ``vi results/system.yaml``
3. Call milabench with by specifying the node list we created.
  - ``docker ... -v $(pwd)/results:/milabench/envs/runs -v <privatekey>:/milabench/id_milabench milabench run ... --system /milabench/envs/runs/system.yaml``

.. notes::

   The main node is the node that will be in charge of managing the other worker nodes.

.. code-block:: yaml

   system:
     sshkey: <privatekey>
     arch: cuda
     docker_image: ghcr.io/mila-iqia/milabench:${system.arch}-nightly

     nodes:
       - name: node1
         ip: 192.168.0.25
         main: true
         port: 8123
         user: <username>
      
       - name: node2
         ip: 192.168.0.26
         main: false
         user: <username>


Then, the command should look like this:

.. code-block:: bash

    # On manager-node:

    # Change if needed
    export SSH_KEY_FILE=$HOME/.ssh/id_rsa
    export MILABENCH_IMAGE=ghcr.io/mila-iqia/milabench:cuda-nightly
    docker run -it --rm --gpus all --network host --ipc=host --privileged \
      -v $SSH_KEY_FILE:/milabench/id_milabench \
      -v $(pwd)/results:/milabench/envs/runs \
      $MILABENCH_IMAGE \
      milabench run --system /milabench/envs/runs/system.yaml \
      --select multinode

The last line (``--select multinode``) specifically selects the multi-node benchmarks. Omit that line to run all benchmarks.

If you need to use more than two nodes, edit or copy ``system.yaml`` and simply add the other nodes' addresses in ``nodes``. 
You will also need to update the benchmark definition and increase the max number of nodes by creating a new ``overrides.yaml`` file.

For example, for 4 nodes:


.. code-block:: yaml

   # Name of the benchmark. You can also override values in other benchmarks.
   opt-6_7b-multinode:
     num_machines: 4
  

.. code-block:: yaml

   system:
     arch: cuda
     docker_image: ghcr.io/mila-iqia/milabench:${system.arch}-nightly

     nodes:
       - name: node1
         ip: 192.168.0.25
         main: true
         port: 8123
         user: <username>
      
       - name: node2
         ip: 192.168.0.26
         main: false
         user: <username>
      
       - name: node3
         ip: 192.168.0.27
         main: false
         user: <username>

       - name: node4
         ip: 192.168.0.28
         main: false
         user: <username>


The command would look like

.. code-block:: bash

   docker ... milabench run ... --system /milabench/envs/runs/system.yaml --overrides /milabench/envs/runs/overrides.yaml


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
