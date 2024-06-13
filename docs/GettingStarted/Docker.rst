Docker
======

`Docker Images <https://github.com/mila-iqia/milabench/pkgs/container/milabench>`_ are created for each release. They come with all the benchmarks installed and the necessary datasets. No additional downloads are necessary.


Setup
------

0. Make sure the machine can ssh between each other without passwords
1. Pull the milabench docker image you would like to run on all machines
  - ``docker pull``
1. Create the output directory
  - ``mkdir -p results``
2. Create a list of nodes that will participate in the benchmark inside a ``results/system.yaml`` file (see example below)
  - ``vi results/system.yaml``
3. Call milabench with by specifying the node list we created.
  - ``docker ... -v $(pwd)/results:/milabench/envs/runs -v <privatekey>:/milabench/id_milabench milabench run ... --system /milabench/envs/runs/system.yaml``


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

   export SSH_KEY_FILE=$HOME/.ssh/id_rsa

   # Choose the image you want to use
   export MILABENCH_IMAGE=ghcr.io/mila-iqia/milabench:cuda-nightly

   # Pull the image we are going to run
   docker pull $MILABENCH_IMAGE

   # Run milabench
   docker run -it --rm --ipc=host --gpus=all --network host --privileged    \
         -v $SSH_KEY_FILE:/milabench/id_milabench                           \
         -v $(pwd)/results:/milabench/envs/runs                             \
         $MILABENCH_IMAGE                                                   \
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

   export SSH_KEY_FILE=$HOME/.ssh/id_rsa

   # Choose the image you want to use
   export MILABENCH_IMAGE=ghcr.io/mila-iqia/milabench:rocm-nightly

   # Pull the image we are going to run
   docker pull $MILABENCH_IMAGE

   # Run milabench
   docker run -it --rm --ipc=host --network host --privileged                       \
         --device=/dev/kfd --device=/dev/dri                                        \
         --security-opt seccomp=unconfined --group-add video                        \
         -v $SSH_KEY_FILE:/milabench/id_milabench                                   \
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


Building images
---------------

Images can be built locally for prototyping and testing.

.. code-block::

   docker build -f docker/Dockerfile-cuda -t milabench:cuda-nightly --build-arg CONFIG=standard.yaml .

Or for ROCm:

.. code-block::

   docker build -f docker/Dockerfile-rocm -t milabench:rocm-nightly --build-arg CONFIG=standard.yaml .
