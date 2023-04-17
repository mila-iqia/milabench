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
   docker run -it --rm --shm-size=8G --gpus=all   \
         -v $(pwd)/results:/milabench/envs/runs   \
         $MILABENCH_IMAGE                         \
         milabench run

You may have to increase the value of ``--shm-size`` if it turns out to be insufficient.

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
   docker run -it --rm --shm-size=8G                         \
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


Building images
---------------

Images can be build locally for prototyping and testing.

.. code-block::

   sudo docker build -t milabench:cuda-nightly --build-arg ARCH=cuda --build-arg CONFIG=standard.yaml .

Set the ``ARCH`` and ``CONFIG`` build arguments to the appropriate values for your use case.
