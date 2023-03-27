Docker
======

`Docker Images <https://github.com/mila-iqia/milabench/pkgs/container/milabench>`_ are created for each release, 
they come with all the benchmarks installed and the necessary datasets.
No additional downloads are necessary.

CUDA
----

Requirements
^^^^^^^^^^^^

To run docker images you will need to install the softwares below.

* nvidia driver
* `docker-ce <https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository>`_
* `nvidia-docker <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>`_


Usage
^^^^^

The commands below will download the lastest cuda container and run milabench right away,
storing the results inside the `results` folder on the host machine.

The last command will generate a json summary of each benchmarks.

.. code-block:: bash

   # Pull the image we are going to run
   sudo docker pull ghcr.io/mila-iqia/milabench:cuda-nightly

   # Run milabench
   sudo docker run -it --rm --shm-size=256M              \
         --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all  \
         -v $(pwd)/results:/milabench/envs/runs          \
         ghcr.io/mila-iqia/milabench:cuda-nightly        \
         milabench run

   # Show Performance Report
   sudo docker run -it --rm --shm-size=256M              \
         --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all  \
         -v $(pwd)/results:/milabench/envs/runs          \
         ghcr.io/mila-iqia/milabench:cuda-nightly        \
         milabench summary /milabench/envs/runs


Publish
^^^^^^^

Images can be build locally for prototyping and testing.

.. code-block::

   sudo docker build -t milabench:cuda-nightly --build-arg ARCH=cuda --build-arg CONFIG=ci.yaml .
