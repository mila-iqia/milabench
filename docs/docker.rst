Docker
======

`Docker Images <https://github.com/mila-iqia/milabench/kgs/container/milabench>`_ are created for each release, 
they come with all the benchmark installed and the necessary dataset.


Requirements
------------

* `docker-ce <https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository>`_
* `nvidia-docker <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>`_


Usage
-----

.. code-block:: bash

   # Pull the image we are going to run
   sudo docker pull ghcr.io/mila-iqia/milabench:cuda-nightly

   # Run milabench
   sudo docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -v $(pwd)/results:/milabench/results ghcr.io/mila-iqia/milabench:cuda-nightly milabench run

   # Show Performance Report
   sudo docker run -it --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -v $(pwd)/results:/milabench/results ghcr.io/mila-iqia/milabench:cuda-nightly milabench report
   

Publish
-------

Images can be build locally for prototyping and testing.

.. code-block::

   sudo docker build -t milabench:cuda-nightly --build-arg ARCH=cuda --build-arg CONFIG=ci-cuda.yaml .
