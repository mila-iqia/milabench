Request For proposal
====================

Preparing
---------

1. Make sure milabench support the targetted hardware

   * NVIDIA
   * AMD

2. Create a milabench configuration for your RFP
   Milabench comes with a wide variety of benchmarks.
   You should select and weight each benchmarks according to your
   target hardware.

.. code-block:: yaml

   include:
     - base.yaml

   llama:
     enabled: true
     weight: 1.0

   resnet50:
     enabled: true
     weight: 1.0


.. code-block:: yaml

   milabench resolve myconfig.yaml > RFP.yaml


3. Prepare a container for your RFP


.. code-block::

   FROM milabench:cuda-v1.2.3

   COPY RFP.yaml .../RFP.yaml

   ENV MILABENCH_CONFIG=".../RFP.yaml

   CMD milabench run


4.  Hot fixes

   * Disable a benchmarks
   * update container


Vendor Instructions
-------------------

1. Vendor needs to create a system configuration that will
   specify the different compute nodes that will be used by milabench

.. code-block::

   system:
      sshkey: <privatekey>
      arch: cuda
      docker_image: ghcr.io/mila-iqia/milabench:cuda-nightly

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


2. Run milabench

.. code-block::

   export MILABENCH_IMAGE=ghcr.io/mila-iqia/milabench:cuda-nightly

   # create ...
   mkdir -p configs
   mkdir -p results

   # put your vendor specific configuration
   vi configs/system.yaml

   #
   docker pull $MILABENCH_IMAGE

   # run milabench
   docker run -it --rm --gpus all --network host --ipc=host --privileged \
      -v $SSH_KEY_FILE:/milabench/id_milabench        \
      -v $(pwd)/results:/milabench/envs/runs          \
      -v $(pwd)/configs:/milabench/envs/configs       \
      $MILABENCH_IMAGE                                \
      milabench run --system /milabench/envs/configs/system.yaml
