Design
======

Milabench aims to simulate research workloads for benchmarking purposes.

* Performance is measured as throughput (samples / secs).
  For example, for a model like resnet the throughput would be image per seconds.

* Single GPU workloads are spawned per GPU to ensure the entire machine is used.
  Simulating something similar to a hyper parameter search.
  The performance of the benchmark is the sum of throughput of each processes.

* Multi GPU workloads

* Multi Nodes


Run
---

* Milabench Manager Process
   * Handles messages from benchmark processes
   * Saves messages into a file for future analysis

* Benchmark processes
   * run using ``voir``
   * voir is configured to intercept and send events during the training process
   * This allow us to add models from git repositories without modification
   * voir sends data through a file descriptor that was created by milabench main process


What milabench is
-----------------

* Training focused
* milabench show candid performance numbers
   * No optimization beyond batch size scaling is performed
   * we want to measure the performance our researcher will see
     not the performance they could get.
* pytorch centric
   * Pytorch has become the defacto library for research
   * We are looking for accelerator with good maturity that can support
     this framework with limited code change.


What milabench is not
---------------------

* milabench goal is not a performance show case of an accelerator.
