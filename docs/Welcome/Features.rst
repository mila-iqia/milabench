Features
========

* non intruisive Instrumentation
* Validation Layers
* Automatic batch resizing
* Docker
* Hardware
   * ROCm 5.7
   * NVIDIA
   * XPU (OneAPI)
   * HPU (Habana)
* Metrics gathering
   * Performance throughput
   * GPU util
   * CPU util
   * IO util

Benchmarks
----------

.. code-block:: text
    +--------------------------+-----------+-----------+-------------+-----------+-------------------+
    |        Benchmark         |   Unit    |  Domain   |   Network   |   Focus   |       Task        |
    +==========================+===========+===========+=============+===========+===================+
    | bf16                     | TFlops    | Synthetic |             | Training  |                   |
    | fp16                     | TFlops    | Synthetic |             | Training  |                   |
    | tf32                     | TFlops    | Synthetic |             | Training  |                   |
    | fp32                     | TFlops    | Synthetic |             | Training  |                   |
    | bert-fp16                |           | NLP       | Transformer | Training  | Language Modeling |
    | bert-fp32                |           | NLP       | Transformer | Training  | Language Modeling |
    | bert-tf32                |           | NLP       | Transformer | Training  | Language Modeling |
    | bert-tf32-fp16           |           | NLP       | Transformer | Training  | Language Modeling |
    | opt-1_3b                 |           | NLP       | Transformer | Training  | Language Modeling |
    | opt-6_7b                 |           | NLP       | Transformer | Training  | Language Modeling |
    | reformer                 |           | NLP       | Transformer | Training  | Language Modeling |
    | rwkv                     |           | NLP       | RNN         | Training  | Language Modeling |
    | llama                    | Token/sec | NLP       | Transformer | Inference | Generation        |
    | dlrm                     |           | NLP       |             | Training  | Recommendation    |
    | convnext_large-fp16      | img/sec   | Vision    | Convolution | Training  | Classification    |
    | convnext_large-fp32      | img/sec   | Vision    | Convolution | Training  | Classification    |
    | convnext_large-tf32      | img/sec   | Vision    | Convolution | Training  | Classification    |
    | convnext_large-tf32-fp16 | img/sec   | Vision    | Convolution | Training  | Classification    |
    | davit_large              | img/sec   | Vision    | Transformer | Training  | Classification    |
    | focalnet                 |           | Vision    | Convolution | Training  | Classification    |
    | davit_large-multi        | img/sec   | Vision    | Transformer | Training  | Classification    |
    | regnet_y_128gf           | img/sec   | Vision    | Convolution | Training  | Classification    |
    | resnet152                | img/sec   | Vision    | Convolution | Training  | Classification    |
    | resnet152-multi          | img/sec   | Vision    | Convolution | Training  | Classification    |
    | resnet50                 | img/sec   | Vision    | Convolution | Training  | Classification    |
    | stargan                  | img/sec   | Vision    | Convolution | Training  | GAN               |
    | super-slomo              | img/sec   | Vision    | Convolution | Training  |                   |
    | t5                       |           | NLP       | Transformer | Training  |                   |
    | whisper                  |           | Audio     |             | Training  |                   |
    +--------------------------+-----------+-----------+-------------+-----------+-------------------+