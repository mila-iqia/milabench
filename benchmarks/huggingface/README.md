
# Huggingface

Transformers benchmarks, based on the `transformers` package from huggingface. There is no data to download: they are run on random batches of data.

The following models are available.

* Opt350m
* GPT2
* GPT2_large
* T5
* T5_base
* T5_large
* Bart
* Reformer
* BigBird
* Albert
* DistilBert
* Longformer
* Bert
* Bert_large

## Usage

For example:

```bash
voir --dash main.py --model T5 --batch-size 128 --with-amp
```
