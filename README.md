
# Training Benchmarks

* Tested on **python 3.7** with **pytorch 1.5**

## Install

1. Install apt requirements

```bash
sudo scripts/install-apt-packages.sh
```

2. Install Poetry

```bash
scripts/install-poetry.sh

# Reload bash so that the poetry command is visible
exec bash
```

3. Install dependencies

```bash
poetry install
```

This assumes Python 3.7 is installed on the system, which it should be if the apt packages were installed in the first step.

Once the software is installed, you can start a shell within the environment as follows:

```bash
poetry shell
```

This will give you access to the `milarun` command.

## Set up

1. Set up the cgroups

```bash
sudo scripts/cgroup_setup.sh
```

2. Download the datasets (this might take a while)

```bash
# Download the data into ~/data, or a directory of your choosing
poetry run scripts/download.sh -d $DATAROOT
```

It is important to run this script before the benchmarks.

## Run the benchmarks

First run `poetry shell` to activate the virtualenv. Alternatively, you can prefix calls to `milarun` with `poetry run`.

```bash
milarun jobs profiles/standard.json -d $DATAROOT -o $OUTDIR

# To repeat 10 times
milarun jobs profiles/standard.json -d $DATAROOT -o $OUTDIR --repeat 10
```

We suggest outputting to a different results directory for each run, for example by embedding the date, like so:

```bash
export DATAROOT=~/data/
export OUTDIR=~/results-$(date '+%Y-%m-%d.%H:%M:%S')/
milarun jobs profiles/standard.json -d $DATAROOT -o $OUTDIR
```

The test results will be stored as json files in the specified outdir, one file for each test. A result will have a name such as `standard.vae.J0.0.20200106-160000-123456.json`, which means the test named `vae` from the jobs file `profiles/standard.json`, run 0, device 0, and then the date and time. If the tests are run 10 times and there are 8 GPUs, you should get 80 of these files for each test (`J0.0` through `J9.7`). If a test fails, the filename will also contain the word `FAIL`.

### Run one test in the suite

Use the `--name` flag to run a specific test (look in `profiles/standard.json` for the available names). You can also change its parameters by putting extra arguments after the double dash separator `--`.

```bash
milarun jobs profiles/standard.json -d $DATAROOT --name vae
milarun jobs profiles/standard.json -d $DATAROOT --name vae -- --batch-size 32 --max-count 3000
```

Note that the `-d` and `-o` flags go before the `--`, because they are parameters to milarun, not parameters to the test. If `-o` is not specified, the JSON will be output on stdout.

All tests take a `--max-count` parameter which controls for how long they run.

### Re-running a test

`milarun rerun` will re-run any individual test. It is possible to change some of the test's parameters by listing them after `--`.

```bash
milarun rerun results/standard.vae.J0.0.20200106-160000-123456.json
milarun rerun results/standard.vae.J0.0.20200106-160000-123456.json -- --batch-size 32 --max-count 3000
```

### Running an individual test (advanced)

Instead of using a jobs file, you can run a test directly with `milarun run`, like so:

```bash
milarun run milabench.models.polynome:main -d $DATAROOT -o polynome-results.json -- --poly-degree 6 --batch-size 1024 --max-count 1_000_000
```

### Tweaks

You can create tweaked baselines by modifying a copy of `standard.json`. These tweaked baselines may be used to either test something different, debug, or demonstrate further capacities, if needed.

```bash
cp profiles/standard.json profiles/tweaked.json

# modify tweaked.json to reflect the device capacity

milarun jobs profiles/tweaked.json -d $DATAROOT -o $OUTDIR
```

# Report the results

This will print out a summary of the results as a table, as well as a global score, which is the weighted geometric mean of the adjusted test scores (the `perf_adj` column) using the provided weights file, and will also output a prettier HTML file (note: all arguments save for `$OUTDIR` are optional).

```bash
milarun report $OUTDIR --html results.html --weights weights/standard.json
```

Notes:

* `perf_adj = perf * (1 - std%) * (1 - fail/n)` -- it is a measure of performance that penalizes variance and test failures.
* `score = exp(sum(log(perf_adj) * weight) / sum(weight))` -- this is the weighted geometric mean of `perf_adj`.


## Compare results

You may compare with a baseline using the `--compare` argument:

```bash
milarun report $OUTDIR --compare baselines/v100.json --html results.html
```

A baseline can be produced with `milarun summary`:

```bash
milarun summary $OUTDIR -o our-config.json
```

You will get a comparison for each test as well as a comparison of the scores in the form of performance ratios (>1.00 = you are doing better than the baseline, <1.00 = you are doing worse).

The command also accepts a `--price` argument (in dollars) to compute the price/score ratio.


## GPU comparisons

The `--compare-gpus` option will provide more information in the form of comparison matrices for individual GPUs (this information will be more reliable if there are multiple runs of the suite, for example if you use `milarun jobs profiles/standard.json -o $OUTDIR --repeat 10`).

```bash
milarun report $OUTDIR --compare-gpus
```


# Tips

* The benchmark starts with two toy examples to make sure everything is setup properly.

* Each bench run `N_GPU` times in parallel with only `N_CPU / N_GPU` and `RAM / N_GPU` to simulate multiple users.
  * if your machine has 16 GPUs and 32 cores, the bench will run in parallel 16 times with only 2 cores for each GPU.

* Some tasks are allowed to use the machine entirely (`scaling`)

* `milarun report $OUTDIR` can be used at any time to check current results.

* Stop a run that is in progress
    * `kill -9 $(ps | grep milarun | awk '{print $1}' | paste -s -d ' ')`


# FAQ

## cgroups

**What does the cgroup script do?**

The cgroups are used to emulate multiple users and force the resources of each user to be clearly segregated, similar to what Slurm does in a HPC cluster.

**Does the cgroups setup affect the results of the benchmark?**

Yes. Because of resource segregation, the multiple experiments launched by `milarun` in parallel will not fight for resources, leading to reduced variance and different performance characteristics (some tests may do a little better, some others may do a little worse). According to our experiments, using the cgroup setup can increase the score by 2 to 3%.

**Can we run the benchmarks without the cgroups?**

You can pass the `--no-cgexec` flag to `milarun`.

**IMPORTANT:** If you run the benchmarks in the context of an RFP, you may be required to run the benchmarks with the cgroups active, so you should check.

**Can we set up the cgroups differently?**

Yes, provided that the constraints below are met:

* 1 student group per GPUs (student0 for GPU 0, ... student31 for GPU 31)
* Each student group need to be allocated an equal amount of RAM. All students should be able to use all the RAM that has been allocated to them without issues.
* Each student group need to be allocated the same amount of threads, the threads need to be mutually exclusive.

## AMD

**Do the benchmarks run on AMD GPUs?**

In principle, they should, provided a compatible version of PyTorch is installed. The benchmarks in their current iteration have not been tested on AMD, however, and we provide no instructions at the moment.

<!-- * When running using the AMD stack the initial compilation of each models can take a significant amount of time. You can remove the compilation step by using Mila's miopen compilation cache. To use it you can simply execute `copy_rocm_cache.sh`. -->

## Other

<!-- 
* If your machine supports SSE vector instructions you are allowed to replace it with pillow-simd for faster load times -->


**Do all these benchmarks run/use GPUs or are some of them solely CPU-centric?**

* `cart` is a simplistic reinforcement learning benchmark that only uses the CPU.
* `loader` mostly measures IO speed loading JPEG images, although it does load them into the GPU.

**convnet and convnet_fp16 seem to be single GPU benchmarks but nvidia-smi shows activity on all GPUs in a node. Are the other GPUs used for workers?**

They use a single GPU, but all scripts using a single GPU are launched N times in parallel where N is the number of GPU on the node. This is done to simulate N users running N models in parallel.

**While running fp16 tasks, the warnings below are shown:**

```
Warning:  multi_tensor_applier fused unscale kernel is unavailable, possibly because apex was installed without --cuda_ext --cpp_ext. Using Python fallback.  Original ImportError was: ModuleNotFoundError("No module named 'amp_C'",)
Attempting to unscale a grad with type torch.cuda.HalfTensor Unscaling non-fp32 grads may indicate an error. When using Amp, you don't need to call .half() on your model.
```

These warnings can be ignored.

<!-- 
* We are using docker and `sudo` is not necessary
    * you can set `export SUDO=''` to not use sudo

* Is there a multi-node benchmark in convnets ? If yes, what was the reference run configuration ?
    * There are no multi-node benchmarks. Only Single Node Multi-GPU -->

**What is standard.scaling.1 to standard.scaling.8 in the report?**

The scaling test performs an experiment on 1 GPU, then 2 GPUs, up to N GPUs. These have obviously different performance characteristics and therefore cannot be aggregated, so `standard.scaling.1` represents the results on 1 GPU,  `standard.scaling.2` represents the results of the algorithm distributed on 2 GPUs, and so on.


# Details

## Datasets

The benchmarks require approximatively 50 GiB of storage:

```
du -hd 2 data
1.2M    data/time_series_prediction
796M    data/coco/annotations
19G     data/coco/train2017
788M    data/coco/val2017
20G     data/coco
16G     data/ImageNet/train
16G     data/ImageNet
1.8G    data/ml-20m
948K    data/wmt16/subword-nmt
205M    data/wmt16/mosesdecoder
4.5G    data/wmt16/data
13G     data/wmt16
117M    data/mnist/MNIST
117M    data/mnist
73M     data/bsds500/BSR
73M     data/bsds500
13M     data/wikitext-2
50G     data
```

Note: the `ImageNet` dataset used for the benchmarks is actually fake data that is generated by the "download" script.

You should run `scripts/download.sh` prior to running the benchmarks.


## Benchmark methodology

For each test we log the number of examples processed per second by the training algorithm, for every 0.5s slice. We sync the GPU at the end of each slice. This logs a graph of performance through training, up to the number of examples set as `--max-count`.

`milarun summary` and `milarun report` then take the average performance over the last 10 seconds of the run, which should correspond to the "steady state" of performance during training (note that we stop training well before any researcher would, because we do not care about the final model). The first few seconds are not reliable indicators of steady performance: they may be slower because of data loading, GPU warmup, initial compilation of the model, and other factors, which is why they are ignored. We also ignore the last second, because we have observed some artifacts due to early stoppage.

Future versions of the benchmark may include further measurements for each test such as pre-processing or loading times.
