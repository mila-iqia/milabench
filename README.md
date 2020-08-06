
# Training Benchmarks

* Tested on **python 3.7** with **pytorch 1.5**

# Install

You can either build a Docker image or install the benchmark software manually.


## Docker

### Build

```bash
docker build -t milabench_img -f Dockerfile .
```

The build might seem to hang at `poetry install`, but just wait it out.


### Flags

The container should be run with the following flags, with appropriate adjustments to the image name or the path to mount:

```bash
# Interactive session (need to specify bash)
sudo docker run --cap-add=SYS_ADMIN --security-opt=apparmor:unconfined --mount type=bind,source=/localnvme/milabench,target=/root/milabench --gpus all --shm-size=16GB -it milabench_img bash

# Run
sudo docker run --cap-add=SYS_ADMIN --security-opt=apparmor:unconfined --mount type=bind,source=/localnvme/milabench,target=/root/milabench --gpus all --shm-size=16GB milabench_img milarun ...
```

(Note: `/localnvme/milabench` is just a path to demonstrate where we expect the data to be physically located, the only thing that matters is that this is on the NVMe drive.)

Further explanations/precisions:

* When using Docker, the data directory for the test, and the output directory for the results, should be outside of the image and mounted using `--mount`.
  * The source *must* be physically located on the local NVMe drive. The path `/localnvme/milabench` is arbitrary, it can be anything as long as it is on the aforementioned drive.
  * The target should be `/root/milabench`. This is where the scripts expect to find the data, and this is where they will save the results.
* Because the benchmarks use cgroups, it is necessary to run the image in privileged mode.
  * The entry point for the image will set up the cgroups automatically.
  * For debugging purposes only: passing `--no-cgexec` to the `milarun jobs` command will deactivate the feature, which lets you run the image in user mode.
* In order for some tests to run it may be necessary to set a sufficient size for `/dev/shm` using `--shm-size`. We're not certain what the minimal size is exactly but 16GB should work.


### Download the datasets

In the container, run:

```bash
scripts/download.sh
```

Verify that the dataset is indeed downloaded into the `data/` subdirectory of the mounted directory.

### Benchmarks and reports

If everything went well, refer to the Benchmarks and Report sections further down.


## Manual install

Skip this section if you are using the Docker set up described above.


1. Install apt requirements

```bash
scripts/install-apt-packages.sh
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

Note: it is also possible to use `conda` (this is what the Docker image does). In that case, create and activate the conda environment before running `poetry install`, and everything will be installed in the conda environment (no need for `poetry shell` in this case, just activate the conda environment).


4. Set up the cgroups

```bash
sudo scripts/cgroup_setup.sh
```


5. Set up environment variables

The benchmarks will need the following two environment variables: `$MILARUN_DATAROOT` is where the datasets will be located, and they must point to a directory which physically resides on the machine's local NVMe drive. `$MILARUN_OUTROOT` is where the results directories will be saved.

For example:

```bash
export MILARUN_DATAROOT=~/milabench/data/
export MILARUN_OUTROOT=~/milabench/
```

The above is equivalent to giving the command line options `-d ~/milabench/data/ -o ~/milabench/results-$(date '+%Y-%m-%d.%H:%M:%S')` to `milarun`.


5. Download the datasets

Once the environment variables are set, run `poetry shell` to activate the virtualenv. Then, run:

```bash
# Download the datasets
scripts/download.sh
```


# Benchmarks

**Note:** Unless you are running Docker, which has an entry script that does this for you, you must first run `poetry shell` to activate the virtualenv. Alternatively, you can prefix calls to `milarun` with `poetry run`.

```bash
milarun jobs profiles/standard.json

# To repeat 5 times
milarun jobs profiles/standard.json --repeat 5
```

This will put the output into `$MILARUN_OUTROOT/results-$(date '+%Y-%m-%d.%H:%M:%S')` (if using Docker, `$MILARUN_OUTROOT` is set to `/root/milabench` which should correspond to the directory you mounted). When debugging, it may be less cumbersome to specify a directory with `-o`:

```bash
# Note that this bypasses $MILARUN_OUTROOT entirely
milarun jobs profiles/standard.json -o test-results
```

The test results will be stored as json files in the specified outdir, one file for each test. A result will have a name such as `standard.vae.J0.0.20200106-160000-123456.json`, which means the test named `vae` from the jobs file `profiles/standard.json`, run 0, device 0, and then the date and time. If the tests are run 10 times and there are 8 GPUs, you should get 80 of these files for each test (`J0.0` through `J9.7`). If a test fails, the filename will also contain the word `FAIL`.

### Run one test in the suite

Use the `--name` flag to run a specific test (look in `profiles/standard.json` for the available names). You can also change its parameters by putting extra arguments after the double dash separator `--`.

```bash
milarun jobs profiles/standard.json --name vae
milarun jobs profiles/standard.json --name vae -- --batch-size 32 --max-count 3000
```

Note that `--name` (and `-d` and `-o` if using these flags) go before the `--`, because they are parameters to milarun, not parameters to the test.

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
milarun run milabench.models.polynome:main -o polynome-results.json -- --poly-degree 6 --batch-size 1024 --max-count 1_000_000
```

Here we only have a single result rather than a directory with multiple results, so we can use the `-o` option to save in a specific `json` file.

### Benchmarks for lower memory

Use `profiles/<N>gb.json` instead of `profiles/standard.json` to run the benchmarks on GPUs with at most N GB of memory. There are also weights in `weights/<N>gb.json` to compute a score for these modified benchmarks.

### Tweaks

You can create tweaked baselines by modifying a copy of `standard.json`. These tweaked baselines may be used to either test something different, debug, or demonstrate further capacities, if needed.

```bash
cp profiles/standard.json profiles/tweaked.json

# modify tweaked.json to reflect the device capacity

milarun jobs profiles/tweaked.json
```

# Report the results

This will print out a summary of the results as a table, as well as a global score, which is the weighted geometric mean of the adjusted test scores (the `perf_adj` column) using the provided weights file, and will also output a prettier HTML file (note: all arguments save for `RESULTS_DIR` are optional).

```bash
milarun report RESULTS_DIR --html results.html --weights weights/standard.json
```

`RESULTS_DIR` should be the path to the results to analyze, e.g. `/root/milabench/results-2020-02-22.22:22:22`. Be careful to use the right directory, if you have several.

Notes:

* `perf_adj = perf * (1 - fail/n)` -- it is a measure of performance that penalizes test failures.
  * If the `--penalize-variance` flag is given (this is off by default): `perf_adj = perf * (1 - std%) * (1 - fail/n)` -- in this case we penalize variance by subtracting one standard deviation.
* `score = exp(sum(log(perf_adj) * weight) / sum(weight))` -- this is the weighted geometric mean of `perf_adj`.


### Compare results

You may compare with a baseline using the `--compare` argument:

```bash
milarun report RESULTS_DIR --compare baselines/standard/v100.json --html results.html
```

A baseline can be produced with `milarun summary`:

```bash
milarun summary RESULTS_DIR -o our-config.json
```

You will get a comparison for each test as well as a comparison of the scores in the form of performance ratios (>1.00 = you are doing better than the baseline, <1.00 = you are doing worse).

The command also accepts a `--price` argument (in dollars) to compute the price/score ratio.


### GPU comparisons

The `--compare-gpus` option will provide more information in the form of comparison matrices for individual GPUs (this information will be more reliable if there are multiple runs of the suite, for example if you use `milarun jobs profiles/standard.json -o RESULTS_DIR --repeat 10`).

```bash
milarun report RESULTS_DIR --compare-gpus
```


# Tips

* The benchmark starts with two toy examples to make sure everything is setup properly.

* Each bench run `N_GPU` times in parallel with only `N_CPU / N_GPU` and `RAM / N_GPU` to simulate multiple users.
  * if your machine has 16 GPUs and 32 cores, the bench will run in parallel 16 times with only 2 cores for each GPU.

* Some tasks are allowed to use the machine entirely (`scaling`)

* `milarun report RESULTS_DIR` can be used at any time to check current results.

* Stop a run that is in progress
    * `kill -9 $(ps | grep milarun | awk '{print $1}' | paste -s -d ' ')`


# FAQ

## cgroups

**What does the cgroup script do?**

The cgroups are used to emulate multiple users and force the resources of each user to be clearly segregated, similar to what Slurm does in a HPC cluster.

**Does the cgroups setup affect the results of the benchmark?**

Yes. Because of resource segregation, the multiple experiments launched by `milarun` in parallel will not fight for resources, leading to reduced variance and different performance characteristics (some tests do a little better, but most do a little worse). According to our experiments, using the cgroup setup increases the score by about 1%.

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


**I get crashes with massive core dumps.**

If using Docker, something like this has happened when there is insufficient space available in `/dev/shm`, which you can increase with `--shm-size` on the docker command.

As a small hack, you can try to run the `milarun` command from a directory that's in the mounted drive, e.g. from `~/milabench/workdir` so that the core dumps are written over there and don't fill up the cointainer.


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
