
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
