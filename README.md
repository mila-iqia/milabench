
# Milabench

[Documentation](https://mila-iqia.github.io/milabench)

The benchmark suite has been validated on the following configurations:

| Python version | GPU | Configuration file |
| - | - | - |
| 3.9.12 (conda) | 4x NVIDIA A100 80GB | config/standard.yaml |
| 3.9.12 (conda) | 4x NVIDIA RTX8000 48GB | config/standard.yaml |
| 3.9.16 (conda) | 2x NVIDIA K80 | config/ci.yaml |
| 3.9.16 (conda) | 2x AMD MI100 | config/ci.yaml |

We are working on validating it on more configurations and will update the above table as we do.


<!--
## Install

To install for development, clone the repo and use branch `v2`:

```bash
git -b v2 clone git@github.com:mila-iqia/milabench.git
cd milabench
# <Activate virtual environment>

# Make sure pip version is high enough to handle pyproject.toml
pip install --upgrade pip

# Install in editable mode
pip install -e .
```

This will install two commands, `milabench` and `voir`.


## Using milabench

To use `milabench`, you need:

* A YAML configuration file to define the benchmarks to install, prepare or run.
* The base directory for code, virtual environments, data and outputs, set either with the `$MILABENCH_BASE` environment variable or the `--base` option. The base directory will be automatically constructed by milabench and will be organized as follows:

```bash
$MILABENCH_BASE/
|- venv/                            # Virtual environments and dependencies
|  |- bench1/                       # venv for benchmark bench1
|  |- ...                           # etc
|- code/                            # Benchmark code
|  |- bench1/                       # Code for benchmark bench1
|  |- ...                           # etc
|- data/                            # Datasets
|  |- dataset1/                     # A dataset
|  |- ...                           # etc
|- runs/                            # Outputs of benchmark runs
   |- calimero.2022-03-30_15:00:00/ # Auto-generated run name
   |  |- bench1.0.json              # Output for the first run of bench1
   |  |- bench1.1.json              # Output for the second run of bench1
   |  |- ...                        # etc
   |- blah/                         # Can set name with --run
```

It is possible to change the structure in the YAML to e.g. force benchmarks to all use the same virtual environment.

### Important options

* Use the `--select` option with a comma-separated list of benchmarks in order to only install/prepare/run these benchmarks (or use `--exclude` to run all benchmarks except a specific set).
* You may use `--use-current-env` to force the use the currently active virtual environment (useful for development).

### milabench install

```bash
milabench install benchmarks.yaml --select mybench
```

* Copies the code for the benchmark (specified in the `definition` field of the benchmark's YAML, relative to the YAML file itself) into `$MILABENCH_BASE/code/mybench`. Only files listed by the `manifest` file are copied.
* Creates/reuses a virtual environment in `$MILABENCH_BASE/venv/mybench` and installs all pip dependencies in it.
* Optionally extracts a shallow git clone of an external repository containing model code into `$MILABENCH_BASE/code/mybench`.

### milabench prepare

```bash
milabench prepare benchmarks.yaml --select mybench
```

* Prepares data for the benchmark into `$MILABENCH_BASE/data/dataset_name`. Multiple benchmarks can share the same data. Some benchmarks need no preparation, so the prepare step does nothing.

### milabench run

```bash
milabench run benchmarks.yaml --select mybench
```

* Creates a certain number of tasks from the benchmark using the `plan` defined in the YAML. For instance, one plan might be to run it in parallel on each GPU on the machine.
* For each task, runs the benchmark installed in `$MILABENCH_BASE/code/mybench` in the appropriate virtual environment.
* The benchmark is run from that directory using a command like `voir [VOIR_OPTIONS] main.py [SCRIPT_OPTIONS]`
  * Both option groups are defined in the YAML.
  * The VOIR_OPTIONS determine which instruments to use and what data to forward to milabench.
  * The SCRIPT_OPTIONS are benchmark dependent.
* Standard output/error and other data (training rates, etc.) are forwarded to the main dispatcher process and saved into `$MILABENCH_BASE/runs/run_name/mybench.run_number.json` (the name of the directory is printed out for easy reference).

### milabench report

TODO.

```bash
milabench report benchmarks.yaml --run <run_name>
```


## Benchmark configuration

The configuration has two sections:

* `defaults` defines a template for benchmarks.
* `benchmarks` defines the benchmarks. Each benchmark may include the defaults with the special directive `<<< *defaults`. Note that the `<<<` operator performs a deep merge. For example:

```yaml
defaults: &defaults
  plan:
    method: njobs
    n: 2

benchmarks:
  test:
    <<<: *defaults
    plan:
      n: 3
```

is equivalent to:

```yaml
benchmarks:
  test:
    plan:
      method: njobs
      n: 3
```

### Fields

Let's say you have the following `benchmark.yaml` configuration:

```yaml
benchmarks:
  mnist:
    definition: ../benchmarks/mnist-pytorch-example

    dirs:
      code: code/{name}
      venv: venv/{name}
      data: data
      runs: runs

    plan:
      method: njobs
      n: 2

    voir:
      --stop: 200
      --forward:
        - "#stdout"
        - "#stderr"
        - loss
        - compute_rate
        - train_rate
        - loading_rate
      --compute-rate: true
      --train-rate: true
      --loading-rate: true

    argv:
      --batch-size: 64
```

* `definition` points to the *definition directory* for the benchmark (more on that later). Important note: the path is *relative to benchmark.yaml*.
* `dirs` defines the directories for the venv, code, data and runs. Normally, this is set in the defaults, but it is technically possible to override it for every benchmark. The paths are relative to `$MILABENCH_BASE` (or the argument to `--base`) `code/{name}` expands to `code/mnist`.
* `plan` describes the way tasks will be created for this benchmark. `nruns` just launches a fixed number of parallel processes.
* `voir` are the arguments given to the `voir` command when running a task. The `--forward` argument is important because it defines what will end up in the final `json` output saved to the disk. Some of them correspond to what other flags output.
* `argv` are the arguments given to the benchmark script.


## Benchmark definition

To define a new benchmark, create a directory with roughly the following files:

```bash
mybench
|- manifest        # Lists the file milabench install should copy (accepts wildcards)
|- benchfile.py    # Benchmark definition file
|- voirfile.py     # Probes and extra instruments
|- prepare.py      # Executed by milabench prepare
|- main.py         # Executed by milabench run
|- dev.yaml        # Bench file to use for development
``` -->
