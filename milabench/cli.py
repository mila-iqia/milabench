import json
import os
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
import getpass

from coleo import Option, config as configuration, default, run_cli, tooled
from omegaconf import OmegaConf
from voir.instruments.gpu import deduce_backend, select_backend

from milabench.alt_async import proceed
from milabench.utils import blabla, validation_layers, multilogger, available_layers

from .metadata import machine_metadata
from .compare import compare, fetch_runs
from .config import build_config, build_system_config
from .fs import XPath
from .log import (
    DataReporter,
    LongDashFormatter,
    ShortDashFormatter,
    TerminalFormatter,
    TextReporter,
)
from .merge import merge
from .multi import MultiPackage
from .report import make_report
from .summary import aggregate, make_summary


def main(argv=None):
    sys.path.insert(0, os.path.abspath(os.curdir))
    if argv is None:
        argv = sys.argv[1:]
    argv = [str(x) for x in argv]
    try:
        sys.exit(run_cli(Main, argv=argv))
    except KeyboardInterrupt:
        pass


def get_pack(defn):
    pack = XPath(defn["definition"])
    pack_glb = runpy.run_path(str(pack / "benchfile.py"))
    pack_cls = pack_glb["__pack__"]
    pack_obj = pack_cls(defn)
    return pack_obj


@tooled
def get_multipack(run_name=None, overrides={}):
    # Configuration file
    config: Option & str = None

    # System Configuration file
    system: Option & str = None

    # Base path for code, venvs, data and runs
    base: Option & str = None

    # Whether to use the current environment
    use_current_env: Option & bool = False

    # Packs to select
    select: Option & str = default("")

    # Packs to exclude
    exclude: Option & str = default("")

    # Override configuration values
    # [action: append]
    override: Option = []

    # Define capabilities
    capabilities: Option = ""

    override = [
        o if re.match(pattern=r"[.\w]+=", string=o) else f"={o}" for o in override
    ]

    override.extend(
        [f"*.capabilities.{entry}" for entry in capabilities.split(",") if entry]
    )

    if override:
        override_obj = OmegaConf.to_object(OmegaConf.from_dotlist(override))
        if "" in override_obj:
            override_obj = merge(override_obj, override_obj.pop(""))
        overrides = merge(overrides, override_obj)

    return _get_multipack(
        config,
        system,
        base,
        use_current_env,
        select,
        exclude,
        run_name=run_name,
        overrides=overrides,
    )


def selection_keys(defn):
    if "group" not in defn:
        raise Exception("Invalid benchmark:", defn["name"])
    sel = {
        "*",
        defn["name"],
        defn["group"],
        defn["install_group"],
        *defn.get("tags", []),
    }
    return sel


def get_base_defaults(base, arch="none", run_name="none"):
    try:
        user = os.getlogin()
    except OSError:
        user = "root"
    return {
        "_defaults": {
            "system": {
                "arch": arch,
                "sshkey": None,
                "nodes": [
                    {
                        "name": "local",
                        "ip": "127.0.0.1",
                        "port": None,
                        "user": user,
                        "main": True,
                    }
                ],
            },
            "dirs": {
                "base": base,
                "venv": "${dirs.base}/venv/${install_group}",
                "data": "${dirs.base}/data",
                "runs": "${dirs.base}/runs",
                "extra": "${dirs.base}/extra/${group}",
                "cache": "${dirs.base}/cache",
            },
            "group": "${name}",
            "install_group": "${group}",
            "install_variant": "${system.arch}",
            "run_name": run_name,
            "enabled": True,
            "capabilities": {
                "nodes": 1,
            },
        }
    }


def deduce_arch():
    """Deduce the arch for installation and preparation purposes"""
    arch_guess = os.environ.get("MILABENCH_GPU_ARCH", None)

    if arch_guess is not None:
        return arch_guess

    return deduce_backend()


def init_arch(arch=None):
    """Initialize the monitor for the given arch"""
    arch = arch or deduce_arch()
    return select_backend(arch)


def _get_multipack(
    config_path,
    system_config_path=None,
    base=None,
    use_current_env=False,
    select="",
    exclude="",
    run_name=None,
    overrides={},
    return_config=False,
):
    if config_path is None:
        config_path = os.environ.get("MILABENCH_CONFIG", None)

    if config_path is None:
        sys.exit("Error: CONFIG argument not provided and no $MILABENCH_CONFIG")

    if select:
        select = set(select.split(","))

    if exclude:
        exclude = set(exclude.split(","))

    if base is None:
        base = os.environ.get("MILABENCH_BASE", None)

    if not base:
        sys.exit("Error: Neither --base nor $MILABENCH_BASE are set.")

    base = base and os.path.abspath(os.path.expanduser(base))

    if use_current_env:
        venv = os.environ.get("CONDA_PREFIX", None)
        if venv is None:
            venv = os.environ.get("VIRTUAL_ENV", None)
        if venv is None:
            print("Could not find virtual environment", file=sys.stderr)
            sys.exit(1)
        overrides = merge(overrides, {"*": {"dirs": {"venv": venv}}})

    if run_name is None:
        run_name = blabla() + ".{time}"

    now = str(datetime.today()).replace(" ", "_")
    run_name = run_name.format(time=now)

    base_defaults = get_base_defaults(base=base, arch=deduce_arch(), run_name=run_name)

    system_config = build_system_config(
        system_config_path, defaults=base_defaults["_defaults"]["system"]
    )
    overrides = merge({"*": {"system": system_config}}, overrides)

    config = build_config(base_defaults, config_path, overrides)

    def is_selected(defn):
        if defn["name"] == "*":
            return False
        keys = selection_keys(defn)
        return (
            defn["enabled"]
            and not defn["name"].startswith("_")
            and defn.get("definition", None)
            and (not select or (keys & select))
            and (not exclude or not (keys & exclude))
        )

    selected_config = {name: defn for name, defn in config.items() if is_selected(defn)}
    if return_config:
        return selected_config
    else:
        return MultiPackage(
            {name: get_pack(defn) for name, defn in selected_config.items()}
        )


def _read_reports(*runs):
    all_data = {}
    for folder in runs:
        for parent, _, filenames in os.walk(folder):
            for file in filenames:
                if not file.endswith(".data"):
                    continue
                pth = XPath(parent) / file
                with pth.open() as f:
                    lines = f.readlines()
                    try:
                        data = [json.loads(line) for line in lines]
                    except Exception:
                        import traceback

                        print(f"Could not parse line inside {pth}\n\t- {line}")
                        traceback.print_exc()
                    else:
                        all_data[str(pth)] = data
    return all_data


def _error_report(reports):
    out = {}
    for r, data in reports.items():
        agg = aggregate(data)
        if not agg:
            continue
        (success,) = agg["data"]["success"]
        if not success:
            out[r] = [line for line in data if "#stdout" in line or "#stderr" in line]
    return out


def run_with_loggers(coro, loggers, mp=None):
    retcode = 0
    loggers = [logger for logger in loggers if logger is not None]

    try:
        with multilogger(*loggers) as log:
            for entry in proceed(coro):
                log(entry)

        retcode = log.result()

    except Exception:
        traceback.print_exc()
        retcode = -1

    finally:
        if mp:
            logdirs = {pack.logdir for pack in mp.packs.values() if pack.logdir}
            for logdir in logdirs:
                print(f"[DONE] Reports directory: {logdir}")

        return retcode


def run_sync(coro, terminal=True):
    return run_with_loggers(coro, [TerminalFormatter()] if terminal else [])


def validation_names(layers):
    if layers is None:
        layers = ""

    layers = layers.split(",")
    all_layers = available_layers()

    if "all" in layers:
        return all_layers

    results = set(["error", "ensure_rate", "version"])
    for l in layers:
        if l in all_layers:
            results.add(l)
        elif l != "":
            print(f"Layer {l} does not exist")

    return results


class Main:
    def run():
        """Run the benchmarks."""

        # Name of the run
        run_name: Option = None

        # Number of times to repeat the benchmark
        repeat: Option & int = 1

        # On error show full stacktrace
        fulltrace: Option & bool = False

        # Do not show a report at the end
        # [negate]
        report: Option & bool = True

        # Which type of dashboard to show (short, long, or no)
        dash: Option & str = os.environ.get("MILABENCH_DASH", "long")

        validations: Option & str = None

        layers = validation_names(validations)

        dash_class = {
            "short": ShortDashFormatter,
            "long": LongDashFormatter,
            "no": None,
        }[dash]

        mp = get_multipack(run_name=run_name)
        arch = next(iter(mp.packs.values())).config["system"]["arch"]

        # Initialize the backend here so we can retrieve GPU stats
        init_arch(arch)

        success = run_with_loggers(
            mp.do_run(repeat=repeat),
            loggers=[
                TerminalFormatter(),
                dash_class and dash_class(),
                TextReporter("stdout"),
                TextReporter("stderr"),
                DataReporter(),
                *validation_layers(*layers, short=not fulltrace),
            ],
            mp=mp,
        )

        if report:
            runs = {pack.logdir for pack in mp.packs.values()}
            compare = None
            compare_gpus = False
            html = None
            price = None

            reports = None
            if runs:
                reports = _read_reports(*runs)
                summary = make_summary(reports.values())

                make_report(
                    summary,
                    compare=compare,
                    html=html,
                    compare_gpus=compare_gpus,
                    price=price,
                    title=None,
                    sources=runs,
                    errdata=reports and _error_report(reports),
                )

        return success

    def prepare():
        """Prepare a benchmark: download datasets, weights etc."""

        mp = get_multipack(run_name="prepare.{time}")

        # On error show full stacktrace
        fulltrace: Option & bool = False

        return run_with_loggers(
            mp.do_prepare(),
            loggers=[
                TerminalFormatter(),
                TextReporter("stdout"),
                TextReporter("stderr"),
                DataReporter(),
                *validation_layers("error", short=not fulltrace),
            ],
            mp=mp,
        )

    def install():
        """Install the benchmarks' dependencies."""

        # Force install
        force: Option & bool = False

        # On error show full stacktrace
        fulltrace: Option & bool = False

        # Install variant
        variant: Option & str = None

        overrides = {"*": {"install_variant": variant}} if variant else {}

        if force:
            mp = get_multipack(run_name="install.{time}", overrides=overrides)
            for pack in mp.packs.values():
                pack.install_mark_file.rm()
                pack.dirs.venv.rm()

        mp = get_multipack(run_name="install.{time}", overrides=overrides)

        return run_with_loggers(
            mp.do_install(),
            loggers=[
                TerminalFormatter(),
                TextReporter("stdout"),
                TextReporter("stderr"),
                DataReporter(),
                *validation_layers("error", short=not fulltrace),
            ],
            mp=mp,
        )

    def pin():
        """Pin the benchmarks' dependencies."""

        # Extra args to pass to pip-compile
        # [nargs: --]
        pip_compile: Option = tuple()

        # Constraints files
        # [options: -c]
        # [nargs: *]
        constraints: Option = tuple()

        # Install variant
        variant: Option & str = None

        # Do not use previous pins if they exist
        from_scratch: Option & bool = False

        overrides = {"*": {"install_variant": variant}} if variant else {}

        if "-h" in pip_compile or "--help" in pip_compile:
            out = (
                subprocess.check_output(
                    ["python3", "-m", "piptools", "compile", "--help"]
                )
                .decode("utf-8")
                .split("\n")
            )
            for i in range(len(out)):
                if out[i].startswith("Usage:"):
                    bin = os.path.basename(sys.argv[0])
                    out[i] = out[i].replace(
                        "Usage: python -m piptools compile",
                        f"usage: {bin} pin [...] --pip-compile",
                    )
            print("\n".join(out))
            exit(0)

        mp = get_multipack(run_name="pin", overrides=overrides)

        return run_with_loggers(
            mp.do_pin(
                pip_compile_args=pip_compile,
                constraints=constraints,
                from_scratch=from_scratch,
            ),
            loggers=[
                TerminalFormatter(),
                TextReporter("stdout"),
                TextReporter("stderr"),
            ],
            mp=mp,
        )

    def dev():
        """Create a shell in a benchmark's environment for development."""

        # The name of the benchmark to develop
        select: Option & str = "*"

        mp = get_multipack(run_name="dev")

        for pack in mp.packs.values():
            if select in selection_keys(pack.config):
                break
        else:
            sys.exit(f"Cannot find a benchmark with selector {select}")

        subprocess.run(
            [os.environ["SHELL"]],
            env=pack.full_env(),
            cwd=pack.dirs.code,
        )

    def summary():
        """Produce a JSON summary of a previous run."""

        # Directory(ies) containing the run data
        # [positional: +]
        runs: Option = []

        # Output file
        # [alias: -o]
        out: Option = None

        all_data = _read_reports(*runs)
        summary = make_summary(all_data.values())

        if out is not None:
            with open(out, "w") as file:
                json.dump(summary, file, indent=4)
        else:
            print(json.dumps(summary, indent=4))

    def compare():
        """Compare all runs with each other."""

        # Parameters
        # ----------
        # folder: str
        #     Folder where milabench results are stored
        # last: int
        #     Number of runs to compare i.e 3 means the 3 latest runs only
        # metric: str
        #     Metric to compare
        # stat: str
        #     statistic name to compare
        # Examples
        # --------
        # >>> milabench compare results/ --last 3 --metric train_rate --stat median
        #                                        |   rufidoko |   sokilupa
        #                                        | 2023-02-23 | 2023-03-09
        # bench                |          metric |   16:16:31 |   16:02:53
        # ----------------------------------------------------------------
        # bert                 |      train_rate |     243.05 |     158.50
        # convnext_large       |      train_rate |     210.11 |     216.23
        # dlrm                 |      train_rate |  338294.94 |  294967.41
        # efficientnet_b0      |      train_rate |     223.56 |     223.48

        # [positional: ?]
        folder: Option = None

        last: Option & int = None

        metric: Option & str = "train_rate"

        stat: Option & str = "median"

        if folder is None:
            base = os.environ.get("MILABENCH_BASE", None)

            if base is not None:
                folder = os.path.join(base, "runs")

        runs = fetch_runs(folder)

        for run in runs:
            all_data = _read_reports(run.path)
            run.summary = make_summary(all_data.values())

        compare(runs, last, metric, stat)

    def report():
        """Generate a report aggregating all runs together into a final report."""

        # Examples
        # --------
        # >>> milabench report --runs results/
        # Source: /home/newton/work/milabench/milabench/../tests/results
        # =================
        # Benchmark results
        # =================
        #                    n fail       perf   perf_adj   std%   sem%% peak_memory
        # bert               2    0     201.06     201.06  21.3%   8.7%          -1
        # convnext_large     2    0     198.62     198.62  19.7%   2.5%       29878
        # td3                2    0   23294.73   23294.73  13.6%   2.1%        2928
        # vit_l_32           2    1     548.09     274.04   7.8%   0.8%        9771
        # <BLANKLINE>
        # Errors
        # ------
        # 1 errors, details in HTML report.

        # Runs directory
        # [action: append]
        runs: Option = []

        # Configuration file (for weights)
        config: Option & str = os.environ.get("MILABENCH_CONFIG", None)

        # Comparison summary
        compare: Option & configuration = None

        # Compare the GPUs
        compare_gpus: Option & bool = False

        # HTML report file
        html: Option = None

        # Price per unit
        price: Option & int = None

        reports = None
        if runs:
            reports = _read_reports(*runs)
            summary = make_summary(reports.values())

        if config:
            config = _get_multipack(config, return_config=True)

        make_report(
            summary,
            compare=compare,
            weights=config,
            html=html,
            compare_gpus=compare_gpus,
            price=price,
            title=None,
            sources=runs,
            errdata=reports and _error_report(reports),
        )

    def pip():
        """Run pip on every pack"""
        # pip arguments
        # [remainder]
        args: Option = []

        mp = get_multipack(run_name="pip")

        for pack in mp.packs.values():
            run_sync(pack.pip_install(*args))

    def slurm_system():
        """Generate a system file based of slurm environment variables"""
        node_list = filter(
            lambda x: len(x) > 0, os.getenv("SLURM_JOB_NODELIST", "").split(",")
        )

        def make_node(i, ip):
            return dict(name=ip, ip=ip, port=22, user=getpass.getuser(), main=i == 0)

        system = dict(
            arch="cuda", nodes=[make_node(i, ip) for i, ip in enumerate(node_list)]
        )

        import yaml

        print(yaml.dump(system))

    def machine():
        """Display machine metadata.
        Used to generate metadata json to back populate archived run

        """
        from bson.json_util import dumps as to_json

        print(to_json(machine_metadata(), indent=2))

    def publish():
        """Publish an archived run to a database"""
        # URI to the database
        #   ex:
        #       - postgresql://user:password@hostname:27017/database
        #       - sqlite:///sqlite.db
        uri: str

        # Run folder to save
        folder: str

        # Json string of file to append to the meta dictionary
        meta: Option & str = None

        from .metrics.archive import publish_archived_run
        from .metrics.sqlalchemy import SQLAlchemy

        if meta is not None:
            with open(meta, "r") as file:
                meta = json.load(file)

        backend = SQLAlchemy(uri, meta_override=meta)
        publish_archived_run(backend, folder)

    def container():
        """Build a container image (might not work properly at the moment)."""

        # Configuration file
        # [positional]
        config_file: Option & str = None

        config = _get_multipack(config, return_config=True)
        config_file = XPath(config["defaults"]["config_file"])
        config_base = XPath(config["defaults"]["config_base"])
        benchmarks = config["benchmarks"]

        # The container type to create
        type: Option & str = None

        # Include the dataset in the image
        include_data: Option & bool = False

        # Optional path to copy build dir to, instead of building the image.
        # This directory must not exist and will be created.
        output_dir: Option & str = None

        # File in which to generate the SIF image (Singularity).
        # Defaults to milabench.sif.
        # [alias: -o]
        output_file: Option & str = None

        # Optional python version to use for the image, ignored for
        # conda-based benchmarks. Can be specified as any of
        # ('3', '3.9', '3.9.2')
        python_version: Option & str = "3.9"

        # Milabench source to clone from
        milabench: Option & str = "v2"

        # The tag for the generated container
        tag: Option & str = None

        if type not in ["docker", "singularity"]:
            sys.exit(f"Unsupported type {type}")

        with tempfile.TemporaryDirectory() as base:
            root = XPath(base)

            common_base = config_base

            # Figure out common base between the benchmark config and all
            # the benchmarks.
            for defn in benchmarks.values():
                pack = XPath(defn["definition"]).expanduser()
                while not pack.is_relative_to(common_base):
                    common_base = common_base.parent

            def _transfer(pth):
                dest = root / pth.relative_to(common_base)
                shutil.copytree(pth, dest, dirs_exist_ok=True)

            for defn in benchmarks.values():
                _transfer(XPath(defn["definition"]))

            _transfer(config_base)

            # We check all configs since they may not have all the same setting
            use_conda = any(
                defn["venv"]["type"] == "conda" for defn in benchmarks.values()
            )

            if "//" not in milabench:
                milabench = (
                    f"git+https://github.com/mila-iqia/milabench.git@{milabench}"
                )

            if type == "docker":
                if output_file is not None:
                    sys.exit("Error: --output-file only valid with Singularity")
                tag = tag or "milabench"
                with (root / "Dockerfile").open("w") as f:
                    f.write(
                        dockerfile_template(
                            milabench_req=milabench,
                            include_data=include_data,
                            use_conda=use_conda,
                            python_version=python_version,
                            config_file=config_file.relative_to(common_base),
                        )
                    )
                if output_dir:
                    root.copy(output_dir)
                else:
                    subprocess.check_call(["docker", "build", ".", "-t", tag], cwd=root)

            elif type == "singularity":
                if tag is not None:
                    sys.exit("Error: --tag only valid with Docker")
                output_file = output_file or "milabench.sif"

                with (root / "milabench.def").open("w") as f:
                    f.write(
                        singularitydef_template(
                            milabench_req=milabench,
                            include_data=include_data,
                            use_conda=use_conda,
                            python_version=python_version,
                            config_file=config_file.relative_to(common_base),
                        )
                    )
                if output_dir:
                    root.copy(output_dir)
                else:
                    user = os.environ["USER"]
                    filename = str(XPath(output_file).absolute())
                    singularity = subprocess.check_output(
                        ["which", "singularity"]
                    ).strip()
                    subprocess.check_call(
                        ["sudo", singularity, "build", filename, "milabench.def"],
                        cwd=root,
                    )
                    subprocess.check_call(["sudo", "chown", f"{user}:{user}", filename])


def dockerfile_template(
    milabench_req, include_data, use_conda, python_version, config_file
):
    conda_clean = "conda clean -a" if use_conda else "echo"
    return f"""
FROM { 'continuumio/miniconda3' if use_conda else f'python:{python_version}-slim' }

RUN apt-get update && apt-get install --no-install-suggests --no-install-recommends -y \
    git \
    wget \
    patch \
 && apt-get clean

RUN mkdir /bench && mkdir /base
ENV MILABENCH_BASE /base
# This is to signal to milabench to use that as fallback
ENV VIRTUAL_ENV /base/venv/_
ENV MILABENCH_CONFIG /bench/{ config_file }
ENV HEADLESS 1
WORKDIR /base

RUN echo '{ milabench_req }' > /version.txt

COPY / /bench

RUN pip install -U pip && \
    pip install -r /version.txt && \
    milabench install && \
    { conda_clean } && \
    pip cache purge

{ 'RUN milabench prepare' if include_data else '' }

CMD ["milabench", "run"]
"""


def singularitydef_template(
    milabench_req, include_data, use_conda, python_version, config_file
):
    conda_clean = "conda clean -a" if use_conda else "echo"
    return f"""\
BootStrap: docker
From: { 'continuumio/miniconda3' if use_conda else f'python:{python_version}-slim' }

%files
    . /bench

%environment
    export MILABENCH_BASE=/base
    export MILABENCH_CONFIG=/bench/{ config_file }
    export HEADLESS=1

%post
    export MILABENCH_BASE=/base
    export MILABENCH_CONFIG=/bench/{ config_file }
    export HEADLESS=1

    apt-get update && apt-get install --no-install-suggests --no-install-recommends -y git wget patch
    apt-get clean

    mkdir /base
    cd /bench

    echo '{ milabench_req }' > /version.txt
    pip install -U pip && \
    pip install -r /version.txt && \
    milabench install && \
    { conda_clean } && \
    pip cache purge
{ '    milabench prepare' if include_data else '' }

    chmod -R o+rwx /base /bench

%runscript
    milabench run
"""
