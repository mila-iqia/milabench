import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
from functools import partial

import yaml
from coleo import Option, config as configuration, default, run_cli, tooled

from .fs import XPath
from .log import simple_dash, simple_report
from .merge import self_merge
from .multi import MultiPackage
from .report import make_report
from .summary import aggregate, make_summary


def main():
    sys.path.insert(0, os.path.abspath(os.curdir))
    run_cli(Main)


def parse_config(config_file, base=None):
    config_file = XPath(config_file).absolute()
    config_base = config_file.parent
    with open(config_file) as cf:
        config = yaml.safe_load(cf)
    config.setdefault("defaults", {})
    config["defaults"]["config_base"] = str(config_base)
    config["defaults"]["config_file"] = str(config_file)

    if base is not None:
        config["defaults"]["dirs"]["base"] = base

    config = self_merge(config)

    for name, defn in config["benchmarks"].items():
        defn.setdefault("name", name)
        defn.setdefault("group", name)
        defn["tag"] = [defn["name"]]

        pack = XPath(defn["definition"]).expanduser()
        if not pack.is_absolute():
            pack = (XPath(defn["config_base"]) / pack).resolve()
            defn["definition"] = str(pack)

    return config


def get_pack(defn):
    pack = XPath(defn["definition"])
    pack_glb = runpy.run_path(str(pack / "benchfile.py"))
    pack_cls = pack_glb["__pack__"]
    pack_obj = pack_cls(defn)
    return pack_obj


@tooled
def _get_multipack(dev=False):
    # Configuration file
    # [positional: ?]
    config: Option & str = None

    if config is None:
        config = os.environ.get("MILABENCH_CONFIG", None)

    if config is None:
        sys.exit("Error: CONFIG argument not provided and no $MILABENCH_CONFIG")

    # Base path for code, venvs, data and runs
    base: Option & str = None

    # Whether to use the current environment
    use_current_env: Option & bool = False

    if dev:
        use_current_env = True

    # Packs to select
    select: Option & str = default("")

    # Packs to exclude
    exclude: Option & str = default("")

    if select:
        select = select.split(",")

    if exclude:
        exclude = exclude.split(",")

    if base is None:
        base = os.environ.get("MILABENCH_BASE", None)

    config = parse_config(config, base)

    if not config["defaults"]["dirs"].get("base", None):
        sys.exit("Error: Neither --base nor $MILABENCH_BASE are set.")

    objects = {}

    for name, defn in config["benchmarks"].items():
        group = defn.get("group", name)
        if select and name not in select and group not in select:
            continue
        if exclude and name in exclude or group in exclude:
            continue

        defn.setdefault("name", name)
        defn["tag"] = [defn["name"]]

        if use_current_env or defn["dirs"].get("venv", None) is None:
            venv = os.environ.get("CONDA_PREFIX", None)
            if venv is None:
                venv = os.environ.get("VIRTUAL_ENV", None)
            if venv is None:
                print("Could not find virtual environment", file=sys.stderr)
                sys.exit(1)
            defn["dirs"]["venv"] = venv

        def _format_path(pth):
            formatted = pth.format(**defn)
            xpth = XPath(formatted).expanduser()
            if formatted.startswith("."):
                xpth = xpth.absolute()
            return xpth

        dirs = {k: _format_path(v) for k, v in defn["dirs"].items()}
        dirs = {
            k: str(v if v.is_absolute() else dirs["base"] / v) for k, v in dirs.items()
        }
        defn["dirs"] = dirs
        objects[name] = get_pack(defn)

    return MultiPackage(objects)


def _read_reports(*runs):
    all_data = {}
    for folder in runs:
        for parent, _, filenames in os.walk(folder):
            for file in filenames:
                if not file.endswith(".json"):
                    continue
                pth = XPath(parent) / file
                with pth.open() as f:
                    lines = f.readlines()
                    try:
                        data = [json.loads(line) for line in lines]
                    except Exception as exc:
                        print(f"Could not parse {pth}")
                    all_data[str(pth)] = data
    return all_data


def _error_report(reports):
    out = {}
    for r, data in reports.items():
        (success,) = aggregate(data)["success"]
        if not success:
            out[r] = [line for line in data if "#stdout" in line or "#stderr" in line]
    return out


class Main:
    def run():
        # Name of the run
        run_name: Option = None

        # Dev mode (adds --sync, current venv, only one run, no logging)
        dev: Option & bool = False

        # Sync changes to the benchmark directory
        sync: Option & bool = False

        # Number of times to repeat the benchmark
        repeat: Option & int = 1

        mp = _get_multipack(dev=dev)

        if dev or sync:
            mp.do_install(dash=simple_dash, sync=True)

        if dev:
            assert repeat == 1
            mp.do_dev(dash=simple_dash)
        else:
            mp.do_run(
                repeat=repeat,
                dash=simple_dash,
                report=partial(simple_report, runname=run_name),
            )

    def prepare():
        # Dev mode (does install --sync, uses current venv)
        dev: Option & bool = False

        # Sync changes to the benchmark directory
        sync: Option & bool = False

        mp = _get_multipack(dev=dev)

        if dev or sync:
            mp.do_install(dash=simple_dash, sync=True)

        mp.do_prepare(dash=simple_dash)

    def install():
        # Force install
        force: Option & bool = False

        # Sync changes to the benchmark directory
        sync: Option & bool = False

        # Dev mode (adds --sync, use current venv)
        dev: Option & bool = False

        mp = _get_multipack(dev=dev)
        mp.do_install(dash=simple_dash, force=force, sync=sync)

    def summary():
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

    def report():
        # Runs directory
        # [action: append]
        runs: Option = []

        # Weights configuration file
        weights: Option & configuration = None

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

        make_report(
            summary,
            compare=compare,
            weights=weights,
            html=html,
            compare_gpus=compare_gpus,
            price=price,
            title=None,
            sources=runs,
            errdata=reports and _error_report(reports),
        )

    def container():
        # Configuration file
        # [positional]
        config_file: Option & str = None

        config = parse_config(config_file)
        config_file = XPath(config["defaults"]["config_file"])
        config_base = XPath(config["defaults"]["config_base"])
        benchmarks = config["benchmarks"]

        # The container type to create
        type: Option & str = None

        # Include the dataset in the image
        include_data: Option & bool = False

        # Optional path to copy build dir to, instead of building the image.
        # This directory must not exist and will be created.
        # [alias: -o]
        output_dir: Option & str = None

        # Optional python version to use for the image, ignored for
        # conda-based benchmarks. Can be specified as any of
        # ('3', '3.9', '3.9.2')
        python_version: Option & str = "3.9"

        # The tag for the generated container
        tag: Option & str = "milabench"

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

            if type == "docker":
                with (root / "Dockerfile").open("w") as f:
                    f.write(
                        dockerfile_template(
                            milabench_req="git+https://github.com/mila-iqia/milabench.git@v2",
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
                raise NotImplementedError(type)


def dockerfile_template(
    milabench_req, include_data, use_conda, python_version, config_file
):
    return f"""
FROM { 'continuumio/miniconda3' if use_conda else f'python:{python_version}-slim' }

RUN apt-get update && apt-get install --no-install-suggests --no-install-recommends -y \
    git \
    patch \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir /bench && mkdir /base
ENV MILABENCH_BASE /base
# This is to signal to milabench to use that as fallback
ENV VIRTUAL_ENV /base/venv
ENV MILABENCH_DEVREQS /version.txt
ENV MILABENCH_CONFIG /bench/{ config_file }
WORKDIR /base

RUN echo '{ milabench_req }' > /version.txt

COPY / /bench

RUN pip install -U pip && \
    pip install -r /version.txt && \
    milabench install && \
    rm -rf /root/.cache/pip

{ 'RUN milabench prepare' if include_data else '' }

CMD ["milabench", "run"]
"""
