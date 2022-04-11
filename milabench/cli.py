import os
import runpy
import sys
from functools import partial
import tempfile
import subprocess
import shutil

from coleo import Option, config as configuration, default, run_cli, tooled, ConfigFile

from .fs import XPath
from .log import simple_dash, simple_report
from .merge import self_merge
from .multi import MultiPackage


def main():
    sys.path.insert(0, os.path.abspath(os.curdir))
    run_cli(Main)


def get_pack(defn):
    pack = XPath(defn["definition"]).expanduser()
    if not pack.is_absolute():
        pack = XPath(defn["config_base"]) / pack
        defn["definition"] = str(pack)
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

    config_base = str(XPath(config).parent.absolute())
    config_file = str(XPath(config).absolute())
    config = configuration(config)
    config["defaults"]["config_base"] = config_base
    config["defaults"]["config_file"] = config_file
    if base is not None:
        config["defaults"]["dirs"]["base"] = base
    elif os.environ.get("MILABENCH_BASE", None):
        config["defaults"]["dirs"]["base"] = os.environ["MILABENCH_BASE"]
    elif not config["defaults"]["dirs"].get("base", None):
        sys.exit("Error: Neither --base nor $MILABENCH_BASE are set.")
    config = self_merge(config)

    objects = {}

    for name, defn in config["benchmarks"].items():
        if select and name not in select:
            continue
        if exclude and name in exclude:
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


class Main:
    def run():
        # Name of the run
        run: Option = None

        # Dev mode (adds --sync, current venv, only one run, no logging)
        dev: Option & bool = False

        # Sync changes to the benchmark directory
        sync: Option & bool = False

        mp = _get_multipack(dev=dev)

        if dev or sync:
            mp.do_install(dash=simple_dash, sync=True)

        if dev:
            mp.do_dev(dash=simple_dash)
        else:
            mp.do_run(dash=simple_dash, report=partial(simple_report, runname=run))

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

    def container():
        # The container type to create
        type: Option & str = None

        # Include the dataset in the image
        include_data: Option & bool = False

        # Optional path to copy build dir to, instead of building the image.
        # This directory must not exist and will be created.
        output_dir: Option & str = None

        # Optional python version to use for the image, ignored for
        # conda-based benchmarks. Can be specified as any of
        # ('3', '3.9', '3.9.2')
        python_version: Option & str = "3.9"

        # The tag for the generated container
        tag: Option & str = "milabench"

        mp = _get_multipack(dev=False)

        if type not in ["docker", "singularity"]:
            sys.exit(f"Unsupported type {type}")

        with tempfile.TemporaryDirectory() as base:
            root = XPath(base)

            # To build containers all your files and directories must be
            # relative to config_base.
            config = next(iter(mp.packs.values())).config
            config_base = XPath(config["config_base"])
            config_file = XPath(config["config_file"])

            # We check all configs since they may not have all the same setting
            use_conda = any(pack.config['venv']['type'] == 'conda' for pack in mp.packs.values())

            shutil.copytree(config_base, root, dirs_exist_ok=True)
            config_file.copy(root / "bench.yaml")

            if type == "docker":
                with (root / "Dockerfile").open("w") as f:
                    f.write(
                        dockerfile_template(
                            milabench_req="git+https://github.com/mila-iqia/milabench.git@container",
                            include_data=include_data,
                            use_conda=use_conda,
                            python_version=python_version,
                        )
                    )
                if output_dir:
                    root.copy(output_dir)
                else:
                    subprocess.check_call(["docker", "build", ".", "-t", tag], cwd=root)
            elif type == "singularity":
                raise NotImplementedError(type)


def dockerfile_template(milabench_req, include_data, use_conda, python_version):
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
WORKDIR /base

RUN echo '{ milabench_req }' > /version.txt

COPY / /bench

RUN pip install -U pip && \
    pip install -r /version.txt && \
    milabench install /bench/bench.yaml && \
    rm -rf /root/.cache/pip

{ 'RUN milabench prepare /bench/bench.yaml' if include_data else '' }

CMD ["milabench", "run", "/bench/bench.yaml"]
"""
