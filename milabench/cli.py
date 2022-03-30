import os
import runpy
import sys
from functools import partial

from coleo import Option, config as configuration, default, run_cli, tooled

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
    pack_glb = runpy.run_path(pack / "benchfile.py")
    pack_cls = pack_glb["__pack__"]
    pack_obj = pack_cls(defn)
    return pack_obj


@tooled
def _get_multipack():
    # Configuration file
    # [positional]
    config: Option & str

    # Base path for code, venvs, data and runs
    base: Option & str = None

    # Whether to use the current environment
    use_current_env: Option & bool = False

    # Packs to select
    select: Option & str = default("")

    # Packs to exclude
    exclude: Option & str = default("")

    if select:
        select = select.split(",")

    if exclude:
        exclude = exclude.split(",")

    config_base = str(XPath(config).parent)
    config = configuration(config)
    config["defaults"]["config_base"] = config_base
    if base is not None:
        config["defaults"]["dirs"]["base"] = base
    elif "MILABENCH_BASE" in os.environ:
        config["defaults"]["dirs"]["base"] = os.environ["MILABENCH_BASE"]
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

        dirs = {
            k: XPath(v.format(**defn)).expanduser() for k, v in defn["dirs"].items()
        }
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

        mp = _get_multipack()
        mp.do_run(dash=simple_dash, report=partial(simple_report, runname=run))

    def prepare():
        mp = _get_multipack()
        mp.do_prepare(dash=simple_dash)

    def install():
        # Force install
        force: Option & bool = False

        mp = _get_multipack()
        mp.do_install(dash=simple_dash, force=force)
