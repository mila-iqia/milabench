import os
import runpy
import sys

from coleo import Option, config as configuration, default, run_cli, tooled

from .fs import XPath
from .log import simple_dash, simple_report
from .merge import self_merge
from .multi import MultiPackage


def main():
    sys.path.insert(0, os.path.abspath(os.curdir))
    run_cli(Main)


def get_pack(defn):
    pack = defn["definition"]
    pack_glb = runpy.run_path(XPath(pack) / "benchfile.py")
    pack_cls = pack_glb["__pack__"]
    pack_obj = pack_cls(defn)
    return pack_obj


@tooled
def _get_multipack():
    # Configuration file
    # [positional]
    config: Option & configuration
    config = self_merge(config)

    # Packs to select
    select: Option & str = default("")

    # Packs to exclude
    exclude: Option & str = default("")

    if select:
        select = select.split(",")

    if exclude:
        exclude = exclude.split(",")

    objects = {}

    for name, defn in config["benchmarks"].items():
        if select and name not in select:
            continue
        if exclude and name in exclude:
            continue

        defn.setdefault("name", name)
        defn["tag"] = [defn["name"]]
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
        mp = _get_multipack()
        mp.do_run(dash=simple_dash, report=simple_report)

    def prepare():
        mp = _get_multipack()
        mp.do_prepare(dash=simple_dash)

    def install():
        mp = _get_multipack()
        mp.do_install(dash=simple_dash)
