import os
import runpy
import sys
from contextlib import contextmanager

from coleo import Option
from coleo import config as configuration
from coleo import default, run_cli, tooled
from ptera import probing

from .bench import make_runner
from .fs import XPath
from .merge import self_merge
from .multi import MultiPackage
from .utils import fetch, resolve


def cli():
    sys.path.insert(0, os.path.abspath(os.curdir))
    runner = run_cli(Main)
    if runner is not None:
        runner()


def get_pack(defn):
    pack = defn["definition"]
    pack_glb = runpy.run_path(pack)
    pack_cls = pack_glb["__pack__"]
    pack_obj = pack_cls(defn)
    return pack_obj


@tooled
def _get_multipack():
    # Configuration file
    # [positional]
    config: Option & configuration
    config = self_merge(config)

    dirs = config["dirs"]

    def _resolvedir(name):
        result = XPath(dirs.get(name, name)).expanduser()
        if result.is_absolute():
            return result
        else:
            base = XPath(dirs["base"]).expanduser()
            return base / result

    dirs["code"] = _resolvedir("code")
    dirs["venv"] = _resolvedir("venv")
    dirs["data"] = _resolvedir("data")
    dirs["runs"] = _resolvedir("runs")

    objects = {}

    for name, defn in config["benchmarks"].items():
        defn.setdefault("name", name)
        defn["tag"] = [defn["name"]]
        defn["dirs"] = {
            "code": str(dirs["code"] / name),
            "venv": str(dirs["venv"] / name),
            "data": str(dirs["data"]),
            "runs": str(dirs["runs"]),
        }
        objects[name] = get_pack(defn)

    return MultiPackage(objects)


def simple_bridge(*selectors):
    @contextmanager
    def bridge(runner, gv):
        with probing(*selectors) as prb:
            prb.give()
            yield

    return bridge


class Main:
    def run():
        # Instrumenting functions
        # [alias: -i]
        # [action: append]
        instrumenter: Option = default([])

        # Bridge
        # [alias: -b]
        # [action: append]
        bridge: Option = default(None)

        # Path to the script
        # [positional]
        script: Option

        # Arguments to the script
        # [positional: --]
        args: Option

        script, field, _ = resolve(script, "__main__")

        bridges = [simple_bridge(b) if ">" in b else fetch(b) for b in bridge]

        return make_runner(
            script=script,
            field=field,
            args=args,
            instruments=[fetch(inst) for inst in instrumenter],
            bridges=bridges,
        )

    def install():
        mp = _get_multipack()
        return mp.do_install

    def jobs():
        mp = _get_multipack()
        return mp.do_run

    def job():
        # Configuration for the job
        # [positional]
        config: Option & configuration

        bench = XPath(config["dirs"]["code"]) / "__bench__.py"
        if not bench.exists():
            sys.exit("Benchmark is not installed")

        pypath = sys.argv[0]
        if not pypath.startswith(config["dirs"]["venv"]):
            sys.exit("Not in venv")

        pack = get_pack(config)
        return pack.main
