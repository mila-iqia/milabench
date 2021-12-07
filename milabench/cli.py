import os
import runpy
import sys

from coleo import Option
from coleo import config as configuration
from coleo import default, run_cli, tooled

from .fs import XPath
from .merge import self_merge
from .multi import MultiPackage
from .runner import make_runner
from .utils import extract_instruments, fetch, resolve, simple_bridge


def main():
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


class Main:
    def run():
        # Instrumenting functions
        # [alias: -i]
        # [action: append]
        instrument: Option = default([])

        # Probe(s) to bridge data collection
        # [alias: -p]
        # [action: append]
        probe: Option = default([])

        # Configuration file
        # [alias: -c]
        config: Option & configuration = default({})

        # Path to the script
        # [positional]
        script: Option

        # Arguments to the script
        # [positional: --]
        args: Option

        script, field, _ = resolve(script, "__main__")

        instruments = extract_instruments(config)
        instruments += [fetch(inst) for inst in instrument]

        if probe:
            instruments.insert(0, simple_bridge(*probe))

        return make_runner(
            script=script,
            field=field,
            args=args,
            instruments=instruments,
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
