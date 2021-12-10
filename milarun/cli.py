import itertools
import json
import os
import random
import runpy
import sys
from contextlib import contextmanager
from datetime import datetime
from functools import partial

from coleo import Option
from coleo import config as configuration
from coleo import default, run_cli, tooled
from giving import ObservableProxy
from giving import operators as op

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

    objects = {}

    for name, defn in config["benchmarks"].items():
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


@contextmanager
def _simple_dash(gv):
    import rich
    from blessed import Terminal

    T = Terminal()

    colors = [T.cyan, T.magenta, T.yellow, T.red, T.green, T.blue]
    headers = {}
    newline = True

    def bold(txt, color=""):
        return T.bold(txt)

    @gv.subscribe
    def _(data):
        nonlocal newline

        def line(*contents, end="\n"):
            print(headers[tg], *contents, end=end)

        data = dict(data)
        run = data.pop("#run")
        pack = data.pop("#pack")
        ks = set(data.keys())
        tg = ".".join(run["tag"])
        if tg not in headers:
            headers[tg] = colors[len(headers) % len(colors)](bold(tg))

        if ks != {"#stdout"} and not newline:
            print()

        if ks == {"#stdout"}:
            txt = data["#stdout"]
            if newline:
                line(">", txt, end="")
            else:
                print(txt, end="")
            newline = txt.endswith("\n")
        elif ks == {"#start"}:
            line(bold("Start:"), datetime.fromtimestamp(data["#start"]))
        elif ks == {"#end", "#return_code"}:
            line(bold("End:"), datetime.fromtimestamp(data["#end"]))
            line(bold("Return code:"), data["#return_code"])
        else:
            rich.print(data)

    yield


vowels = list("aeiou")
consonants = list("bdfgjklmnprstvz")
syllables = ["".join(letters) for letters in itertools.product(consonants, vowels)]


def blabla(n=4):
    return "".join([random.choice(syllables) for _ in range(n)])


@contextmanager
def _simple_report(gv):
    def _to_line(data):
        data = dict(data)
        data.pop("#run")
        data.pop("#pack")
        return json.dumps(data) + "\n"

    now = str(datetime.today()).replace(" ", "_")
    bla = blabla()
    rundir = f"{bla}.{now}"

    grouped = gv.group_by(lambda data: tuple(data["#run"]["tag"]))

    @grouped.subscribe
    def _(stream):
        stream = ObservableProxy(stream)

        batches = stream.buffer_with_time(1.0).filter(lambda entries: len(entries) > 0)

        @batches.subscribe
        def __(entries):
            d0 = entries[0]
            tag = ".".join(d0["#run"]["tag"])
            base = d0["#pack"].dirs.runs / rundir
            os.makedirs(base, exist_ok=True)

            entries = [_to_line(data) for data in entries]
            with open(base / f"{tag}.json", "a", encoding="utf8") as f:
                f.writelines(entries)

    yield


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
        return partial(mp.do_install, dash=_simple_dash)

    def jobs():
        mp = _get_multipack()
        return partial(mp.do_run, dash=_simple_dash, report=_simple_report)

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
        return pack.do_main
