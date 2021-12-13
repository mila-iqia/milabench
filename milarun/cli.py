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

REAL_STDOUT = sys.stdout


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

    # Packs to select
    select: Option & str = default("")

    # Packs to exclude
    # [option: --except]
    except_: Option & str = default("")

    if select:
        select = select.split(",")

    if except_:
        except_ = except_.split(",")

    objects = {}

    for name, defn in config["benchmarks"].items():
        if select and name not in select:
            continue
        if except_ and name in except_:
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


@contextmanager
def _simple_dash(gv):
    import rich
    from blessed import Terminal

    T = Terminal()

    colors = [T.cyan, T.magenta, T.yellow, T.red, T.green, T.blue]
    headers = {}
    newline = True
    rc = None

    @gv.subscribe
    def _(data):
        nonlocal newline, rc

        def pr(*args, **kwargs):
            print(*args, **kwargs, file=REAL_STDOUT)

        def line(*contents, end="\n"):
            pr(headers[tg], *contents, end=end)

        data = dict(data)
        run = data.pop("#run", None)
        pack = data.pop("#pack", None)
        ks = set(data.keys())
        tg = ".".join(run["tag"]) if run else pack.config["name"]
        if tg not in headers:
            headers[tg] = colors[len(headers) % len(colors)](T.bold(tg))

        if ks != {"#stdout"} and not newline:
            pr()

        if ks == {"#stdout"}:
            txt = data["#stdout"]
            if newline:
                line(">", txt, end="")
            else:
                pr(txt, end="")
            newline = txt.endswith("\n")
        elif ks == {"#start"}:
            line(
                T.bold_green("Start:"),
                T.green(str(datetime.fromtimestamp(data["#start"]))),
            )
        elif ks == {"#error"}:
            rc = "ERROR"
            line("", end="")
            rich.print(data, file=REAL_STDOUT)
        elif ks == {"#end", "#return_code"}:
            if rc is None:
                rc = data["#return_code"]
            if rc == 0:
                line(
                    T.bold_green("End:"),
                    T.green(str(datetime.fromtimestamp(data["#end"]))),
                )
            else:
                line(
                    T.bold_red(f"End ({rc}):"),
                    T.red(str(datetime.fromtimestamp(data["#end"]))),
                )
        else:
            line("", end="")
            rich.print(data, file=REAL_STDOUT)

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
