from collections import defaultdict
from dataclasses import dataclass, field
import itertools
import json
import os
import random
import sys
from contextlib import contextmanager
from datetime import datetime

from giving import ObservableProxy

from milabench.utils import REAL_STDOUT

from .fs import XPath


@contextmanager
def simple_dash(gv):
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
            txt = str(data["#stdout"])
            if newline:
                line(">", txt, end="")
            else:
                pr(txt, end="")
            newline = txt.endswith("\n")
        elif ks == {"#stderr"}:
            txt = str(data["#stderr"])
            if newline:
                line(">!", txt, end="")
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
def simple_report(gv, rundir, runname=None):
    def _to_line(data):
        data = dict(data)
        data.pop("#run", None)
        data.pop("#pack", None)
        if set(data.keys()) == {"descr", "progress", "total"}:
            return ""
        return json.dumps(data) + "\n"

    now = str(datetime.today()).replace(" ", "_")

    if runname is None:
        bla = blabla()
        runname = f"{bla}.{now}"

    rundir = XPath(rundir) / runname
    if XPath(rundir).exists():
        print(f"Rundir {rundir} already exists", file=sys.stderr)
        sys.exit(1)
    os.makedirs(rundir, exist_ok=True)

    print(f"[BEGIN] Reports directory: {rundir}")

    grouped = gv.group_by(lambda data: tuple(data["#run"]["tag"]))

    @grouped.subscribe
    def _(stream):
        stream = ObservableProxy(stream)

        batches = stream.buffer_with_time(1.0).filter(lambda entries: len(entries) > 0)

        @batches.subscribe
        def __(entries):
            d0 = entries[0]
            tag = ".".join(d0["#run"]["tag"])
            entries = [_to_line(data) for data in entries]
            with open(rundir / f"{tag}.json", "a", encoding="utf8") as f:
                f.writelines(entries)

    yield

    print(f"[DONE] Reports directory: {rundir}")


@dataclass
class PackError:
    stderr: list[str] = field(default_factory=list)
    code: int = 0
    message: str = None


@contextmanager
def error_capture(gv):
    errors = defaultdict(PackError)

    @gv.subscribe
    def _(data):
        data = dict(data)
        run = data.pop("#run", None)
        pack = data.pop("#pack", None)

        tg = ".".join(run["tag"]) if run else pack.config["name"]

        ks = set(data.keys())
        error = errors[tg]

        if ks == {"#stderr"}:
            txt = str(data["#stderr"])
            error.stderr.append(txt)

        elif ks == {"#error"}:
            error.message = data

        elif ks == {"#end", "#return_code"}:
            error.code = data["#return_code"]

    yield errors
