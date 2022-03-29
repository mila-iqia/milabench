import json
import os
import subprocess
import time
from copy import deepcopy

from giving import give, given
from voir.forward import MultiReader

from .merge import merge
from .utils import give_std

planning_methods = {}


def get_planning_method(name):
    name = name.replace("-", "_")
    return planning_methods[name]


def planning_method(f):
    planning_methods[f.__name__] = f


def clone_with(cfg, new_cfg):
    return merge(deepcopy(cfg), new_cfg)


@planning_method
def per_gpu(cfg):
    import GPUtil as gu

    gpus = gu.getGPUs()

    ids = [gpu.id for gpu in gpus] or [0]

    for gid in ids:
        gcfg = {
            "tag": [f"D{gid}"],
            "device": gid,
            "devices": [gid] if ids else [],
            "env": {"CUDA_VISIBLE_DEVICES": str(gid)},
        }
        yield clone_with(cfg, gcfg)


@planning_method
def njobs(cfg, n):
    for i in range(n):
        gcfg = {
            "tag": [f"{i}"],
            "job-number": i,
        }
        yield clone_with(cfg, gcfg)


def multiread(pipes):
    for (_, __, p) in pipes.values():
        os.set_blocking(p.stdout.fileno(), False)

    while pipes:
        for tag, (pack, run, proc) in list(pipes.items()):

            def _give(data):
                data["#pack"] = pack
                data["#run"] = run
                give(**data)

            while True:
                line = proc.stdout.readline()
                if not line:
                    ret = proc.poll()
                    if ret is not None:
                        # We do not read a line and the process is over
                        del pipes[tag]
                        _give({"#end": time.time(), "#return_code": ret})
                    break
                line = line.decode("utf8")
                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        data = {"#stdout": data}
                except json.JSONDecodeError:
                    data = {"#stdout": line}
                _give(data)
        time.sleep(0.1)


def _assemble_options(options):
    args = []
    for k, v in options.items():
        if v is None:
            continue
        if v is True:
            args.append(k)
        elif v is False:
            raise ValueError()
        else:
            args.append(k)
            args.append(",".join(map(str, v)) if isinstance(v, list) else str(v))
    return args


class MultiPackage:
    def __init__(self, packs):
        self.packs = packs
        (self.rundir,) = {p.dirs.runs for p in packs.values()}

    def do_install(self, dash, force=False):
        with given() as gv, dash(gv), give_std():
            for pack in self.packs.values():
                with give.inherit(**{"#pack": pack}):
                    pack.do_install(force=force)

    def do_prepare(self, dash):
        with given() as gv, dash(gv), give_std():
            for pack in self.packs.values():
                with give.inherit(**{"#pack": pack}):
                    mr = MultiReader()
                    process = pack.prepare()
                    if isinstance(process, subprocess.Popen):
                        mr.add_process(process, info={"#pack": pack})
                        for _ in mr:
                            time.sleep(0.1)

    def do_run(self, dash, report):
        with given() as gv, dash(gv), report(gv, self.rundir):
            for pack in self.packs.values():
                cfg = pack.config
                plan = deepcopy(cfg["plan"])
                method = get_planning_method(plan.pop("method"))
                mr = MultiReader()
                for run in method(cfg, **plan):
                    give(**{"#pack": pack, "#run": run, "#start": time.time()})
                    voirargs = _assemble_options(run.get("voir", {}))
                    args = _assemble_options(run.get("argv", {}))
                    env = run.get("env", {})

                    process = pack.launch(args=args, voirargs=voirargs, env=env)
                    mr.add_process(process, info={"#pack": pack, "#run": run})

                for _ in mr:
                    time.sleep(0.1)
